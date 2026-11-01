"""Potential energy and forces, written from scratch in JAX.

Functional form (AMBER-style, as used by ff14SB):

    E = sum_bonds   k/2 (r - r0)^2
      + sum_angles  k/2 (theta - theta0)^2
      + sum_torsions k (1 + cos(n phi - phase))
      + sum_{i<j, not excluded} [ 4 eps_ij ((s_ij/r)^12 - (s_ij/r)^6) + k_e q_i q_j / r ]
      + sum_{1-4 pairs} (same, with scaled parameters)
      + E_GB  (generalized Born, OBC2 Born radii)          polar solvation
      + E_ACE (Schaefer-style surface term)               nonpolar solvation

Forces are exact gradients obtained by automatic differentiation, so every
energy term automatically has a consistent force. Pair terms are evaluated
densely (all pairs, no cutoff), which is exact for implicit solvent and fast
for systems up to a few thousand atoms. Neighbor lists are on the roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..constants import (
    ACE_PROBE_RADIUS,
    OBC2_ALPHA,
    OBC2_BETA,
    OBC2_GAMMA,
    OBC_DIELECTRIC_OFFSET,
    ONE_4PI_EPS0,
)
from .system import MolecularSystem

jax.config.update("jax_enable_x64", True)

TERM_NAMES = (
    "bond", "angle", "torsion", "lj", "coulomb", "lj14", "coulomb14", "gb_polar", "gb_nonpolar",
)


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def _norm(v):
    return jnp.sqrt(jnp.sum(v * v, axis=-1))


def bond_lengths(x, idx):
    return _norm(x[idx[:, 1]] - x[idx[:, 0]])


def angle_values(x, idx):
    u = x[idx[:, 0]] - x[idx[:, 1]]
    v = x[idx[:, 2]] - x[idx[:, 1]]
    # atan2 form is well conditioned near 0 and pi (arccos is not)
    return jnp.arctan2(_norm(jnp.cross(u, v)), jnp.sum(u * v, axis=-1))


def dihedral_values(x, idx):
    """IUPAC dihedral angle in (-pi, pi] for atoms (a, b, c, d)."""
    b1 = x[idx[:, 1]] - x[idx[:, 0]]
    b2 = x[idx[:, 2]] - x[idx[:, 1]]
    b3 = x[idx[:, 3]] - x[idx[:, 2]]
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    y = _norm(b2) * jnp.sum(b1 * n2, axis=-1)
    xx = jnp.sum(n1 * n2, axis=-1)
    return jnp.arctan2(y, xx)


# ----------------------------------------------------------------------------
# energy model
# ----------------------------------------------------------------------------
@dataclass
class EnergyModel:
    """Compiled energy/force functions for one MolecularSystem.

    Attributes
    ----------
    energy(x) -> scalar total potential energy (kJ/mol), x in nm, shape (N, 3)
    terms(x)  -> dict of per-term energies
    energy_and_forces(x) -> (E, F) with F = -dE/dx (kJ/mol/nm)
    """

    system: MolecularSystem
    dtype: object
    energy: Callable
    terms: Callable
    energy_and_forces: Callable
    born_radii: Callable | None


def build_energy_model(system: MolecularSystem, dtype=jnp.float64, use_gb: bool | None = None) -> EnergyModel:
    if use_gb is None:
        use_gb = system.has_gb
    N = system.n_atoms
    def f(a):
        return jnp.asarray(np.asarray(a, dtype=np.float64), dtype=dtype)

    def i_(a, width):
        return jnp.asarray(np.asarray(a, dtype=np.int32).reshape(-1, width))

    # ---- bonded parameters -------------------------------------------------
    b_idx, b_k, b_r0 = i_(system.bonds, 2), f(system.bond_k), f(system.bond_r0)
    a_idx, a_k, a_t0 = i_(system.angles, 3), f(system.angle_k), f(system.angle_t0)
    t_idx, t_k, t_ph = i_(system.torsions, 4), f(system.torsion_k), f(system.torsion_phase)
    t_n = f(system.torsion_n)

    # ---- nonbonded: dense pair matrices with exclusions folded in ----------
    q = np.asarray(system.charges, dtype=np.float64)
    sig = np.asarray(system.sigma, dtype=np.float64)
    eps = np.asarray(system.epsilon, dtype=np.float64)
    pair_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    exc = np.asarray(system.exceptions, dtype=np.int64).reshape(-1, 2)
    if len(exc):
        pair_mask[exc[:, 0], exc[:, 1]] = False
        pair_mask[exc[:, 1], exc[:, 0]] = False
    sig_ij = 0.5 * (sig[:, None] + sig[None, :])
    eps4_ij = np.where(pair_mask, 4.0 * np.sqrt(eps[:, None] * eps[None, :]), 0.0)
    kqq_ij = np.where(pair_mask, ONE_4PI_EPS0 * q[:, None] * q[None, :], 0.0)
    pair_mask_j = jnp.asarray(pair_mask)
    sig2_ij_j = f(sig_ij**2)
    eps4_ij_j, kqq_ij_j = f(eps4_ij), f(kqq_ij)

    # 1-4 (and any other nonzero) exceptions
    exc_qq = np.asarray(system.exc_qq, dtype=np.float64)
    exc_eps = np.asarray(system.exc_eps, dtype=np.float64)
    keep = (np.abs(exc_qq) > 0) | (np.abs(exc_eps) > 0)
    e_idx = jnp.asarray(exc[keep].astype(np.int32).reshape(-1, 2))
    e_kqq = f(ONE_4PI_EPS0 * exc_qq[keep])
    e_eps4 = f(4.0 * exc_eps[keep])
    e_sig2 = f(np.asarray(system.exc_sigma, dtype=np.float64)[keep] ** 2)

    # ---- GB-OBC2 -------------------------------------------------------------
    if use_gb:
        if not system.has_gb:
            raise ValueError("system has no GB parameters")
        rad = np.asarray(system.gb_radius, dtype=np.float64)
        or_ = rad - OBC_DIELECTRIC_OFFSET              # offset radius
        sr_ = np.asarray(system.gb_scale, dtype=np.float64) * or_   # scaled (descreening) radius
        rad_j, or_j, sr_j = f(rad), f(or_), f(sr_)
        offdiag = jnp.asarray(~np.eye(N, dtype=bool))
        engulfed_on = bool(system.gb_engulfed_correction)
        gb_pre = -system.gb_coulomb_constant * (1.0 / system.solute_dielectric - 1.0 / system.solvent_dielectric)
        qq_full = f(gb_pre * q[:, None] * q[None, :])
        ace_pref = f(4.0 * np.pi * system.surface_area_energy * (rad + ACE_PROBE_RADIUS) ** 2)
    one = jnp.asarray(1.0, dtype=dtype)

    def _born_radii(r2):
        r = jnp.sqrt(jnp.where(offdiag, r2, one))
        U = r + sr_j[None, :]
        cond = offdiag & (or_j[:, None] < U)
        L = jnp.maximum(or_j[:, None], jnp.abs(r - sr_j[None, :]))
        Ls, Us, rs = jnp.where(cond, L, one), jnp.where(cond, U, one), jnp.where(cond, r, one)
        iL, iU = 1.0 / Ls, 1.0 / Us
        term = (
            iL - iU
            + 0.25 * rs * (iU * iU - iL * iL)
            + 0.5 / rs * jnp.log(iU / iL)
            + 0.25 * (sr_j[None, :] ** 2) / rs * (iL * iL - iU * iU)
        )
        if engulfed_on:  # HCT correction when atom i sits entirely inside j's sphere
            engulfed = cond & (or_j[:, None] < (sr_j[None, :] - r))
            term = term + jnp.where(engulfed, 2.0 * (1.0 / or_j[:, None] - iL), 0.0)
        I = jnp.sum(jnp.where(cond, term, 0.0), axis=1)
        psi = 0.5 * or_j * I
        th = jnp.tanh(OBC2_ALPHA * psi - OBC2_BETA * psi**2 + OBC2_GAMMA * psi**3)
        return 1.0 / (1.0 / or_j - th / rad_j)

    def _terms(x):
        x = x.astype(dtype)
        out = {}
        # bonded
        if b_idx.shape[0]:
            out["bond"] = jnp.sum(0.5 * b_k * (bond_lengths(x, b_idx) - b_r0) ** 2)
        else:
            out["bond"] = jnp.zeros((), dtype)
        if a_idx.shape[0]:
            out["angle"] = jnp.sum(0.5 * a_k * (angle_values(x, a_idx) - a_t0) ** 2)
        else:
            out["angle"] = jnp.zeros((), dtype)
        if t_idx.shape[0]:
            out["torsion"] = jnp.sum(t_k * (1.0 + jnp.cos(t_n * dihedral_values(x, t_idx) - t_ph)))
        else:
            out["torsion"] = jnp.zeros((), dtype)
        # nonbonded (dense, exclusions folded into the parameter matrices)
        d = x[:, None, :] - x[None, :, :]
        r2 = jnp.sum(d * d, axis=-1)
        r2m = jnp.where(pair_mask_j, r2, one)
        inv_r2 = 1.0 / r2m
        s6 = (sig2_ij_j * inv_r2) ** 3
        out["lj"] = jnp.sum(eps4_ij_j * (s6 * s6 - s6))
        out["coulomb"] = jnp.sum(kqq_ij_j * jnp.sqrt(inv_r2))
        if e_idx.shape[0]:
            de = x[e_idx[:, 0]] - x[e_idx[:, 1]]
            er2 = jnp.sum(de * de, axis=-1)
            es6 = (e_sig2 / er2) ** 3
            out["lj14"] = jnp.sum(e_eps4 * (es6 * es6 - es6))
            out["coulomb14"] = jnp.sum(e_kqq / jnp.sqrt(er2))
        else:
            out["lj14"] = out["coulomb14"] = jnp.zeros((), dtype)
        # implicit solvent
        if use_gb:
            R = _born_radii(r2)
            RR = R[:, None] * R[None, :]
            fgb = jnp.sqrt(r2 + RR * jnp.exp(-r2 / (4.0 * RR)))  # diagonal -> R_i (self energy)
            out["gb_polar"] = 0.5 * jnp.sum(qq_full / fgb)
            out["gb_nonpolar"] = jnp.sum(ace_pref * (rad_j / R) ** 6)
        else:
            out["gb_polar"] = out["gb_nonpolar"] = jnp.zeros((), dtype)
        return out

    def _energy(x):
        t = _terms(x)
        return sum(t[k] for k in TERM_NAMES)

    vg = jax.value_and_grad(_energy)

    def _energy_and_forces(x):
        e, g = vg(x)
        return e, -g

    born = None
    if use_gb:
        def born(x):  # noqa: E306
            x = x.astype(dtype)
            d = x[:, None, :] - x[None, :, :]
            return _born_radii(jnp.sum(d * d, axis=-1))
        born = jax.jit(born)

    return EnergyModel(
        system=system,
        dtype=dtype,
        energy=jax.jit(_energy),
        terms=jax.jit(_terms),
        energy_and_forces=jax.jit(_energy_and_forces),
        born_radii=born,
    )
