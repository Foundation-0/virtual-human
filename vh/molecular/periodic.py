"""Explicit solvent: periodic boundaries and exact long-range electrostatics.

Every water molecule is present; the simulation box repeats infinitely in all
directions. Coulomb interactions are long-ranged (1/r), so a cutoff is not
acceptable -- they are summed over all periodic images with Ewald summation,
split into three convergent pieces:

    E_coul = E_real (short range, erfc(alpha r)/r, cut at r_c)
           + E_recip (smooth long range, summed in Fourier space)
           + E_self (removes each charge's interaction with its own Gaussian)
           + E_excl (removes reciprocal-space interactions of bonded pairs)

Two reciprocal-space solvers:
  * "ewald": the exact structure-factor sum over reciprocal lattice vectors
    (O(N^1.5)); the reference.
  * "pme":   smooth particle-mesh Ewald (Essmann et al. 1995): charges spread on
    a grid with cardinal B-splines and convolved by FFT (O(N log N)).

Lennard-Jones is truncated at r_c with the standard analytic long-range
dispersion correction. Forces are exact gradients by autodiff.

Alchemistry: a set of solute atoms can be decoupled from everything else with
lambda_elec (scales solute charges) and lambda_vdw (soft-core LJ between solute
and environment, Beutler et al. 1994). dU/dlambda is exact via autodiff, which
is what thermodynamic integration needs.

Units: nm, ps, kJ/mol, e. Orthorhombic boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf, erfc

from ..constants import ONE_4PI_EPS0
from .energy import angle_values, bond_lengths, dihedral_values
from .system import MolecularSystem

jax.config.update("jax_enable_x64", True)
SQRT_PI = float(np.sqrt(np.pi))


def ewald_alpha(cutoff: float, tol: float) -> float:
    """Splitting parameter so the real-space error at the cutoff is ~tol (OpenMM convention)."""
    return float(np.sqrt(-np.log(2.0 * tol)) / cutoff)


def pme_grid_size(box, alpha, tol, order=5):
    """Grid dimensions as OpenMM chooses them for PME (minimal FFT-friendly sizes)."""
    out = []
    for L in box:
        n = int(np.ceil(2.0 * alpha * L / (3.0 * tol ** 0.2)))
        out.append(max(n, 2 * order))
    return tuple(_fft_friendly(n) for n in out)


def _fft_friendly(n):
    while True:
        m = n
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


@dataclass
class PeriodicSpec:
    cutoff: float = 0.9              # nm, real-space cutoff for Coulomb and LJ
    tol: float = 5e-6                # Ewald error tolerance (sets alpha)
    method: str = "pme"              # "ewald" or "pme"
    kmax: tuple | None = None        # Ewald: max |n| per dimension (None = choose from tol)
    pme_grid: tuple | None = None    # PME: grid points per dimension (None = choose from tol)
    pme_order: int = 5               # PME B-spline order (OpenMM default)
    dispersion_correction: bool = True
    softcore_alpha: float = 0.5      # Beutler soft-core parameter

    @property
    def alpha(self):
        return ewald_alpha(self.cutoff, self.tol)


# ----------------------------------------------------------------------------
# B-splines for PME
# ----------------------------------------------------------------------------
def _cardinal_bspline(x, n):
    """Cardinal B-spline M_n(x), support (0, n)."""
    if n == 2:
        return jnp.where((x >= 0.0) & (x <= 2.0), 1.0 - jnp.abs(x - 1.0), 0.0)
    return (x * _cardinal_bspline(x, n - 1) + (n - x) * _cardinal_bspline(x - 1.0, n - 1)) / (n - 1)


def _bspline_moduli(K: int, order: int) -> np.ndarray:
    """|b(m)|^2 for m = 0..K-1 (Essmann eq. 4.4), with the standard fix where the
    Euler spline denominator vanishes."""
    k = np.arange(order - 1)
    Mk = np.array([float(_cardinal_bspline(jnp.asarray(kk + 1.0), order)) for kk in k])
    m = np.arange(K)
    den = np.array([np.abs(np.sum(Mk * np.exp(2j * np.pi * mm * k / K))) ** 2 for mm in m])
    mod = np.zeros(K)
    small = den < 1e-7
    mod[~small] = 1.0 / den[~small]
    for i in np.where(small)[0]:     # average of neighbors (Essmann's fix)
        mod[i] = 0.5 * (mod[(i - 1) % K] + mod[(i + 1) % K])
    return mod


def _fft_freqs(K):
    return np.fft.fftfreq(K, d=1.0 / K)   # integers m in FFT order


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
@dataclass
class PeriodicModel:
    """Compiled periodic energy. Every function takes a neighbor list `nbr` for the
    real-space Coulomb sum: `model.full_nbr` (all pairs, exact, slower) or one from
    `model.neighbor_list(x, box)` (pairs within cutoff + skin; exact as long as no
    atom moves more than skin/2 before the list is rebuilt)."""

    system: MolecularSystem
    spec: PeriodicSpec
    energy: Callable            # (x, box, lam_elec, lam_vdw, nbr) -> E
    terms: Callable             # -> dict
    energy_and_forces: Callable  # (x, box, lam_elec, lam_vdw, nbr) -> (E, F)
    dudl: Callable              # (x, box, lam_elec, lam_vdw, nbr) -> (dU/dlam_elec, dU/dlam_vdw)
    solute: np.ndarray
    full_nbr: tuple
    _build_nbr: Callable
    skin: float = 0.12
    capacity: int = 0

    def neighbor_list(self, x, box):
        """Pairs within cutoff+skin, padded to a fixed capacity (grown if needed)."""
        if self.capacity == 0:
            n = int(self._count(x, box))
            self.capacity = int(n * 1.12) + 256       # liquid density is nearly constant
        ni, nj, valid, n = self._build_nbr(jnp.asarray(x), jnp.asarray(box), self.capacity)
        if int(n) > self.capacity:                 # overflow: grow and rebuild (recompiles once)
            self.capacity = int(int(n) * 1.12) + 256
            ni, nj, valid, n = self._build_nbr(jnp.asarray(x), jnp.asarray(box), self.capacity)
        return (ni, nj, valid)

    def _count(self, x, box):
        ci, cj, _ = self.full_nbr
        d = np.asarray(x)[np.asarray(ci)] - np.asarray(x)[np.asarray(cj)]
        d -= np.asarray(box) * np.round(d / np.asarray(box))
        return np.sum(np.sum(d * d, axis=1) < (self.spec.cutoff + self.skin) ** 2)

    # convenience wrappers using the exact full pair list
    def energy_exact(self, x, box, lam_elec=1.0, lam_vdw=1.0):
        return self.energy(x, box, lam_elec, lam_vdw, self.full_nbr)

    def ef_exact(self, x, box, lam_elec=1.0, lam_vdw=1.0):
        return self.energy_and_forces(x, box, lam_elec, lam_vdw, self.full_nbr)


def build_periodic_model(system: MolecularSystem, box0, spec: PeriodicSpec | None = None,
                         solute=None) -> PeriodicModel:
    """Compile energy/force functions for a periodic system.

    box0 is used only to size the reciprocal-space grid / k-vector set; the
    returned functions take the (possibly changing) box as an argument.
    solute: atom indices that the lambda parameters decouple (None = no alchemy).
    """
    spec = spec or PeriodicSpec()
    N = system.n_atoms
    box0 = np.asarray(box0, dtype=np.float64)
    rc, alpha = spec.cutoff, spec.alpha
    if rc > 0.5 * box0.min() + 1e-9:
        raise ValueError("cutoff exceeds half the box (minimum image breaks)")

    q = np.asarray(system.charges, dtype=np.float64)
    sig = np.asarray(system.sigma, dtype=np.float64)
    eps = np.asarray(system.epsilon, dtype=np.float64)

    solute = np.zeros(0, dtype=np.int64) if solute is None else np.asarray(solute, dtype=np.int64)
    is_sol = np.zeros(N, dtype=bool)
    is_sol[solute] = True
    sol_j = jnp.asarray(is_sol)

    # ---- flat pair lists (i<j, exclusions removed) ----------------------------
    # Coulomb runs over every non-excluded pair; Lennard-Jones only over pairs
    # where both atoms carry LJ (e.g. O-O in rigid water: ~10x fewer), split
    # into plain pairs and solute-environment soft-core pairs.
    pair = np.triu(np.ones((N, N), dtype=bool), k=1)
    exc = np.asarray(system.exceptions, dtype=np.int64).reshape(-1, 2)
    pair[exc[:, 0], exc[:, 1]] = False
    pair[exc[:, 1], exc[:, 0]] = False
    pi_np, pj_np = np.nonzero(pair)
    cP_i, cP_j = jnp.asarray(pi_np.astype(np.int32)), jnp.asarray(pj_np.astype(np.int32))
    has_lj = eps > 0
    lj_mask = has_lj[pi_np] & has_lj[pj_np]
    cross_mask = is_sol[pi_np] ^ is_sol[pj_np]
    def _lj_list(sel):
        a, b = pi_np[sel], pj_np[sel]
        s = 0.5 * (sig[a] + sig[b])
        return (jnp.asarray(a.astype(np.int32)), jnp.asarray(b.astype(np.int32)),
                jnp.asarray(s * s), jnp.asarray(np.sqrt(eps[a] * eps[b])))
    L_i, L_j, L_s2, L_e = _lj_list(lj_mask & ~cross_mask)      # plain LJ
    S_i, S_j, S_s2, S_e = _lj_list(lj_mask & cross_mask)       # soft-core LJ

    # exceptions: remove their reciprocal-space part, add their own params
    e_i, e_j = jnp.asarray(exc[:, 0]), jnp.asarray(exc[:, 1])
    exc_qq = jnp.asarray(np.asarray(system.exc_qq, dtype=np.float64))
    exc_sig = jnp.asarray(np.asarray(system.exc_sigma, dtype=np.float64))
    exc_eps = jnp.asarray(np.asarray(system.exc_eps, dtype=np.float64))
    if np.any(is_sol[exc[:, 0]] != is_sol[exc[:, 1]]):
        raise ValueError("bonded exceptions between solute and environment are not supported")

    # ---- dispersion correction: OpenMM convention (all particles, average over
    # the N(N+1)/2 unordered pairs including self pairs) --------------------
    def disp_coeff(mask):
        idx = np.where(mask)[0]
        if not spec.dispersion_correction or len(idx) == 0:
            return 0.0
        classes = {}
        for i in idx:
            key = (round(float(sig[i]), 12), round(float(eps[i]), 12))
            classes[key] = classes.get(key, 0) + 1
        keys = list(classes)
        s1 = s2 = 0.0
        for a in range(len(keys)):
            for b in range(a, len(keys)):
                (sa, ea), (sb, eb) = keys[a], keys[b]
                na, nb = classes[keys[a]], classes[keys[b]]
                count = na * (na + 1) / 2 if a == b else na * nb
                s_ = 0.5 * (sa + sb)
                e_ = np.sqrt(ea * eb)
                s1 += count * e_ * s_**12
                s2 += count * e_ * s_**6
        n = len(idx)
        n_int = n * (n + 1) / 2
        return 8.0 * n * n * np.pi * (s1 / n_int / (9 * rc**9) - s2 / n_int / (3 * rc**3))

    disp_full = disp_coeff(np.ones(N, dtype=bool))
    disp_env = disp_coeff(~is_sol)        # tail with the solute's LJ fully off
    # solute's share of the tail, scaled linearly with lambda_vdw
    disp_sol = disp_full - disp_env

    # ---- bonded (for solutes with internal flexibility) --------------------
    b_idx = jnp.asarray(np.asarray(system.bonds, dtype=np.int32).reshape(-1, 2))
    b_k, b_r0 = jnp.asarray(system.bond_k), jnp.asarray(system.bond_r0)
    a_idx = jnp.asarray(np.asarray(system.angles, dtype=np.int32).reshape(-1, 3))
    a_k, a_t0 = jnp.asarray(system.angle_k), jnp.asarray(system.angle_t0)
    t_idx = jnp.asarray(np.asarray(system.torsions, dtype=np.int32).reshape(-1, 4))
    t_k, t_n, t_ph = jnp.asarray(system.torsion_k), jnp.asarray(np.asarray(system.torsion_n, float)), jnp.asarray(system.torsion_phase)

    q_j = jnp.asarray(q)
    one = jnp.asarray(1.0)
    sc_a = spec.softcore_alpha

    # ---- reciprocal space ---------------------------------------------------
    if spec.method == "ewald":
        if spec.kmax is None:
            kcut = 2.0 * alpha * np.sqrt(-np.log(spec.tol))       # |k| beyond which terms < tol
            kmax = tuple(int(np.ceil(kcut * L / (2 * np.pi))) for L in box0)
        else:
            kmax = tuple(spec.kmax)
        g = [np.arange(-k, k + 1) for k in kmax]
        n = np.array(np.meshgrid(*g, indexing="ij")).reshape(3, -1).T
        half = (n[:, 0] > 0) | ((n[:, 0] == 0) & (n[:, 1] > 0)) | ((n[:, 0] == 0) & (n[:, 1] == 0) & (n[:, 2] > 0))
        nvec = jnp.asarray(n[half].astype(np.float64))

        def recip(x, box, qe):
            k = 2.0 * jnp.pi * nvec / box                        # (Nk, 3)
            k2 = jnp.sum(k * k, axis=1)
            phase = k @ x.T                                       # (Nk, N)
            c = jnp.cos(phase) @ qe
            s = jnp.sin(phase) @ qe
            V = box[0] * box[1] * box[2]
            return ONE_4PI_EPS0 * (4.0 * jnp.pi / V) * jnp.sum(jnp.exp(-k2 / (4 * alpha**2)) / k2 * (c * c + s * s))
    elif spec.method == "pme":
        order = spec.pme_order
        K = spec.pme_grid or pme_grid_size(box0, alpha, spec.tol, order)
        K = tuple(int(k) for k in K)
        Kj = jnp.asarray(np.array(K, dtype=np.float64))
        mods = [jnp.asarray(_bspline_moduli(k, order)) for k in K]
        Bmod = mods[0][:, None, None] * mods[1][None, :, None] * mods[2][None, None, :]
        mfreq = [jnp.asarray(_fft_freqs(k)) for k in K]
        offs = jnp.arange(order)

        def recip(x, box, qe):
            u = x / box * Kj                                      # grid units
            fl = jnp.floor(jax.lax.stop_gradient(u))
            w = u - fl                                            # in [0,1)
            # weights for grid points fl - j, j = 0..order-1 : M_p(w + j)
            W = _cardinal_bspline(w[:, :, None] + offs[None, None, :], order)   # (N, 3, order)
            idx = (fl[:, :, None].astype(jnp.int32) - offs[None, None, :]) % jnp.asarray(K)[None, :, None]
            wx, wy, wz = W[:, 0, :], W[:, 1, :], W[:, 2, :]
            ix, iy, iz = idx[:, 0, :], idx[:, 1, :], idx[:, 2, :]
            val = qe[:, None, None, None] * wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
            IX = jnp.broadcast_to(ix[:, :, None, None], val.shape)
            IY = jnp.broadcast_to(iy[:, None, :, None], val.shape)
            IZ = jnp.broadcast_to(iz[:, None, None, :], val.shape)
            Q = jnp.zeros(K).at[IX.ravel(), IY.ravel(), IZ.ravel()].add(val.ravel())
            F = jnp.fft.fftn(Q)
            mx = mfreq[0][:, None, None] / box[0]
            my = mfreq[1][None, :, None] / box[1]
            mz = mfreq[2][None, None, :] / box[2]
            m2 = mx * mx + my * my + mz * mz
            m2s = jnp.where(m2 > 0, m2, 1.0)
            V = box[0] * box[1] * box[2]
            G = jnp.where(m2 > 0, jnp.exp(-jnp.pi**2 * m2s / alpha**2) / m2s, 0.0)
            return ONE_4PI_EPS0 / (2.0 * jnp.pi * V) * jnp.sum(G * Bmod * (F.real**2 + F.imag**2))
    else:
        raise ValueError(spec.method)

    # ---- assembled energy -----------------------------------------------------
    def _terms(x, box, lam_elec, lam_vdw, nbr):
        x = jnp.asarray(x)
        box = jnp.asarray(box)
        qe = q_j * jnp.where(sol_j, lam_elec, 1.0)
        out = {}

        def mind(i, j):                       # minimum-image squared distance
            d = x[i] - x[j]
            d = d - box * jnp.round(d / box)
            return jnp.sum(d * d, axis=-1)

        # real-space Coulomb over the neighbor list (or the full pair list)
        ni, nj, valid = nbr
        r2 = mind(ni, nj)
        inside = valid & (r2 < rc * rc)
        r = jnp.sqrt(jnp.where(inside, r2, one))
        out["coul_real"] = ONE_4PI_EPS0 * jnp.sum(jnp.where(inside, qe[ni] * qe[nj] * erfc(alpha * r) / r, 0.0))
        # plain Lennard-Jones (pairs that both carry LJ)
        r2 = mind(L_i, L_j)
        inside = r2 < rc * rc
        s6 = (L_s2 / jnp.where(inside, r2, one)) ** 3
        lj = jnp.sum(jnp.where(inside, 4.0 * L_e * (s6 * s6 - s6), 0.0))
        # soft-core LJ between solute and environment: 4 eps lam [1/D^2 - 1/D], D = a(1-lam) + (r/sigma)^6
        if S_i.shape[0]:
            r2 = mind(S_i, S_j)
            inside = r2 < rc * rc
            rs6 = (jnp.where(inside, r2, one) / S_s2) ** 3
            dsc = sc_a * (1.0 - lam_vdw) + rs6
            lj = lj + jnp.sum(jnp.where(inside, 4.0 * S_e * lam_vdw * (1.0 / (dsc * dsc) - 1.0 / dsc), 0.0))
        out["lj"] = lj
        # reciprocal, self, exclusion corrections
        out["coul_recip"] = recip(x, box, qe)
        out["coul_self"] = -ONE_4PI_EPS0 * alpha / SQRT_PI * jnp.sum(qe * qe)
        V = box[0] * box[1] * box[2]
        out["coul_net"] = -ONE_4PI_EPS0 * jnp.pi * jnp.sum(qe) ** 2 / (2.0 * V * alpha**2)
        de = x[e_i] - x[e_j]
        de = de - box * jnp.round(de / box)
        er = jnp.sqrt(jnp.sum(de * de, axis=-1))
        qq_full = qe[e_i] * qe[e_j]
        es6 = (exc_sig * exc_sig / (er * er)) ** 3
        out["coul_excl"] = ONE_4PI_EPS0 * jnp.sum(-qq_full * erf(alpha * er) / er + exc_qq * jnp.where(sol_j[e_i], lam_elec**2, 1.0) / er)
        out["lj_excl"] = jnp.sum(4.0 * exc_eps * (es6 * es6 - es6))
        out["lj_dispersion"] = (disp_env + lam_vdw * disp_sol) / V
        # bonded
        out["bond"] = jnp.sum(0.5 * b_k * (bond_lengths(x, b_idx) - b_r0) ** 2) if b_idx.shape[0] else jnp.zeros(())
        out["angle"] = jnp.sum(0.5 * a_k * (angle_values(x, a_idx) - a_t0) ** 2) if a_idx.shape[0] else jnp.zeros(())
        out["torsion"] = jnp.sum(t_k * (1.0 + jnp.cos(t_n * dihedral_values(x, t_idx) - t_ph))) if t_idx.shape[0] else jnp.zeros(())
        return out

    def _energy(x, box, lam_elec, lam_vdw, nbr):
        return sum(_terms(x, box, lam_elec, lam_vdw, nbr).values())

    vg = jax.value_and_grad(_energy)

    def _ef(x, box, lam_elec, lam_vdw, nbr):
        e, g = vg(x, box, lam_elec, lam_vdw, nbr)
        return e, -g

    _dudl = jax.grad(_energy, argnums=(2, 3))

    # neighbor-list construction over the full flat pair list
    rl2 = (spec.cutoff + 0.12) ** 2

    def _build(x, box, capacity, rl2=rl2):
        d = x[cP_i] - x[cP_j]
        d = d - box * jnp.round(d / box)
        within = jnp.sum(d * d, axis=-1) < rl2
        n = jnp.sum(within)
        (idx,) = jnp.nonzero(within, size=capacity, fill_value=0)
        valid = jnp.arange(capacity) < n
        return cP_i[idx], cP_j[idx], valid, n

    full = (cP_i, cP_j, jnp.ones(cP_i.shape[0], dtype=bool))
    return PeriodicModel(system, spec, jax.jit(_energy), jax.jit(_terms), jax.jit(_ef), jax.jit(_dudl), solute,
                         full, jax.jit(_build, static_argnums=2), skin=0.12)
