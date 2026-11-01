"""Holonomic constraints for rigid molecules (rigid water: SHAKE + RATTLE).

TIP3P is parameterized as a RIGID molecule: the two O-H bonds and the H-H
distance are fixed. Rigid water also allows a 2 fs time step. We enforce the
three distance constraints per molecule with:

  SHAKE (positions):  x_i = x_i* + sum_c lambda_c s_c (d_{i,a_c} - d_{i,b_c}) / m_i
                      where s_c is the constraint vector at the previous step;
                      the 3 multipliers per molecule are solved by Newton
                      iteration to machine precision (a 3x3 solve per molecule).
  RATTLE (velocities): removes the velocity components along the constraints
                      with one exact linear solve per molecule.

Everything is vectorized over molecules (one batched 3x3 solve per iteration).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .system import MolecularSystem

# local constraint topology of a 3-site rigid molecule: (0,1), (0,2), (1,2)
_PAIRS = np.array([[0, 1], [0, 2], [1, 2]])


class RigidTriangles:
    """Constraints for molecules made of exactly 3 atoms and 3 distance constraints."""

    def __init__(self, system: MolecularSystem, iterations: int = 8):
        if system.constraints is None or len(system.constraints) == 0:
            raise ValueError("system has no constraints")
        cons = np.asarray(system.constraints)
        dist = np.asarray(system.constraint_dist)
        mol = np.asarray(system.molecule_ids)
        groups = {}
        for (a, b), d in zip(cons, dist):
            if mol[a] != mol[b]:
                raise ValueError("constraint spans two molecules")
            groups.setdefault(int(mol[a]), []).append((int(a), int(b), float(d)))
        atoms, dists = [], []
        for m_id, cs in sorted(groups.items()):
            members = sorted({a for a, _, _ in cs} | {b for _, b, _ in cs})
            if len(cs) != 3 or len(members) != 3:
                raise NotImplementedError("only 3-atom, 3-constraint rigid molecules are supported")
            loc = {g: k for k, g in enumerate(members)}
            dd = np.zeros(3)
            for a, b, d in cs:
                pair = sorted((loc[a], loc[b]))
                c = int(np.where((_PAIRS[:, 0] == pair[0]) & (_PAIRS[:, 1] == pair[1]))[0][0])
                dd[c] = d
            atoms.append(members)
            dists.append(dd)
        self.atoms = jnp.asarray(np.array(atoms, dtype=np.int32))          # (M,3)
        self.d2 = jnp.asarray(np.array(dists) ** 2)                        # (M,3)
        m = np.asarray(system.masses)[np.array(atoms)]                     # (M,3)
        self.inv_m = jnp.asarray(1.0 / m)
        # coef[M, c, e] = (δ(a_c,a_e) - δ(a_c,b_e))/m_{a_c} - (δ(b_c,a_e) - δ(b_c,b_e))/m_{b_c}
        a_, b_ = _PAIRS[:, 0], _PAIRS[:, 1]
        dlt = lambda i, j: (i[:, None] == j[None, :]).astype(float)        # noqa: E731
        A = dlt(a_, a_) - dlt(a_, b_)       # (c, e) for atom a_c
        B = dlt(b_, a_) - dlt(b_, b_)       # (c, e) for atom b_c
        im = 1.0 / m
        self.coef = jnp.asarray(A[None] * im[:, a_][:, :, None] - B[None] * im[:, b_][:, :, None])  # (M,3,3)
        # per-atom incidence of constraint e: +1 if atom is a_e, -1 if b_e
        inc = np.zeros((3, 3))
        for e, (a, b) in enumerate(_PAIRS):
            inc[a, e] += 1.0
            inc[b, e] -= 1.0
        self.inc = jnp.asarray(inc)          # (atom_local, constraint)
        self.iterations = iterations
        self.n_constraints = int(len(cons))

    # ------------------------------------------------------------------------
    def _vecs(self, P):
        return P[:, _PAIRS[:, 0], :] - P[:, _PAIRS[:, 1], :]                 # (M,3c,3xyz)

    def shake(self, x_new, x_ref):
        """Project unconstrained positions x_new onto the constraint surface,
        moving along the constraint directions at x_ref (SHAKE)."""
        P = x_new[self.atoms]                 # (M,3,3)
        S = self._vecs(x_ref[self.atoms])     # reference constraint vectors
        lam = jnp.zeros(self.d2.shape)

        def body(_, lam):
            # positions for current multipliers
            disp = jnp.einsum("ae,me,mex->max", self.inc, lam[:, :], S) * self.inv_m[:, :, None]
            r = self._vecs(P + disp)                                        # (M,3,3)
            g = jnp.sum(r * r, axis=-1) - self.d2                           # (M,3)
            J = 2.0 * jnp.einsum("mcx,mex->mce", r, S) * self.coef          # (M,3,3)
            return lam - jnp.linalg.solve(J, g[..., None])[..., 0]

        lam = jax.lax.fori_loop(0, self.iterations, body, lam)
        disp = jnp.einsum("ae,me,mex->max", self.inc, lam, S) * self.inv_m[:, :, None]
        return x_new.at[self.atoms].set(P + disp)

    def rattle(self, x, v):
        """Remove velocity components that would change constrained distances."""
        P = x[self.atoms]
        V = v[self.atoms]
        r = self._vecs(P)                                                    # (M,3,3)
        dv = V[:, _PAIRS[:, 0], :] - V[:, _PAIRS[:, 1], :]
        rhs = -jnp.sum(r * dv, axis=-1)                                      # (M,3)
        K = jnp.einsum("mcx,mex->mce", r, r) * self.coef                     # (M,3,3)
        mu = jnp.linalg.solve(K, rhs[..., None])[..., 0]                     # (M,3)
        dV = jnp.einsum("ae,me,mex->max", self.inc, mu, r) * self.inv_m[:, :, None]
        return v.at[self.atoms].set(V + dV)

    def max_violation(self, x):
        r = self._vecs(x[self.atoms])
        return jnp.max(jnp.abs(jnp.sqrt(jnp.sum(r * r, axis=-1)) - jnp.sqrt(self.d2)))
