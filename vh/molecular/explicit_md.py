"""Explicit-solvent dynamics: constrained Langevin (g-BAOAB) + Monte Carlo barostat.

Integrator (Leimkuhler & Matthews' g-BAOAB with one SHAKE/RATTLE projection
after every sub-step):

    B  v += dt/2 F/m                ; RATTLE
    A  x' = SHAKE(x + dt/2 v), v = (x' - x)/(dt/2) ; RATTLE
    O  v = c1 v + c2 sqrt(kT/m) xi  ; RATTLE
    A  (as above)
    F  = F(x)
    B  v += dt/2 F/m                ; RATTLE

Steps run in compiled blocks of `block` steps sharing one neighbor list; the
list is rebuilt between blocks, and every block checks that no atom moved more
than skin/2 (the condition under which the list is exact). A violated block is
rejected and re-run with fresh lists, so the neighbor list never changes the
physics.

Constant pressure: a Monte Carlo barostat proposes isotropic volume changes,
scaling molecular centers of mass (molecules stay rigid) and accepting with the
NPT Metropolis criterion  dU + P dV - N_mol kT ln(V'/V).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from ..constants import KB
from .constraints import RigidTriangles
from .periodic import PeriodicModel

BAR_TO_KJ_MOL_NM3 = 0.0602214076      # 1 bar in kJ/mol/nm^3
AMU_PER_NM3_TO_G_CM3 = 1.66053906660e-3


@dataclass
class ExplicitState:
    x: jnp.ndarray
    v: jnp.ndarray
    f: jnp.ndarray
    e: float
    key: jnp.ndarray
    box: jnp.ndarray
    nbr: tuple
    time: float = 0.0
    steps: int = 0
    stats: dict = field(default_factory=lambda: {"rejected_blocks": 0, "max_disp": 0.0,
                                                  "baro_tries": 0, "baro_accepts": 0})


class ExplicitLangevin:
    def __init__(self, model: PeriodicModel, constraints: RigidTriangles | None, dt=0.002, temperature=298.15,
                 friction=1.0, lam_elec=1.0, lam_vdw=1.0, block=10):
        self.model, self.cons = model, constraints
        self.dt, self.T, self.gamma = float(dt), float(temperature), float(friction)
        self.lam = (float(lam_elec), float(lam_vdw))
        self.block = int(block)
        sysm = model.system
        self.masses = jnp.asarray(np.asarray(sysm.masses))
        self.inv_m = 1.0 / self.masses[:, None]
        n_cons = constraints.n_constraints if constraints is not None else 0
        self.dof = 3 * sysm.n_atoms - n_cons
        self.mol_ids = np.asarray(sysm.molecule_ids)
        self.n_mol = int(self.mol_ids.max()) + 1
        self._compile()

    def _compile(self):
        dt, inv_m = self.dt, self.inv_m
        c1 = float(np.exp(-self.gamma * dt))
        c2 = float(np.sqrt(1 - c1 * c1))
        sig = jnp.sqrt(KB * self.T * inv_m)
        cons = self.cons
        ef = self.model.energy_and_forces
        proj = (lambda x, v: cons.rattle(x, v)) if cons is not None else (lambda x, v: v)
        shake = (lambda xn, xr: cons.shake(xn, xr)) if cons is not None else (lambda xn, xr: xn)

        def block_fn(x, v, f, e, key, box, nbr, le, lv, n):
            x0 = x

            def drift(x, v):
                xn = shake(x + 0.5 * dt * v, x)
                v = (xn - x) / (0.5 * dt)
                return xn, proj(xn, v)

            def step(carry, _):
                x, v, f, e, key = carry
                v = proj(x, v + 0.5 * dt * f * inv_m)
                x, v = drift(x, v)
                key, sub = jax.random.split(key)
                v = proj(x, c1 * v + c2 * sig * jax.random.normal(sub, v.shape))
                x, v = drift(x, v)
                e, f = ef(x, box, le, lv, nbr)
                v = proj(x, v + 0.5 * dt * f * inv_m)
                return (x, v, f, e, key), None

            (x, v, f, e, key), _ = jax.lax.scan(step, (x, v, f, e, key), None, length=n)
            disp = jnp.max(jnp.sqrt(jnp.sum((x - x0) ** 2, axis=-1)))
            return x, v, f, e, key, disp

        self._block = jax.jit(block_fn, static_argnums=9)

    # ------------------------------------------------------------------------
    def init_state(self, x, box, seed=0, v=None) -> ExplicitState:
        x = jnp.asarray(x)
        box = jnp.asarray(box)
        key = jax.random.PRNGKey(seed)
        key, sub = jax.random.split(key)
        if v is None:
            v = jax.random.normal(sub, x.shape) * jnp.sqrt(KB * self.T * self.inv_m)
        v = jnp.asarray(v)
        v = self.cons.rattle(x, v) if self.cons is not None else v
        nbr = self.model.neighbor_list(x, box)
        e, f = self.model.energy_and_forces(x, box, *self.lam, nbr)
        return ExplicitState(x, v, f, float(e), key, box, nbr)

    def run(self, st: ExplicitState, n_steps: int, barostat: "MonteCarloBarostat | None" = None,
            report_every: int | None = None, callback=None) -> ExplicitState:
        """Advance n_steps (rounded to whole blocks)."""
        skin_half = 0.5 * self.model.skin
        target = st.steps + n_steps
        blocks_done = 0
        while st.steps < target:
            n = min(self.block, target - st.steps)
            while True:                             # never accept a block that outran the list
                x, v, f, e, key, disp = self._block(st.x, st.v, st.f, st.e, st.key, st.box, st.nbr, *self.lam, n)
                disp = float(disp)
                if disp <= skin_half or n == 1:
                    break
                st.stats["rejected_blocks"] += 1
                n = max(1, n // 2)
                self.block = max(1, min(self.block, n))   # adapt: shorter blocks from now on
            if disp > skin_half:
                raise RuntimeError(f"atoms move {disp:.3f} nm in one step; neighbor-list skin too small")
            st.stats["max_disp"] = max(st.stats["max_disp"], disp)
            st.x, st.v, st.f, st.e, st.key = x, v, f, float(e), key
            st.steps += n
            st.time = st.steps * self.dt
            st.nbr = self.model.neighbor_list(st.x, st.box)
            blocks_done += 1
            if barostat is not None and blocks_done % barostat.every_blocks == 0:
                barostat.attempt(self, st)
            if callback is not None and report_every and st.steps % report_every < n:
                callback(st)
        return st

    def kinetic_temperature(self, st):
        ek = 0.5 * float(jnp.sum(self.masses[:, None] * st.v ** 2))
        return 2 * ek / (self.dof * KB)

    def density(self, st):
        return float(jnp.sum(self.masses)) * AMU_PER_NM3_TO_G_CM3 / float(jnp.prod(st.box))


class MonteCarloBarostat:
    """Isotropic MC barostat acting on molecular centers of mass."""

    def __init__(self, pressure_bar=1.0, every_blocks=2, max_dv_frac=0.01, seed=0):
        self.P = pressure_bar * BAR_TO_KJ_MOL_NM3
        self.every_blocks = every_blocks
        self.max_dv_frac = max_dv_frac
        self.rng = np.random.default_rng(seed)
        self._com_w = None

    def attempt(self, integ: ExplicitLangevin, st: ExplicitState):
        model = integ.model
        kT = KB * integ.T
        if self._com_w is None:
            m = np.asarray(integ.masses)
            ids = integ.mol_ids
            tot = np.bincount(ids, weights=m)
            self._ids = jnp.asarray(ids)
            self._w = jnp.asarray(m / tot[ids])
        V = float(jnp.prod(st.box))
        dV = self.rng.uniform(-1, 1) * self.max_dv_frac * V
        Vn = V + dV
        s = (Vn / V) ** (1.0 / 3.0)
        com = jax.ops.segment_sum(st.x * self._w[:, None], self._ids, num_segments=integ.n_mol)
        xn = st.x + (s - 1.0) * com[self._ids]
        boxn = st.box * s
        nbrn = model.neighbor_list(xn, boxn)
        en, fn = model.energy_and_forces(xn, boxn, *integ.lam, nbrn)
        w = float(en) - st.e + self.P * dV - integ.n_mol * kT * np.log(Vn / V)
        st.stats["baro_tries"] += 1
        self._com_w = True
        if w <= 0 or self.rng.random() < np.exp(-w / kT):
            st.x, st.box, st.nbr, st.e, st.f = xn, boxn, nbrn, float(en), fn
            st.stats["baro_accepts"] += 1
            # velocities of rigid molecules unchanged; keep them tangent to constraints
            if integ.cons is not None:
                st.v = integ.cons.rattle(st.x, st.v)
        # adapt the step for ~25-75% acceptance (as OpenMM does)
        tries, acc = st.stats["baro_tries"], st.stats["baro_accepts"]
        if tries % 10 == 0:
            rate = acc / tries
            if rate < 0.25:
                self.max_dv_frac *= 0.9
            elif rate > 0.75:
                self.max_dv_frac = min(self.max_dv_frac * 1.1, 0.3)
