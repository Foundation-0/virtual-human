"""Equations of motion: velocity Verlet (NVE) and Langevin BAOAB (NVT).

Both integrators run many steps inside one compiled `lax.scan`, so the Python
loop is only hit once per reporting interval.

BAOAB (Leimkuhler & Matthews 2013) splits the Langevin equation into
kick (B), drift (A) and Ornstein-Uhlenbeck (O) pieces. It samples
configurations very accurately at large time steps and is the same scheme as
OpenMM's LangevinMiddleIntegrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..constants import KB
from .energy import EnergyModel


@dataclass
class MDState:
    x: jnp.ndarray        # positions, nm
    v: jnp.ndarray        # velocities, nm/ps
    f: jnp.ndarray        # forces, kJ/mol/nm
    e_pot: jnp.ndarray    # potential energy, kJ/mol
    key: jnp.ndarray      # PRNG state
    time: float = 0.0     # ps


def n_dof(n_atoms: int, remove_com: bool = True) -> int:
    return 3 * n_atoms - (3 if remove_com else 0)


def kinetic_energy(v, masses):
    return 0.5 * jnp.sum(masses[:, None] * v * v)


def temperature(v, masses, dof):
    return 2.0 * kinetic_energy(v, masses) / (dof * KB)


def maxwell_boltzmann(key, masses, T, dtype=jnp.float64, remove_com=True):
    m = jnp.asarray(masses, dtype=dtype)[:, None]
    v = jax.random.normal(key, (m.shape[0], 3), dtype=dtype) * jnp.sqrt(KB * T / m)
    if remove_com:
        v = v - jnp.sum(m * v, axis=0) / jnp.sum(m)
    return v


class Integrator:
    """Base class: holds the compiled multi-step function."""

    def __init__(self, model: EnergyModel, dt: float):
        self.model = model
        self.dt = float(dt)
        self.dtype = model.dtype
        self.masses = jnp.asarray(np.asarray(model.system.masses), dtype=self.dtype)
        self._run = None

    def init_state(self, x, T: float | None = None, seed: int = 0, v=None) -> MDState:
        x = jnp.asarray(x, dtype=self.dtype)
        key = jax.random.PRNGKey(seed)
        key, sub = jax.random.split(key)
        if v is None:
            v = maxwell_boltzmann(sub, self.masses, T or 0.0, dtype=self.dtype)
        e, f = self.model.energy_and_forces(x)
        return MDState(x=x, v=jnp.asarray(v, dtype=self.dtype), f=f, e_pot=e, key=key)

    def run(self, state: MDState, n_steps: int) -> MDState:
        x, v, f, e, key = self._run(state.x, state.v, state.f, state.e_pot, state.key, n_steps)
        return MDState(x=x, v=v, f=f, e_pot=e, key=key, time=state.time + n_steps * self.dt)


class VelocityVerlet(Integrator):
    """Microcanonical (NVE) integration. Used to test energy conservation."""

    def __init__(self, model: EnergyModel, dt: float):
        super().__init__(model, dt)
        dt_, inv_m, ef = self.dt, 1.0 / self.masses[:, None], model.energy_and_forces

        def step(carry, _):
            x, v, f, e, key = carry
            v = v + 0.5 * dt_ * f * inv_m
            x = x + dt_ * v
            e, f = ef(x)
            v = v + 0.5 * dt_ * f * inv_m
            return (x, v, f, e, key), None

        def run(x, v, f, e, key, n):
            return jax.lax.scan(step, (x, v, f, e, key), None, length=n)[0]

        self._run = jax.jit(run, static_argnums=5)


class LangevinBAOAB(Integrator):
    """Canonical (NVT) Langevin dynamics with the BAOAB splitting."""

    def __init__(self, model: EnergyModel, dt: float, temperature: float, friction: float = 1.0):
        super().__init__(model, dt)
        self.temperature, self.friction = float(temperature), float(friction)
        dt_, inv_m, ef = self.dt, 1.0 / self.masses[:, None], model.energy_and_forces
        c1 = float(np.exp(-friction * dt))
        c2 = float(np.sqrt(1.0 - c1 * c1))
        sigma_v = jnp.sqrt(KB * temperature * inv_m)

        def step(carry, _):
            x, v, f, e, key = carry
            v = v + 0.5 * dt_ * f * inv_m                      # B
            x = x + 0.5 * dt_ * v                              # A
            key, sub = jax.random.split(key)
            v = c1 * v + c2 * sigma_v * jax.random.normal(sub, v.shape, dtype=v.dtype)  # O
            x = x + 0.5 * dt_ * v                              # A
            e, f = ef(x)
            v = v + 0.5 * dt_ * f * inv_m                      # B
            return (x, v, f, e, key), None

        def run(x, v, f, e, key, n):
            return jax.lax.scan(step, (x, v, f, e, key), None, length=n)[0]

        self._run = jax.jit(run, static_argnums=5)


def simulate(integrator: Integrator, state: MDState, n_steps: int, report_every: int, callback=None):
    """Run n_steps, recording positions and energies every `report_every` steps.

    Returns (final_state, frames (F,N,3) nm, log dict of arrays).
    """
    frames, log = [], {"time_ps": [], "e_pot": [], "e_kin": [], "T": []}
    dof = n_dof(integrator.model.system.n_atoms)
    for _ in range(n_steps // report_every):
        state = integrator.run(state, report_every)
        ek = float(kinetic_energy(state.v, integrator.masses))
        frames.append(np.asarray(state.x))
        log["time_ps"].append(state.time)
        log["e_pot"].append(float(state.e_pot))
        log["e_kin"].append(ek)
        log["T"].append(2.0 * ek / (dof * KB))
        if callback is not None:
            callback(state, log)
    return state, np.array(frames), {k: np.array(v) for k, v in log.items()}
