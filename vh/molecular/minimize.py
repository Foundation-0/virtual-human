"""Energy minimization with L-BFGS (SciPy driver, JAX energies/gradients)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize as _scipy_minimize

from .energy import EnergyModel


def minimize(model: EnergyModel, x0, tolerance: float = 10.0, max_iterations: int = 2000):
    """Minimize the potential energy.

    tolerance: stop when the RMS force falls below this (kJ/mol/nm), similar
    in spirit to OpenMM's `minimizeEnergy(tolerance=10 kJ/mol/nm)`.
    Returns (x_min (N,3) nm, info dict).
    """
    shape = np.shape(x0)
    n = shape[0]

    def fun(flat):
        e, f = model.energy_and_forces(flat.reshape(shape))
        return float(e), -np.asarray(f, dtype=np.float64).ravel()

    # L-BFGS's gtol is a max-component criterion; convert the RMS target.
    res = _scipy_minimize(
        fun, np.asarray(x0, dtype=np.float64).ravel(), jac=True, method="L-BFGS-B",
        options={"maxiter": max_iterations, "gtol": tolerance / np.sqrt(3.0), "maxcor": 30},
    )
    x = res.x.reshape(shape)
    e, f = model.energy_and_forces(x)
    f = np.asarray(f)
    return x, {
        "energy": float(e),
        "rms_force": float(np.sqrt(np.mean(np.sum(f * f, axis=1)))),
        "max_force": float(np.max(np.linalg.norm(f, axis=1))),
        "iterations": int(res.nit),
        "converged": bool(res.success),
        "n_atoms": n,
    }
