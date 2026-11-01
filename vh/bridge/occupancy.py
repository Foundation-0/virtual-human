"""The molecular -> body bridge: target engagement over time.

The body model gives the unbound drug concentration next to the target
(e.g. brain interstitial water for a CNS receptor). Molecular-scale binding
parameters turn that into the fraction of receptors occupied:

    equilibrium:  occupancy = Cu / (Cu + Kd)
    kinetic:      dO/dt = kon * Cu * (1 - O) - koff * O,   kon = koff / Kd

Kd is exactly the quantity an atomistic binding free-energy calculation
produces: Kd = c0 * exp(dG_bind / RT) with c0 = 1 M. Today Kd comes from
experiment; the roadmap computes it with vh.molecular (alchemical free
energy), at which point this module becomes the hand-off point.
"""

from __future__ import annotations

import numpy as np

from ..constants import KB


def kd_from_binding_free_energy(dg_kj_mol: float, temperature_k: float = 310.15) -> float:
    """Binding free energy (kJ/mol, negative = favorable) -> Kd in nM (1 M standard state)."""
    return float(np.exp(dg_kj_mol / (KB * temperature_k)) * 1e9)


def binding_free_energy_from_kd(kd_nm: float, temperature_k: float = 310.15) -> float:
    return float(KB * temperature_k * np.log(kd_nm * 1e-9))


def mg_per_l_to_nm(c_mg_l, mw_g_mol):
    return np.asarray(c_mg_l) / mw_g_mol * 1e6


def equilibrium_occupancy(cu_nm, kd_nm):
    cu = np.asarray(cu_nm, dtype=float)
    return cu / (cu + kd_nm)


def kinetic_occupancy(t_h, cu_nm, kd_nm, koff_per_h, o0=0.0):
    """Exact piecewise solution assuming Cu is constant within each interval."""
    t = np.asarray(t_h, dtype=float)
    cu = np.asarray(cu_nm, dtype=float)
    kon = koff_per_h / kd_nm
    o = np.empty_like(t)
    o[0] = o0
    for i in range(1, len(t)):
        c = 0.5 * (cu[i] + cu[i - 1])
        k = kon * c + koff_per_h
        o_inf = kon * c / k
        o[i] = o_inf + (o[i - 1] - o_inf) * np.exp(-k * (t[i] - t[i - 1]))
    return o


def target_engagement(model, sim, koff_per_h: float | None = None) -> dict:
    """Unbound concentration at the compound's target tissue and receptor occupancy."""
    tgt = model.compound.target
    if tgt is None:
        raise ValueError(f"{model.compound.name} has no target defined")
    p = model.p
    c_tissue = sim[tgt.tissue]
    cu = c_tissue * p.fu / p.kp[tgt.tissue]             # unbound, in equilibrium with plasma water
    cu_nm = mg_per_l_to_nm(cu, model.compound.mw)
    occ = (equilibrium_occupancy(cu_nm, tgt.kd_nm) if koff_per_h is None
           else kinetic_occupancy(sim["time_h"], cu_nm, tgt.kd_nm, koff_per_h))
    return {"time_h": sim["time_h"], "cu_nm": cu_nm, "occupancy": occ, "target": tgt.name, "kd_nm": tgt.kd_nm,
            "dg_bind_kj_mol": binding_free_energy_from_kd(tgt.kd_nm)}
