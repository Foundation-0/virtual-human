"""Independent referee: evaluate the same system with OpenMM's Reference platform.

Used only by tests and validation scripts. OpenMM's Reference platform is a
double-precision CPU implementation, so agreement with it to ~1e-6 relative
shows that our engine implements exactly the same physics.
"""

from __future__ import annotations

import numpy as np

# map our term names onto OpenMM force classes
GROUPS = {
    "HarmonicBondForce": 0,
    "HarmonicAngleForce": 1,
    "PeriodicTorsionForce": 2,
    "NonbondedForce": 3,
    "GBSAOBCForce": 4,
    "CustomGBForce": 4,
}
OUR_TERMS_FOR_GROUP = {
    0: ("bond",),
    1: ("angle",),
    2: ("torsion",),
    3: ("lj", "coulomb", "lj14", "coulomb14"),
    4: ("gb_polar", "gb_nonpolar"),
}
GROUP_LABEL = {0: "bonds", 1: "angles", 2: "torsions", 3: "LJ + Coulomb (incl. 1-4)", 4: "GB-OBC2 solvation"}


class OpenMMReferee:
    def __init__(self, omm_system, platform: str = "Reference"):
        import copy

        import openmm as mm

        self.system = copy.deepcopy(omm_system)
        for force in self.system.getForces():
            force.setForceGroup(GROUPS.get(type(force).__name__, 31))
        self.context = mm.Context(
            self.system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(platform)
        )

    def evaluate(self, x_nm):
        from openmm import unit

        self.context.setPositions(np.asarray(x_nm, dtype=np.float64))
        per_group = {}
        for g in sorted(set(GROUPS.values())):
            st = self.context.getState(getEnergy=True, groups={g})
            per_group[g] = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        st = self.context.getState(getEnergy=True, getForces=True)
        total = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = np.asarray(st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
        return total, per_group, forces


def compare(model, referee: OpenMMReferee, x_nm) -> dict:
    """Per-term and force comparison between our engine and OpenMM."""
    terms = {k: float(v) for k, v in model.terms(x_nm).items()}
    e_ours, f_ours = model.energy_and_forces(x_nm)
    f_ours = np.asarray(f_ours, dtype=np.float64)
    e_ref, groups, f_ref = referee.evaluate(x_nm)
    rows = []
    for g, names in OUR_TERMS_FOR_GROUP.items():
        ours = sum(terms[n] for n in names)
        ref = groups[g]
        rows.append({
            "term": GROUP_LABEL[g], "ours": ours, "openmm": ref,
            "abs_err": abs(ours - ref), "rel_err": abs(ours - ref) / max(abs(ref), 1e-12),
        })
    df = f_ours - f_ref
    return {
        "terms": rows,
        "total_ours": float(e_ours),
        "total_openmm": e_ref,
        "total_rel_err": abs(float(e_ours) - e_ref) / max(abs(e_ref), 1e-12),
        "force_max_abs_err": float(np.max(np.abs(df))),
        "force_rms_err": float(np.sqrt(np.mean(df**2))),
        "force_rms": float(np.sqrt(np.mean(f_ref**2))),
        "force_rel_rms_err": float(np.sqrt(np.mean(df**2)) / np.sqrt(np.mean(f_ref**2))),
    }
