"""Explicit-solvent engine (rung 3): Ewald/PME electrostatics and rigid-water
constraints, checked against OpenMM as an independent referee."""

import copy

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from vh.molecular.prep import water_box  # noqa: E402
from vh.molecular.periodic import PeriodicSpec, build_periodic_model  # noqa: E402
from vh.molecular.constraints import RigidTriangles  # noqa: E402


def _omm_nonbonded(omm, pos, method, tol):
    s = copy.deepcopy(omm)
    for f in s.getForces():
        f.setForceGroup(1 if isinstance(f, openmm.NonbondedForce) else 2)
        if isinstance(f, openmm.NonbondedForce):
            f.setNonbondedMethod(method)
            f.setEwaldErrorTolerance(tol)
    c = openmm.Context(s, openmm.VerletIntegrator(1e-3), openmm.Platform.getPlatformByName("Reference"))
    c.setPositions(pos)
    st = c.getState(getEnergy=True, getForces=True, groups={1})
    e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f = st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    nb = [x for x in s.getForces() if isinstance(x, openmm.NonbondedForce)][0]
    params = nb.getPMEParametersInContext(c) if method == openmm.NonbondedForce.PME else None
    return e, np.asarray(f), params


@pytest.fixture(scope="module")
def box():
    system, pos, top, omm = water_box(1.8, 0.8, ewald_tol=5e-4)
    return system, pos, omm


def test_ewald_matches_openmm(box):
    system, pos, omm = box
    e_o, f_o, _ = _omm_nonbonded(omm, pos, openmm.NonbondedForce.Ewald, 1e-9)
    m = build_periodic_model(system, system.box, PeriodicSpec(cutoff=0.8, tol=1e-9, method="ewald"))
    e, f = m.ef_exact(pos, system.box)
    e = float(e)
    f = np.asarray(f)
    assert abs(e - e_o) / abs(e_o) < 1e-8
    assert np.sqrt(((f - f_o) ** 2).mean() / (f_o ** 2).mean()) < 1e-8


def test_pme_matches_openmm_pme(box):
    system, pos, omm = box
    e_o, f_o, params = _omm_nonbonded(omm, pos, openmm.NonbondedForce.PME, 5e-4)
    a, nx, ny, nz = params
    spec = PeriodicSpec(cutoff=0.8, tol=0.5 * np.exp(-(a * 0.8) ** 2), method="pme", pme_grid=(nx, ny, nz))
    m = build_periodic_model(system, system.box, spec)
    e, f = m.ef_exact(pos, system.box)
    assert abs(float(e) - e_o) / abs(e_o) < 1e-8
    assert np.sqrt(((np.asarray(f) - f_o) ** 2).mean() / (f_o ** 2).mean()) < 1e-8


def test_neighbor_list_is_exact(box):
    system, pos, omm = box
    m = build_periodic_model(system, system.box, PeriodicSpec(cutoff=0.8, tol=5e-4, method="pme"))
    e_full, f_full = m.ef_exact(pos, system.box)
    nbr = m.neighbor_list(pos, system.box)
    e_nl, f_nl = m.energy_and_forces(pos, system.box, 1.0, 1.0, nbr)
    assert abs(float(e_full) - float(e_nl)) < 1e-6
    assert np.abs(np.asarray(f_full) - np.asarray(f_nl)).max() < 1e-6


def test_rigid_water_constraints_hold_under_dynamics(box):
    import jax.numpy as jnp
    from vh.molecular.explicit_md import ExplicitLangevin
    system, pos, omm = box
    m = build_periodic_model(system, system.box, PeriodicSpec(cutoff=0.8, tol=5e-4, method="pme"))
    cons = RigidTriangles(system)
    # the freshly-solvated box has clashes; relax cold before sampling (as in real use)
    relax = ExplicitLangevin(m, cons, dt=0.0005, temperature=100.0, friction=50.0, block=5)
    st = relax.init_state(pos, system.box, seed=1)
    st = relax.run(st, 400)
    md = ExplicitLangevin(m, cons, dt=0.002, temperature=298.15, friction=1.0, block=5)
    st = md.init_state(st.x, system.box, seed=2)
    st = md.run(st, 300)
    assert float(cons.max_violation(st.x)) < 1e-6          # SHAKE/RATTLE keep water rigid
    assert np.all(np.isfinite(np.asarray(st.x)))           # integrator is stable
    assert 150 < md.kinetic_temperature(st) < 500          # thermostat in a sane range after relaxation
