"""Molecular engine tests. Our engine is checked against OpenMM's double-precision
Reference platform, which acts purely as an independent referee."""

from pathlib import Path

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from vh.molecular import VelocityVerlet, build_energy_model, minimize, simulate  # noqa: E402
from vh.molecular.prep import load_protein  # noqa: E402
from vh.molecular.reference import OpenMMReferee, compare  # noqa: E402
from vh.molecular.system import MolecularSystem  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "structures"


@pytest.fixture(scope="module")
def chignolin():
    system, pos, top, omm = load_protein(DATA / "1uao_chignolin.pdb")
    return system, pos, omm, build_energy_model(system)


@pytest.mark.parametrize("pdb", ["1uao_chignolin.pdb", "1l2y_trpcage.pdb"])
def test_every_energy_term_matches_openmm(pdb):
    system, pos, _, omm = load_protein(DATA / pdb)
    model, ref = build_energy_model(system), OpenMMReferee(omm)
    rng = np.random.default_rng(0)
    for x in (pos, pos + rng.normal(0.0, 0.01, pos.shape)):
        c = compare(model, ref, x)
        for row in c["terms"]:
            assert row["rel_err"] < 1e-10 or row["abs_err"] < 1e-8, row
        assert c["force_rel_rms_err"] < 1e-10
        assert c["force_max_abs_err"] < 1e-8


def test_dihedral_sign_convention_with_asymmetric_phase():
    rng = np.random.default_rng(3)
    xs = rng.normal(0, 0.15, (4, 3))
    s = openmm.System()
    for _ in range(4):
        s.addParticle(12.0)
    tf = openmm.PeriodicTorsionForce()
    tf.addTorsion(0, 1, 2, 3, 3, 1.0, 5.0)
    s.addForce(tf)
    c = openmm.Context(s, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    c.setPositions(xs)
    st = c.getState(getEnergy=True, getForces=True)
    pairs = np.array([[i, j] for i in range(4) for j in range(i + 1, 4)])
    ms = MolecularSystem(
        masses=np.full(4, 12.0), charges=np.zeros(4), sigma=np.full(4, 0.3), epsilon=np.zeros(4),
        bonds=np.zeros((0, 2), int), bond_k=np.zeros(0), bond_r0=np.zeros(0),
        angles=np.zeros((0, 3), int), angle_k=np.zeros(0), angle_t0=np.zeros(0),
        torsions=np.array([[0, 1, 2, 3]]), torsion_k=np.array([5.0]), torsion_n=np.array([3]),
        torsion_phase=np.array([1.0]), exceptions=pairs, exc_qq=np.zeros(6), exc_sigma=np.ones(6), exc_eps=np.zeros(6),
    )
    e, f = build_energy_model(ms).energy_and_forces(xs)
    assert abs(float(e) - st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)) < 1e-10
    f_ref = st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    assert np.abs(np.asarray(f) - f_ref).max() < 1e-10


def test_trajectory_matches_openmm_verlet(chignolin):
    system, pos, omm, model = chignolin
    x0, _ = minimize(model, pos)
    dt = 0.0005
    vv = VelocityVerlet(model, dt=dt)
    st0 = vv.init_state(x0, T=300.0, seed=7)
    _, f0 = model.energy_and_forces(st0.x)
    v_leapfrog = np.asarray(st0.v) - 0.5 * dt * np.asarray(f0) / np.asarray(system.masses)[:, None]
    ctx = openmm.Context(omm, openmm.VerletIntegrator(dt), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(np.asarray(st0.x))
    ctx.setVelocities(v_leapfrog)
    ctx.getIntegrator().step(500)
    st = vv.run(st0, 500)
    x_ref = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    assert np.abs(np.asarray(st.x) - x_ref).max() < 1e-9


def test_nve_energy_conservation(chignolin):
    system, pos, omm, model = chignolin
    x0, _ = minimize(model, pos)
    vv = VelocityVerlet(model, dt=0.0005)
    st = vv.init_state(x0, T=300.0, seed=2)
    _, _, log = simulate(vv, st, n_steps=4000, report_every=50)
    etot = log["e_pot"] + log["e_kin"]
    assert etot.std() < 0.05 * log["e_kin"].std()
    drift = abs(np.polyfit(log["time_ps"], etot, 1)[0])
    assert drift < 0.5  # kJ/mol/ps


def test_serialization_roundtrip(chignolin, tmp_path):
    system, pos, omm, model = chignolin
    path = tmp_path / "sys.json"
    system.save_json(path)
    again = MolecularSystem.load_json(path)
    e1 = float(model.energy(pos))
    e2 = float(build_energy_model(again).energy(pos))
    assert abs(e1 - e2) < 1e-9 * abs(e1)
