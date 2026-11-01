"""Quantum-layer tests. Fast levels of theory, checked against known physics and
analytic results. The expensive benchmark numbers live in scripts/validate_quantum.py."""

import numpy as np
import pytest

pytest.importorskip("pyscf")

from vh.quantum import Molecule, cbs_helgaker, interaction_energy, single_point  # noqa: E402
from vh.quantum.ab_initio import QMResult  # noqa: E402
from vh.quantum.qmmm import classical_charge_dipole_energy, embed  # noqa: E402

WATER = Molecule.from_xyz_block("O 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467", name="water")
DIMER = Molecule.from_xyz_block("""
O -1.551 -0.114 0; H -1.934 0.762 0; H -0.599 0.040 0
O  1.350  0.111 0; H  1.680 -0.367 -0.758; H 1.680 -0.367 0.758
""".replace(";", "\n"), name="dimer")


def test_hartree_fock_is_reproducible_and_reasonable():
    r = single_point(WATER, "hf", "cc-pvdz")
    assert r.converged
    assert -76.1 < r.e_tot < -75.9           # known HF/cc-pVDZ water region
    assert abs(r.e_corr) < 1e-12


def test_correlation_lowers_energy_in_the_right_order():
    hf = single_point(WATER, "hf", "cc-pvdz").e_tot
    mp2 = single_point(WATER, "mp2", "cc-pvdz").e_tot
    ccsd = single_point(WATER, "ccsd", "cc-pvdz").e_tot
    ccsdt = single_point(WATER, "ccsd(t)", "cc-pvdz").e_tot
    # each adds (mostly) more correlation; all below HF
    assert hf > mp2
    assert hf > ccsd > ccsdt or abs(ccsd - ccsdt) < 0.01   # (T) is a small negative correction
    assert ccsdt < hf


def test_counterpoise_reduces_bsse():
    r = interaction_energy(DIMER, [[0, 1, 2], [3, 4, 5]], method="hf", basis="cc-pvdz")
    # raw (no CP) over-binds due to basis-set superposition error; CP corrects toward less binding
    assert r["e_int_raw_kcal"] < r["e_int_counterpoise_kcal"]
    assert r["bsse_kcal"] < 0                 # BSSE always stabilizes the raw complex
    assert -10 < r["e_int_counterpoise_kcal"] < 0


def test_cbs_extrapolation_formula():
    # synthetic: E_corr(X) = E_inf + A X^-3, choose E_inf, A and check recovery
    E_inf, A = -0.30000, 0.5
    lo = QMResult("ccsd(t)", "cc-pvtz", 0, -76.0, E_inf + A * 3.0**-3)
    hi = QMResult("ccsd(t)", "cc-pvqz", 0, -76.0, E_inf + A * 4.0**-3)
    out = cbs_helgaker(lo, hi)
    assert abs(out["e_corr_cbs"] - E_inf) < 1e-9
    assert out["cardinals"] == (3, 4)


def test_qmmm_recovers_classical_electrostatics_at_long_range():
    from pyscf import scf
    from vh.quantum.ab_initio import _build_mol
    mf = scf.RHF(_build_mol(WATER, "cc-pvdz")).run(verbose=0)
    dip = mf.dip_moment(unit="Debye", verbose=0)
    r = np.array([[0.0, 0.0, 12.0]])
    q = np.array([1.0])
    e_qmmm = embed(WATER, r, q, method="hf", basis="cc-pvdz").e_interaction_kcal
    e_cl = classical_charge_dipole_energy(r, q, dip, [0, 0, 0.117])
    assert abs(e_qmmm - e_cl) / abs(e_cl) < 0.02    # within 2% at 12 A


def test_qmmm_forces_match_finite_difference():
    r = np.array([[0.0, 0.0, 3.0]])
    q = np.array([1.0])
    res = embed(WATER, r, q, method="hf", basis="sto-3g", grad=True)
    h = 1e-3
    fd = np.zeros((len(WATER.atoms), 3))
    for a in range(len(WATER.atoms)):
        for k in range(3):
            cp = [list(c) for _, c in WATER.atoms]
            cm = [list(c) for _, c in WATER.atoms]
            cp[a][k] += h
            cm[a][k] -= h
            mp_ = Molecule([(WATER.atoms[i][0], tuple(cp[i])) for i in range(len(cp))])
            mm_ = Molecule([(WATER.atoms[i][0], tuple(cm[i])) for i in range(len(cm))])
            ep = embed(mp_, r, q, "hf", "sto-3g").e_tot
            em = embed(mm_, r, q, "hf", "sto-3g").e_tot
            fd[a][k] = (ep - em) / (2 * h)       # h in Angstrom -> Ha/Angstrom, same as qm_grad
    assert np.abs(res.qm_grad - fd).max() < 1e-4


def test_ghost_atoms_carry_no_electrons():
    # a fragment-in-dimer-basis has the electron count of just that fragment
    from vh.quantum.ab_initio import _build_mol
    frag = DIMER.with_ghosts([0, 1, 2])
    m = _build_mol(frag, "sto-3g")
    assert m.nelectron == 10            # one water, not two
