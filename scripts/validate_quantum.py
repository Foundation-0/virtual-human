"""Quantum validation: prove the highest-accuracy layer on the smallest real
parts of the body, against values the scientific community treats as truth.

Targets (each a genuine building block of the human body):
  1. Water           -- ~60% of the body. Geometry and dipole vs experiment.
  2. Water dimer     -- the hydrogen bond behind solvation and protein folding.
                        CCSD(T)/CBS interaction energy vs the accepted ~ -5.0 kcal/mol.
  3. Peptide bond    -- links every amino acid. Planarity and partial double-bond
                        character (formamide), and the C-N rotation barrier.
  4. Glycine         -- the simplest amino acid; a real optimized structure.

CCSD(T)/CBS is obtained by the focal-point additivity scheme:
    E(CCSD(T)/CBS) ~= E(MP2/CBS) + [E(CCSD(T)) - E(MP2)]_(triple-zeta)
because the coupled-cluster correction beyond MP2 converges far faster with
basis set than the MP2 correlation energy itself. All interaction energies are
counterpoise (BSSE) corrected.

    python scripts/validate_quantum.py [--section water|dimer|peptide|glycine|all]

Results are cached to docs/validation_quantum.json so slow sections can run
independently and the report assembles from whatever is done.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np

from vh.quantum import Molecule, interaction_energy, optimize_geometry, single_point
from vh.quantum.ab_initio import HARTREE_TO_KCAL, geometry_parameters

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CACHE = DOCS / "validation_quantum.json"

# Reference values (experiment or accepted high-level theory), with sources.
REF = {
    "water_rOH": (0.9572, "A", "experiment, Benedict 1956 (r_e)"),
    "water_angle": (104.52, "deg", "experiment, Benedict 1956 (r_e)"),
    "water_dipole": (1.855, "D", "experiment, Clough 1973 (gas-phase)"),
    "dimer_De": (-5.0, "kcal/mol", "CCSD(T)/CBS benchmark, Tschumper 2002 / Klopper 2000 (electronic De ~ -4.99 to -5.02)"),
    "peptide_CN": (1.36, "A", "typical peptide-bond C-N length (between C-N single ~1.47 and C=N double ~1.28)"),
    "formamide_barrier": ((16.0, 19.0), "kcal/mol", "formamide C-N rotation barrier, experiment/high-level ~16-19"),
}


def load_cache():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def save(cache):
    DOCS.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=1))


def quiet(fn, *a, **k):
    """Run fn swallowing the geomeTRIC/PySCF banner noise on stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


WATER = Molecule.from_xyz_block("O 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467", name="water")

# A good water-dimer starting geometry (Cs, near equilibrium), Angstrom.
WATER_DIMER = Molecule.from_xyz_block("""
O  -1.5510  -0.1140   0.0000
H  -1.9340   0.7620   0.0000
H  -0.5990   0.0400   0.0000
O   1.3500   0.1110   0.0000
H   1.6800  -0.3670  -0.7580
H   1.6800  -0.3670   0.7580
""", name="water dimer")

# Formamide HCON H2 -- the minimal model of the peptide bond (atoms: C O N H_C H_N H_N)
FORMAMIDE = Molecule.from_xyz_block("""
C   0.0000   0.4300   0.0000
O   1.1600   0.6600   0.0000
N  -0.9400  -0.5400   0.0000
H  -0.6200   1.1300   0.0000
H  -1.9100  -0.2800   0.0000
H  -0.6700  -1.5000   0.0000
""", name="formamide")

GLYCINE = Molecule.from_xyz_block("""
N   1.4900  -0.1000   0.2000
C   0.2600   0.5900  -0.2000
C  -0.9600  -0.3000   0.0300
O  -2.0800   0.1500  -0.1500
O  -0.8200  -1.5200   0.4200
H   0.2400   1.5100   0.3900
H   0.2600   0.8700  -1.2600
H   1.5300  -1.0000  -0.2700
H   2.2900   0.4300  -0.0900
H  -1.6500  -1.9900   0.5200
""", name="glycine")


def sec_water(cache):
    print("== water (the body is ~60% water) ==")
    t = time.time()
    opt = quiet(optimize_geometry, WATER, method="mp2", basis="cc-pvtz")
    g = geometry_parameters(opt)
    rOH = 0.5 * (g["dist"](0, 1) + g["dist"](0, 2))
    ang = g["angle"](1, 0, 2)
    # dipole: HF overestimates; correlation corrects it toward experiment
    from pyscf import scf
    from vh.quantum.ab_initio import _build_mol
    m = _build_mol(opt, "aug-cc-pvtz")
    mf = quiet(scf.RHF(m).run)
    dip_hf = float(np.linalg.norm(mf.dip_moment(unit="Debye", verbose=0)))
    from pyscf import cc
    mc = quiet(cc.CCSD(mf).run)
    dm_mo = mc.make_rdm1()                      # CCSD 1-RDM in the MO basis
    mo = mf.mo_coeff
    dm_ao = mo @ dm_mo @ mo.conj().T            # transform to AO basis
    dip_ccsd = float(np.linalg.norm(_dipole_from_dm(m, dm_ao)))
    cache["water"] = {
        "rOH": rOH, "angle": ang, "dipole_hf": dip_hf, "dipole_ccsd": dip_ccsd,
        "level_geom": "MP2/cc-pVTZ", "level_dipole": "CCSD/aug-cc-pVTZ", "seconds": time.time() - t,
    }
    print(f"  r(OH) {rOH:.4f} A (expt {REF['water_rOH'][0]}), angle {ang:.2f} deg (expt {REF['water_angle'][0]})")
    print(f"  dipole: HF {dip_hf:.3f} -> CCSD {dip_ccsd:.3f} D (expt {REF['water_dipole'][0]})")
    save(cache)


def _dipole_from_dm(m, dm_ao):
    """Dipole (Debye) from an AO-basis 1-RDM whose trace is the electron count."""
    with m.with_common_orig((0, 0, 0)):
        ao_dip = m.intor_symmetric("int1e_r", comp=3)
    el = np.einsum("xij,ji->x", ao_dip, dm_ao).real
    nucl = np.einsum("i,ix->x", m.atom_charges(), m.atom_coords())
    return (nucl - el) * 2.541746051  # a.u. -> Debye


def _cp_components(complex_mol, fragments, method, basis):
    """Counterpoise-corrected (HF, correlation) interaction energy parts, Hartree."""
    ec = single_point(complex_mol, method, basis)
    hf = ec.e_hf
    tot = ec.e_tot
    for idx in fragments:
        f = single_point(complex_mol.with_ghosts(idx), method, basis)
        hf -= f.e_hf
        tot -= f.e_tot
    return hf, tot - hf


def sec_dimer(cache):
    print("== water dimer (the hydrogen bond: solvation, folding) ==")
    t = time.time()
    frags = [[0, 1, 2], [3, 4, 5]]
    comp = {}
    for method, basis in [("mp2", "cc-pvtz"), ("mp2", "cc-pvqz"), ("ccsd(t)", "cc-pvtz")]:
        hf, corr = quiet(_cp_components, WATER_DIMER, frags, method, basis)
        comp[f"{method}/{basis}"] = {"hf": hf, "corr": corr}
        print(f"  {method}/{basis}: HF {hf * HARTREE_TO_KCAL:.3f}, corr {corr * HARTREE_TO_KCAL:.3f} kcal/mol")
    # MP2/CBS (T->Q) correlation extrapolation; HF at QZ
    X, Y = 3, 4
    mp2_tz, mp2_qz = comp["mp2/cc-pvtz"], comp["mp2/cc-pvqz"]
    corr_cbs = (Y**3 * mp2_qz["corr"] - X**3 * mp2_tz["corr"]) / (Y**3 - X**3)
    hf_cbs = mp2_qz["hf"]
    e_mp2_cbs = hf_cbs + corr_cbs
    # focal-point: add [CCSD(T) - MP2] at TZ
    delta = comp["ccsd(t)/cc-pvtz"]["corr"] - comp["mp2/cc-pvtz"]["corr"]
    e_ccsdt_cbs = e_mp2_cbs + delta
    cache["dimer"] = {
        "components_kcal": {k: {"hf": v["hf"] * HARTREE_TO_KCAL, "corr": v["corr"] * HARTREE_TO_KCAL} for k, v in comp.items()},
        "e_mp2_cbs_kcal": e_mp2_cbs * HARTREE_TO_KCAL,
        "delta_ccsdt_mp2_tz_kcal": delta * HARTREE_TO_KCAL,
        "e_ccsdt_cbs_kcal": e_ccsdt_cbs * HARTREE_TO_KCAL,
        "scheme": "focal-point: MP2/CBS(T,Q) + [CCSD(T)-MP2]/cc-pVTZ, counterpoise-corrected",
        "seconds": time.time() - t,
    }
    print(f"  MP2/CBS {e_mp2_cbs * HARTREE_TO_KCAL:.3f} + delta(T) {delta * HARTREE_TO_KCAL:.3f} "
          f"-> CCSD(T)/CBS {e_ccsdt_cbs * HARTREE_TO_KCAL:.3f} kcal/mol (benchmark {REF['dimer_De'][0]})")
    save(cache)


def sec_peptide(cache):
    print("== peptide bond (links every amino acid; formamide model) ==")
    t = time.time()
    # accuracy-sensitive numbers (C-N length, planarity) at MP2/cc-pVTZ
    opt = quiet(optimize_geometry, FORMAMIDE, method="mp2", basis="cc-pvtz")
    g = geometry_parameters(opt)
    c_n = g["dist"](0, 2)
    improper = _dihedral(opt.coords, 1, 0, 3, 2)  # O-C-H(C)-N, ~0 for a planar peptide unit
    # rotation barrier: optimize planar and the 90-deg-rotated TS at MP2/cc-pVDZ,
    # then refine both with MP2/cc-pVTZ single points.
    con = "$set\ndihedral 5 3 1 2 90.0\n"  # 1-indexed: HN(5)-N(3)-C(1)-O(2)
    try:
        planar = quiet(optimize_geometry, FORMAMIDE, method="mp2", basis="cc-pvdz")
        ts = quiet(optimize_geometry, planar, method="mp2", basis="cc-pvdz", constraints=con)
        e_planar = quiet(single_point, planar, "mp2", "cc-pvtz").e_tot
        e_ts = quiet(single_point, ts, "mp2", "cc-pvtz").e_tot
        barrier = (e_ts - e_planar) * HARTREE_TO_KCAL
    except Exception as e:
        barrier = None
        print("  (barrier scan skipped:", str(e)[:80], ")")
    cache["peptide"] = {
        "cn_length": c_n, "improper_deg": improper, "barrier_kcal": barrier,
        "level": "MP2/cc-pVTZ geometry; barrier MP2/cc-pVTZ//cc-pVDZ", "seconds": time.time() - t,
    }
    print(f"  C-N bond {c_n:.3f} A (single ~1.47, double ~1.28; peptide ~{REF['peptide_CN'][0]}) -> partial double bond")
    print(f"  N out-of-plane {abs(improper):.1f} deg (planar peptide unit ~ 0)")
    if barrier is not None:
        print(f"  C-N rotation barrier {barrier:.1f} kcal/mol (expt {REF['formamide_barrier'][0]})")
    save(cache)


def sec_glycine(cache):
    print("== glycine (the simplest amino acid) ==")
    t = time.time()
    opt = quiet(optimize_geometry, GLYCINE, method="mp2", basis="cc-pvdz")
    g = geometry_parameters(opt)
    cache["glycine"] = {
        "cc_bond": g["dist"](1, 2), "c_n_bond": g["dist"](0, 1),
        "co_double": g["dist"](2, 3), "co_single": g["dist"](2, 4),
        "level": "MP2/cc-pVDZ", "seconds": time.time() - t,
    }
    print(f"  Calpha-C {g['dist'](1, 2):.3f} A, Calpha-N {g['dist'](0, 1):.3f} A, "
          f"C=O {g['dist'](2, 3):.3f} / C-O {g['dist'](2, 4):.3f} A (carboxyl: one short, one long)")
    save(cache)


def _dihedral(c, i, j, k, l):
    b1, b2, b3 = c[j] - c[i], c[k] - c[j], c[l] - c[k]
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x, y = np.dot(n1, n2), np.dot(m1, n2)
    return float(np.degrees(np.arctan2(y, x)))


SECTIONS = {"water": sec_water, "dimer": sec_dimer, "peptide": sec_peptide, "glycine": sec_glycine}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all", choices=["all", *SECTIONS])
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))
    cache = load_cache()
    cache["_reference_values"] = {k: {"value": v[0], "unit": v[1], "source": v[2]} for k, v in REF.items()}
    todo = SECTIONS if a.section == "all" else {a.section: SECTIONS[a.section]}
    for name, fn in todo.items():
        fn(cache)
    save(cache)
    print("wrote", CACHE)


if __name__ == "__main__":
    main()
