"""QM/MM electrostatic embedding: a quantum region inside a classical environment.

The quantum region (a drug, a reacting group, a chromophore) is treated by real
electronic structure; its surroundings (protein, water) enter as classical point
charges that polarize the quantum wavefunction, and feel its electrostatic field
in return. This is how molecular-accuracy physics reaches into a system far too
large to treat entirely with quantum mechanics -- the bridge between
`vh.quantum` and the classical `vh.molecular` world.

Built on PySCF's `qmmm` electrostatic embedding. Energies in Hartree, forces in
Hartree/Angstrom, geometries in Angstrom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ab_initio import HARTREE_TO_KCAL, Molecule, _build_mol


@dataclass
class QMMMResult:
    e_tot: float                 # Hartree, total QM energy in the MM field
    e_qm_isolated: float         # Hartree, same QM region with no MM charges
    e_interaction_kcal: float    # (e_tot - e_qm_isolated) * conversion
    qm_grad: np.ndarray | None = None   # Hartree/Angstrom on QM atoms
    mm_grad: np.ndarray | None = None   # Hartree/Angstrom on MM point charges


def embed(qm: Molecule, mm_coords, mm_charges, method="hf", basis="cc-pvdz",
          xc="b3lyp", grad=False, verbose=0) -> QMMMResult:
    """Energy (and optional forces) of the QM region polarized by MM point charges.

    mm_coords: (M,3) Angstrom, mm_charges: (M,) in e.
    """
    from pyscf import cc, dft, mp, qmmm, scf

    mm_coords = np.asarray(mm_coords, dtype=float)
    mm_charges = np.asarray(mm_charges, dtype=float)
    m = _build_mol(qm, basis, verbose)
    method = method.lower()
    base = (dft.RKS(m) if m.spin == 0 else dft.UKS(m)) if method == "dft" else (scf.RHF(m) if m.spin == 0 else scf.UHF(m))
    if method == "dft":
        base.xc = xc
    mf = qmmm.mm_charge(base, mm_coords, mm_charges, unit="Angstrom").run(verbose=verbose)
    e_embedded_scf = float(mf.e_tot)

    if method in ("mp2", "ccsd", "ccsd(t)"):
        post = {"mp2": mp.MP2, "ccsd": cc.CCSD, "ccsd(t)": cc.CCSD}[method](mf).run(verbose=verbose)
        e_tot = float(post.e_tot)
        if method == "ccsd(t)":
            e_tot += float(post.ccsd_t())
    else:
        e_tot = e_embedded_scf

    # isolated QM reference (same method, no MM charges)
    from .ab_initio import single_point
    e_iso = single_point(qm, method, basis, xc=xc, verbose=verbose).e_tot

    qm_grad = mm_grad = None
    if grad:
        scanner = mf.nuc_grad_method()
        qm_grad = np.asarray(scanner.run(verbose=verbose).de) / 0.529177210903  # Ha/Bohr -> Ha/Angstrom
        # force on MM charges from the QM electrostatic potential + nuclei
        mm_grad = _mm_gradient(mf, m, mm_coords, mm_charges) / 0.529177210903

    return QMMMResult(e_tot, e_iso, (e_tot - e_iso) * HARTREE_TO_KCAL, qm_grad, mm_grad)


def _mm_gradient(mf, mol, mm_coords, mm_charges):
    """Gradient (Hartree/Bohr) on the MM point charges: interaction with the QM
    electron density and nuclei."""
    from pyscf import df
    coords_b = np.asarray(mm_coords) / 0.529177210903
    dm = mf.make_rdm1()
    # electronic part: d/dR_mm of -∫ rho(r) q/|r-R| dr
    g = np.zeros_like(coords_b)
    for i, (r, q) in enumerate(zip(coords_b, mm_charges)):
        with mol.with_rinv_origin(r):
            v = mol.intor("int1e_iprinv", comp=3)   # d/dr 1/|r-R| integrals
        # force on the charge is +q * d/dR (attraction/repulsion); sign bookkeeping below
        f_el = q * np.einsum("xij,ji->x", v, dm).real
        g[i] += f_el
    # nuclear part
    qm_charges = mol.atom_charges()
    qm_coords = mol.atom_coords()  # Bohr
    for i, (r, q) in enumerate(zip(coords_b, mm_charges)):
        d = r - qm_coords
        dist = np.linalg.norm(d, axis=1)
        g[i] += -q * np.einsum("a,ax,a->x", qm_charges, d, 1 / dist**3)
    return g


def classical_charge_dipole_energy(mm_coords, mm_charges, dipole_debye, dipole_center):
    """Asymptotic classical interaction of a set of point charges with a fixed
    dipole (Debye) at `dipole_center` -- the long-range limit QM/MM must recover.
    Returns kcal/mol.
    """
    ONE_4PI_EPS0_KCAL = 332.0637128   # e^2/Angstrom -> kcal/mol
    mu = np.asarray(dipole_debye) / 4.80320471257   # Debye -> e*Angstrom
    e = 0.0
    for r, q in zip(np.asarray(mm_coords), np.asarray(mm_charges)):
        d = np.asarray(r) - np.asarray(dipole_center)   # field point (charge) relative to dipole
        dist = np.linalg.norm(d)
        e += ONE_4PI_EPS0_KCAL * q * np.dot(mu, d) / dist**3
    return e
