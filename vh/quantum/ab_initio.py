"""Ab initio electronic structure via PySCF.

A thin, well-defined interface over PySCF that exposes exactly the methods whose
accuracy is understood and ordered:

    HF  <  DFT  <  MP2  <  CCSD  <  CCSD(T)          (roughly, for closed-shell
                                                      main-group molecules)

and the two corrections needed to make a number trustworthy to sub-kcal/mol:

    - counterpoise (BSSE) correction for interaction energies, and
    - complete-basis-set (CBS) extrapolation.

Energies are returned in Hartree (atomic units); helpers convert to kcal/mol and
kJ/mol. Geometries are Angstrom.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HARTREE_TO_KCAL = 627.509474063   # CODATA-consistent
HARTREE_TO_KJ = 2625.499639479
BOHR_TO_ANGSTROM = 0.529177210903


@dataclass
class Molecule:
    """A molecule: atoms as (element, x, y, z) in Angstrom, plus charge and spin.

    `ghost` marks atoms that keep their basis functions but carry no electrons or
    nuclear charge -- used for the counterpoise correction.
    """

    atoms: list                      # [(symbol, (x, y, z)), ...]
    charge: int = 0
    spin: int = 0                    # 2S = n_alpha - n_beta
    name: str = "molecule"
    ghost: list = field(default_factory=list)   # indices of ghost atoms

    @classmethod
    def from_xyz_block(cls, block: str, **kw) -> "Molecule":
        atoms = []
        for line in block.strip().splitlines():
            p = line.split()
            if len(p) >= 4:
                atoms.append((p[0], (float(p[1]), float(p[2]), float(p[3]))))
        return cls(atoms=atoms, **kw)

    def pyscf_atom(self):
        out = []
        for i, (sym, xyz) in enumerate(self.atoms):
            label = f"ghost-{sym}" if i in self.ghost else sym
            out.append([label, tuple(xyz)])
        return out

    def with_ghosts(self, real_indices) -> "Molecule":
        """Same geometry, but every atom not in `real_indices` becomes a ghost
        (keeps its basis, loses its electrons). For counterpoise fragments."""
        real = set(real_indices)
        charge = self.charge if set(range(len(self.atoms))) == real else 0
        return Molecule(atoms=self.atoms, charge=charge if real == set(range(len(self.atoms))) else 0,
                        spin=self.spin if real == set(range(len(self.atoms))) else 0,
                        name=self.name, ghost=[i for i in range(len(self.atoms)) if i not in real])

    @property
    def coords(self):
        return np.array([a[1] for a in self.atoms], dtype=float)


def _build_mol(mol: Molecule, basis: str, verbose=0):
    from pyscf import gto
    return gto.M(atom=mol.pyscf_atom(), basis=basis, charge=mol.charge, spin=mol.spin,
                 unit="Angstrom", verbose=verbose)


@dataclass
class QMResult:
    method: str
    basis: str
    e_tot: float                 # Hartree
    e_hf: float
    e_corr: float = 0.0
    converged: bool = True
    extra: dict = field(default_factory=dict)

    def kcal(self):
        return self.e_tot * HARTREE_TO_KCAL


def single_point(mol: Molecule, method: str = "ccsd(t)", basis: str = "cc-pvtz",
                 xc: str = "b3lyp", density_fit: bool = False, verbose: int = 0) -> QMResult:
    """One energy evaluation. method in {hf, dft, mp2, ccsd, ccsd(t)} (case-insensitive)."""
    from pyscf import cc, dft, mp, scf

    m = _build_mol(mol, basis, verbose)
    method = method.lower()
    restricted = (m.spin == 0)
    if method == "dft":
        mf = (dft.RKS(m) if restricted else dft.UKS(m))
        mf.xc = xc
        if density_fit:
            mf = mf.density_fit()
        mf.run()
        return QMResult("dft/" + xc, basis, float(mf.e_tot), float(mf.e_tot), 0.0, bool(mf.converged))

    mf = (scf.RHF(m) if restricted else scf.UHF(m))
    if density_fit:
        mf = mf.density_fit()
    mf.run()
    e_hf = float(mf.e_tot)
    if method == "hf":
        return QMResult("hf", basis, e_hf, e_hf, 0.0, bool(mf.converged))
    if method == "mp2":
        pt = mp.MP2(mf).run()
        return QMResult("mp2", basis, float(pt.e_tot), e_hf, float(pt.e_corr), True)
    if method in ("ccsd", "ccsd(t)"):
        mc = cc.CCSD(mf).run()
        e_corr = float(mc.e_corr)
        e_tot = float(mc.e_tot)
        extra = {}
        if method == "ccsd(t)":
            et = float(mc.ccsd_t())
            e_corr += et
            e_tot += et
            extra["e_triples"] = et
        return QMResult(method, basis, e_tot, e_hf, e_corr, bool(mc.converged), extra)
    raise ValueError(f"unknown method {method!r}")


def interaction_energy(complex_mol: Molecule, fragments: list, method="ccsd(t)", basis="cc-pvtz",
                       counterpoise: bool = True, verbose: int = 0) -> dict:
    """Interaction energy of a complex split into fragments (list of atom-index lists).

    With counterpoise=True, each fragment is evaluated in the FULL basis of the
    complex (the other atoms present as ghosts), which removes basis-set
    superposition error -- the standard rigorous treatment.

    Returns energies in Hartree and kcal/mol, plus the BSSE.
    """
    e_complex = single_point(complex_mol, method, basis, verbose=verbose).e_tot
    e_frags_cp, e_frags_nocp = 0.0, 0.0
    for idx in fragments:
        e_frags_cp += single_point(complex_mol.with_ghosts(idx), method, basis, verbose=verbose).e_tot
        sub = Molecule(atoms=[complex_mol.atoms[i] for i in idx], name="frag")
        e_frags_nocp += single_point(sub, method, basis, verbose=verbose).e_tot
    e_int_cp = e_complex - e_frags_cp
    e_int_nocp = e_complex - e_frags_nocp
    out = {
        "method": method, "basis": basis,
        "e_int_hartree": e_int_cp if counterpoise else e_int_nocp,
        "e_int_kcal": (e_int_cp if counterpoise else e_int_nocp) * HARTREE_TO_KCAL,
        "e_int_counterpoise_kcal": e_int_cp * HARTREE_TO_KCAL,
        "e_int_raw_kcal": e_int_nocp * HARTREE_TO_KCAL,
        "bsse_kcal": (e_int_nocp - e_int_cp) * HARTREE_TO_KCAL,
    }
    return out


# cardinal numbers for Dunning cc-pVXZ / aug-cc-pVXZ bases
_CARDINAL = {"d": 2, "t": 3, "q": 4, "5": 5, "6": 6}


def _cardinal(basis: str) -> int:
    b = basis.lower().replace("aug-", "")
    key = b.split("pv")[-1][0] if "pv" in b else None
    if key not in _CARDINAL:
        raise ValueError(f"cannot infer cardinal number from basis {basis!r}")
    return _CARDINAL[key]


def cbs_helgaker(result_low: QMResult, result_high: QMResult) -> dict:
    """Two-point complete-basis-set extrapolation (Helgaker scheme).

    HF energy converges exponentially; the correlation energy converges as
    X**-3. Using two consecutive cardinal numbers X < Y:

        E_corr(CBS) = (Y**3 E_corr(Y) - X**3 E_corr(X)) / (Y**3 - X**3)
        E_HF(CBS)   = E_HF(Y)   (the larger-basis HF, already near-converged)

    The CBS total energy is the quantity closest to the exact non-relativistic
    Born-Oppenheimer energy for the given method.
    """
    X, Y = _cardinal(result_low.basis), _cardinal(result_high.basis)
    if not X < Y:
        raise ValueError("result_low must use the smaller basis")
    c = Y**3 / (Y**3 - X**3)
    e_corr_cbs = result_high.e_corr + (result_high.e_corr - result_low.e_corr) * (X**3) / (Y**3 - X**3)
    # equivalently (Y^3 Ec(Y) - X^3 Ec(X))/(Y^3 - X^3); keep the stable form:
    e_corr_cbs = (Y**3 * result_high.e_corr - X**3 * result_low.e_corr) / (Y**3 - X**3)
    e_hf_cbs = result_high.e_hf
    return {
        "e_hf_cbs": e_hf_cbs, "e_corr_cbs": e_corr_cbs, "e_tot_cbs": e_hf_cbs + e_corr_cbs,
        "e_tot_cbs_kcal": (e_hf_cbs + e_corr_cbs) * HARTREE_TO_KCAL,
        "cardinals": (X, Y), "coefficient": c,
    }


def interaction_energy_cbs(complex_mol, fragments, method="ccsd(t)",
                           bases=("cc-pvtz", "cc-pvqz"), counterpoise=True, verbose=0) -> dict:
    """Counterpoise-corrected interaction energy extrapolated to the CBS limit.

    HF interaction is taken at the larger basis; correlation interaction is
    X**-3 extrapolated. This is the standard recipe behind benchmark hydrogen-bond
    and binding energies.
    """
    def e_int_components(basis):
        ec = single_point(complex_mol, method, basis, verbose=verbose)
        hf = ec.e_hf - sum(single_point(complex_mol.with_ghosts(idx) if counterpoise
                                         else Molecule([complex_mol.atoms[i] for i in idx]),
                                         "hf", basis, verbose=verbose).e_hf for idx in fragments)
        tot = ec.e_tot - sum(single_point(complex_mol.with_ghosts(idx) if counterpoise
                                           else Molecule([complex_mol.atoms[i] for i in idx]),
                                           method, basis, verbose=verbose).e_tot for idx in fragments)
        return hf, tot - hf  # (HF interaction, correlation interaction)

    (hf_lo, corr_lo), (hf_hi, corr_hi) = e_int_components(bases[0]), e_int_components(bases[1])
    X, Y = _cardinal(bases[0]), _cardinal(bases[1])
    corr_cbs = (Y**3 * corr_hi - X**3 * corr_lo) / (Y**3 - X**3)
    e_int_cbs = hf_hi + corr_cbs
    return {
        "method": method, "bases": bases, "counterpoise": counterpoise,
        "e_int_cbs_kcal": e_int_cbs * HARTREE_TO_KCAL,
        "hf_int_kcal": hf_hi * HARTREE_TO_KCAL,
        "corr_int_cbs_kcal": corr_cbs * HARTREE_TO_KCAL,
        f"e_int_{bases[0]}_kcal": (hf_lo + corr_lo) * HARTREE_TO_KCAL,
        f"e_int_{bases[1]}_kcal": (hf_hi + corr_hi) * HARTREE_TO_KCAL,
    }


def optimize_geometry(mol: Molecule, method="mp2", basis="cc-pvtz", xc="b3lyp", verbose=0,
                      maxsteps: int = 50, constraints: str | None = None) -> Molecule:
    """Relax the geometry to a stationary point on the chosen surface.

    Supports hf, dft, mp2, ccsd (all have analytic gradients in PySCF). For
    CCSD(T) accuracy, optimize at mp2 or ccsd and take a CCSD(T) single point --
    the standard, and much cheaper, division of labor. `constraints` is an
    optional geomeTRIC constraint file body (e.g. to freeze a dihedral).

    Returns a new Molecule with optimized coordinates (Angstrom).
    """
    import tempfile

    from pyscf import cc, dft, mp, scf
    from pyscf.geomopt.geometric_solver import optimize as geomopt

    m = _build_mol(mol, basis, verbose)
    method = method.lower()
    restricted = (m.spin == 0)
    if method == "dft":
        obj = (dft.RKS(m) if restricted else dft.UKS(m)); obj.xc = xc
    elif method == "hf":
        obj = scf.RHF(m) if restricted else scf.UHF(m)
    elif method == "mp2":
        obj = mp.MP2(scf.RHF(m) if restricted else scf.UHF(m))
    elif method == "ccsd":
        obj = cc.CCSD(scf.RHF(m) if restricted else scf.UHF(m))
    else:
        raise ValueError(f"geometry optimization not supported for {method!r}; "
                         "optimize at mp2/ccsd and take a CCSD(T) single point")
    kw = {"maxsteps": maxsteps}
    cfile = None
    if constraints:
        cfile = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        cfile.write(constraints); cfile.flush()
        kw["constraints"] = cfile.name
    opt = geomopt(obj, **kw)
    coords = opt.atom_coords(unit="Angstrom")
    return Molecule(atoms=[(mol.atoms[i][0], tuple(coords[i])) for i in range(len(mol.atoms))],
                    charge=mol.charge, spin=mol.spin, name=mol.name + "_opt")


def geometry_parameters(mol: Molecule) -> dict:
    """Common internal coordinates for validation (bond lengths and angles)."""
    c = mol.coords
    def dist(i, j):
        return float(np.linalg.norm(c[i] - c[j]))
    def angle(i, j, k):
        u, v = c[i] - c[j], c[k] - c[j]
        return float(np.degrees(np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))))
    return {"dist": dist, "angle": angle}
