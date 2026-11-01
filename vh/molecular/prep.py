"""Structure preparation and force-field parameter assignment.

This is the one place the project leans on an external package: OpenMM's
`ForceField` is used as a *parameter compiler*. It reads a PDB, adds missing
atoms and hydrogens, matches residues to AMBER ff14SB templates and assigns
GB-OBC2 radii. We then copy every number out into our own `MolecularSystem`.
No simulation physics is delegated: energies, forces and dynamics are all
computed by `vh.molecular.energy` / `vh.molecular.integrate`.

Replacing this step with a native template matcher is on the roadmap
(docs/roadmap.md); OpenMM would then only be used as an independent referee
in the test suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .system import MolecularSystem

DEFAULT_FORCEFIELD = ("amber14-all.xml", "implicit/obc2.xml")


def _require_openmm():
    try:
        import openmm  # noqa: F401
        from openmm import app  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("structure preparation needs `pip install openmm pdbfixer`") from e


def prepare_pdb(path: str | Path, ph: float = 7.0, forcefield=DEFAULT_FORCEFIELD):
    """PDB file -> (openmm Topology, positions in nm (N,3), openmm System).

    Heterogens and waters are removed, missing heavy atoms are added, and all
    hydrogens are rebuilt at the requested pH so that names match ff14SB.
    """
    _require_openmm()
    from openmm import app, unit
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(path))
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.missingResidues = {}          # never invent missing loops
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    modeller = app.Modeller(fixer.topology, fixer.positions)
    hydrogens = [a for a in modeller.topology.atoms() if a.element is not None and a.element.symbol == "H"]
    modeller.delete(hydrogens)
    ff = app.ForceField(*forcefield)
    modeller.addHydrogens(ff, pH=ph)
    omm_system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )
    pos = np.array(modeller.positions.value_in_unit(unit.nanometer), dtype=np.float64)
    return modeller.topology, pos, omm_system


_NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def _parse_obc2_customgb(force) -> dict:
    """Read OpenMM's implicit/obc2.xml CustomGBForce and verify it is the OBC2
    model we implement. Returns per-atom parameters and the constants used.

    We check the expression structure (whitespace-insensitive) rather than
    trusting the file name, so a different GB variant fails loudly instead of
    being silently mis-simulated.
    """
    import re

    names = [force.getPerParticleParameterName(i) for i in range(force.getNumPerParticleParameters())]
    if names != ["charge", "or", "sr"] or force.getNumComputedValues() != 2 or force.getNumEnergyTerms() != 3:
        raise NotImplementedError("unrecognized CustomGBForce (expected OpenMM implicit/obc2.xml)")
    strip = lambda s: re.sub(r"\s+", "", s)  # noqa: E731
    i_expr = strip(force.getComputedValueParameters(0)[1])
    b_expr = strip(force.getComputedValueParameters(1)[1])
    e_self = strip(force.getEnergyTermParameters(0)[0])
    e_sa = strip(force.getEnergyTermParameters(1)[0])
    e_pair = strip(force.getEnergyTermParameters(2)[0])
    expected_i = strip("select(step(r+sr2-or1), 0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r), 0);U=r+sr2;L=max(or1, D);D=abs(r-sr2)")
    if i_expr != expected_i:
        raise NotImplementedError(f"unexpected Born integral expression: {i_expr}")
    m = re.fullmatch(
        r"1/\(1/or-tanh\(psi-" + _NUM + r"\*psi\^2\+" + _NUM + r"\*psi\^3\)/radius\);psi=I\*or;radius=or\+offset;offset=" + _NUM,
        b_expr,
    )
    if not m:
        raise NotImplementedError(f"unexpected Born radius expression: {b_expr}")
    beta, gamma, offset = map(float, m.groups())
    k_self = float(re.match(r"-0\.5\*" + _NUM, e_self).group(1))
    k_pair = float(re.match(r"-" + _NUM, e_pair).group(1))
    sa_m = re.match(_NUM + r"\*\(radius\+" + _NUM + r"\)\^2\*\(radius/B\)\^6", e_sa)
    solvent = float(re.search(r"solventDielectric=" + _NUM, e_pair).group(1))
    solute = float(re.search(r"soluteDielectric=" + _NUM, e_pair).group(1))
    kappa = float(re.search(r"kappa=" + _NUM, e_pair).group(1))
    if not (sa_m and abs(k_self - k_pair) < 1e-12 and kappa == 0.0 and abs(beta - 0.8) < 1e-12 and abs(gamma - 4.85) < 1e-12):
        raise NotImplementedError("unsupported OBC2 variant (salt screening or modified coefficients)")
    sa_factor, probe = map(float, sa_m.groups())
    if abs(probe - 0.14) > 1e-12:
        raise NotImplementedError("unsupported ACE probe radius")
    p = np.array([force.getParticleParameters(i) for i in range(force.getNumParticles())], dtype=np.float64)
    charge, orad, srad = p[:, 0], p[:, 1], p[:, 2]
    return dict(
        charge=charge, radius=orad + offset, scale=srad / orad,
        solute=solute, solvent=solvent, sa=sa_factor / (4.0 * np.pi),
        k_gb=k_pair, offset=offset, engulfed=False,
    )


def system_from_openmm(omm_system, topology, name: str = "system") -> MolecularSystem:
    """Copy every parameter out of an OpenMM System into a MolecularSystem."""
    _require_openmm()
    import openmm as mm
    from openmm import unit

    N = omm_system.getNumParticles()
    masses = np.array([omm_system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(N)])
    out = dict(masses=masses)
    gb = None
    seen = set()
    for force in omm_system.getForces():
        kind = type(force).__name__
        seen.add(kind)
        if isinstance(force, mm.HarmonicBondForce):
            rows = [force.getBondParameters(i) for i in range(force.getNumBonds())]
            out["bonds"] = np.array([[r[0], r[1]] for r in rows], dtype=np.int64).reshape(-1, 2)
            out["bond_r0"] = np.array([r[2].value_in_unit(unit.nanometer) for r in rows])
            out["bond_k"] = np.array([r[3].value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2) for r in rows])
        elif isinstance(force, mm.HarmonicAngleForce):
            rows = [force.getAngleParameters(i) for i in range(force.getNumAngles())]
            out["angles"] = np.array([[r[0], r[1], r[2]] for r in rows], dtype=np.int64).reshape(-1, 3)
            out["angle_t0"] = np.array([r[3].value_in_unit(unit.radian) for r in rows])
            out["angle_k"] = np.array([r[4].value_in_unit(unit.kilojoule_per_mole / unit.radian**2) for r in rows])
        elif isinstance(force, mm.PeriodicTorsionForce):
            rows = [force.getTorsionParameters(i) for i in range(force.getNumTorsions())]
            out["torsions"] = np.array([[r[0], r[1], r[2], r[3]] for r in rows], dtype=np.int64).reshape(-1, 4)
            out["torsion_n"] = np.array([r[4] for r in rows], dtype=np.int64)
            out["torsion_phase"] = np.array([r[5].value_in_unit(unit.radian) for r in rows])
            out["torsion_k"] = np.array([r[6].value_in_unit(unit.kilojoule_per_mole) for r in rows])
        elif isinstance(force, mm.NonbondedForce):
            if force.getNonbondedMethod() not in (mm.NonbondedForce.NoCutoff, mm.NonbondedForce.Ewald, mm.NonbondedForce.PME):
                raise NotImplementedError("supported nonbonded methods: NoCutoff (implicit), Ewald / PME (explicit)")
            p = [force.getParticleParameters(i) for i in range(N)]
            out["charges"] = np.array([x[0].value_in_unit(unit.elementary_charge) for x in p])
            out["sigma"] = np.array([x[1].value_in_unit(unit.nanometer) for x in p])
            out["epsilon"] = np.array([x[2].value_in_unit(unit.kilojoule_per_mole) for x in p])
            e = [force.getExceptionParameters(i) for i in range(force.getNumExceptions())]
            out["exceptions"] = np.array([[x[0], x[1]] for x in e], dtype=np.int64).reshape(-1, 2)
            out["exc_qq"] = np.array([x[2].value_in_unit(unit.elementary_charge**2) for x in e])
            out["exc_sigma"] = np.array([x[3].value_in_unit(unit.nanometer) for x in e])
            out["exc_eps"] = np.array([x[4].value_in_unit(unit.kilojoule_per_mole) for x in e])
        elif isinstance(force, mm.CustomGBForce):
            gb = _parse_obc2_customgb(force)
        elif isinstance(force, mm.GBSAOBCForce):
            p = [force.getParticleParameters(i) for i in range(N)]
            gb = dict(
                charge=np.array([x[0].value_in_unit(unit.elementary_charge) for x in p]),
                radius=np.array([x[1].value_in_unit(unit.nanometer) for x in p]),
                scale=np.array([float(x[2]) for x in p]),
                solute=force.getSoluteDielectric(),
                solvent=force.getSolventDielectric(),
                sa=force.getSurfaceAreaEnergy().value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2),
                k_gb=138.935456, engulfed=True,  # native OpenMM GBSAOBCForce conventions
            )
        elif isinstance(force, mm.CMMotionRemover):
            continue
        else:
            raise NotImplementedError(f"force type {kind} is not supported in v0.1")

    if gb is not None:
        if not np.allclose(gb["charge"], out["charges"], atol=1e-12):
            raise ValueError("GB and nonbonded charges differ")
        out.update(
            gb_radius=gb["radius"], gb_scale=gb["scale"],
            solute_dielectric=gb["solute"], solvent_dielectric=gb["solvent"],
            surface_area_energy=gb["sa"], gb_coulomb_constant=gb["k_gb"],
            gb_engulfed_correction=gb["engulfed"],
        )

    atoms = list(topology.atoms())
    out.update(
        elements=[a.element.symbol if a.element is not None else "X" for a in atoms],
        atom_names=[a.name for a in atoms],
        residue_names=[a.residue.name for a in atoms],
        residue_ids=[int(a.residue.index) for a in atoms],
        name=name,
    )

    # explicit solvent: periodic box, constraints, molecule membership
    if omm_system.usesPeriodicBoundaryConditions():
        bv = omm_system.getDefaultPeriodicBoxVectors()
        box = np.array([bv[i][i].value_in_unit(unit.nanometer) for i in range(3)])
        off = [bv[i][j].value_in_unit(unit.nanometer) for i in range(3) for j in range(3) if i != j]
        if np.max(np.abs(off)) > 1e-12:
            raise NotImplementedError("only orthorhombic boxes are supported")
        out["box"] = box
    nc = omm_system.getNumConstraints()
    if nc:
        rows = [omm_system.getConstraintParameters(i) for i in range(nc)]
        out["constraints"] = np.array([[r[0], r[1]] for r in rows], dtype=np.int64)
        out["constraint_dist"] = np.array([r[2].value_in_unit(unit.nanometer) for r in rows])
    # molecules = connected components of bonds + constraints + exceptions
    parent = list(range(N))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    links = [out.get("bonds", np.zeros((0, 2), int)), out.get("exceptions", np.zeros((0, 2), int))]
    if nc:
        links.append(out["constraints"])
    for arr in links:
        for a, b in np.asarray(arr).reshape(-1, 2):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb
    roots = {}
    out["molecule_ids"] = np.array([roots.setdefault(find(i), len(roots)) for i in range(N)], dtype=np.int64)
    for key in ("torsions", "angles", "bonds"):
        out.setdefault(key, np.zeros((0, {"bonds": 2, "angles": 3, "torsions": 4}[key]), dtype=np.int64))
    out.setdefault("torsion_n", np.zeros(0, dtype=np.int64))
    for key in ("bond_k", "bond_r0", "angle_k", "angle_t0", "torsion_k", "torsion_phase"):
        out.setdefault(key, np.zeros(0))
    return MolecularSystem(**out)


def water_box(box_nm: float = 1.9, cutoff_nm: float = 0.9, model: str = "tip3p", ewald_tol: float = 5e-6):
    """A cubic box of rigid water, parameterized by OpenMM's ForceField and copied
    into a MolecularSystem. Returns (system, positions nm, topology, openmm System).

    The OpenMM System (Ewald electrostatics, rigid water, dispersion correction)
    is returned so tests can use it as an independent referee.
    """
    _require_openmm()
    import openmm as mm
    from openmm import app, unit

    ff = app.ForceField("amber14-all.xml", f"amber14/{model}.xml")
    mod = app.Modeller(app.Topology(), [])
    mod.addSolvent(ff, boxSize=mm.Vec3(box_nm, box_nm, box_nm) * unit.nanometer, model=model)
    omm_system = ff.createSystem(mod.topology, nonbondedMethod=app.Ewald, nonbondedCutoff=cutoff_nm * unit.nanometer,
                                 rigidWater=True, ewaldErrorTolerance=ewald_tol, removeCMMotion=False)
    pos = np.array(mod.positions.value_in_unit(unit.nanometer), dtype=np.float64)
    system = system_from_openmm(omm_system, mod.topology, name=f"{model} water box")
    return system, pos, mod.topology, omm_system


def load_protein(path: str | Path, ph: float = 7.0, name: str | None = None):
    """Convenience: PDB -> (MolecularSystem, positions nm, openmm topology, openmm System)."""
    topology, pos, omm_system = prepare_pdb(path, ph=ph)
    system = system_from_openmm(omm_system, topology, name=name or Path(path).stem)
    return system, pos, topology, omm_system
