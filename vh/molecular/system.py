"""Engine-native representation of a molecular system.

A `MolecularSystem` is just arrays: per-atom properties plus lists of bonded
terms. It is independent of how it was built (our own builders, or the
OpenMM-based parameter compiler in `prep.py`), and it is what both the JAX
engine and the JavaScript port consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np


@dataclass
class MolecularSystem:
    # --- per atom -----------------------------------------------------------
    masses: np.ndarray            # (N,) amu
    charges: np.ndarray           # (N,) e
    sigma: np.ndarray             # (N,) nm      Lennard-Jones
    epsilon: np.ndarray           # (N,) kJ/mol  Lennard-Jones
    # --- bonded terms -------------------------------------------------------
    bonds: np.ndarray             # (Nb, 2) int
    bond_k: np.ndarray            # (Nb,) kJ/mol/nm^2   E = k/2 (r - r0)^2
    bond_r0: np.ndarray           # (Nb,) nm
    angles: np.ndarray            # (Na, 3) int
    angle_k: np.ndarray           # (Na,) kJ/mol/rad^2  E = k/2 (t - t0)^2
    angle_t0: np.ndarray          # (Na,) rad
    torsions: np.ndarray          # (Nt, 4) int   (propers and impropers)
    torsion_k: np.ndarray         # (Nt,) kJ/mol        E = k (1 + cos(n phi - phase))
    torsion_n: np.ndarray         # (Nt,) int
    torsion_phase: np.ndarray     # (Nt,) rad
    # --- nonbonded exceptions ----------------------------------------------
    # Every pair listed here is removed from the normal LJ/Coulomb sum.
    # Pairs with nonzero qq/eps (1-4 pairs) are added back with these params.
    exceptions: np.ndarray        # (Ne, 2) int
    exc_qq: np.ndarray            # (Ne,) e^2 (already scaled, e.g. 1/1.2 for 1-4)
    exc_sigma: np.ndarray         # (Ne,) nm
    exc_eps: np.ndarray           # (Ne,) kJ/mol (already scaled, e.g. 1/2 for 1-4)
    # --- implicit solvent (GB-OBC2) ----------------------------------------
    gb_radius: np.ndarray | None = None   # (N,) nm intrinsic radius
    gb_scale: np.ndarray | None = None    # (N,) HCT overlap scale factor
    solute_dielectric: float = 1.0
    solvent_dielectric: float = 78.5
    surface_area_energy: float = 2.25936  # kJ/mol/nm^2 (ACE nonpolar term)
    # OpenMM's implicit/obc2.xml uses 138.935485 in the GB terms (older value of
    # 1/(4 pi eps0)) and omits the HCT "engulfed atom" correction; both are
    # recorded so the engine reproduces whichever variant was parameterized.
    gb_coulomb_constant: float = 138.935485
    gb_engulfed_correction: bool = False
    # --- explicit solvent ------------------------------------------------------
    box: np.ndarray | None = None              # (3,) nm, orthorhombic periodic box
    constraints: np.ndarray | None = None      # (Nc, 2) int, fixed-distance pairs (rigid water)
    constraint_dist: np.ndarray | None = None  # (Nc,) nm
    molecule_ids: np.ndarray | None = None     # (N,) int, molecule each atom belongs to
    # --- labels (for output only) ------------------------------------------
    elements: list = field(default_factory=list)
    atom_names: list = field(default_factory=list)
    residue_names: list = field(default_factory=list)
    residue_ids: list = field(default_factory=list)
    name: str = "system"

    @property
    def n_atoms(self) -> int:
        return int(self.masses.shape[0])

    @property
    def has_gb(self) -> bool:
        return self.gb_radius is not None

    @property
    def net_charge(self) -> float:
        return float(np.sum(self.charges))

    def summary(self) -> str:
        return (
            f"{self.name}: {self.n_atoms} atoms, {len(set(self.residue_ids))} residues, "
            f"net charge {self.net_charge:+.2f} e | {len(self.bonds)} bonds, "
            f"{len(self.angles)} angles, {len(self.torsions)} torsions, "
            f"{len(self.exceptions)} exceptions | solvent: "
            f"{'GB-OBC2 implicit water' if self.has_gb else 'vacuum'}"
        )

    # --- serialization (used by the JS port and for caching) ---------------
    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = v.tolist() if isinstance(v, np.ndarray) else v
        return out

    @property
    def is_periodic(self) -> bool:
        return self.box is not None

    @property
    def n_molecules(self) -> int:
        return 0 if self.molecule_ids is None else int(np.max(self.molecule_ids)) + 1

    @classmethod
    def from_dict(cls, d: dict) -> "MolecularSystem":
        int_fields = {"bonds", "angles", "torsions", "exceptions", "torsion_n", "constraints", "molecule_ids"}
        kw = {}
        for f in fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            if isinstance(v, list) and f.name not in {"elements", "atom_names", "residue_names", "residue_ids"}:
                v = np.asarray(v, dtype=np.int64 if f.name in int_fields else np.float64)
            kw[f.name] = v
        return cls(**kw)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load_json(cls, path: str | Path) -> "MolecularSystem":
        return cls.from_dict(json.loads(Path(path).read_text()))
