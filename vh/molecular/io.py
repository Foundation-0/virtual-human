"""Output: multi-model PDB trajectories (open in PyMOL / ChimeraX / VMD) and
compact JSON for the browser demo."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .system import MolecularSystem


def write_pdb_trajectory(path: str | Path, system: MolecularSystem, frames_nm, times_ps=None) -> None:
    frames = np.asarray(frames_nm)
    if frames.ndim == 2:
        frames = frames[None]
    lines = [f"REMARK   1 {system.name} written by virtual-human vh.molecular"]
    for k, x in enumerate(frames):
        lines.append(f"MODEL     {k + 1:4d}")
        if times_ps is not None:
            lines.append(f"REMARK   2 t = {float(times_ps[k]):.3f} ps")
        res_serial = {}
        for i in range(system.n_atoms):
            rid = system.residue_ids[i] if system.residue_ids else 0
            res_serial.setdefault(rid, len(res_serial) + 1)
            name = system.atom_names[i] if system.atom_names else "X"
            el = system.elements[i] if system.elements else "X"
            rname = system.residue_names[i] if system.residue_names else "UNK"
            name_f = f" {name:<3s}" if len(name) < 4 else name[:4]
            xa = x[i] * 10.0  # nm -> Angstrom
            lines.append(
                f"ATOM  {i + 1:5d} {name_f} {rname:>3s} A{res_serial[rid]:4d}    "
                f"{xa[0]:8.3f}{xa[1]:8.3f}{xa[2]:8.3f}  1.00  0.00          {el:>2s}"
            )
        lines.append("ENDMDL")
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def write_energy_log(path: str | Path, log: dict) -> None:
    keys = list(log.keys())
    rows = [",".join(keys)]
    for i in range(len(log[keys[0]])):
        rows.append(",".join(f"{log[k][i]:.6g}" for k in keys))
    Path(path).write_text("\n".join(rows) + "\n")


def trajectory_json(system: MolecularSystem, frames_nm, times_ps, decimals: int = 3) -> dict:
    """Compact trajectory for the web viewer: coordinates in Angstrom, rounded."""
    frames = np.round(np.asarray(frames_nm) * 10.0, decimals)
    return {
        "name": system.name,
        "elements": list(system.elements),
        "atom_names": list(system.atom_names),
        "residue_names": list(system.residue_names),
        "residue_ids": [int(r) for r in system.residue_ids],
        "bonds": np.asarray(system.bonds).tolist(),
        "times_ps": [round(float(t), 3) for t in times_ps],
        "frames": [f.ravel().tolist() for f in frames],
    }


def save_trajectory_json(path, system, frames_nm, times_ps) -> None:
    Path(path).write_text(json.dumps(trajectory_json(system, frames_nm, times_ps), separators=(",", ":")))
