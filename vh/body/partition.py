"""Tissue partition coefficients by the mechanistic method of Rodgers & Rowland.

Rodgers, Leahy & Rowland 2005 (J Pharm Sci 94:1259) for moderate-to-strong
bases (pKa >= 7) and Rodgers & Rowland 2006 (J Pharm Sci 95:1238) for acids,
weak bases, neutrals. Kpu (tissue : unbound plasma) is built from how a drug
dissolves in tissue water, partitions into neutral lipids and phospholipids,
binds acidic phospholipids (bases) and binds extracellular proteins
(albumin for acids, lipoproteins for neutrals and weak bases).

Tissue composition values are read from data/tissue_composition_rr.json with
per-value provenance (open-access transcriptions of the original tables).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "tissue_composition_rr.json"

# which composition is used for each model compartment
COMPOSITION_FOR = {
    "lung": "lung", "heart": "heart", "brain": "brain", "muscle": "muscle", "adipose": "adipose",
    "skin": "skin", "bone": "bone", "kidney": "kidney", "liver": "liver", "gut": "gut",
    "spleen": "spleen", "pancreas": "pancreas", "rest": "muscle",
}


@lru_cache(maxsize=1)
def tissue_data() -> dict:
    return json.loads(DATA.read_text())


def _v(entry):
    return entry["value"] if isinstance(entry, dict) else entry


@dataclass
class Ionization:
    """Simple ionization model: 'neutral', 'acid', 'base' (monoprotic)."""
    kind: str
    pka: float | None = None

    def x_y(self, ph_in: float, ph_out: float):
        """Ionization factors X (inside) and Y (outside), 1 + 10^(+-(pKa - pH))."""
        if self.kind == "neutral" or self.pka is None:
            return 1.0, 1.0
        if self.kind == "acid":
            return 1 + 10 ** (ph_in - self.pka), 1 + 10 ** (ph_out - self.pka)
        return 1 + 10 ** (self.pka - ph_in), 1 + 10 ** (self.pka - ph_out)


def _lipid_term(P, f_nl, f_np):
    return P * f_nl + (0.3 * P + 0.7) * f_np


def predict_bp(logp: float, ion: Ionization, fu: float, hematocrit: float) -> float:
    """Blood:plasma ratio from erythrocyte partitioning (not valid for strong bases,
    which need a measured B:P to calibrate acidic-phospholipid binding)."""
    d = tissue_data()
    ery, ph_p = d["erythrocytes"], _v(d["plasma"]["ph"])
    P = 10.0**logp
    X, Y = ion.x_y(_v(ery["ph"]), ph_p)
    kpu_bc = X * _v(ery["f_iw"]) / Y + _lipid_term(P, _v(ery["f_nl"]), _v(ery["f_np"])) / Y
    return hematocrit * fu * kpu_bc + (1.0 - hematocrit)


def kpu_all(logp: float, ion: Ionization, fu: float, bp: float, hematocrit: float) -> dict:
    """Tissue : unbound-plasma partition coefficients for every compartment."""
    d = tissue_data()
    ph_p = _v(d["plasma"]["ph"])
    ph_iw = _v(d["intracellular"]["ph"])
    P = 10.0**logp
    P_vo = 10.0 ** (_v(d["adipose"]["vo_slope"]) * logp + _v(d["adipose"]["vo_intercept"]))
    X, Y = ion.x_y(ph_iw, ph_p)
    strong_base = ion.kind == "base" and ion.pka is not None and ion.pka >= 7.0

    if strong_base:
        # affinity for acidic phospholipids, calibrated on red blood cells (B:P)
        ery = d["erythrocytes"]
        Xbc, _ = ion.x_y(_v(ery["ph"]), ph_p)
        kpu_bc = (hematocrit - 1.0 + bp) / (hematocrit * fu)
        ka_ap = (kpu_bc - Xbc * _v(ery["f_iw"]) / Y - _lipid_term(P, _v(ery["f_nl"]), _v(ery["f_np"])) / Y) * Y / (_v(ery["ap"]) * Xbc)
        ka_ap = max(ka_ap, 0.0)
    else:
        f_nl_p, f_np_p = _v(d["plasma"]["f_nl"]), _v(d["plasma"]["f_np"])
        protein_capacity = max(1.0 / fu - 1.0 - _lipid_term(P, f_nl_p, f_np_p) / Y, 0.0)

    out = {}
    for comp, key in COMPOSITION_FOR.items():
        t = d["tissues"][key]
        Pt = P_vo if key == "adipose" else P
        f_ew, f_iw, f_nl, f_np = (_v(t[k]) for k in ("f_ew", "f_iw", "f_nl", "f_np"))
        kpu = f_ew + X * f_iw / Y + _lipid_term(Pt, f_nl, f_np) / Y
        if strong_base:
            kpu += ka_ap * _v(t["ap"]) * X / Y
        else:
            ratio = _v(t["albumin"]) if ion.kind == "acid" else _v(t["lipoprotein"])
            kpu += protein_capacity * ratio
        out[comp] = float(kpu)
    return out


def kp_all(logp, ion, fu, bp, hematocrit) -> dict:
    """Tissue : total-plasma partition coefficients (Kp = Kpu * fu)."""
    return {k: v * fu for k, v in kpu_all(logp, ion, fu, bp, hematocrit).items()}


def vss(kp: dict, volumes: dict, bp: float, v_blood: float, hematocrit: float) -> float:
    """Steady-state volume of distribution referenced to plasma (L)."""
    return float(sum(kp[t] * volumes[t] for t in kp) + v_blood * bp)
