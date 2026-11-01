"""Export engine data for the browser demo (web/data/*.json).

- proteins: full parameter set + minimized coordinates + reference energy/forces
  (the JS engine is tested against these in web/test_engines.mjs)
- body: per-individual physiology and drug parameters, clinical observations,
  plus reference concentration profiles for the JS PBPK test.

    python scripts/export_web.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from vh.body import MIDAZOLAM, Dose, PBPKModel, make_individual, reference_individual
from vh.body.pbpk import (GASTRIC_HALF_LIFE_H, N_TRANSIT, SI_TRANSIT_H)
from vh.body.physiology import PORTAL, TISSUES
from vh.molecular import build_energy_model, minimize
from vh.molecular.prep import load_protein

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "data"


def export_protein(pdb, key, label):
    system, pos, _, _ = load_protein(ROOT / "data/structures" / pdb, name=label)
    model = build_energy_model(system, dtype=jnp.float64)
    x0, info = minimize(model, pos)
    e, f = model.energy_and_forces(x0)
    rng = np.random.default_rng(5)
    xt = x0 + rng.normal(0, 0.005, x0.shape)
    et, ft = model.energy_and_forces(xt)
    terms = {k: float(v) for k, v in model.terms(xt).items()}
    ca = [i for i, n in enumerate(system.atom_names) if n == "CA"]
    d = system.to_dict()
    d.update({
        "key": key, "label": label, "x0": np.round(x0, 6).tolist(), "ca_indices": ca,
        "reference": {"x": xt.tolist(), "energy": float(et), "forces": np.asarray(ft).tolist(), "terms": terms,
                      "energy_x0": float(e)},
        "minimization": info,
    })
    (OUT / f"{key}.json").write_text(json.dumps(d, separators=(",", ":")))
    print(f"{key}: {system.n_atoms} atoms, minimized E {info['energy']:.1f} kJ/mol")


def individual_payload(model: PBPKModel):
    ind, p = model.ind, model.p
    return {
        "label": ind.label, "sex": ind.sex, "age": ind.age, "body_mass": ind.body_mass, "height": ind.height,
        "volumes": {t: ind.volumes[t] for t in TISSUES}, "v_arterial": ind.v_arterial, "v_venous": ind.v_venous,
        "flows": {t: ind.flows[t] for t in TISSUES}, "cardiac_output": ind.cardiac_output,
        "liver_flow_total": ind.liver_flow_total, "gfr": ind.gfr,
        "kp": p.kp, "bp": p.bp, "fu": p.fu, "clint_hep": p.clint_hep_l_h, "cl_renal": p.cl_renal_l_h,
        "ka": p.ka_h, "fa": p.fa, "fg": p.fg, "q_gut": p.q_gut_l_h, "clint_gut": p.clint_gut_l_h,
        "vss": p.vss_l, "summary": model.summary(),
    }


def export_body():
    calib = json.loads((ROOT / "docs/validation_body.json").read_text())
    scale = calib["clint_scale_middle_out"]
    presets = [
        ("ref_male", reference_individual("male"), "Reference man (ICRP 89)"),
        ("ref_female", reference_individual("female"), "Reference woman (ICRP 89)"),
        ("obese_female", make_individual("female", 36, 136, 163), "Obese woman, 136 kg"),
        ("elderly_male", make_individual("male", 75, 78, 172), "Older man, 75 y"),
        ("small_female", make_individual("female", 25, 48, 155), "Small woman, 48 kg"),
    ]
    individuals = {}
    for key, ind, label in presets:
        ind.label = label
        individuals[key] = individual_payload(PBPKModel(MIDAZOLAM, ind, clint_scale=scale))

    obs = json.loads((ROOT / "data/clinical/midazolam_osp_observed.json").read_text())["datasets"]
    wanted = [("Olkkola 1993", "IV"), ("Olkkola 1993", "PO"), ("Hohmann 2015", "IV"), ("Backman 1998", "PO"),
              ("Kharasch 2011", "IV"), ("Bornemann 1986", "PO")]
    datasets = []
    for study, route in wanted:
        for d in obs:
            if d["study"] == study and d["route"] == route and d["data_type"] == "aggregated":
                f = {"mg/l": 1.0, "ng/ml": 1e-3}.get(d["conc_unit"].lower())
                if f is None:
                    continue
                datasets.append({"study": study, "route": route, "dose": d["dose"], "n": d["n"],
                                 "reference": d["reference"], "time": d["time"],
                                 "conc_ng_ml": [c * f * 1000 for c in d["conc"]]})
                break

    # reference profile for the JS test (reference male, 2 mg IV over 2 min then 7.5 mg oral at 6 h)
    m = PBPKModel(MIDAZOLAM, reference_individual("male"), clint_scale=scale)
    sim = m.simulate([Dose(0, 2.0, "iv", 2 / 60), Dose(6.0, 7.5, "oral")], t_end_h=12, dt_h=0.25)
    payload = {
        "compound": {"name": "midazolam", "mw": MIDAZOLAM.mw, "logp": MIDAZOLAM.logp, "fu": MIDAZOLAM.fu,
                     "target": {"name": MIDAZOLAM.target.name, "tissue": MIDAZOLAM.target.tissue,
                                "kd_nm": MIDAZOLAM.target.kd_nm, "koff_per_h": 60.0, "source": MIDAZOLAM.target.source},
                     "sources": MIDAZOLAM.sources},
        "constants": {"n_transit": N_TRANSIT, "si_transit_h": SI_TRANSIT_H, "gastric_half_life_h": GASTRIC_HALF_LIFE_H,
                      "tissues": list(TISSUES), "portal": list(PORTAL)},
        "clint_scale": scale,
        "individuals": individuals,
        "observed": datasets,
        "validation": calib["summary"],
        "reference_profile": {"individual": "ref_male", "doses": [[0, 2.0, "iv", 2 / 60], [6.0, 7.5, "oral"]],
                              "time_h": sim["time_h"].tolist(), "plasma": sim["plasma"].tolist(),
                              "brain": sim["brain"].tolist()},
    }
    (OUT / "body.json").write_text(json.dumps(payload, separators=(",", ":")))
    print(f"body: {len(individuals)} individuals, {len(datasets)} observed datasets")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    export_protein("1uao_chignolin.pdb", "chignolin", "Chignolin (PDB 1UAO)")
    export_protein("1l2y_trpcage.pdb", "trpcage", "Trp-cage (PDB 1L2Y)")
    export_body()
