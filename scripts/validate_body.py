"""Whole-body engine validation against clinical midazolam data.

Every plasma dataset digitized by the Open Systems Pharmacology project is
predicted twice:

  bottom-up   no clinical data used at all: in vitro enzyme kinetics scaled to
              the liver, physicochemistry -> tissue partitioning.
  middle-out  ONE factor on intrinsic clearance is fitted to the IV datasets
              only; the oral datasets are then predicted blind.

Writes docs/validation_body.{json,md} and figures in docs/img/.

    python scripts/validate_body.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

from vh.body.compound import MIDAZOLAM  # noqa: E402
from vh.body.pbpk import Dose, PBPKModel  # noqa: E402
from vh.body.physiology import make_individual, reference_individual  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS, IMG = ROOT / "docs", ROOT / "docs" / "img"
IV_INFUSION_H = 2.0 / 60.0   # IV doses modeled as 2-minute infusions (durations not recorded in the source)


def to_mg_l(values, unit, mw):
    v = np.asarray(values, dtype=float)
    u = unit.lower().replace("µ", "u")
    if u == "mg/l":
        return v
    if u in ("ng/ml", "ug/l"):
        return v * 1e-3
    if u == "umol/l":
        return v * mw * 1e-3
    raise ValueError(unit)


def individual_for(ds):
    m = re.search(r"\b([mf]), (\d+)kg, (\d+)y", ds["subject"] or "")
    if m:
        sex = "male" if m.group(1) == "m" else "female"
        return make_individual(sex, age=float(m.group(3)), body_mass=float(m.group(2)), label=ds["subject"])
    return reference_individual("male")


def dose_for(ds, ind):
    m = re.match(r"([\d.]+)\s*(mg/kg|mg|µg|ug)", ds["dose"])
    factor = {"mg/kg": ind.body_mass, "mg": 1.0, "µg": 1e-3, "ug": 1e-3}[m.group(2)]
    amount = float(m.group(1)) * factor
    if ds["route"] == "IV":
        return Dose(0.0, amount, "iv", IV_INFUSION_H)
    return Dose(0.0, amount, "oral")


def auc(t, c):
    return float(np.trapezoid(c, t))


def evaluate(ds, clint_scale=1.0):
    ind = individual_for(ds)
    model = PBPKModel(MIDAZOLAM, ind, clint_scale=clint_scale)
    t = np.asarray(ds["time"], dtype=float)
    c_obs = to_mg_l(ds["conc"], ds["conc_unit"], MIDAZOLAM.mw)
    keep = (c_obs > 0) & (t > 0)
    t, c_obs = t[keep], c_obs[keep]
    sim = model.simulate([dose_for(ds, ind)], t_end_h=float(t.max()) + 0.5, dt_h=0.01)
    c_pred = np.interp(t, sim["time_h"], sim["plasma"])
    dense = sim["time_h"] <= t.max()
    return {
        "t": t, "obs": c_obs, "pred": c_pred, "sim_t": sim["time_h"], "sim_c": sim["plasma"],
        "auc_obs": auc(t, c_obs), "auc_pred": auc(t, c_pred),
        "cmax_obs": float(c_obs.max()), "cmax_pred": float(sim["plasma"][dense].max()),
        "aafe": float(10 ** np.mean(np.abs(np.log10(c_pred / c_obs)))),
        "individual": ind.label, "dose_mg": dose_for(ds, ind).amount_mg,
    }


def summarize(rows, key):
    r = np.array([x[key + "_pred"] / x[key + "_obs"] for x in rows])
    return {
        "n": len(r), "gmfe": float(10 ** np.mean(np.abs(np.log10(r)))),
        "median_ratio": float(np.median(r)),
        "within_1.5x": float(np.mean((r >= 1 / 1.5) & (r <= 1.5))),
        "within_2x": float(np.mean((r >= 0.5) & (r <= 2.0))),
    }


def main():
    data = json.loads((ROOT / "data/clinical/midazolam_osp_observed.json").read_text())["datasets"]
    usable = [d for d in data if d["data_type"] in ("aggregated", "typical representative")]
    iv = [d for d in usable if d["route"] == "IV"]
    po = [d for d in usable if d["route"] == "PO"]

    # --- middle-out calibration on IV only --------------------------------
    def iv_loss(log_s):
        s = float(np.exp(log_s))
        return float(np.mean([np.log(evaluate(d, s)["auc_pred"] / evaluate(d, s)["auc_obs"]) ** 2 for d in iv]))

    res = minimize_scalar(iv_loss, bounds=(np.log(0.2), np.log(10.0)), method="bounded", options={"xatol": 1e-3})
    scale = float(np.exp(res.x))
    print(f"middle-out: CLint scale fitted on {len(iv)} IV datasets = {scale:.2f}")

    results = {}
    for label, s in (("bottom_up", 1.0), ("middle_out", scale)):
        rows = {"IV": [], "PO": []}
        for d in usable:
            e = evaluate(d, s)
            e["name"], e["study"], e["route"] = d["name"], d["study"], d["route"]
            rows[d["route"]].append(e)
        results[label] = rows
        for route in ("IV", "PO"):
            a = summarize(rows[route], "auc")
            line = f"[{label}] {route}: AUC within 2x {a['within_2x']:.0%} (GMFE {a['gmfe']:.2f}, median ratio {a['median_ratio']:.2f})"
            if route == "PO":
                cm = summarize(rows[route], "cmax")
                line += f"; Cmax within 2x {cm['within_2x']:.0%} (GMFE {cm['gmfe']:.2f})"
            print(line)

    ref = PBPKModel(MIDAZOLAM, reference_individual("male"))
    cal = PBPKModel(MIDAZOLAM, reference_individual("male"), clint_scale=scale)
    report = {
        "compound": "midazolam", "n_iv": len(iv), "n_po": len(po), "clint_scale_middle_out": scale,
        "iv_infusion_h": IV_INFUSION_H,
        "reference_individual": {"bottom_up": ref.summary(), "middle_out": cal.summary()},
        "summary": {
            lab: {r: {"auc": summarize(results[lab][r], "auc"), "cmax": summarize(results[lab][r], "cmax")} for r in ("IV", "PO")}
            for lab in results
        },
        "datasets": [
            {k: v for k, v in e.items() if k not in ("t", "obs", "pred", "sim_t", "sim_c")} | {"scenario": lab}
            for lab in results for r in ("IV", "PO") for e in results[lab][r]
        ],
    }
    DOCS.mkdir(exist_ok=True)
    IMG.mkdir(exist_ok=True)
    (DOCS / "validation_body.json").write_text(json.dumps(report, indent=1))
    make_figures(results, scale)
    write_markdown(report)
    print("wrote docs/validation_body.{json,md}, docs/img/body_*.png")


def make_figures(results, scale):
    # goodness of fit: AUC and Cmax
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, key, title in ((axes[0], "auc", "AUC (mg·h/L)"), (axes[1], "cmax", "Cmax (mg/L)")):
        for lab, marker, color in (("bottom_up", "o", "#9aa4b2"), ("middle_out", "s", "#2563eb")):
            for route, fill in (("IV", True), ("PO", False)):
                rows = results[lab][route]
                if key == "cmax" and route == "IV":
                    continue
                x = [r[key + "_obs"] for r in rows]
                y = [r[key + "_pred"] for r in rows]
                ax.scatter(x, y, marker=marker, s=26, edgecolors=color, facecolors=color if fill else "none",
                           label=f"{lab.replace('_', '-')} {route}", alpha=0.85, linewidths=1.2)
        lo, hi = ax.get_xlim()[0] * 0.5, ax.get_xlim()[1] * 2
        g = np.array([lo, hi])
        ax.plot(g, g, "k-", lw=0.8)
        ax.plot(g, 2 * g, "k--", lw=0.6)
        ax.plot(g, g / 2, "k--", lw=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("observed")
        ax.set_ylabel("predicted")
        ax.set_title(title)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Midazolam: predicted vs observed (dashed = 2-fold)", fontsize=11)
    fig.tight_layout()
    fig.savefig(IMG / "body_goodness_of_fit.png", dpi=140)
    plt.close(fig)

    # representative concentration-time profiles
    pick = []
    for route in ("IV", "PO"):
        seen = set()
        for e in results["middle_out"][route]:
            if e["study"] not in seen:
                seen.add(e["study"])
                pick.append((route, e["name"]))
            if len(seen) >= 6:
                break
    fig, axes = plt.subplots(3, 4, figsize=(14, 9.5))
    for ax, (route, name) in zip(axes.ravel(), pick):
        bu = next(e for e in results["bottom_up"][route] if e["name"] == name)
        mo = next(e for e in results["middle_out"][route] if e["name"] == name)
        ax.plot(bu["sim_t"], bu["sim_c"] * 1000, "--", color="#9aa4b2", lw=1.4, label="bottom-up")
        ax.plot(mo["sim_t"], mo["sim_c"] * 1000, "-", color="#2563eb", lw=1.6, label="middle-out")
        ax.plot(mo["t"], mo["obs"] * 1000, "o", color="#111827", ms=4, label="observed")
        ax.set_yscale("log")
        ax.set_title(f"{mo['study']} · {route} {mo['dose_mg']:.2g} mg", fontsize=9)
        ax.set_xlabel("time (h)", fontsize=8)
        ax.set_ylabel("plasma (ng/mL)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes.ravel()[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Midazolam plasma profiles: model vs clinical data (middle-out CLint x{scale:.2f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(IMG / "body_profiles.png", dpi=130)
    plt.close(fig)


def write_markdown(r):
    s = r["summary"]

    def row(lab, route, key):
        x = s[lab][route][key]
        return f"| {lab.replace('_', '-')} | {route} | {key.upper()} | {x['n']} | {x['within_1.5x']:.0%} | {x['within_2x']:.0%} | {x['gmfe']:.2f} | {x['median_ratio']:.2f} |"

    L = ["# Whole-body engine validation: midazolam", "",
         "Generated by `scripts/validate_body.py`. Observed data: plasma concentration–time profiles from "
         f"{r['n_iv']} IV and {r['n_po']} oral clinical datasets (aggregated means and typical subjects), digitized by the "
         "Open Systems Pharmacology project from the original publications (see `data/clinical/`).", "",
         "## Scenarios", "",
         "- **Bottom-up**: no clinical data used. Liver clearance is scaled from in vitro microsomal kinetics "
         "(CYP3A4 Vmax 850 pmol/min/mg, Km 4 µM; UGT1A4 276 pmol/min/mg, Km 37.8 µM; 32 mg microsomal protein/g liver). "
         "Tissue distribution comes from physicochemistry (Rodgers & Rowland). Gut-wall first pass from the Qgut model.",
         f"- **Middle-out**: one multiplier on intrinsic clearance (×{r['clint_scale_middle_out']:.2f}) fitted to the IV "
         "datasets only, then the oral datasets are predicted blind. Under-prediction of clearance by microsomal IVIVE "
         "is a well-known systematic effect.", "",
         f"Reference male, bottom-up: {r['reference_individual']['bottom_up']}", "",
         f"Reference male, middle-out: {r['reference_individual']['middle_out']}", "",
         "## Accuracy", "",
         "| Scenario | Route | Metric | Datasets | Within 1.5× | Within 2× | GMFE | Median pred/obs |",
         "|---|---|---|---|---|---|---|---|"]
    for lab in ("bottom_up", "middle_out"):
        L += [row(lab, "IV", "auc"), row(lab, "PO", "auc"), row(lab, "PO", "cmax")]
    L += ["", "GMFE = geometric mean fold error (1.0 = perfect). The 2-fold criterion is the conventional acceptance "
          "bar for PBPK predictions.", "",
          "![goodness of fit](img/body_goodness_of_fit.png)", "", "![profiles](img/body_profiles.png)", "",
          "## Known limitations (v0.1)", "",
          f"- IV doses are modeled as {r['iv_infusion_h'] * 60:.0f}-minute infusions (actual durations were not recorded in the source data).",
          "- Subjects are the ICRP reference male unless the dataset gives sex/weight/age (Greenblatt 1984).",
          "- Linear kinetics (concentrations far below Km at these doses). Fed-state effects are not modeled.",
          "- Intestinal permeability is an assumed value (BCS class I); it sets absorption rate and gut-wall extraction.", ""]
    (DOCS / "validation_body.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
