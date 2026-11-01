"""Assemble the self-contained browser demo from the tested engine files.

    python scripts/build_demo.py

Inlines web/vh-md.js, web/vh-pbpk.js, web/demo-app.js and the exported data
(web/data/*.json, validation summaries) into:
  web/dist/artifact.html   body-only page (published as a claude.ai artifact)
  web/dist/index.html      standalone page (open locally in any browser)
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def protein(key):
    d = json.loads((WEB / "data" / f"{key}.json").read_text())
    for k in ("reference", "minimization"):
        d.pop(k, None)
    return d


def evidence():
    mol = json.loads((ROOT / "docs/validation_molecular.json").read_text())
    body = json.loads((ROOT / "docs/validation_body.json").read_text())
    worst_e = max(s["total_rel_err"] for s in mol["static"])
    worst_f = max(s["force_rel_rms_err"] for s in mol["static"])
    traj = next(t for t in mol["trajectory_vs_openmm"] if t["steps"] == 1000)
    nve = next(n for n in mol["nve"] if n["dt_fs"] == 1.0)
    ours = next(n for n in mol["nvt_ours"] if n["dt_fs"] == 1.0)
    omm = mol["nvt_openmm"][0]
    s = body["summary"]
    return {
        "mol": {"worst_energy_rel": worst_e, "worst_force_rel": worst_f, "traj_1000_dx_nm": traj["max_abs_dx_nm"],
                "nve_drift": nve["drift_kj_mol_ps"], "u_ours": ours["mean_U"], "u_ours_se": ours["se_U"],
                "u_omm": omm["mean_U"], "u_omm_se": omm["se_U"]},
        "body": {"n_iv": body["n_iv"], "n_po": body["n_po"],
                 "bu_iv_auc_2x": s["bottom_up"]["IV"]["auc"]["within_2x"],
                 "bu_po_auc_2x": s["bottom_up"]["PO"]["auc"]["within_2x"],
                 "bu_po_cmax_2x": s["bottom_up"]["PO"]["cmax"]["within_2x"],
                 "mo_iv_auc_2x": s["middle_out"]["IV"]["auc"]["within_2x"],
                 "mo_po_auc_2x": s["middle_out"]["PO"]["auc"]["within_2x"],
                 "mo_po_gmfe": s["middle_out"]["PO"]["auc"]["gmfe"],
                 "bu_iv_gmfe": s["bottom_up"]["IV"]["auc"]["gmfe"],
                 "clint_scale": body["clint_scale_middle_out"]},
    }


def main():
    data = {"chignolin": protein("chignolin"), "trpcage": protein("trpcage"),
            "body": json.loads((WEB / "data" / "body.json").read_text()), "evidence": evidence()}
    data_js = "window.VH_DATA = " + json.dumps(data, separators=(",", ":")) + ";"
    template = (WEB / "demo_template.html").read_text()
    scripts = "\n".join(
        f"<script>\n{s}\n</script>" for s in (
            data_js, (WEB / "vh-md.js").read_text(), (WEB / "vh-pbpk.js").read_text(), (WEB / "demo-app.js").read_text())
    )
    page = template.replace("<!--SCRIPTS-->", scripts)
    out = WEB / "dist"
    out.mkdir(exist_ok=True)
    (out / "artifact.html").write_text(page)
    standalone = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                  "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
                  "</head>\n<body>\n" + page + "\n</body>\n</html>\n")
    (out / "index.html").write_text(standalone)
    print(f"wrote web/dist/artifact.html ({len(page) / 1024:.0f} KB) and web/dist/index.html")


if __name__ == "__main__":
    main()
