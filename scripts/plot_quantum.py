"""Figure: accuracy converging to truth as the method climbs the ladder.

Left: the water-dimer hydrogen-bond energy at HF, MP2 and CCSD(T), each toward
the complete-basis-set limit, approaching the accepted benchmark.
Right: computed-vs-reference for the smallest parts of the body, as % error
against a 1%/chemical-accuracy band.

    python scripts/plot_quantum.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "docs/validation_quantum.json").read_text())
IMG = ROOT / "docs/img"
IMG.mkdir(parents=True, exist_ok=True)

INK, ACCENT, GOOD, WARN = "#131A2B", "#2B44C4", "#1A8A54", "#B26A12"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.0))

# ---- left: dimer binding energy converging to the benchmark -----------------
comp = d["dimer"]["components_kcal"]
hf_tz = comp["mp2/cc-pvtz"]["hf"]
hf_qz = comp["mp2/cc-pvqz"]["hf"]
mp2_tz = hf_tz + comp["mp2/cc-pvtz"]["corr"]
mp2_qz = hf_qz + comp["mp2/cc-pvqz"]["corr"]
ccsdt_tz = hf_tz + comp["ccsd(t)/cc-pvtz"]["corr"]
mp2_cbs = d["dimer"]["e_mp2_cbs_kcal"]
ccsdt_cbs = d["dimer"]["e_ccsdt_cbs_kcal"]
bench = d["_reference_values"]["dimer_De"]["value"]

series = {
    "Hartree-Fock\n(no correlation)": [("TZ", hf_tz), ("QZ", hf_qz)],
    "MP2": [("TZ", mp2_tz), ("QZ", mp2_qz), ("CBS", mp2_cbs)],
    "CCSD(T)\n(gold standard)": [("TZ", ccsdt_tz), ("CBS", ccsdt_cbs)],
}
colors = {"Hartree-Fock\n(no correlation)": "#9aa4b2", "MP2": ACCENT, "CCSD(T)\n(gold standard)": GOOD}
x = 0
xticks, xlabels = [], []
for name, pts in series.items():
    xs = list(range(x, x + len(pts)))
    axL.plot(xs, [p[1] for p in pts], "o-", color=colors[name], lw=2, ms=7, label=name.replace("\n", " "))
    for xi, (lab, val) in zip(xs, pts):
        xticks.append(xi); xlabels.append(lab)
        axL.annotate(f"{val:.2f}", (xi, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=colors[name])
    x += len(pts) + 1
axL.axhline(bench, color=INK, lw=1.3, ls="--")
axL.annotate(f"accepted benchmark {bench:.1f} kcal/mol", (xticks[-1], bench), textcoords="offset points",
             xytext=(-4, -14), ha="right", fontsize=9, color=INK)
axL.axhspan(bench - 1, bench + 1, color=GOOD, alpha=0.07)
axL.set_xticks(xticks); axL.set_xticklabels(xlabels, fontsize=8)
axL.set_ylabel("water-dimer H-bond energy (kcal/mol)")
axL.set_title("Climbing the ladder converges to the benchmark", fontsize=11)
axL.legend(fontsize=8, frameon=False, loc="upper right")
axL.text(0.5, -0.14, "basis set: TZ = triple-zeta, QZ = quadruple-zeta, CBS = complete-basis limit",
         transform=axL.transAxes, ha="center", fontsize=8, color="#586379")

# ---- right: computed vs reference, as signed % error ------------------------
rows = [
    ("Water O-H length", d["water"]["rOH"], d["_reference_values"]["water_rOH"]["value"]),
    ("Water H-O-H angle", d["water"]["angle"], d["_reference_values"]["water_angle"]["value"]),
    ("Water dipole (CCSD)", d["water"]["dipole_ccsd"], d["_reference_values"]["water_dipole"]["value"]),
    ("H-bond energy (CCSD(T)/CBS)", ccsdt_cbs, bench),
    ("Peptide C-N length", d["peptide"]["cn_length"], d["_reference_values"]["peptide_CN"]["value"]),
]
labels = [r[0] for r in rows]
errs = [100 * (r[1] - r[2]) / abs(r[2]) for r in rows]
y = np.arange(len(rows))[::-1]
axR.axvspan(-1, 1, color=GOOD, alpha=0.12, label="within 1%")
axR.axvline(0, color=INK, lw=1)
axR.barh(y, errs, color=[GOOD if abs(e) <= 1 else WARN for e in errs], height=0.55)
for yi, e in zip(y, errs):
    axR.annotate(f"{e:+.2f}%", (e, yi), textcoords="offset points",
                 xytext=(6 if e >= 0 else -6, 0), va="center", ha="left" if e >= 0 else "right", fontsize=8)
axR.set_yticks(y); axR.set_yticklabels(labels, fontsize=9)
axR.set_xlabel("error vs experiment / benchmark (%)")
axR.set_title("Computed from first principles vs truth", fontsize=11)
axR.set_xlim(-max(3, 1.4 * max(abs(e) for e in errs)), max(3, 1.4 * max(abs(e) for e in errs)))
axR.legend(fontsize=8, frameon=False, loc="lower right")

fig.suptitle("Quantum layer: the smallest parts of the body at the accuracy ceiling", fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(IMG / "quantum_accuracy.png", dpi=140, bbox_inches="tight")
print("wrote docs/img/quantum_accuracy.png")
print(f"CCSD(T)/CBS H-bond {ccsdt_cbs:.2f} vs benchmark {bench}; water dipole {d['water']['dipole_ccsd']:.3f} vs 1.855")
