"""Generate plain-language explainer charts from the flow validation report.

    python scripts/plot_flow_explainer.py

Writes:
  docs/img/flow_explained_venous_pressure.png
  docs/img/flow_explained_terms.png
  docs/img/flow_explained_scorecard.png

These are for humans, not for the validation itself: they turn the measured
numbers in docs/validation_flow.json into three pictures -- why standing venous
pressure is high, which term dominates in a vein vs a stenosis, and how close
each solver is to its exact authority.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from vh.graph.flow import (  # noqa: E402
    MMHG,
    RHO_BLOOD,
    hydrostatic_head,
    kinetic_pressure_drop,
    poiseuille_resistance,
)

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
DATA = json.loads((ROOT / "docs" / "validation_flow.json").read_text())

INK = "#33415C"
RED = "#C2373A"
BLUE = "#2B44C4"
GREEN = "#2E8B57"


# ---------------------------------------------------------------------------
# 1. Why standing venous pressure is high
# ---------------------------------------------------------------------------
def venous_pressure_chart():
    pred = DATA["vein_network"]["predictions"]
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(12.4, 5.6), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.18]},
    )

    # -- left: a vertical blood column with a pressure gauge at the foot -----
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.16)
    ax.axis("off")
    ax.set_title("Why standing makes foot pressure high", fontsize=12.5,
                 loc="left", color=INK)

    col_x, col_w, col_bot, col_top = 0.62, 0.075, 0.13, 1.03
    ax.add_patch(Rectangle((col_x, col_bot), col_w, col_top - col_bot,
                           facecolor="#DCE8F5", edgecolor="#6C8EBF", lw=2,
                           zorder=2))
    # right atrium
    ax.add_patch(Rectangle((0.735, 0.965), 0.245, 0.105,
                           facecolor="#F6D5D5", edgecolor=RED, lw=1.5,
                           zorder=3))
    ax.text(0.8575, 1.017, "right atrium\n5 mmHg", ha="center", va="center",
            fontsize=8.5, color="#7A1F1F", zorder=4)
    # foot
    ax.add_patch(Circle((col_x + col_w / 2, col_bot), 0.032,
                        facecolor=BLUE, edgecolor="white", lw=1.2, zorder=4))
    ax.text(col_x + col_w + 0.025, col_bot - 0.005,
            f"foot\n{pred['foot_standing_mmhg']:.1f} mmHg",
            ha="left", va="center", fontsize=8.5, color="#1B2E8A")

    # height arrow and rotated label, well left of the vessel
    ax.annotate("", xy=(0.545, col_top), xytext=(0.545, col_bot),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    ax.text(0.525, (col_top + col_bot) / 2,
            "1.03 m blood column\n≈ 80 mmHg",
            rotation=90, ha="right", va="center", fontsize=8.5, color=INK)

    # one-way valves
    for y in np.linspace(0.24, 0.94, 5):
        ax.plot([col_x, col_x + col_w / 2, col_x],
                [y + 0.018, y, y + 0.018],
                color="#6C8EBF", lw=1.6, zorder=3)
    ax.text(col_x + col_w + 0.025, 0.66, "one-way\nvalves",
            fontsize=8, color="#44546A", va="center")

    # prose, in a clear left column
    ax.text(0.0, 1.06, "Standing (90°)", fontsize=10.5, weight="bold",
            color=RED)
    ax.text(0.0, 0.96,
            "The heart is ~1 m above\n"
            "the floor. Blood in the\n"
            "leg vein is a tall liquid\n"
            "column, so gravity adds\n"
            "~80 mmHg from foot\n"
            "to heart.",
            fontsize=8.8, va="top", color=INK, linespacing=1.45)
    ax.text(0.0, 0.50, "Supine (0°)", fontsize=10.5, weight="bold",
            color=BLUE)
    ax.text(0.0, 0.40,
            "The column is horizontal:\n"
            "height difference ≈ 0, so\n"
            "foot pressure is only\n"
            f"{pred['foot_supine_mmhg']:.1f} mmHg.",
            fontsize=8.8, va="top", color=INK, linespacing=1.45)

    # -- right: prediction vs the physiological reference -------------------
    rows = [
        ("supine — foot", pred["foot_supine_mmhg"], BLUE),
        ("supine — ankle", pred["ankle_supine_mmhg"], BLUE),
        ("standing — foot", pred["foot_standing_mmhg"], RED),
        ("standing — ankle", pred["ankle_standing_mmhg"], RED),
    ]
    ys = np.arange(len(rows))[::-1]
    bx.axvspan(5, 10, color=GREEN, alpha=0.16, zorder=0)
    bx.axvspan(85, 90, color=RED, alpha=0.14, zorder=0)
    bx.text(7.5, 3.36, "5–10", ha="center", va="top",
            fontsize=7.5, color="#1E6B41", weight="bold")
    bx.text(87.5, 3.36, "85–90", ha="center", va="top",
            fontsize=7.5, color="#8A2022", weight="bold")
    for y, (label, value, color) in zip(ys, rows):
        bx.scatter([value], [y], s=130, color=color, zorder=3,
                   edgecolor="white", linewidth=1.2)
        if value > 50:
            bx.text(value - 3.0, y, f"{value:.1f}", va="center", ha="right",
                    fontsize=9.5, color=color, weight="bold")
        else:
            bx.text(value + 3.0, y, f"{value:.1f}", va="center", ha="left",
                    fontsize=9.5, color=color, weight="bold")
    bx.set_yticks(ys)
    bx.set_yticklabels([r[0] for r in rows], fontsize=9)
    bx.set_xlim(0, 105)
    bx.set_ylim(-0.45, 3.45)
    bx.set_xlabel("venous pressure (mmHg)", fontsize=9.5)
    bx.set_title("Prediction vs physiological range (not fitted)", fontsize=12.5,
                 loc="left", color=INK)
    bx.grid(axis="x", alpha=0.22)
    bx.legend(handles=[
        Line2D([0], [0], color=GREEN, lw=8, alpha=0.35,
               label="supine reference 5–10 mmHg"),
        Line2D([0], [0], color=RED, lw=8, alpha=0.35,
               label="standing reference 85–90 mmHg"),
    ], frameon=False, loc="center right", fontsize=8, bbox_to_anchor=(1.0, 0.62))

    fig.text(0.5, -0.02,
             "No parameter was fitted to these ranges. The model knows only "
             "anatomy, blood properties, and a 0.5 L/min capillary inflow.",
             ha="center", va="top", fontsize=8.5, color="#44546A")

    fig.savefig(IMG / "flow_explained_venous_pressure.png", dpi=150,
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Which term dominates: vein vs stenosis
# ---------------------------------------------------------------------------
def terms_chart():
    # Vein: 5 mm radius, 0.2 m long, 0.5 L/min; gravity over 1.05 m.
    q_vein, r_vein, l_vein = 0.5 / 60000.0, 0.005, 0.2
    v_vein = q_vein / (np.pi * r_vein ** 2)
    vein = {
        "gravity": abs(hydrostatic_head(0.0, 1.05)) / MMHG,
        "viscous": poiseuille_resistance(l_vein, r_vein) * q_vein / MMHG,
        "kinetic": 0.5 * RHO_BLOOD * v_vein ** 2 / MMHG,
    }

    # Stenosis: 5 mm -> 2 mm, 20 mL/s, R = 1e6 Pa s/m3.
    q_st, r1, r2, r_visc = 2.0e-5, 5.0e-3, 2.0e-3, 1.0e6
    a1, a2 = np.pi * r1 ** 2, np.pi * r2 ** 2
    stenosis = {
        "kinetic": kinetic_pressure_drop(q_st, a1, a2) / MMHG,
        "viscous": r_visc * q_st / MMHG,
    }

    colors = {"gravity": RED, "viscous": BLUE, "kinetic": GREEN}
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(12.0, 4.9), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.05, 1.0]},
    )

    # -- left: share of the total pressure drop -----------------------------
    labels = ["vein\n(0.5 L/min)", "stenosis\n(1.2 L/min)"]
    scenarios = [vein, stenosis]
    for y, values in zip((1, 0), scenarios):
        total = sum(values.values())
        left = 0.0
        for term in ("gravity", "viscous", "kinetic"):
            if term not in values:
                continue
            share = values[term] / total
            ax.barh(y, share, left=left, height=0.5, color=colors[term],
                    alpha=0.88, edgecolor="white", lw=1.5)
            if share > 0.05:
                ax.text(left + share / 2, y, f"{share:.1%}",
                        ha="center", va="center", color="white",
                        fontsize=10, weight="bold")
            left += share
    ax.set_yticks([1, 0])
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("share of the total pressure drop", fontsize=9.5)
    ax.set_title("Which term does the work?", fontsize=12, loc="left",
                 color=INK)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", alpha=0.2)
    ax.legend(handles=[
        Line2D([0], [0], color=colors[t], lw=9, label=t)
        for t in ("gravity", "viscous", "kinetic")
    ], frameon=False, ncol=3, loc="upper center",
        bbox_to_anchor=(0.5, 1.02), fontsize=9)

    # -- right: absolute values, dot plot on a log axis ---------------------
    rows = [
        ("vein", "gravity", vein["gravity"]),
        ("vein", "viscous", vein["viscous"]),
        ("vein", "kinetic", vein["kinetic"]),
        ("stenosis", "kinetic", stenosis["kinetic"]),
        ("stenosis", "viscous", stenosis["viscous"]),
    ]
    ys = np.arange(len(rows))[::-1]
    for y, (_, term, value) in zip(ys, rows):
        bx.scatter([value], [y], s=95, color=colors[term], zorder=3,
                   edgecolor="white", linewidth=1.0)
        bx.text(value * 1.35, y, f"{value:.3g} mmHg", va="center", ha="left",
                fontsize=8.5, color=INK)
    bx.set_yticks(ys)
    bx.set_yticklabels(
        [f"{group}: {term}" for group, term, _ in rows], fontsize=9)
    bx.set_xscale("log")
    bx.set_xlim(1e-2, 1e3)
    bx.set_ylim(-0.55, 4.55)
    bx.set_xlabel("absolute pressure term (mmHg, log scale)", fontsize=9.5)
    bx.set_title("How big each term is", fontsize=12, loc="left", color=INK)
    bx.grid(axis="x", which="both", alpha=0.2)

    fig.text(0.5, -0.025,
             "Vein: 5 mm radius, 0.2 m long, 0.5 L/min. Stenosis: 5→2 mm, "
             "20 mL/s through R = 10⁶ Pa·s/m³. Gravity uses a 1.05 m column.",
             ha="center", va="top", fontsize=8.3, color="#44546A")

    fig.savefig(IMG / "flow_explained_terms.png", dpi=150,
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Validation scorecard
# ---------------------------------------------------------------------------
def scorecard_chart():
    checks = [
        ("Poiseuille flow", DATA["steady"]["poiseuille"]["relerr"]),
        ("Effective resistance",
         DATA["vein_network"]["effective_resistance"]["relerr"]),
        ("Starling waterfall",
         DATA["steady"]["starling_max_abs_error"] / 0.0015),
        ("Bernoulli residual",
         abs(DATA["steady"]["bernoulli_stenosis"]["bernoulli_residual_pa"])
         / DATA["steady"]["bernoulli_stenosis"]["p_up"]),
        ("R–L step response", DATA["transient"]["rl_step"]["relerr"]),
        ("Windkessel (implicit Euler)",
         DATA["transient"]["windkessel"]["implicit_euler"]["0.001"]["abs_error"]
         / DATA["transient"]["windkessel"]["analytic_pa"]),
        ("Womersley, α=0.48",
         DATA["womersley"]["table"][2]["relerr"]),
        ("Womersley, α=5",
         DATA["womersley"]["table"][5]["relerr"]),
    ]
    checks.sort(key=lambda item: item[1], reverse=True)

    def color(v):
        if v < 1e-2:
            return GREEN
        if v < 5e-2:
            return "#E6A700"
        return RED

    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    fig.subplots_adjust(left=0.30, right=0.985, top=0.90, bottom=0.24)
    ys = np.arange(len(checks))
    vals = [v for _, v in checks]
    ax.barh(ys, vals, color=[color(v) for v in vals], alpha=0.9, height=0.6)
    for y, v in zip(ys, vals):
        ax.text(v * 1.35, y, f"{v:.2g}", va="center", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1e-17, 2.0)
    ax.set_ylim(-0.6, len(checks) - 0.1)
    ax.set_yticks(ys)
    ax.set_yticklabels([name for name, _ in checks], fontsize=9.5)
    for x, label, style in ((1e-2, "1%", ":"), (5e-2, "5%", "--")):
        ax.axvline(x, color=INK, ls=style, lw=1, alpha=0.65)
        ax.text(x, len(checks) - 0.35, label, ha="center", va="bottom",
                fontsize=8, color=INK)
    ax.set_xlabel(
        "relative error vs the exact authority (log scale, lower is better)\n\n"
        "green: better than 1%   •   amber: 1–5%   •   red: the known "
        "high-Womersley-number limit, reported not hidden",
        fontsize=9.0,
    )
    ax.set_title("Validation scorecard — every solver checked against an exact solution",
                 fontsize=11.5, loc="left", color=INK)
    ax.grid(axis="x", which="both", alpha=0.2)

    fig.savefig(IMG / "flow_explained_scorecard.png", dpi=150)
    plt.close(fig)


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    venous_pressure_chart()
    terms_chart()
    scorecard_chart()
    print("wrote docs/img/flow_explained_{venous_pressure,terms,scorecard}.png")


if __name__ == "__main__":
    main()
