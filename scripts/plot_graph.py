"""Figures for the graph-formalization README section.

  docs/img/pbpk_graph.png      the whole-body model rendered as a flow graph
  docs/img/graph_results.png   two graph-algorithm outputs on that graph:
                               blood delivered to each organ, and betweenness
                               centrality (which compartments are the hubs)

Single-series magnitude comparisons -> horizontal bars, one hue each, sorted,
values direct-labeled, recessive axes, no legend. White background so the figure
reads on GitHub in either theme.

    python scripts/plot_graph.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vh.body import MIDAZOLAM, PBPKModel, reference_individual  # noqa: E402
from vh.graph import algorithms as A  # noqa: E402
from vh.graph.physiology_graph import build_pbpk_graph  # noqa: E402
from vh.graph.render import render  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
INK, MUTED, GRID = "#131A2B", "#586379", "#E3E8F0"
FLOW_HUE, HUB_HUE = "#2B44C4", "#1A8A54"


def _barpanel(ax, labels, values, hue, unit, title):
    y = range(len(labels))
    ax.barh(list(y), values, color=hue, height=0.62, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    vmax = max(values)
    for yi, v in zip(y, values):
        ax.text(v + vmax * 0.015, yi, f"{v:,.1f}" if v >= 1 else f"{v:.2f}",
                va="center", ha="left", fontsize=8.5, color=INK, zorder=4)
    ax.set_xlim(0, vmax * 1.18)
    ax.set_xlabel(unit, fontsize=9, color=MUTED)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
    ax.tick_params(length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    g, _ = build_pbpk_graph(PBPKModel(MIDAZOLAM, reference_individual("male")))

    # (A) blood delivered into each organ compartment = sum of inbound flow edges
    inflow = {}
    for n in g.nodes_of_type("compartment"):
        inflow[n.label] = sum(float(e.attrs.get("flow", 0.0)) for e in g.in_edges(n.id, ("flow",)))
    flows = sorted(inflow.items(), key=lambda x: -x[1])

    # (B) betweenness centrality on the flow subgraph -> structural hubs
    bc = A.betweenness_centrality(g.edge_view("flow"), ("flow",))
    name = {nid: g.node(nid).label for nid in g.nodes}
    hubs = sorted(((name[k], v) for k, v in bc.items() if v > 0), key=lambda x: -x[1])[:8]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="white")
    _barpanel(axA, [k for k, _ in flows], [v for _, v in flows], FLOW_HUE,
              "blood flow in  (L/h)", "Blood delivered to each organ")
    _barpanel(axB, [k for k, _ in hubs], [v for _, v in hubs], HUB_HUE,
              "betweenness centrality", "Structural hubs (betweenness)")
    fig.suptitle("Graph algorithms on the whole-body model", fontsize=13, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.005, "Every number comes from the validated PBPK model; blood flow is conserved at every node to 1.5e-16 (Kirchhoff).",
             fontsize=8, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(IMG / "graph_results.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote docs/img/graph_results.png")

    # the network diagram as PNG (SVG also written for docs)
    render(g, str(IMG / "pbpk_graph.png"), direction="LR", title="Midazolam whole-body graph")
    render(g, str(IMG / "pbpk_graph.svg"), direction="LR", title="Midazolam whole-body graph")
    print("wrote docs/img/pbpk_graph.png / .svg")


if __name__ == "__main__":
    main()
