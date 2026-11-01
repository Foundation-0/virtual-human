"""Validate the graph formalization and show what graph algorithms answer.

Proves, on the real validated engines:
  1. the whole-body graph obeys blood conservation (Kirchhoff) -- a physical law
     checked by a graph algorithm;
  2. structural pharmacology falls out of topology (IV reach, oral first pass,
     recirculation, hubs);
  3. partial verification: a subgraph's elimination matches the full model;
  4. a pure graph BFS on the bond graph reproduces the force-field exclusion list
     exactly -- the graph verifying the physics engine;
  5. renders the body graph and a subgraph to diagrams.

    python scripts/validate_graph.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vh.body import MIDAZOLAM, PBPKModel, reference_individual
from vh.graph import algorithms as A
from vh.graph.physiology_graph import build_molecular_graph, build_pbpk_graph, exclusions_from_bond_graph
from vh.graph.render import render
from vh.molecular.prep import load_protein

ROOT = Path(__file__).resolve().parents[1]
DOCS, IMG = ROOT / "docs", ROOT / "docs" / "img"
report: dict = {}


def whole_body():
    m = PBPKModel(MIDAZOLAM, reference_individual("male"))
    g, schema = build_pbpk_graph(m)
    print(g.summary())
    problems = g.validate(schema)
    print("schema + invariants:", "all pass" if not problems else problems)

    bal = A.flow_balance(g, "flow", ("flow",))
    print(f"[1 blood conservation] worst residual {bal['max_abs_residual']:.2e} L/h "
          f"(relative {bal['max_rel_residual']:.1e}) -> balanced = {bal['balanced']}")

    iv = A.reachable(g, "venous", ("flow",))
    oral = A.all_paths(g, "gut_lumen", "liver", ("absorbs", "flow"), max_length=4)
    loops = A.recirculation_loops(g.edge_view("flow"), ("flow",))
    bc = A.betweenness_centrality(g.edge_view("flow"), ("flow",))
    hubs = sorted(bc.items(), key=lambda x: -x[1])[:3]
    print(f"[2 structure] IV reaches {len(iv)} compartments (brain: {'brain' in iv}); "
          f"oral first-pass path {oral[0] if oral else None}; "
          f"{len(loops)} recirculation loop (size {max(len(c) for c in loops)}); "
          f"hubs {[h[0] for h in hubs]}")

    # elimination split from the clearance edges
    cl = {e.attrs['organ']: e.attrs['cl'] for e in g.edges_of_type('clears')}
    print(f"[2 elimination] hepatic {cl['liver']:.2f} L/h vs renal {cl['kidney']:.2f} L/h "
          f"-> liver handles {100*cl['liver']/sum(cl.values()):.0f}% (blood clearance)")

    report["whole_body"] = {
        "summary": g.summary(), "schema_ok": not problems,
        "blood_conservation_rel_residual": bal["max_rel_residual"],
        "iv_reaches": len(iv), "brain_reachable": "brain" in iv,
        "oral_first_pass_path": oral[0] if oral else None,
        "recirculation_loop_size": max(len(c) for c in loops),
        "hubs": [h[0] for h in hubs],
        "hepatic_fraction": cl["liver"] / sum(cl.values()),
    }
    g.save_json(DOCS / "pbpk_graph.json")
    render(g, str(IMG / "pbpk_graph.svg"), direction="LR", title="Midazolam whole-body graph")

    # 3. partial verification: the elimination subgraph alone reproduces total clearance
    sub = g.subgraph(["arterial", "liver", "kidney", "venous", "gut", "spleen", "pancreas", "eliminated"],
                     name="elimination-subgraph")
    sub_cl = sum(e.attrs["cl"] for e in sub.edges_of_type("clears"))
    full_cl = m.p.cl_hep_blood_l_h + m.p.cl_renal_l_h
    print(f"[3 partial verification] elimination subgraph total CL {sub_cl:.3f} L/h "
          f"= full model {full_cl:.3f} L/h  (match: {abs(sub_cl-full_cl) < 1e-9})")
    render(sub, str(IMG / "pbpk_elimination_subgraph.svg"), direction="LR",
           title="Elimination subgraph (verified in isolation)")
    report["partial_verification"] = {"subgraph_cl": sub_cl, "full_cl": full_cl,
                                      "match": abs(sub_cl - full_cl) < 1e-9}


def molecular():
    system, pos, top, omm = load_protein(ROOT / "data/structures/1l2y_trpcage.pdb")
    g, schema = build_molecular_graph(system)
    print("\n" + g.summary())
    comps = A.connected_components(g, ("bond",))
    cyc = [c for c in A.simple_cycles(g.edge_view("bond"), ("bond",), max_cycles=300) if len(c) - 1 >= 3]
    rings = sorted({len(c) - 1 for c in cyc})
    graph_excl = exclusions_from_bond_graph(system, g)
    engine_excl = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in np.asarray(system.exceptions)}
    ok = graph_excl == engine_excl
    print(f"[4 graph verifies the engine] bond-graph components {len(comps)}; ring sizes {rings}; "
          f"exclusion pairs: graph {len(graph_excl)} == engine {len(engine_excl)} -> {ok} "
          f"(symmetric difference {len(graph_excl ^ engine_excl)})")
    report["molecular"] = {"n_atoms": system.n_atoms, "components": len(comps), "ring_sizes": rings,
                           "graph_exclusions": len(graph_excl), "engine_exclusions": len(engine_excl),
                           "exclusions_match": ok}


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    whole_body()
    molecular()
    def _clean(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)
    (DOCS / "validation_graph.json").write_text(json.dumps(report, indent=1, default=_clean))
    print("\nwrote docs/validation_graph.json, docs/pbpk_graph.json, docs/img/pbpk_graph.*")


if __name__ == "__main__":
    main()
