# Formalizing the body as a graph — and what that lets code solve

> Short answer to "if we formalize the body into a graph, can we solve problems
> with code?": **yes, for a specific and useful class of problems** — the
> structural and relational ones — and the graph also becomes the scaffold that
> wires the physics engines together and lets any part be verified on its own.
> It does **not** turn the body's quantitative, continuous core (binding,
> kinetics, folding) into something solvable by symbol-pushing; that stays
> physics. The power is the combination.

## The idea

The body is already a graph at every scale:

- **organism / organ**: compartments (organs, blood pools) joined by **blood flow**;
- **molecule**: species joined by **reactions / transport / binding**;
- **atom**: atoms joined by **bonds**.

`vh.graph` is one typed property graph for all of them. Nodes and edges carry a
`type`, a `scale`, free-form attributes, and — where it matters — the name of the
**validated computational model** behind an edge (a PBPK blood flow, a
Rodgers–Rowland partition coefficient, a quantum binding constant). So the graph
is a *map* of the system that is also partially *executable* and, crucially,
*verifiable in pieces*.

A `Schema` states what a valid formalization is: which node/edge types are legal,
what attributes they must carry, and **structural invariants** — e.g. "blood flow
is conserved at every compartment." Invariants are checked by graph algorithms,
not taken on trust.

## What graph algorithms actually answer (exactly)

Built from scratch in `vh.graph.algorithms` and demonstrated in
`scripts/validate_graph.py` on the real, validated midazolam PBPK model:

| Question | Algorithm | Result on the real model |
|---|---|---|
| Is the formalization physically consistent? | flow balance (Kirchhoff) | blood conserved at every compartment to **1.5×10⁻¹⁶** (relative) |
| Where can a drug reach? | reachability (BFS/DFS) | IV drug reaches all 14 compartments incl. brain |
| What route does an oral drug take first? | path enumeration | `gut lumen → gut → liver` — **first-pass metabolism falls out of the topology** |
| Does anything recirculate? | strongly-connected components | one loop spanning 15 compartments — the systemic circulation |
| Where are the hubs / bottlenecks? | betweenness centrality | arterial, venous, lung (all of cardiac output passes through) |
| Which reported clearance parameter is larger? | edge analysis | liver 99% vs kidney 1% of the plasma-referenced reporting sum; see the distinction below |

These are not approximations — they are the exact answers the graph structure
determines.

## Clearance reporting and independent verification

Because the formalization is compositional, any subgraph can be pulled out and
checked on its own:

- **render** any subgraph to a diagram (`vh.graph.render`, via Mermaid) and
  inspect the [whole-body source](img/pbpk_graph.mmd) or
  [elimination subgraph source](img/pbpk_elimination_subgraph.mmd);
- **check the export**: elimination edges carry explicit `units="L/h"` and
  `reference_basis="plasma"`. Since `C_blood = bp * C_plasma`, conserving
  elimination rate requires `CL_plasma = bp * CL_blood`. The schema rejects
  missing/mixed reference bases, incorrect units, and nonfinite or negative
  clearances. The exported sum is **17.104405 L/h** for reference male midazolam,
  matching the body's *reported parameter sum*. This replaces a previous
  **28.488253 L/h** sum that mixed hepatic blood and renal plasma references.
- **verify independently**: the script integrates the linear ODE analytically,
  `integral(x dt) = -M_active^-1 x(0)`, for a 1 mg IV bolus. Dose divided by
  venous plasma AUC is **17.103070 L/h**, agreeing with the exact well-stirred
  organ mass balance to less than **1e-10 L/h**. Tests separately check exported
  hepatic rates against exact rational mass balance at blood:plasma ratios of
  0.5, 1 and 2, so a ratio of 1 cannot hide the reference error.

These checks answer different questions. Renal `cl` is a **nominal filtration
parameter** (`fu * GFR`), whereas hepatic `cl` is a **well-stirred effective
clearance**. Each edge records its `clearance_kind` and source. The finite renal
blood flow makes exact systemic clearance slightly smaller than the reporting
sum: **0.001336 L/h** in this example. With `q = renal blood flow * bp`, the
effective renal clearance is `q * cl_renal / (q + cl_renal)`. The shared plasma
basis makes export aggregation well defined, but does not make these parameters
identical to the ODE's local tissue elimination coefficients or independently
validate the biology. This change leaves the PBPK ODE and its inputs unchanged.

```mermaid
flowchart LR
  liver -->|"CL 16.86 L/h, plasma: effective hepatic"| eliminated
  kidney -->|"CL 0.24 L/h, plasma: nominal filtration"| eliminated
```

The full diagrams remain reproducible from the linked Mermaid sources. Old
raster/vector images with mixed reference values have been removed.

## The graph verifying the physics engine

The sharpest demonstration that the formalization is faithful: a **pure graph
algorithm reproduces part of the force field**. In molecular mechanics, the
atoms that must be excluded or down-weighted in the nonbonded sum are exactly the
1-2, 1-3 and 1-4 neighbors — i.e. atom pairs within **graph distance ≤ 3 on the
bond graph**. Running breadth-first search on the bond graph of Trp-cage and
chignolin recovers the engine's exclusion list **exactly** (1687 pairs for
Trp-cage, symmetric difference 0). A topological fact, computed by a graph
algorithm, matches the physics engine's own bookkeeping — each checks the other.

Rings (aromatic side chains, proline) show up as the cycles of the same graph;
molecules show up as its connected components.

## So, can we "solve the body with code"?

**What a graph solves, exactly:**
- structure and reachability — what connects to what, what can affect what;
- pathways and flows — routes, first-pass effects, dominant fluxes, bottlenecks;
- conservation and consistency — mass/flow balance, no orphan compartments,
  schema conformance — as checkable invariants;
- recirculation and feedback — cycles and strongly-connected components;
- composition and partial verification — check a part without the whole.

**What a graph does not solve by itself:**
- *how much / how fast / how strong* — binding free energies, reaction rates,
  folding. Those are continuous physics, answered by `vh.quantum`,
  `vh.molecular`, `vh.body`, not by topology. A graph that *claimed* to settle
  them by structure alone would be lying.

**Why the combination is the real answer.** The graph is where the scales meet:
an edge labeled `binds` carries a `Kd` that a quantum free-energy calculation
produces; an edge labeled `flow` carries a number the PBPK model validated
against clinical data; an edge labeled `clears` carries an IVIVE clearance. The
graph routes information between these models, lets each be swapped or refined
independently, checks that the whole stays consistent, and lets you verify any
piece on its own. That is a real, buildable path to "solving problems with code"
about the body — not by replacing the physics with a graph, but by giving the
physics a formal, inspectable, verifiable skeleton to hang on.

## Try it

```bash
python scripts/validate_graph.py     # runs every algorithm above, renders the diagrams
pytest -q tests/test_graph.py tests/test_graph_clearance.py
```
