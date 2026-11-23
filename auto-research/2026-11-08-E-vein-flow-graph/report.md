# Conserved-flow graphs: from topology to a validated venous pressure prediction

- **Track**: E — Grow the graph formalization
- **Author**: OpenCode agent (DeepSeek V4.1 Flash Free), directed by Rohan Arun
- **Date**: 2026-11-08
- **Branch / PR**: `auto/E-vein-flow-graph`
- **Code added**: `vh/graph/flow.py`, `vh/graph/algorithms.py` (weighted
  Laplacian, effective resistance, current flow), `vh/graph/render.py` (conduit
  labels/styles), `tests/test_flow.py`, `scripts/validate_flow.py`,
  `docs/bernoulli_veins.md`, `docs/prediction_methods.md`,
  `docs/validation_flow.md`, `docs/validation_flow.json`,
  `docs/flow_explained.md`, `scripts/plot_flow_explainer.py`,
  `docs/img/womersley_vs_lumped.png`,
  `docs/img/vein_pressure_profile.png`, `docs/img/vein_graph.png`,
  `docs/img/flow_explained_venous_pressure.png`,
  `docs/img/flow_explained_terms.png`,
  `docs/img/flow_explained_scorecard.png`.

## Question

Can a typed graph do more than say *what is connected*? Can it predict real
pressures and flows — specifically, can it predict venous pressure and re-derive
Bernoulli's principle, including the terms where classical Bernoulli fails? The
existing graph layer answered structural questions exactly; this contribution
adds the constitutive physics that makes the same graph quantitative.

## Method

### The graph

`vh/graph/flow.py` builds a typed graph from `FlowNodeSpec` and `VesselSpec`
lists. A node is a `reservoir` (fixed pressure), `compartment` (finite
compliance), or `junction` (zero compliance). A conduit edge carries:

- `R = 8μL/(πr⁴)` (Hagen–Poiseuille; explicit value allowed),
- `L = ρL/A` (inertance),
- `C = 2πr³L/(Eh)` assigned to a compartment endpoint (thin-wall hoop strain),
- `h = ρg(z_dst-z_src)` (gravity head),
- areas for the Bernoulli kinetic term and minor losses,
- optional `collapsible`/`external_pressure_pa` (Starling resistor) and `valve`
  (one-way) behaviour.

The solver enforces conservation at every node and the integrated momentum
balance on every edge. Steady linear networks are solved as one square linear
system; kinetic/minor terms use a nonlinear root finder; collapsible veins and
valves use an active-set complementarity loop (open edges obey the pressure-flow
law, closed edges carry zero flow, and edges open/close when the driving pressure
or flow changes sign). Transients use an exact matrix exponential for linear R–C
networks (all `L=0`, positive compliance) and implicit Euler otherwise.

### The independent referees

1. **Exact analytic solutions**: Hagen–Poiseuille, the Windkessel exponential
   `qR(1-e^{-t/RC})`, the R–L step `(Δp/R)(1-e^{-Rt/L})`, and the Starling
   waterfall `max(0,(p_up-max(p_down,p_ext))/R)`.
2. **The exact Womersley solution** of the pulsatile Navier–Stokes equations for
   a rigid circular tube:
   `Z = iωρL/(A F)`, `F = 1 - 2J₁(z)/(zJ₀(z))`, `z = α√(-i)`. This is
   independent of the graph's lumped `R+iωL` edge and is used to measure where
   that edge stops being valid.
3. **Physiological reference ranges** for venous pressure, used only after the
   solve and never to set a parameter: supine foot/ankle 5–10 mmHg, standing
   85–90 mmHg.

### The venous network

A lower-limb network with 9 nodes and 9 valved conduits: foot → ankle → calf →
popliteal → femoral → iliac → IVC → right atrium, plus a superficial saphenous
path from ankle to femoral. Geometry is anatomical (1.75 m person, right atrium
1.05 m above the floor, ankle 0.06 m), not fitted to the pressure target. The
capillary inflow (0.5 L/min per leg) is an input. Posture is modeled by rotating
the height coordinate, `z(θ)=z₀ sin θ`; the pressures are then predictions.

## Validation — the part that matters

| Quantity | This work | Independent authority | Error |
|---|---|---|---|
| Steady Poiseuille flow on a graph | 6.6667×10⁻⁴ m³/s | Δp/R | **1.6×10⁻¹⁶ relative** |
| Gravity-driven flow, 1 m column | 1.0395×10⁻² m³/s | ρgΔz/R | **0 absolute** |
| Bernoulli stenosis pressure | 1328.14 Pa | Rq + ½ρq²(1/A₂²−1/A₁²) | **0 relative**; Bernoulli residual 0 Pa |
| Starling waterfall, 4 regimes | 1.0/1.5/0/0 ×10⁻³ m³/s | analytic waterfall | **2.2×10⁻¹⁹ m³/s max absolute** |
| Series/parallel equivalent resistance | 5.0/1.0 MΩ | Laplacian / circuit rules | **2.3×10⁻¹⁰ Pa s/m³** |
| Windkessel transient, exact | 993.262053 Pa | qR(1−e^{−t/RC}) | **0 Pa** |
| Windkessel, implicit Euler | 993.09 Pa at dt=1 ms | same | 0.169 Pa; observed order **1.01** |
| R–L step response | 9.93245×10⁻⁶ m³/s | (Δp/R)(1−e^{−Rt/L}) | **1.7×10⁻⁵ relative** |
| Lumped R–L vs exact Womersley | — | exact Womersley Z(ω) | <1% up to **α=0.48**; 4.1% at α=1; 20.7% at α=5 |
| Venous pressure, supine | foot **8.4** mmHg, ankle 7.2 | 5–10 mmHg | inside range |
| Venous pressure, standing | foot **88.7** mmHg, ankle 84.4 | 85–90 mmHg | foot inside; ankle 0.6 mmHg below |
| Standing−supine hydrostatic difference | **80.3** mmHg | ρgΔz (81.9 mmHg for 1.05 m) | 1.6 mmHg |
| Effective resistance cross-check (0° network) | Δp/q = 5.376338×10⁷ | Laplacian R_eff | **3.5×10⁻¹⁵ relative** |
| Extended Bernoulli on every vein edge | edge momentum balance | **2.0×10⁻¹² Pa** max residual |

**Blind prediction.** No parameter was fitted to the venous pressure reference.
The geometry and capillary inflow were fixed from anatomy; the reference range
was consulted only after the solve. The standing foot pressure landed inside the
published range, and the supine value landed inside its range. This is the
successful prediction the contribution is built around.

**What is new vs reused.** The flow graph, solvers, Bernoulli/Womersley helpers,
venous network, tests, and validation script are new. The typed `Graph`, schema
machinery, renderer, and the PBPK graph are reused unchanged (apart from adding
conduit labels/styles to the renderer).

## Limits

- **The venous prediction is steady-state.** It does not model the calf muscle
  pump or ambulatory venous pressure (the post-contraction fall to ~20–30 mmHg).
  That needs transient collapse and valve opening/closing.
- **Transient valve/collapse is not implemented.** The steady solver uses
  active-set complementarity; the transient solver raises `NotImplementedError`
  for these edges. A closed valve can isolate a segment whose pressure is set by
  stored volume, which the current steady formulation does not represent.
- **The lumped R–L edge is a low-α model.** It is within 1% only for α≤0.48;
  large veins at 1 Hz (α≈7–14) have 12–18% pulsatile-impedance error. The steady
  pressure result is unaffected because steady flow has ω=0. The next rung is a
  frequency-domain graph solve with the exact Womersley `Z(ω)` on each edge.
- **Geometry is approximate.** Segment lengths/radii are order-of-magnitude
  values; the reference ranges are broad. The result establishes the hydrostatic
  mechanism and the solver's correctness, not patient-specific accuracy.
- **Blood is Newtonian by default.** At low shear in small veins the apparent
  viscosity is higher; the API accepts an explicit resistance/viscosity.
- **No wave propagation.** A node is a lumped compliance, not a distributed
  tube. Wave speed and reflection sites need the 1-D PDE rung.
- **Active-set is a local method.** It converged on all tested Starling/valve
  cases, but a non-smooth complementarity problem can have multiple solutions;
  the solver reports the one reached from the linear seed.

## Reproduce

```bash
pip install -e ".[dev]"                 # numpy, scipy, ...
python scripts/validate_flow.py          # writes validation_flow.json + 3 figures
pytest -q tests/test_flow.py             # 24 tests against exact authorities
pytest -q                                # full suite: 41 passed, 6 skipped
```

The validation report is `docs/validation_flow.json`; the derivation is
`docs/bernoulli_veins.md`; the method ladder is `docs/prediction_methods.md`.

## References

- Womersley, J. R. (1955). Method for the calculation of velocity, rate of flow
  and viscous drag in arteries when the pressure gradient is known.
  *Journal of Physiology* 127(3), 553–563.
- Nichols, W. W., O'Rourke, M. F., & Vlachopoulos, C. (2011).
  *McDonald's Blood Flow in Arteries*, 6th ed.
- Fung, Y. C. (1997). *Biomechanics: Circulation*, 2nd ed. Springer.
- Shapiro, A. H. (1977). Steady flow in collapsible tubes.
  *Journal of Biomechanical Engineering* 99(3), 126–147.
- Guyton, A. C., & Hall, J. E. *Textbook of Medical Physiology*, 13th/14th ed.
  (venous pressure in the standing and recumbent positions).
- Pedley, T. J. (1980). *The Fluid Mechanics of Large Blood Vessels*.
  Cambridge University Press.
