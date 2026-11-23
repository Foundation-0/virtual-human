# How to predict a flow: the method ladder

A graph says *what is connected*. Prediction needs two more things: a
**constitutive law** on every edge (how much flow a pressure difference buys) and
a **solver** that enforces conservation everywhere at once. There is a ladder of
ways to supply those, from a topological query to a patient-specific digital
twin. Each rung answers a different question, has a different independent
authority, and fails in a different place. Choosing the highest rung is not the
goal; choosing the rung that matches the question is.

| Rung | Method | What it predicts | Independent authority | Cost | Fails when |
|---:|---|---|---|---|---|
| 0 | **Topology / graph algorithms** | reachability, bottlenecks, conservation, equivalent resistance | graph identities (Kirchhoff, Laplacian) | microseconds | you need a pressure or a flow value |
| 1 | **Lumped graph (0-D ODEs)** | organ/segment pressures and flows in time; Windkessel; Starling waterfall; valve gating | exact Hagen–Poiseuille, Windkessel, Starling, Womersley limits | milliseconds–seconds | wave travel, high $\alpha$, local 3-D geometry matter |
| 2 | **1-D distributed PDE** | pressure/flow waveforms, wave speed, reflection sites, impedance spectra | exact Womersley for rigid tubes; measured wave speed | seconds–minutes | curvature, branching, aneurysm/stenosis detail matter |
| 3 | **3-D CFD / FSI** | wall shear stress, recirculation, local pressure loss, aneurysm stress | mesh convergence, analytic limits, experiment | hours–days | geometry/boundary conditions are uncertain; many queries |
| 4 | **Data-driven surrogate** | fast approximation of rung 2/3 over a parameter range | held-out simulations or experiments **not used to train** | ms after training | out of distribution; conservation not guaranteed |
| 5 | **Causal / interventional graph** | effect of clamping, valve failure, ablation, drug | randomized/perturbation experiments or known mechanism | depends on data | confounding; no interventional data |
| 6 | **Hybrid digital twin** | patient-specific trajectory under intervention | prospective measurements not used to fit | real-time + calibration | model misspecification; sensor noise |

## Rung 0 — topology: exact answers without physics

A typed graph already answers questions that do not need a constitutive law:

- **Reachability**: which organs a drug or embolus can reach.
- **Path enumeration**: first-pass route, collateral paths.
- **Cycles / SCCs**: recirculation, feedback loops.
- **Flow balance (Kirchhoff)**: is the formalization physically consistent?
- **Effective resistance** from the weighted Laplacian:
  $R_{\text{eff}}=(e_s-e_t)^\top L^+ (e_s-e_t)$.
- **Current flow**: the unit-injection flow field, which shows the main paths and
  bottlenecks with conductances, not just hop counts.

These are exact for the structural question, and cheap. They are the right first
check on any model: if flow is not conserved at a node, no solver can rescue it.
But topology alone cannot tell you whether the portal vein pressure is 8 or
18 mmHg. That needs a law.

## Rung 1 — lumped graph: the current `vh.graph.flow`

Every node is a storage volume or junction; every edge is a conduit with
resistance $R$, compliance $C$ (assigned to a node), inertance $L$, gravity head
$h$, kinetic/Bernoulli term, and optional Starling collapse or valve behaviour:

$$
C_i\dot p_i=\sum_e B_{ie}q_e+s_i,\qquad
L_e\dot q_e=p_{\text{src}}-p_{\text{dst}}-h_e-R_eq_e-\Delta p_{\text{kinetic}}-\Delta p_{\text{minor}}.
$$

This is a lumped 1-D Navier–Stokes solver. It predicts:

- steady pressures and flows in a network of veins, arteries, or pipes;
- the Windkessel transient (compliance filling and draining);
- the Starling waterfall in a collapsible vein;
- one-way valve gating;
- the low-frequency pulsatile impedance $R+i\omega L$.

The independent authorities are exact analytic solutions, and the measured errors
are small:

| Check | Authority | Measured error |
|---|---|---|
| Poiseuille flow on a graph | $q=\Delta p/R$ | $1.6\times10^{-16}$ relative |
| Gravity-driven flow | $\rho g\Delta z/R$ | 0 |
| Windkessel transient | $qR(1-e^{-t/RC})$ | 0 (matrix exponential); implicit Euler order 1.01 |
| R–L step response | $({\Delta p}/{R})(1-e^{-Rt/L})$ | $1.7\times10^{-5}$ relative |
| Starling waterfall | $\max\!\left(0,\bigl(p_{\text{up}}-\max(p_{\text{down}},p_{\text{ext}})\bigr)/R\right)$ | $2.2\times10^{-19}$ m³/s |
| Series/parallel $R_{\text{eff}}$ | Laplacian / circuit rules | $2.3\times10^{-10}$ Pa s/m³ |
| Standing venous pressure | physiology reference 85–90 mmHg | foot 88.7 mmHg (inside range) |

**Use it when** you need organ- or segment-level pressures and flows, real-time
or many-query behavior, or a vein/Starling/valve question. **Do not use it when**
the pulse wave shape, wave speed, or local 3-D flow details are the answer.

The known limit inside rung 1 is the pulsatile edge. The lumped $R+i\omega L$
impedance is within 1 % only for Womersley number $\alpha\le0.48$; a 1 cm vena
cava at 1 Hz has $\alpha\approx13.8$ and ~11.5 % pulsatile-impedance error. The
steady pressure prediction is unaffected because steady flow has $\omega=0$.
For accurate pulsatility, move to rung 2 (exact $Z(\omega)$ on each edge) or use a
ladder approximation with more R–L rungs.

## Rung 2 — 1-D distributed PDE: waves and waveforms

Keep the cross-sectional area $A(x,t)$ and the cross-sectionally averaged
velocity $u(x,t)$ as fields along the vessel:

$$
\frac{\partial A}{\partial t}+\frac{\partial (Au)}{\partial x}=0,
\qquad
\frac{\partial u}{\partial t}+u\frac{\partial u}{\partial x}
=-\frac{1}{\rho}\frac{\partial p}{\partial x}
+\frac{1}{\rho A}\frac{\partial}{\partial x}\!\left(\mu A\frac{\partial u}{\partial x}\right),
$$

closed by a wall law $p(A;x)$ and boundary conditions at junctions and the heart.
Linearising gives the wave equation with wave speed
$c=\sqrt{(A/\rho)(\partial p/\partial A)}$; in the frequency domain each segment
carries the exact Womersley impedance and a propagation constant. This is the
rung that predicts **pulse wave velocity, reflection sites, systolic
augmentation, and impedance spectra**.

Authorities: the exact Womersley solution for a rigid tube (already implemented
as the referee for rung 1), measured wave speed in a tube, and in vivo
pressure/flow waveforms not used to fit. The cost is seconds to minutes for a
network; the failure modes are branching/curvature losses and uncertain wall
properties.

## Rung 3 — 3-D CFD / fluid–structure interaction

Solve the full Navier–Stokes equations on a patient-specific geometry, with a
moving wall if the question is wall stress. This predicts **wall shear stress,
recirculation, stenosis pressure recovery, aneurysm wall stress, and thrombus
risk** — quantities that are genuinely three-dimensional and cannot be recovered
from a 1-D model. Authorities are mesh and time-step convergence, analytic
Poiseuille/Womersley limits, and experiment. The cost is hours to days per case;
the dominant uncertainty is usually the geometry and boundary conditions, not the
solver.

## Rung 4 — data-driven surrogates

A Gaussian process, neural operator, or reduced-order model learns the map from
parameters/geometry to rung 2/3 outputs. It is fast enough for optimization,
uncertainty quantification, or control. The authority must be **held-out
simulations or experiments**, never the training set. The honest caveats:
out-of-distribution inputs can fail silently, and an unconstrained surrogate need
not conserve mass. Prefer architectures that build conservation in (divergence-
free fields, port-Hamiltonian forms) and report a blind test.

## Rung 5 — causal / interventional graphs

A flow graph predicts the effect of an intervention only if the intervention is
represented. Clamping a vessel, failing a valve, or ablating a tissue is a
**do**-operation, not an observation. The graph supplies the mechanism
(which paths are cut, which compartments lose inflow); causal inference supplies
the identification assumptions. Authorities are randomized/perturbation
experiments or a known mechanism. Without interventional data, a model can
describe but not identify the effect.

## Rung 6 — hybrid digital twin

Combine a physics rung with data assimilation (state estimation, parameter
calibration) and update the model from measurements. This is the only rung that
can be patient-specific in a meaningful sense. The authority is **prospective**
data not used to fit the model, and the honest output is a distribution, not a
point. It inherits every failure mode of the physics rung underneath it, plus
sensor noise and model misspecification.

## How to choose

1. **Is the question structural?** (reachability, consistency, bottleneck) → rung 0.
2. **Do you need organ/segment pressures and flows, or a vein/valve/Starling
   question?** → rung 1.
3. **Is the pulse waveform or wave speed the answer?** → rung 2.
4. **Is local 3-D geometry (stenosis, aneurysm, wall shear) the answer?** → rung 3.
5. **Do you need thousands of queries over a parameter range?** → rung 4, trained
   on rung 2/3, validated on held-out cases.
6. **Is the question “what happens if we clamp/ablate/fail this valve”?** → rung 5,
   with the mechanism from the graph.
7. **Is it a specific patient, now?** → rung 6, with prospective validation.

## What the graph contributes at every rung

- **Shared topology.** The same typed graph is the skeleton for the lumped
  solver, the 1-D network, and the 3-D case; parameters and provenance stay in
  one place.
- **Conservation invariants.** Flow balance and schema checks catch modeling
  errors before a solver runs.
- **Boundary conditions and junctions.** The incidence matrix $B$ is the same
  object at every rung; only the edge law changes.
- **Composability.** A subgraph can be solved and validated alone, then plugged
  back in (the PBPK elimination subgraph is the existing example).
- **Sensitivity and interventions.** Effective resistance, current flow, and
  path algorithms say which edges matter before an expensive simulation.

## Current status in `virtual-human`

| Rung | Status |
|---:|---|
| 0 | built: `vh.graph.algorithms` (reachability, paths, cycles, flow balance, centrality, Laplacian, effective resistance, current flow) |
| 1 | built: `vh.graph.flow` (R/C/L, gravity, Bernoulli, Starling, valves; steady, exact R–C, implicit Euler); validated in `docs/validation_flow.json` |
| 2 | not built; the exact Womersley impedance already exists as the referee |
| 3 | external engines/CFD, not in this repository |
| 4 | not built |
| 5 | mechanism from the graph + PBPK interventions; formal causal layer not built |
| 6 | not built |

The next high-value step is rung 2: a frequency-domain graph solve using the
exact $Z(\omega)$ on each edge. It would remove the rung-1 pulsatile limit while
reusing the same graph, schema, and conservation machinery.
