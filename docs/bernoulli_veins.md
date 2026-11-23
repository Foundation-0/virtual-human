# Bernoulli's principle in veins — derived, extended, and where it fails

This document derives Bernoulli's principle from the flow graph, shows exactly
which terms the graph keeps, and measures where the classical (inviscid, steady)
form is wrong in veins. The executable version lives in
[`vh/graph/flow.py`](../vh/graph/flow.py); the numbers below are reproduced by
[`scripts/validate_flow.py`](../scripts/validate_flow.py) and stored in
[`validation_flow.json`](validation_flow.json).

## 1. Start from the real equations

For an incompressible Newtonian fluid, conservation of momentum and mass are

$$
\rho\left(\frac{\partial \mathbf u}{\partial t}
+(\mathbf u\cdot\nabla)\mathbf u\right)
= -\nabla p + \rho \mathbf g + \mu \nabla^2 \mathbf u,
\qquad \nabla\cdot\mathbf u = 0 .
$$

Using

$$
(\mathbf u\cdot\nabla)\mathbf u
= \nabla\!\left(\tfrac12 \mathbf u^2\right) - \mathbf u\times\boldsymbol\omega,
$$

and integrating along a streamtube from cross-section 1 to cross-section 2 gives
the **extended Bernoulli equation**:

$$
\underbrace{p_1 + \tfrac12\rho v_1^2 + \rho g z_1}_{H_1}
=
\underbrace{p_2 + \tfrac12\rho v_2^2 + \rho g z_2}_{H_2}
+ \underbrace{\rho\!\int_1^2 \frac{\partial v}{\partial t}\,ds}_{\text{unsteady}}
+ \underbrace{\Delta p_{\text{loss}}}_{\text{viscous + minor}} .
$$

The total head $H$ is **not** constant in a real vessel: it decreases by the
losses and by the unsteady acceleration term. Classical Bernoulli is the special
case where both are zero. For a uniform tube the unsteady term is
$\rho L\,\dot v = (\rho L/A)\,\dot q = L_{\text{tube}}\,\dot q$, and for
Poiseuille flow the viscous loss is
$\Delta p_{\text{loss}} = Rq = 8\mu L q/(\pi r^4)$.

For a non-uniform velocity profile, replace $\tfrac12\rho v^2$ by
$\alpha\,\tfrac12\rho v^2$, with $\alpha=1$ for a flat profile and $\alpha=2$
for a fully developed parabolic profile. The graph exposes this as
`kinetic_factor`.

## 2. The graph edge is exactly this equation

A flow graph stores one momentum balance on every conduit edge and one storage
balance at every node:

$$
\begin{aligned}
\text{node }i:\quad & C_i\,\frac{dp_i}{dt} = \sum_e B_{ie}q_e + s_i, \\[2pt]
\text{edge }e:\quad & L_e\,\frac{dq_e}{dt}
= p_{\text{src}} - p_{\text{dst}} - h_e - R_e q_e \\
&\qquad - \tfrac12\rho\,q_e|q_e|
  \left(\frac{1}{A_{\text{dst}}^2}-\frac{1}{A_{\text{src}}^2}\right)
  - \Delta p_{\text{minor}} - \Delta p_{\text{collapse/valve}},
\end{aligned}
$$

with $h_e=\rho g(z_{\text{dst}}-z_{\text{src}})$,
$L_e=\rho \ell/A$, $R_e=8\mu\ell/(\pi r^4)$,
$C_e=2\pi r^3\ell/(Eh)$ for a thin-walled elastic tube, and $B$ the oriented
incidence matrix ($+1$ at the destination, $-1$ at the source). Rearranging the
edge equation:

$$
p_{\text{src}} + \tfrac12\rho v_{\text{src}}^2 + \rho g z_{\text{src}}
=
p_{\text{dst}} + \tfrac12\rho v_{\text{dst}}^2 + \rho g z_{\text{dst}}
+ L_e\dot q_e + R_e q_e + \Delta p_{\text{minor}} .
$$

That is precisely extended Bernoulli. The graph is therefore not a metaphor for
the physics: it is a lumped, 1-D Navier–Stokes solver whose topology is the typed
graph and whose constitutive laws live on the edges.

The node equation is the other half: conservation of the conserved quantity
(volume here, charge or mass in another domain). The two equations together are
Kirchhoff's current law and the constitutive branch relations of a nonlinear
circuit — which is why the same solver works for blood, water, current, and heat.

## 3. Classical Bernoulli, re-derived

Set $L_e=0$ (steady), $R_e=0$ (inviscid), no minor losses, and a flat profile
($\alpha=1$). The edge equation collapses to

$$
p_1 + \tfrac12\rho v_1^2 + \rho g z_1
=
p_2 + \tfrac12\rho v_2^2 + \rho g z_2 .
$$

For a horizontal stenosis ($z_1=z_2$), continuity gives $v_2=v_1 A_1/A_2$, so

$$
p_1-p_2 = \tfrac12\rho v_1^2\left[\left(\frac{A_1}{A_2}\right)^2-1\right].
$$

The graph's [`kinetic_pressure_drop`](../vh/graph/flow.py) is this expression
written in terms of $q$; [`extended_bernoulli_residual`](../vh/graph/flow.py)
returns zero when the full balance (including losses and unsteady terms) holds.

## 4. Where classical Bernoulli fails in veins

Veins are a deliberately hostile case for the inviscid, steady, fixed-geometry
assumption.

| Effect | What classical Bernoulli does | What the flow graph does | Size in a vein |
|---|---|---|---|
| **Viscosity** | drops $Rq$ entirely | keeps $Rq$, with $R=8\mu\ell/\pi r^4$ | 0.18 mmHg per 0.2 m of 5 mm vein at 0.5 L/min |
| **Gravity** | keeps $\rho g z$ but one often omits it | keeps $h=\rho g\Delta z$ | **82 mmHg per 1.05 m** of blood |
| **Unsteady flow** | assumes $\partial v/\partial t=0$ | keeps $L\dot q$ | important in breathing, walking, cardiac pulsatility |
| **Velocity profile** | assumes flat profile ($\alpha=1$) | exposes `kinetic_factor` ($\alpha=2$ for Poiseuille) | small in veins, large in stenoses |
| **Collapse** | assumes a fixed tube | Starling waterfall: $q=\max(0,(p_{\text{up}}-\max(p_{\text{down}},p_{\text{ext}}))/R)$ | veins collapse at low transmural pressure |
| **Valves** | time-reversible | one-way complementarity; closed edge carries $q=0$ | prevents reflux |
| **Non-Newtonian blood** | constant $\mu$ | constant $\mu$ by default; supply an apparent viscosity | matters at low shear in small veins |

### The venous scale: gravity dominates, kinetic is negligible

Take a 5 mm-radius vein, 0.2 m long, carrying 0.5 L/min:

| Term | Value |
|---|---:|
| velocity $v=q/A$ | 0.106 m/s |
| kinetic $\tfrac12\rho v^2$ | 5.97 Pa = **0.045 mmHg** |
| viscous $Rq$ | 23.8 Pa = **0.178 mmHg** |
| gravity over 1.05 m | 10 915 Pa = **81.9 mmHg** |

Gravity is ~460× the viscous term and ~1 800× the kinetic term. In a large vein
at rest, classical Bernoulli's *gravity* term is the one that matters; dropping
it is the biggest error. But the same equation predicts the wrong answer if the
vein collapses, because then the downstream pressure stops setting the flow.

### The stenosis scale: kinetic dominates

For a 5 mm → 2 mm narrowing carrying 20 mL/s (1.2 L/min):

| Term | Value |
|---|---:|
| $v_1$ / $v_2$ | 0.255 / 1.592 m/s |
| kinetic pressure drop | 1 308 Pa = **9.81 mmHg** |
| viscous drop ($R=10^6$ Pa s/m³) | 20 Pa = **0.15 mmHg** |
| kinetic fraction | **98.5 %** |

Here classical Bernoulli is almost the whole story, and viscous loss is a 1.5 %
correction. The same graph edge handles both limits: it is one equation, not two
models.

### The unsteady scale: the lumped R–L edge has a measured limit

The graph's transient edge is $L\dot q+Rq=\Delta p$. Its sinusoidal impedance is
$R+i\omega L$. The exact Womersley solution of the pulsatile Navier–Stokes
equations for a rigid circular tube is

$$
Z(\omega)=\frac{i\omega\rho L}{A\,F(\alpha)},\qquad
F(\alpha)=1-\frac{2J_1(z)}{zJ_0(z)},\qquad
z=\alpha\sqrt{-i},\qquad
\alpha=r\sqrt{\frac{\omega\rho}{\mu}} .
$$

The exact result is implemented as [`womersley_impedance`](../vh/graph/flow.py)
and used as the independent referee. Measured relative error of the graph's
lumped edge:

| $\alpha$ | 0.1 | 0.5 | 1 | 2 | 5 |
|---|---:|---:|---:|---:|---:|
| $|Z_{\text{exact}}-Z_{\text{lumped}}|/|Z_{\text{exact}}|$ | 0.04 % | 1.04 % | 4.1 % | 13.6 % | 20.7 % |

The lumped edge is within 1 % for $\alpha\le0.48$ and within 5 % for
$\alpha\le1.08$. At 1 Hz a 1 cm vena cava has $\alpha\approx13.8$ (11.5 %
pulsatile-impedance error) and a 5 mm femoral vein has $\alpha\approx6.9$
(18.2 % error). The error peaks in the transition $\alpha\approx2$–$5$ and
slowly decreases at very high $\alpha$, where both models approach pure
inertance.

This is a real limit, not a rounding error. It does **not** affect the steady
venous-pressure prediction in §5, because steady flow has $\omega=0$ and the
inertance term vanishes; it does mean the current time-domain edge is a
small-$\alpha$ model for pulsatile flow. The next rung is a frequency-domain
graph solve with the exact $Z(\omega)$ on each edge (see
[`prediction_methods.md`](prediction_methods.md)).

## 5. A worked prediction: standing venous pressure

The lower-limb network in [`scripts/validate_flow.py`](../scripts/validate_flow.py)
has anatomical geometry (1.75 m person, right atrium 1.05 m above the floor,
ankle 0.06 m), Poiseuille resistances, gravity heads, one-way valves, and a
0.5 L/min capillary inflow per leg. The geometry and the inflow are inputs; the
pressures are predictions.

| Posture | Foot pressure (model) | Ankle pressure (model) | Reference (not used to fit) |
|---|---:|---:|---:|
| Supine (0°) | 8.4 mmHg | 7.2 mmHg | 5–10 mmHg |
| Standing (90°) | **88.7 mmHg** | 84.4 mmHg | 85–90 mmHg |

The standing−supine difference is 80.3 mmHg, essentially the hydrostatic column
$\rho g\Delta z$ (81.9 mmHg for 1.05 m) minus the small viscous and posture
terms. The foot value lands inside the physiological range; the ankle value is
0.6 mmHg below the lower edge. The model is predicting the column, not fitting
it: no parameter was adjusted to the 85–90 mmHg target.

The same network gives an exact cross-check of the graph algorithms: at 0° the
network is horizontal, so $\Delta p/q$ must equal the Laplacian effective
resistance. It does, to $3.5\times10^{-15}$ relative. On every edge of the
standing network, the extended-Bernoulli residual
$\Delta H-\Delta p_{\text{loss}}$ is at most $2.0\times10^{-12}$ Pa.

## 6. Limits

- **Steady pressure only.** The prediction above is a steady-state result. It
  does not model the calf muscle pump or ambulatory venous pressure (the
  post-contraction fall to ~20–30 mmHg), which needs transient collapse and
  valve opening/closing.
- **Transient valves/collapse are not implemented.** The steady solver uses an
  active-set complementarity method; the transient solver currently raises
  `NotImplementedError` for collapsible/valve edges. A closed valve can isolate a
  segment whose pressure is then set by its stored volume, which the present
  steady formulation does not represent.
- **Lumped R–L is not a high-$\alpha$ model.** See the table above. Use the exact
  Womersley impedance in the frequency domain, or a ladder approximation, when
  $\alpha\gtrsim1$.
- **Geometry is approximate.** Segment lengths and radii are order-of-magnitude
  values; the reference ranges are broad. The result establishes the hydrostatic
  mechanism and the solver's correctness, not patient-specific accuracy.
- **Blood is treated as Newtonian.** At low shear in small veins the apparent
  viscosity is higher; supply it explicitly if that matters.
- **No wave propagation.** The node is a lumped compliance, not a distributed
  tube. For wave travel times and reflection sites, use a 1-D PDE model.

## 7. Reproduce

```bash
python scripts/validate_flow.py          # writes validation_flow.json + figures
pytest -q tests/test_flow.py             # 24 tests against exact authorities
```

The key functions:

```python
from vh.graph.flow import (
    FlowNodeSpec, VesselSpec, build_flow_graph, FlowNetwork,
    poiseuille_resistance, tube_inertance, tube_compliance,
    hydrostatic_head, kinetic_pressure_drop, starling_flow,
    womersley_impedance, extended_bernoulli_residual,
)

# A vein segment: geometry -> R, L, C, gravity head; then solve the graph.
g, schema = build_flow_graph(nodes, vessels)
net = FlowNetwork.from_graph(g)
sol = net.steady_state()
print(sol.pressure_mmhg(), sol.flow_l_min())
```

## References

- Womersley, J. R. (1955). Method for the calculation of velocity, rate of flow
  and viscous drag in arteries when the pressure gradient is known.
  *Journal of Physiology* 127(3), 553–563.
- Nichols, W. W., O'Rourke, M. F., & Vlachopoulos, C. (2011).
  *McDonald's Blood Flow in Arteries: Theoretical, Experimental and Clinical
  Principles*, 6th ed.
- Fung, Y. C. (1997). *Biomechanics: Circulation*, 2nd ed. Springer.
- Shapiro, A. H. (1977). Steady flow in collapsible tubes.
  *Journal of Biomechanical Engineering* 99(3), 126–147.
- Guyton, A. C., & Hall, J. E. *Textbook of Medical Physiology*, 13th/14th ed.
  (venous pressure in the standing and recumbent positions).
- Pedley, T. J. (1980). *The Fluid Mechanics of Large Blood Vessels*.
  Cambridge University Press.
