# Architecture

Three layers, each the right kind of model for its scale, connected so a
molecular quantity can change a whole-body outcome.

```
                         vh.molecular  (atoms, JAX)
   PDB ─► prep.py ─► MolecularSystem ─► energy.py ─► integrate.py / minimize.py
   (ff14SB+OBC2 params)   (arrays)     (E, forces)   (BAOAB / Verlet / L-BFGS)
                                           │
                                           │  Kd = c0 · exp(ΔG_bind / RT)      (future: from FEP)
                                           ▼
                         vh.bridge   occupancy.py
                                           │  receptor occupancy O(t)
                                           ▲
                                           │  unbound drug at the target organ
                         vh.body  (organs, exact linear ODE)
   Compound + Individual ─► pbpk.py ─► M (state matrix) ─► expm ─► concentrations(t)
      (partition.py: Rodgers & Rowland Kp; physiology.py: ICRP 89 organs)
```

## vh.molecular — atoms

- **`system.py`** `MolecularSystem`: plain arrays (per-atom parameters + bonded term lists + exceptions + GB
  parameters). The one representation both the JAX engine and the JS port consume; JSON-serializable.
- **`energy.py`** The physics, from scratch. Potential energy as the AMBER functional form (harmonic bonds and
  angles, cosine torsions, Lennard-Jones + Coulomb with 1-4 exceptions, GB-OBC2 polar solvation with analytic
  Born radii, ACE nonpolar term). Forces are exact gradients by JAX autodiff, so every energy term has a
  matching force for free. `jit`-compiled.
- **`prep.py`** The only external dependency in the hot path at v0.1: OpenMM's `ForceField` as a *parameter
  compiler* (PDB → hydrogens → ff14SB/OBC2 numbers), copied into a `MolecularSystem`. The OBC2 `CustomGBForce`
  expression is parsed and structurally verified, not trusted by filename. No simulation physics is delegated.
- **`integrate.py`** Langevin BAOAB (NVT) and velocity Verlet (NVE); many steps per compiled `lax.scan`.
- **`minimize.py`** L-BFGS (SciPy driver, JAX energy/grad).
- **`reference.py`** OpenMM Reference platform as an independent referee for tests.

## vh.body — organs

- **`physiology.py`** ICRP 89 reference adults; scaling to any body size (lean/fat/BSA), virtual populations
  with enzyme-activity variability.
- **`partition.py`** Rodgers & Rowland mechanistic tissue:plasma partition coefficients from physicochemistry;
  blood:plasma prediction.
- **`compound.py`** Drug definition with a source string on every parameter; IVIVE from in vitro enzyme kinetics.
- **`pbpk.py`** Assembles the perfusion-limited model as a linear system `dx/dt = M x + u`. Because it's linear,
  it's solved **exactly** by matrix exponential (`scipy.linalg.expm`) — any step size, no integration error.
  `RealTimeSimulator` steps a patient forward in arbitrary increments with live dosing: the demo's interface.

## vh.bridge — coupling

- **`occupancy.py`** Takes the unbound drug concentration at the target organ from the body model and the binding
  Kd from the molecular world, returns receptor occupancy over time (equilibrium or kinetic). `Kd ↔ ΔG_bind` is
  the exact hand-off: ΔG is what an alchemical free-energy calculation in `vh.molecular` will produce (roadmap).

## web — real-time, in the browser

`vh-md.js` and `vh-pbpk.js` reimplement the two engines in plain JavaScript (MD forces hand-derived and analytic,
including the chain rule through the Born radii; PBPK via a Padé matrix exponential). `scripts/export_web.py`
writes reference states from the Python engines, and `web/test_engines.mjs` checks the JS output against them to
≤1e-12. `scripts/build_demo.py` inlines everything into one self-contained page.

## Design choices

- **From scratch, checked against a referee.** The value is an engine we fully control and can differentiate,
  proven correct against an independent implementation rather than wrapping one.
- **Differentiable throughout** (JAX). Gradients of an outcome w.r.t. parameters are the foundation for
  calibration, sensitivity, and design (roadmap).
- **Provenance on every input.** Each physiological and drug parameter carries where it came from; validation
  reports regenerate from code.
- **Exact where the math allows** (linear PBPK via matrix exponential) and **convergence-tested where it
  doesn't** (MD time step, thermostat).
