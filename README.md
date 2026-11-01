# Virtual Human

A multi-scale simulation of the human body for testing medicines and proteins,
built from scratch and validated at every level. It spans from **coupled-cluster
quantum mechanics** on the molecules the body is built from, up through an
**atom-level physics engine** for proteins and drugs, to a **whole-body engine**
that predicts what a drug does in a person — with a **real-time browser demo**.

> Status: under active development. Every accuracy claim below is backed by a test
> or a validation report in `docs/`, checked against an independent engine, an
> exact analytic result, or experiment.

## Two directions at once

- **Up for coverage**: molecule → cell → organ → whole body, so a result can
  reach a clinical endpoint. The whole-body layer runs faster than real time.
- **Down for accuracy**: start at the *highest accuracy physics allows* on the
  *smallest real parts of the body* (water, the peptide bond, amino acids),
  prove them correct against known truth, then carry that accuracy upward. This
  is the [accuracy ladder](docs/accuracy_ladder.md), and its top rung —
  CCSD(T) at the complete-basis limit — is the standard that "molecular accuracy"
  is itself measured against. We reproduce experiment there to a fraction of a
  percent with no fitting.

## Why multi-scale (and not "every atom of a body")

A body has ~7×10²⁷ atoms. Atomic simulation needs ~5×10¹⁴ time steps per
simulated second, so one second of a whole body is ≳10⁴⁵ floating-point
operations. On today's fastest supercomputers (~2×10¹⁸ FLOP/s), that's billions
of times the age of the universe, and a single human cell for one second would
take ~800,000 years. See [`docs/feasibility.md`](docs/feasibility.md).

So molecular accuracy goes where molecules decide the outcome (a drug binding its
target, a protein's shape and motion). The numbers it produces are passed up to
the body model, which works in organs, blood flows and concentrations and runs
faster than real time.

```
 vh.molecular   atoms        proteins + drugs in water, femtosecond steps
      │  binding strength, unbound fraction, ...
      ▼
 vh.bridge      target engagement (receptor occupancy over time)
      ▲
      │  unbound drug concentration at the target organ
 vh.body        organs       whole-body drug distribution and elimination (hours/days)
```

## What exists today

| Layer | What it does | How it's validated |
|---|---|---|
| `vh.quantum` | **The accuracy ceiling.** Ab initio electronic structure (HF, DFT, MP2, CCSD, **CCSD(T)**) via PySCF, with counterpoise (BSSE) correction, complete-basis-set extrapolation, geometry optimization, and QM/MM electrostatic embedding | Water geometry & dipole, the water-dimer hydrogen bond, the peptide bond and glycine reproduced from first principles to a fraction of a percent vs experiment/benchmark; QM/MM vs the exact classical limit and finite-difference forces ([`docs/accuracy_ladder.md`](docs/accuracy_ladder.md)) |
| `vh.molecular` | From-scratch JAX engine: AMBER ff14SB force field, GB-OBC2 implicit water, autodiff forces, Langevin (BAOAB) and Verlet dynamics, L-BFGS minimization | Matches OpenMM (used only as a referee) term by term to ~1e-14 relative on real proteins; trajectories identical step by step; energy conserved ([`docs/validation_molecular.md`](docs/validation_molecular.md)) |
| `vh.body` | From-scratch whole-body PBPK: 13 organs + blood, Rodgers & Rowland tissue partitioning, liver (well-stirred, IVIVE), kidney (filtration), oral absorption (stomach + 7-segment intestine) with gut-wall first-pass, virtual populations; solved exactly with matrix exponentials | Bottom-up midazolam predictions vs 100 clinical datasets (IV and oral) digitized by the Open Systems Pharmacology project |
| `vh.bridge` | Unbound drug at the target organ → receptor occupancy over time; Kd ↔ binding free energy | Analytic round-trip; kinetic occupancy |
| `vh.graph` | **Formalizes the body as a typed graph** across scales (organs↔blood flow, atoms↔bonds); graph algorithms for reachability, pathways, cycles, flow balance, bottlenecks; renders and verifies subgraphs | Blood conservation (Kirchhoff) to 1e-16 on the real PBPK model; a pure graph BFS reproduces the force field's exclusion list exactly ([`docs/graph_formalization.md`](docs/graph_formalization.md)) |
| `web/` | Real-time demo: atom-level protein dynamics and a virtual patient, live in the browser | JS engine checked against the Python engine to ≤1e-12 |

The explicit-solvent engine (`vh.molecular.periodic`, `.constraints`, `.explicit_md`) adds rung 3 of the accuracy ladder: every water molecule, exact long-range electrostatics (Ewald / smooth PME), rigid-water SHAKE/RATTLE and a Monte-Carlo barostat — validated term-by-term against OpenMM Ewald (energy to 1e-13, PME to 1e-11, constraints to 1e-16).

See [`docs/accuracy_ladder.md`](docs/accuracy_ladder.md) for the full five-rung ladder and what each rung is validated against.

## Quick start

```bash
pip install -e ".[dev]"              # numpy, scipy, jax (+ openmm/pdbfixer for structure prep & tests)
pip install -e ".[quantum]"          # pyscf + geometric, for the vh.quantum accuracy ceiling
pytest -q                            # full test suite

# the accuracy ceiling: water, the hydrogen bond, the peptide bond, glycine, from first principles
python scripts/validate_quantum.py
python scripts/plot_quantum.py       # -> docs/img/quantum_accuracy.png

# simulate a protein atom by atom (writes a multi-model PDB you can open in PyMOL/ChimeraX)
python -m vh.molecular.run data/structures/1l2y_trpcage.pdb --ps 50 --temp 300

# re-run the molecular / body validation reports
python scripts/validate_molecular.py
python scripts/validate_body.py
```

## Where the inputs come from

- **Force field**: AMBER ff14SB + OBC2 parameters, assigned via OpenMM's `ForceField`
  used purely as a parameter compiler (`vh/molecular/prep.py`). All energies, forces
  and dynamics are computed by our engine.
- **Structures**: chignolin (PDB 1UAO) and Trp-cage (PDB 1L2Y), first NMR models.
- **Physiology**: ICRP Publication 89 reference adults.
- **Tissue composition**: Rodgers & Rowland tables via open-access transcriptions
  (see `data/tissue_composition_rr.json` for per-value provenance).
- **Clinical data**: Open Systems Pharmacology model library (each dataset cites its
  original publication).

## Layout

```
vh/quantum     ab_initio.py (HF/DFT/MP2/CCSD(T), CBS, BSSE), qmmm.py (QM/MM embedding)
vh/molecular   energy.py (force field), integrate.py, minimize.py, prep.py, io.py, run.py, reference.py
               periodic.py (Ewald/PME explicit solvent), constraints.py (SHAKE/RATTLE), explicit_md.py
vh/body        physiology.py, partition.py, compound.py, pbpk.py
vh/bridge      molecular → body coupling (Kd ↔ ΔG, receptor occupancy)
vh/graph       core.py (typed property graph), algorithms.py, physiology_graph.py, render.py
tests/         pytest suite
scripts/       validation report + figure generators
docs/          accuracy_ladder, graph_formalization, feasibility, validation reports, architecture, roadmap
web/           real-time browser engine + demo
data/          structures, tissue composition, clinical observations
```

## Not a medical device

This is research software for hypothesis generation. It is not validated for
clinical decision-making, and no simulation replaces clinical testing.
