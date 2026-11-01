# Virtual Human

A multi-scale simulation of the human body for testing medicines and proteins,
built from scratch: an **atom-level physics engine** for proteins and drugs,
coupled to a **whole-body physiology engine** that predicts what a drug does in
a person, with a **real-time browser demo**.

> Status: v0.1, under active development. Every accuracy claim below is backed by a
> test or a validation report in `docs/`.

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
| `vh.molecular` | From-scratch JAX engine: AMBER ff14SB force field, GB-OBC2 implicit water, autodiff forces, Langevin (BAOAB) and Verlet dynamics, L-BFGS minimization | Matches OpenMM (used only as a referee) term by term to ~1e-14 relative on real proteins; trajectories identical step by step; energy conserved ([`docs/validation_molecular.md`](docs/validation_molecular.md)) |
| `vh.body` | From-scratch whole-body PBPK: 13 organs + blood, Rodgers & Rowland tissue partitioning, liver (well-stirred, IVIVE), kidney (filtration), oral absorption (stomach + 7-segment intestine) with gut-wall first-pass, virtual populations; solved exactly with matrix exponentials | Bottom-up midazolam predictions vs 100 clinical datasets (IV and oral) digitized by the Open Systems Pharmacology project |
| `vh.bridge` | Unbound drug at the target organ → receptor occupancy over time | (in progress) |
| `web/` | Real-time demo: atom-level protein dynamics and a virtual patient, live in the browser | JS engine checked against the Python engine |

## Quick start

```bash
pip install -e ".[dev]"              # numpy, scipy, jax (+ openmm/pdbfixer for structure prep & tests)
pytest -q                            # full test suite

# simulate a protein atom by atom (writes a multi-model PDB you can open in PyMOL/ChimeraX)
python -m vh.molecular.run data/structures/1l2y_trpcage.pdb --ps 50 --temp 300

# re-run the validation reports
python scripts/validate_molecular.py
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
vh/molecular   energy.py (force field), integrate.py, minimize.py, prep.py, io.py, run.py, reference.py
vh/body        physiology.py, partition.py, compound.py, pbpk.py
vh/bridge      molecular → body coupling
tests/         pytest suite
scripts/       validation report generators
docs/          feasibility, validation reports, roadmap
web/           real-time browser engine + demo
data/          structures, tissue composition, clinical observations
```

## Not a medical device

This is research software for hypothesis generation. It is not validated for
clinical decision-making, and no simulation replaces clinical testing.
