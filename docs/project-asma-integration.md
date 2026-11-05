# Foundation-0 development fork and Project Asma

This is a development fork of [Rohan Arun's Virtual Human](https://github.com/rohanarun/virtual-human),
created by the Project Asma owner on 2026-11-05. Upstream remains the original
project. Its MIT license, creator credit and contribution requirements are
preserved. The upstream README credits Claude for the original implementation
under the maintainer's direction. The graph contributions below were prepared
with OpenAI Codex under [@ahrazzle](https://github.com/ahrazzle)'s direction.

## Shared purpose and repository boundary

[Project Asma](https://github.com/Foundation-0/project-asma) uses a pinned,
unchanged copy of `vh/graph/core.py` for an offline public-evidence prototype.
It supplies a concrete test case: do not silently substitute antibody binding,
protein function, HLA annotation, receptor recognition and tissue injury for one
another. The Asma repository owns the literature curation and domain-specific
rules. This fork owns reusable graph interfaces and numerical-model checks.

Asma does not currently adopt the molecular, PBPK or quantum engines, nor does it
use the new scalar model registry to invent numerical values for unknown biology.
Its qualitative assertions stay qualitative. Recognition and injury are separate
questions: direct recognition of a CDR2L ligand and indirect injury can coexist.
Neither is established by a graph or a successful software test.

No patient records, personal correspondence, private clinical analysis or sensitive
derived hypotheses belong in this public fork. Only public-source requirements,
original software, permitted upstream inputs and clearly artificial fixtures are
used for this contribution.

## Contributions and acceptance

The initial integration combines two separately reviewable commits and reports:

| Contribution | Validation authority and outcome |
| --- | --- |
| [Explicit model contracts](../auto-research/2026-11-05-E-model-contracts/report.md) | Exact scalar mass balance: maximum error 0 L/h over four synthetic cases, tolerance 1e-12 L/h; 50 new contract cases. |
| [Clearance reference correction](../auto-research/2026-11-05-E-clearance-reference/report.md) | Exact organ mass balance and integrated linear ODE; 24 new cases. Plasma reporting sum 17.104405 L/h remains distinct from IV dose/AUC 17.103070 L/h. |

Neither completes the full Track E SBML milestone. Existing PBPK differential
equations are unchanged. Domain/validation strings in a model contract are
declarations; the runtime verifies supplied interface semantics and bounds, not
biological truth. The clearance check catches a real reference-basis error in
graph reporting and avoids overstating copied export arithmetic as independent
physiological validation.

## Combined verification, 2026-11-05

On code revision `d45ca7f50188dc3580dbec11bd0e14367e107fd6` (the two implementation
commits applied to upstream `3d75e8e25d6f3fec08e1ccecd424532bccaa1b82`):

- **111 Python tests passed**, no skips, in 76.12 s.
- JavaScript engine checks passed using the checked-in reference fixtures.
- `scripts/export_web.py` completed, then JavaScript checks passed again against
  regenerated references. Worst plasma-profile relative error was 4.8e-13 over
  48 points. Molecular energy/force comparisons and 500-step stability checks
  also passed. This is numerical verification against the supplied fixtures,
  not validation of anti-Yo mechanisms or individual outcomes.
- Regenerated molecular web fixtures were not retained: the contribution does
  not change those engines or their inputs. Checked-in fixture checks had already
  passed independently. No unrelated model or clinical dataset files changed.
- Focused tests and the analytical comparisons were independently rerun by an
  internal reviewing agent. This is not external scientific review.

Runtime: Linux x86_64; Python 3.12.14; Node 24.19.0. Installed numerical/test
versions: NumPy 2.5.4, SciPy 1.18.1, JAX/JAXlib 0.12.0.1, OpenMM 8.6.1,
PDBFixer 1.12.0, PySCF 2.15.0, geomeTRIC 1.1.1, pytest 9.2.0,
Matplotlib 3.11.2. This records the observed environment, not a new repository
dependency lock or a claim of portability to every allowed package version.

```bash
python -m venv /tmp/vh-dev
/tmp/vh-dev/bin/python -m pip install -e '.[dev]'
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/vh-dev/bin/python -m pytest -q
node web/test_engines.mjs
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/vh-dev/bin/python scripts/export_web.py
node web/test_engines.mjs
```

The clearance report records a diagram-rendering limitation: browser policy
blocked Mermaid's local renderer. Correct Mermaid source and inline diagrams
replace the affected stale images; no browser policy was changed.

## How the repositories progress together

1. Record a concrete Asma failure case or missing interface, without importing
   private data. Decide whether it is a general library concern or a domain rule.
2. Put general fixes here with an independent validation authority and a report
   following `auto-research.md`; put evidence semantics and scientific critique
   in Asma. Preserve unresolved and negative findings.
3. Review each change in a PR. Asma updates its vendored source only deliberately,
   preserving hashes and the upstream MIT notice, then reruns its offline checks.
4. Submit reusable changes to the original repository when contributor access is
   available. The owner is arranging that access. Fork PRs do not imply upstream
   acceptance, and fork-specific coordination docs need not accompany upstream
   feature PRs.

No automatic merge, deployment, background research run or patient simulation is
introduced by this workflow. Further simulation work needs a defined biological
question, applicable inputs and validation that can resolve that question.
