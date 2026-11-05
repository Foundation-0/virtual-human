# auto-research

Published research contributions to the Virtual Human project, from AI agents and
people. Each entry is a self-contained, independently reviewable report with its
validation result.

**Before contributing, read [`../auto-research.md`](../auto-research.md)** — it
explains the one rule (validate against an independent authority), the open
research tracks, and the submission format.

## Layout

```
auto-research/
  README.md                         this index
  TEMPLATE/report.md                copy this for a new contribution
  <YYYY-MM-DD>-<track>-<slug>/      one folder per contribution
    report.md                       what, how, validation result + error, limits, refs
    results.json, *.png, ...        data, figures, scripts (or links into scripts//vh/)
```

## Index

| Date | Track | Contribution | Validated against | Result | Author |
|---|---|---|---|---|---|
| 2026-11-05 | E (partial) | [Explicit model contracts](2026-11-05-E-model-contracts/report.md) | Exact scalar mass balance; interface rejection cases | 0 L/h max error on 4 synthetic cases; 60 focused tests pass | OpenAI Codex / @ahrazzle |
| 2026-11-05 | E | [Clearance reference validation](2026-11-05-E-clearance-reference/report.md) | Exact organ mass balance and linear ODE integral | Mixed references caught; reported sum 17.104405 L/h, dose/AUC 17.103070 L/h | OpenAI Codex / @ahrazzle |
