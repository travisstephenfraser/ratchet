# loop-eval

A portable eval scaffold: a stable core (`loop_eval/`) you import, plus a thin per-project
layer (`projects/<name>/`) you write. Ports the patterns from `loop-eng` (the Rubrica
grading-prompt search) without touching it.

- **Verifier** — hash train/holdout split, holdout vault, anomaly/overfit guards, pluggable
  objective (`within_tol` / `prf1` / `judge` / custom).
- **Loop** — hill-climb a candidate on train (constraints threaded in as policy), persist
  every result with its regime stamp, escalate the winner to the holdout gate.
- **Bench** — compare fixed candidates on one frozen eval set under one enforced regime.
- **Versioning** — every run computes a regime; a silent comparability change BLOCKS until a
  ledger rationale is recorded; cross-regime comparison is refused.

See `docs/PORTING.md` to stand up a project. Run `python -m pytest` for the self-test.
