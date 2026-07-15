# Porting ratchet to a new project

Copy `projects/_template/` to `projects/<name>/` and fill in the adapters. You never
edit `ratchet/` (the core).

## The authoring workflow (start minimal, grow by failure mode)

1. **Minimal base.** `base.txt` = role, data, task + definition-of-done, output shape.
2. **Ingest your ground truth.** Implement `ingest()` — you export the truth yourself.
   Data under `data/` (gitignored). Truth values are strings; item inputs must be
   JSON-serializable so ratchet can fingerprint the exact evaluated content.
3. **Pick an objective.** `within_tol` (numeric closeness; `params.climb: within|mae` —
   use `mae` when within-rate ceilings and gives no gradient), `prf1` (classification/
   extraction), or `judge` (open-ended; add `judge.py:judge_fn(pred, rubric) -> float`).
   Need something else? `objective.name: custom` + `objective.py:make_objective(params)`.
4. **Establish the baseline.** Run `python -m ratchet.loop_cli --project
   projects/<name> --establish-baseline`. This reads the holdout once and logs that access;
   paste the printed values into `guards.baseline`, then record the regime rationale.
5. **Run the eval, read failure modes.** `python -m ratchet.loop_cli --project projects/<name>`.
   Ask how each failure generalizes beyond the one case.
6. **Add mutations to address failures — structurally.** A field, a criterion, a tool.
   Not exhortation ("NEVER", "CRITICAL"). Long ban-lists backfire on strong models.
7. **Escalate.** `--escalate` grades the winner on the holdout and runs regression,
   anomaly, coverage, and overfit guards. Survives → real. Fails → inspect the named guard.

## Two meanings of "candidate"

In the **loop**, the candidate is the searched prompt (model fixed). In the **bench**,
the candidate is a model id (prompt fixed). Your `runner.run` decides which by context —
if you do both, branch on whether `candidate` looks like a model id.

## Feedback re-enters as constraints, not patches

Reviewer verdicts go in `constraints.jsonl` via `add_constraint(...)` — they prepend to
every candidate as policy (the runner passes `policy` into `assemble`). Do NOT paste them
into `base.txt`. Write them **two-sided** (cost of escalating AND cost of a wrong answer),
never one-sided. Periodically run `python -m ratchet.constraints_cli --project
projects/<name> --review` to catch duplicates and one-sided language, then
`--consolidate "<why>"` to record what you cleaned up.

## Versioning

Changing the salt, objective, guards/baselines, model, truth, item contents, runner/core
scoring source, declared environment, or eval-set contents changes the **regime**.
Any scoring command BLOCKS until you record why:
`python -m ratchet.regime_cli --project projects/<name> --why "..." --impact "..."`.
The version number points; the ledger explains. Cross-regime results are never pooled.

## CLI reference

All five entry points, one line each:

- `python -m ratchet.loop_cli --project projects/<name> [--escalate|--establish-baseline]` — establish frozen split baselines, hill-climb on train, or grade the winner on holdout and report all guards.
- `python -m ratchet.bench_cli --project projects/<name>` — frozen-param bench comparison across model candidates on the eval set.
- `python -m ratchet.verify --project projects/<name> --predictions <csv> --split train|holdout|gap` — score predictions against truth and frozen baselines; exits 2 on regression, anomaly, overfit, or low coverage.
- `python -m ratchet.constraints_cli --project projects/<name> --review|--consolidate "<why>"` — constraints hygiene: flag duplicates and one-sided language, or record a consolidation.
- `python -m ratchet.regime_cli --project projects/<name> --why "..." --impact "..."` — record a regime bump so the next scoring command unblocks.
