# Porting loop-eval to a new project

Copy `projects/_template/` to `projects/<name>/` and fill in the adapters. You never
edit `loop_eval/` (the core).

## The authoring workflow (start minimal, grow by failure mode)

1. **Minimal base.** `base.txt` = role, data, task + definition-of-done, output shape.
2. **Ingest your ground truth.** Implement `ingest()` — you export the truth yourself.
   Data under `data/` (gitignored). Truth values are strings.
3. **Pick an objective.** `within_tol` (numeric closeness; `params.climb: within|mae` —
   use `mae` when within-rate ceilings and gives no gradient), `prf1` (classification/
   extraction), or `judge` (open-ended; add `judge.py:judge_fn(pred, rubric) -> float`).
   Need something else? `objective.name: custom` + `objective.py:make_objective(params)`.
4. **Run the eval, read failure modes.** `python -m loop_eval.loop_cli --project projects/<name>`.
   Ask how each failure generalizes beyond the one case.
5. **Add mutations to address failures — structurally.** A field, a criterion, a tool.
   Not exhortation ("NEVER", "CRITICAL"). Long ban-lists backfire on strong models.
6. **Escalate.** `--escalate` grades the winner on the holdout and runs the overfit gate.
   Survives → real. Fails → it memorized the train split.

## Two meanings of "candidate"

In the **loop**, the candidate is the searched prompt (model fixed). In the **bench**,
the candidate is a model id (prompt fixed). Your `runner.run` decides which by context —
if you do both, branch on whether `candidate` looks like a model id.

## Feedback re-enters as constraints, not patches

Reviewer verdicts go in `constraints.jsonl` via `add_constraint(...)` — they prepend to
every candidate as policy (the runner passes `policy` into `assemble`). Do NOT paste them
into `base.txt`. Write them **two-sided** (cost of escalating AND cost of a wrong answer),
never one-sided. Periodically run `python -m loop_eval.constraints_cli --project
projects/<name> --review` to catch contradictions and one-sided language, then
`--consolidate "<why>"` to record what you cleaned up.

## Versioning

Changing the salt, objective, guards, model, or eval-set contents changes the **regime**.
Any scoring command BLOCKS until you record why:
`python -m loop_eval.regime_cli --project projects/<name> --why "..." --impact "..."`.
The version number points; the ledger explains. Cross-regime results are never pooled.
