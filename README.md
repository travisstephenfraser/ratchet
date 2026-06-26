# loop-eval

A small, portable scaffold for **evaluating and improving LLM prompts and models against your own ground truth**, with the comparability guarantees that make the numbers trustworthy.

The same labeled set, objective, and frozen configuration power two directions off one project instance:

- **Defend** (`verify`) — score a candidate against a baseline on a frozen set; exit non-zero if it regresses, leaks, or overfits. This is a regression gate you can drop into CI.
- **Improve** (`loop`) — hill-climb a prompt against the objective, then make the winner survive a held-out overfit gate before you trust it.

The pivot is that *the same baseline is both the floor you cannot drop below and the bar to beat*. Author the labeled set once; get both directions for free. Quality stops being a hope and becomes a ratchet: never down, systematically up.

It is intentionally tiny (pure Python, one dependency) and unopinionated about how you call your model. You bring a project; the core brings the discipline.

---

## Why this exists

Most teams that tune an LLM prompt end up hand-rolling the same scaffolding: a scorer, a train/holdout split, some "did it get better?" comparison, and a pile of ad-hoc rules pasted into the prompt. That scaffolding is easy to get subtly wrong in ways that quietly invalidate the results:

- The split leaks, so a prompt that *memorized* the examples looks like a prompt that *generalized*.
- Someone changes the model, the tolerance, or the eval set, and yesterday's score is silently compared against today's under different rules.
- Reviewer feedback gets pasted into the prompt as `NEVER DO X`, the prompt rots into a contradictory ban-list, and strong models start ignoring all of it.

loop-eval makes each of those failure modes either impossible or loud. The design choices below are all in service of *the score means what you think it means*.

---

## Architecture: a stable core, a thin per-project layer

```
loop_eval/            # the core — you import it, you never edit it
projects/<name>/      # the per-project layer — you write this
```

The core has no knowledge of your domain. A project supplies five small adapters and a config file; the core supplies the split discipline, the guards, the objectives, the search loop, the persistence, and the versioning. Porting to a new domain means copying `projects/_template/` and filling in the adapters, never touching `loop_eval/`.

The five adapters a project implements (by shape, not by import; see `loop_eval/adapter.py`):

| File | Contract |
|---|---|
| `config.yaml` | salt, holdout %, objective, guards, search params, model, bench set |
| `ingest.py` | `ingest() -> (items, truth)` — **you** export your ground truth |
| `runner.py` | `Runner.run(candidate, item, policy) -> prediction` — the one place that calls your model |
| `mutations.py` | `MUTATIONS = [(name, transform), ...]` — the moves the hill-climb can make |
| `base.txt` | the starting prompt |

---

## The design choices, and why

### Split by stable hash, vault the holdout

`split_ids` buckets each item by `sha256(salt:id)`, so the train/holdout assignment is deterministic and independent of order, insertion, or run. The holdout is never touched during search; every read of it is appended to `holdout_access.log`. If you can't generalize past the data you trained on, you didn't learn anything, and the gate is what proves it.

### Missing predictions are misses

If the runner produces no prediction for an item, that item still counts in the denominator. A prompt that silently drops the hard cases cannot inflate its score by answering only the easy ones.

### The runner fails loud

If your model's response can't be parsed into a valid prediction, the runner **raises**. It never resolves a malformed answer to a silent `0`/miss. A silent zero makes a good mutation look like a regression and corrupts the hill-climb, so a parse failure is a crash, not a data point. Arithmetic (sums, clamps) happens in your code; the model emits judgments only.

### Two anti-leak guards, both direction-aware

Every score carries two flags:

- **anomaly** — the result is implausibly good (above `anomaly_at`), the classic signature of a verifier leak where the answer is reachable from the input.
- **overfit** — the train-vs-holdout gap exceeds `overfit_gap`, the signature of memorization.

Both respect the objective's direction (`max` for within-tolerance/F1/judge, `min` for MAE), so "better" is never hard-coded. `verify --split gap` and the escalation gate exit `2` when either trips, so CI catches it.

### Objectives are pluggable, and they own their direction

Three are built in, and a project can register its own:

- `within_tol` — numeric closeness. Within-rate ceilings on easy items (everything inside tolerance gives no gradient), so you can climb on MAE instead via `params.climb: mae`.
- `prf1` — precision/recall/F1 for classification and extraction. A missing prediction is the negative label, so failing to flag a positive is a recall miss.
- `judge` — open-ended generation scored by an injected `judge_fn(pred, rubric) -> float`. The function is injected by the project loader, not expressed in YAML, so the core never makes a network call in tests.

Direction lives on the objective instance, not in config, so it can never drift out of sync with the metric.

### Feedback re-enters as constraints, not as patches

When a reviewer (human or model) finds a problem, the verdict goes into `constraints.jsonl`, and the loop prepends it to **every** candidate as a `<policy>` block. It does **not** get pasted into the searched prompt. This keeps the thing you are searching over clean, and it keeps policy separate from instructions separate from data in the assembled prompt (`prompt.py`).

Constraints are append-only, so they rot if untended. The tooling fights that: entries are dated and attributed, `constraints_cli --review` flags duplicates and one-sided language (`NEVER`/`ALWAYS`/`CRITICAL`), and a consolidation is recorded in a ledger. Write constraints **two-sided** (the cost of escalating *and* the cost of a wrong answer); one-sided absolutes are exactly what strong models learn to ignore.

### Mutations prefer structure over exhortation

A good mutation adds a *criterion*, a *field*, or a *tool*, not a louder nag. The template ships with the anti-pattern commented out so it's clear what not to copy. Long ban-lists backfire on capable models; a structural change to the task gives the model something to actually do.

### Versioning is first-class, and it blocks

A **regime** is the fingerprint of everything that determines whether two results are comparable: the salt, the objective and its params, the holdout %, the guards, the model, the eval-set contents (tracked by content hash, not path, so swapping which items you bench changes the regime), and the constraints version.

Any scoring command computes the current regime and compares it to the last one on disk. If it changed and no ledger entry explains why, the command **exits 2 and refuses to run**:

```
regime changed without a ledger rationale:
  frozen.model.name: 'qwen2.5' -> 'qwen3'
Record it: python -m loop_eval.regime_cli --project projects/<name> --why '...' --impact '...'
```

Cross-regime results are never pooled. The version number points; the ledger explains. This turns "don't silently change the frozen params" from a discipline you have to remember into something the core won't let you skip.

---

## Quickstart

Requires Python 3.12+ (developed on 3.14). One dependency: `pyyaml`.

```bash
pip install pyyaml
python -m pytest          # 52 tests, the full self-test, runs in well under a second
```

The repo ships a self-contained **toy project** (`projects/toy/`) with 40 deterministic synthetic exams and a synthetic grader, so the whole pipeline runs with no model and no network:

```bash
# Improve: hill-climb the grading prompt on the train split, then escalate the winner
python -m loop_eval.loop_cli --project projects/toy --escalate

# Defend: score predictions against ground truth (exits 2 on anomaly/overfit)
python -m loop_eval.verify --project projects/toy --predictions <preds.csv> --split gap

# Bench: compare fixed candidates on one frozen set under one regime
python -m loop_eval.bench_cli --project projects/toy
```

The toy exercises the real code path end to end: structured prompt assembly, the policy/constraints channel, the hill-climb, persistence with regime stamps, and the holdout gate.

---

## Porting to your own project

Copy the template and fill in the adapters. The full walkthrough is in [`docs/PORTING.md`](docs/PORTING.md); the short version:

```bash
cp -r projects/_template projects/my-eval
```

1. **Start minimal.** `base.txt` = role, data, task, definition of done, output shape.
2. **Ingest your ground truth.** Implement `ingest()`. You export the truth yourself; keep data under `data/` (gitignored). Truth values are strings.
3. **Pick an objective.** `within_tol`, `prf1`, `judge`, or `custom`.
4. **Run it, read the failure modes.** Ask how each failure generalizes beyond the one case.
5. **Add mutations that address failures structurally.** A field, a criterion, a tool, not a `NEVER`.
6. **Escalate.** Grade the winner on the holdout and run the overfit gate. Survives, it's real; fails, it memorized the train split.

---

## CLI reference

| Command | What it does |
|---|---|
| `python -m loop_eval.loop_cli --project <p> [--escalate]` | Hill-climb search on the train split; `--escalate` grades the winner on the holdout and runs the overfit gate. |
| `python -m loop_eval.verify --project <p> --predictions <csv> --split train\|holdout\|gap` | Score predictions against ground truth; exits 2 on anomaly or overfit. |
| `python -m loop_eval.bench_cli --project <p>` | Frozen-param comparison of fixed candidates on the eval set, under one enforced regime. |
| `python -m loop_eval.constraints_cli --project <p> --review \| --consolidate "<why>"` | Constraints hygiene: flag contradictions and one-sided language, or record a consolidation. |
| `python -m loop_eval.regime_cli --project <p> --why "..." --impact "..."` | Record a regime bump so the next scoring command unblocks. |

---

## Repository layout

```
loop_eval/              core (do not edit per-project)
  adapter.py            the five seam contracts a project satisfies
  config.py             typed load of config.yaml
  prompt.py             structured policy / instructions / data assembly
  verifier.py           split, scoring, anomaly + overfit guards
  loop.py               hill-climb + escalation gate
  bench.py              frozen-param model comparison
  constraints.py        the feedback channel + hygiene + ledger
  regime.py             regime fingerprint, diff, comparison guard, ledger
  regime_state.py       enforce versioning at the entry points
  results.py            persistence (every result stamped with its regime)
  objectives/           within_tol, prf1, judge (pluggable)
  *_cli.py, verify.py   the five entry points
projects/
  _template/            copy this to start a new project
  toy/                  self-contained synthetic e2e example
tests/                  52 tests, the self-test
docs/PORTING.md         the porting guide
```

---

## License

MIT. See [`LICENSE`](LICENSE).
