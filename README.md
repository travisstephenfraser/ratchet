# ratchet

[![test](https://github.com/travisstephenfraser/ratchet/actions/workflows/test.yml/badge.svg)](https://github.com/travisstephenfraser/ratchet/actions/workflows/test.yml)

A small, portable scaffold for **evaluating and improving LLM prompts and models against your own ground truth**, with the comparability guarantees that make the numbers trustworthy.

The same labeled set, objective, and frozen configuration power two directions off one project instance:

- **Defend** (`verify`) — score a candidate against a baseline on a frozen set; exit non-zero if it regresses, leaks, or overfits. This is a regression gate you can drop into CI.
- **Improve** (`loop`) — hill-climb a prompt against the objective, then make the winner survive held-out regression, anomaly, coverage, and overfit guards before you trust it.

The pivot is that *the same baseline is both the floor you cannot drop below and the bar to beat*. Author the labeled set once; get both directions for free. Quality stops being a hope and becomes a ratchet: never down, systematically up.

It is intentionally tiny (pure Python, one dependency) and unopinionated about how you call your model. You bring a project; the core brings the discipline.

---

## Why this exists

Most teams that tune an LLM prompt end up hand-rolling the same scaffolding: a scorer, a train/holdout split, some "did it get better?" comparison, and a pile of ad-hoc rules pasted into the prompt. That scaffolding is easy to get subtly wrong in ways that quietly invalidate the results:

- The split leaks, so a prompt that *memorized* the examples looks like a prompt that *generalized*.
- Someone changes the model, the tolerance, or the eval set, and yesterday's score is silently compared against today's under different rules.
- Reviewer feedback gets pasted into the prompt as `NEVER DO X`, the prompt rots into a contradictory ban-list, and strong models start ignoring all of it.

ratchet makes each of those failure modes either impossible or loud. The design choices below are all in service of *the score means what you think it means*.

---

## Lineage: the same loop, a harder kind of truth

ratchet was built from the ground up for a different problem, the grading-accuracy challenge Rubrica ran into, and it was derived independently rather than forked. It lands on the same shape as Andrej Karpathy's [`autoresearch`](https://github.com/karpathy/autoresearch), which is a clean public expression of the idea and a fair inspiration to name: an LLM proposes a change, you run it against a frozen eval, you keep it only if a single scalar improves, and you let it run overnight. autoresearch points this at ML research, where an agent edits a training script and *validation loss* is the fitness signal. That signal is objective, cheap, and regenerable; if you want more truth, you just train again. So it needs almost no guardrails. A lower number simply *is* better, and a bad edit is one `git reset` away.

ratchet aims the same shape at a harder target: **subjective judgment measured against external ground truth you cannot regenerate**, such as human grades, telemetry labels, or a finite hand-built set. That truth is noisy, expensive, and *gameable*; a prompt can memorize the split or leak the answer and score brilliantly while being wrong. So the loop is the easy part, and the guards are the product. The held-out gate catches memorization, the anomaly ceiling catches leakage, the overfit gap catches train/holdout divergence, the regime fingerprint refuses to compare scores minted under different rules (you can't just re-mint the labels), and the runner fails loud because a *missing* judgment is ambiguous in a way a crashed training run never is. Same skeleton, opposite problem, which is exactly why the rules below exist.

---

## Architecture: a stable core, a thin per-project layer

```
ratchet/            # the core: you import it, never edit it
projects/<name>/      # the per-project layer: you write this
```

The core has no knowledge of your domain. A project supplies five project files; the core supplies the split discipline, the guards, the objectives, the search loop, the persistence, and the versioning. Porting to a new domain means copying `projects/_template/` and filling in those files, never touching `ratchet/`.

The table below spans three core modules. `ratchet/adapter.py` defines the runner, ingest,
and mutation shapes; `ratchet/config.py` validates `config.yaml`; `ratchet/project.py`
loads the project modules and `base.txt`.

| File | Contract |
|---|---|
| `config.yaml` | salt, holdout %, objective, frozen baseline, guards, search params, model, bench set |
| `ingest.py` | `ingest() -> (items, truth)` — **you** export JSON-serializable inputs and ground truth |
| `runner.py` | `Runner.run(candidate, item, policy) -> prediction` — the one place that calls your model |
| `mutations.py` | `MUTATIONS = [(name, transform), ...]` — the moves the hill-climb can make |
| `base.txt` | the starting prompt |

---

## The design choices, and why

### Split by stable hash, vault the holdout

`split_ids` buckets each item by `sha256(salt:id)`, so the train/holdout assignment is deterministic and independent of order, insertion, or run. The holdout is never touched during search; every read of it is appended to `holdout_access.log`. If you can't generalize past the data you trained on, you didn't learn anything, and the gate is what proves it.

Search reads only the train split. `loop --escalate` sends one candidate to the holdout: the final winner. The gate reruns that winner on both train and holdout under the same regime, so it does not combine cached train predictions with a fresh holdout run.

### Missing predictions are misses

If the runner produces no prediction for an item, that item still counts in the denominator. A prompt that silently drops the hard cases cannot inflate its score by answering only the easy ones.

### The runner fails loud

If your model's response can't be parsed into a valid prediction, the runner raises `Unparseable`; the core records that item as an explicit miss. Any other exception—transport, timeout, or runner integration bug—propagates and halts the run. An optional `max_miss_rate` turns systematic parse failure on a known-good candidate into a loud halt. Arithmetic (sums, clamps) happens in your code; the model emits judgments only.

### Anti-leak and coverage guards

Every gated verify/escalation report carries these flags:

- **regressed** — the objective fell below the frozen split baseline (or rose above it for a minimizing objective).
- **anomaly** — the result is implausibly good (above `anomaly_at`), the classic signature of a verifier leak where the answer is reachable from the input.
- **overfit** — the train-vs-holdout gap exceeds `overfit_gap`, the signature of memorization.
- **min_coverage** (optional) — minimum fraction of split ids a candidate must answer for its score to count. Closes a Goodhart hole in ratio objectives: `mae` averages only the items a candidate answered, so answering one easy item posts `mae=0.0` and would win the climb. With the floor set, a low-coverage split is flagged (`low_coverage`), the candidate is disqualified from the hill-climb, and `verify` exits 2. Opt-in: projects that don't set it are unaffected.

The anomaly and overfit checks respect the objective's direction (`max` for within-tolerance/F1/judge, `min` for MAE), so "better" is never hard-coded.

Bench rows are informational — the floor gates at verify and in the loop, not in bench ranking.

Bench validates its candidates and eval ids before it can write regime state, a ledger entry, or a result. It requires at least one nonblank candidate plus a nonempty eval set of unique, known ids. Invalid bench input exits `2` without creating those files.

**The exit code lives in `verify`, not the loop.** `verify` is the gate: it exits `2` on regression, anomaly, overfit, or low coverage, so it's the thing you wire into CI. The `loop` surfaces the same flags in its output and on the escalation report, but it exits `0` regardless; it's a search tool that explores, and explorers are allowed to find candidates that trip a guard. The discipline is: **search with `loop`, gate with `verify`.** Don't gate on the loop's exit code; have CI run `verify` against the candidate the loop produced.

### Objectives are pluggable, and they own their direction

Three are built in, and a project can register its own:

- `within_tol` — numeric closeness. Within-rate ceilings on easy items (everything inside tolerance gives no gradient), so you can climb on MAE instead via `params.climb: mae`. MAE averages answered items; if a split has zero predictions, it is unscorable rather than `0` or infinity. The loop skips an unscorable mutation, but an unscorable base stops the run. `verify` reports the error and exits `2`.
- `prf1` — precision/recall/F1 for classification and extraction. A missing prediction is the negative label, so failing to flag a positive is a recall miss.
- `judge` — open-ended generation scored by an injected `judge_fn(pred, rubric) -> float`. The function is injected by the project loader, not expressed in YAML, so the core never makes a network call in tests.

Direction lives on the objective instance, not in config, so it can never drift out of sync with the metric.

### Feedback re-enters as constraints, not as patches

When a reviewer (human or model) finds a problem, the verdict goes into `constraints.jsonl`, and the loop prepends it to **every** candidate as a `<policy>` block. It does **not** get pasted into the searched prompt. This keeps the thing you are searching over clean, and it keeps policy separate from instructions separate from data in the assembled prompt (`prompt.py`).

Constraints are append-only, so they rot if untended. The tooling fights that: entries are dated and attributed, `constraints_cli --review` flags duplicates and one-sided language (`NEVER`/`ALWAYS`/`CRITICAL`), and a consolidation is recorded in a ledger. Write constraints **two-sided** (the cost of escalating *and* the cost of a wrong answer); one-sided absolutes are exactly what strong models learn to ignore.

### Mutations prefer structure over exhortation

A good mutation adds a *criterion*, a *field*, or a *tool*, not a louder nag. The template ships with the anti-pattern commented out so it's clear what not to copy. Long ban-lists backfire on capable models; a structural change to the task gives the model something to actually do.

### Versioning is first-class, and it blocks

A **regime** is the fingerprint of everything that determines whether two results are comparable: the salt, objective and params, holdout %, guards and frozen baselines, model, derived truth, evaluated item contents, runner source, active core scoring source/version, bench-set contents, declared environment knobs, and constraints version. Item inputs must therefore be JSON-serializable; unsupported objects fail loudly instead of producing an unstable fingerprint.

The core fingerprint covers every `ratchet/**/*.py` path and its bytes, with length framing between each path and content block. It also includes the package version and requires named source files to be present. The regime records two runtime fields: Python major/minor and PyYAML version. A change to either recorded value changes the regime fingerprint.

Any scoring command computes the current regime and compares it to the last one on disk. If it changed and no ledger entry explains why, the command **exits 2 and refuses to run**:

```
regime changed without a ledger rationale:
  frozen.model.name: 'qwen2.5' -> 'qwen3'
Record it: python -m ratchet.regime_cli --project projects/<name> --why '...' --impact '...'
```

Cross-regime results are never pooled. The version number points; the ledger explains. This turns "don't silently change the frozen params" from a discipline you have to remember into something the core won't let you skip.

Guarded comparisons require prediction provenance. Ratchet-generated predictions carry a regime
stamp; `verify` rejects unstamped files by default. Its `--allow-unstamped` flag is for legacy or
external files: it warns and permits a missing stamp, but never a present mismatch.

Direct library callers make the guard decision explicitly. Guarded `score_split(...)` calls pass a
nonblank `expected_regime`. Passing `expected_regime=None` selects an unguarded low-level
calculation; it is not a migration shortcut and must not be used for regression or holdout
comparisons. `gap_report(...)` is always guarded and requires a nonblank `expected_regime`.
The separate library keyword `allow_unstamped=True` is the unsafe legacy/external prediction
override. It warns and permits only a missing stamp; a present mismatch still fails. The CLI
spelling is `--allow-unstamped`, not `allow_unstamped`.

---

## Quickstart

Requires Python 3.12, 3.13, or 3.14. One runtime dependency: `pyyaml`.

For a contributor checkout:

```bash
python -m pip install -e ".[dev]"
python -m pytest          # 264 tests, the full self-test
```

The source checkout ships a self-contained **toy project** (`projects/toy/`) with 40 deterministic synthetic exams and a synthetic grader, so the whole pipeline runs with no model and no network:

```bash
# Improve: hill-climb the grading prompt on the train split, then escalate the winner
python -m ratchet.loop_cli --project projects/toy --escalate

# Defend: score predictions against truth and the frozen baseline (exits 2 on any guard)
python -m ratchet.verify --project projects/toy --predictions <preds.csv> --split gap

# Bench: compare fixed candidates on one frozen set under one regime
python -m ratchet.bench_cli --project projects/toy
```

The toy exercises the real code path end to end: structured prompt assembly, the policy/constraints channel, the hill-climb, persistence with regime stamps, and the holdout gate.

`projects/toy/` and `projects/_template/` are checkout examples, not data files in the wheel. Copy either project from a checkout when exercising an installed wheel.

### CI artifact proof

The `test` workflow runs the source suite on Python 3.12, 3.13, and 3.14, compiles the core and both checkout examples, then builds `dist/ratchet-0.1.0-py3-none-any.whl`. For each Python version it checks that exact archive, installs that exact wheel into a clean `/tmp` environment, and runs import, core-fingerprint, CLI, bench, and escalation checks from outside the checkout. No `PYTHONPATH` or checkout virtual environment is involved in the installed-wheel checks.

---

## Porting to your own project

Copy the template and fill in the adapters. The full walkthrough is in [`docs/PORTING.md`](docs/PORTING.md); the short version:

```bash
cp -r projects/_template projects/my-eval
```

1. **Start minimal.** `base.txt` = role, data, task, definition of done, output shape.
2. **Ingest your ground truth.** Implement `ingest()`. You export JSON-serializable items and truth yourself; keep data under `data/` (gitignored). Truth values are strings.
3. **Pick an objective.** `within_tol`, `prf1`, `judge`, or `custom`.
4. **Establish the frozen baseline.** Run `python -m ratchet.loop_cli --project projects/<name> --establish-baseline`, then paste the printed train/holdout objective values into `guards.baseline` and record the regime.
5. **Run it, read the failure modes.** Ask how each failure generalizes beyond the one case.
6. **Add mutations that address failures structurally.** A field, a criterion, a tool, not a `NEVER`.
7. **Escalate.** Grade the winner on the holdout and run the regression/overfit guards. Survives, it's real; fails, it regressed, leaked, or memorized the train split.

---

## CLI reference

| Command | What it does |
|---|---|
| `python -m ratchet.loop_cli --project <p> [--escalate\|--establish-baseline]` | Hill-climb on train; establish frozen split baselines; or grade the winner on holdout and report all guards. |
| `python -m ratchet.verify --project <p> --predictions <csv> --split train\|holdout\|gap [--allow-unstamped]` | Score predictions against truth and frozen baselines; exits 2 on provenance failure, regression, anomaly, overfit, or low coverage. |
| `python -m ratchet.bench_cli --project <p>` | Frozen-param comparison on an explicit eval set, or all ingested ids only when `eval_set: null`. Configured missing, empty, or unknown-id sets fail closed. |
| `python -m ratchet.constraints_cli --project <p> --review \| --consolidate "<why>"` | Constraints hygiene: flag duplicates and one-sided language, or record a consolidation. |
| `python -m ratchet.regime_cli --project <p> --why "..." --impact "..."` | Record a regime bump so the next scoring command unblocks. |

---

## Repository layout

```
ratchet/              core (do not edit per-project)
  adapter.py            runner, ingest, and mutation shapes
  config.py             validation and typed load of config.yaml
  project.py            project module and base.txt loading
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
tests/                  264 tests, the self-test
docs/PORTING.md         the porting guide
```

---

## License

MIT. See [`LICENSE`](LICENSE).
