# ratchet Red-Team Fixes (Wave 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the six verified CRITICAL/HIGH red-team findings that are small and self-contained: the Instructor transport-laundering bug, the holdout-truth leak to the objective, and four missing fail-loud/guard checks across escalate, bench, and gen_preds.

**Architecture:** Surgical changes to existing files. Two touch the dependency-light core (`ratchet/verifier.py`, `ratchet/loop.py`, `ratchet/bench.py`), two touch the cosmos-direction project layer (`runner.py`, `gen_preds.py`). No new modules, no signature changes that ripple to callers except adding one keyword pass-through.

**Tech Stack:** Python 3.12+ (repo runs 3.14), pytest, the repo venv at `/Users/travis/Developer/ratchet/.venv`. Core stays stdlib + pyyaml only.

## Global Constraints

- **Core stays dependency-light.** `ratchet/` must import only stdlib + pyyaml. No `pydantic`/`instructor`/`openai` in core, ever. All tests for core code must pass under `/Users/travis/Developer/ratchet/.venv` (which has NO pydantic).
- **Run tests with:** `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest <path> -v` from the repo root.
- **Fail-loud contract (adapter.py):** only `adapter.Unparseable` is demoted to a per-item miss; every other exception must propagate and halt the run. Do not broaden any `except`.
- **Match existing comment style:** terse, lowercase-leaning inline comments; no em dashes in code or comments (use commas/semicolons/parentheses).
- **Commit trailers:** end every commit message with the two lines
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01KnQsvsU7cyqTgE7gKr9E5W`.
- **Branch:** do this work on a branch off `master` (e.g. `redteam-fixes-wave1`), not directly on `master`.

---

### Task 1: Instructor cause-chain classification (CRITICAL)

The cosmos structured runner translates `InstructorRetryException -> Unparseable` by matching the exception TYPE. But instructor 1.15.4 (`instructor/v2/core/retry.py:261`) wraps EVERY failure, including `openai.APIConnectionError` and HTTP 500, in `InstructorRetryException`. So a dead/crashing LM Studio mid-run is laundered into candidate misses, which is the exact bug the fail-loud contract exists to prevent. Fix: classify by the wrapped CAUSE chain, demote only genuine schema/JSON parse failures, re-raise everything else. The classifier is pure name-matching (no imports), so it is testable under the dep-light venv.

**Files:**
- Modify: `projects/cosmos-direction/runner.py` (add `_is_parse_failure` helper near the other module helpers ~line 43; change the structured `except` block ~line 90-95)
- Test: `projects/cosmos-direction/test_runner_regex.py` (dep-light; pydantic-free)

**Interfaces:**
- Produces: `_is_parse_failure(exc) -> bool` in `runner.py`.

- [ ] **Step 1: Write the failing test**

Add to `projects/cosmos-direction/test_runner_regex.py`:

```python
def test_is_parse_failure_classifies_by_cause_chain():
    import runner
    class InstructorRetryException(Exception):
        pass
    class ValidationError(Exception):
        pass
    class APIConnectionError(Exception):
        pass

    # schema-validation cause -> a real parse miss -> demote
    try:
        try:
            raise ValidationError("does not match DirectionRead")
        except ValidationError as v:
            raise InstructorRetryException("retries exhausted") from v
    except InstructorRetryException as e:
        assert runner._is_parse_failure(e) is True

    # transport cause -> NOT a parse miss -> must halt
    try:
        try:
            raise APIConnectionError("connection refused")
        except APIConnectionError as a:
            raise InstructorRetryException("retries exhausted") from a
    except InstructorRetryException as e:
        assert runner._is_parse_failure(e) is False

    # implicit chaining (raise without `from`) is followed too
    try:
        try:
            raise ValidationError("bad json")
        except ValidationError:
            raise InstructorRetryException("wrapped implicitly")
    except InstructorRetryException as e:
        assert runner._is_parse_failure(e) is True

    # an unrelated exception is never a parse failure
    assert runner._is_parse_failure(ValueError("x")) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest projects/cosmos-direction/test_runner_regex.py::test_is_parse_failure_classifies_by_cause_chain -v`
Expected: FAIL with `AttributeError: module 'runner' has no attribute '_is_parse_failure'`.

- [ ] **Step 3: Write minimal implementation**

Add this helper in `runner.py` (near `_temperature`, before `class Runner`):

```python
_PARSE_CAUSE_NAMES = {"ValidationError", "JSONDecodeError"}


def _is_parse_failure(exc) -> bool:
    """Whether `exc` is an Instructor retry failure caused by a SCHEMA/JSON parse error
    (demote to a miss) rather than a transport/infra fault (must halt). instructor wraps
    EVERY failure, including openai.APIConnectionError and HTTP 500, in
    InstructorRetryException (instructor/v2/core/retry.py), so matching the outer type is
    not enough. Classify by walking the wrapped cause/context chain; err toward NOT-a-parse
    (halt loudly) when the chain shows no validation error."""
    if not any(c.__name__ == "InstructorRetryException" for c in type(exc).__mro__):
        return False
    seen, stack = set(), [exc.__cause__, exc.__context__]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        if any(c.__name__ in _PARSE_CAUSE_NAMES for c in type(cur).__mro__):
            return True
        stack.extend([cur.__cause__, cur.__context__])
    return False
```

Then change the structured `except` block in `Runner.run` from:

```python
            except Exception as e:
                if any(c.__name__ == "InstructorRetryException" for c in type(e).__mro__):
                    raise Unparseable(f"structured parse failed for {frame.name}: {e}") from e
                raise
```

to:

```python
            except Exception as e:
                # instructor wraps transport/infra faults in InstructorRetryException too,
                # so only a validation/JSON cause is a real parse miss; everything else halts.
                if _is_parse_failure(e):
                    raise Unparseable(f"structured parse failed for {frame.name}: {e}") from e
                raise
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest projects/cosmos-direction/test_runner_regex.py -v`
Expected: PASS (all tests in the file, including the two existing regex tests).

- [ ] **Step 5: Commit**

```bash
git add projects/cosmos-direction/runner.py projects/cosmos-direction/test_runner_regex.py
git commit -m "fix(cosmos): classify Instructor failures by cause, not type

instructor wraps transport/HTTP-500 errors in InstructorRetryException, so the
type-match laundered infra faults into candidate misses (the exact bug the
fail-loud contract prevents). Classify by the wrapped validation/JSON cause;
re-raise transport faults so they halt.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KnQsvsU7cyqTgE7gKr9E5W"
```

---

### Task 2: Slice truth to the scored ids in score_split (HIGH)

`hill_climb` threads the FULL `truth` dict (holdout labels included) to `objective.score` on every candidate. Stock objectives only index their own ids, but a project-authored objective is handed the entire vault during search. Slice the truth to the ids being scored before dispatch so the objective structurally cannot read holdout labels during a train-split scoring call.

**Files:**
- Modify: `ratchet/verifier.py:33` (inside `score_split`)
- Test: `tests/test_verifier.py`

**Interfaces:**
- Consumes: `objective.score(preds, truth, ids)` (unchanged signature; `truth` is now pre-sliced).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_verifier.py`:

```python
def test_score_split_slices_truth_to_ids():
    from ratchet.verifier import score_split

    class _Snoop:
        direction = "max"
        seen_keys = None
        def score(self, preds, truth, ids):
            _Snoop.seen_keys = set(truth)
            return {"objective": 1.0}

    truth = {"a": "1", "b": "1", "HOLDOUT": "9"}
    score_split({"a": "1"}, truth, ["a", "b"], _Snoop(), anomaly_at=0.98)
    # the objective must NOT see the holdout label during a train-split scoring call
    assert _Snoop.seen_keys == {"a", "b"}
    assert "HOLDOUT" not in _Snoop.seen_keys
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_verifier.py::test_score_split_slices_truth_to_ids -v`
Expected: FAIL, `assert {'a','b','HOLDOUT'} == {'a','b'}` (the objective currently sees the full dict).

- [ ] **Step 3: Write minimal implementation**

In `ratchet/verifier.py`, change the body of `score_split` from:

```python
    base = objective.score(preds, truth, ids)
```

to:

```python
    # hand the objective only the labels for the ids being scored, never the whole vault
    scoped_truth = {i: truth[i] for i in ids if i in truth}
    base = objective.score(preds, scoped_truth, ids)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_verifier.py -v`
Expected: PASS (new test plus all existing verifier tests, which index `truth[i] for i in ids` and are unaffected).

- [ ] **Step 5: Commit**

```bash
git add ratchet/verifier.py tests/test_verifier.py
git commit -m "fix(core): score_split hands the objective only the scored ids' truth

Least-privilege: hill_climb passed the full truth dict (holdout labels included)
to a project-authored objective on every candidate. Slice truth to ids before
dispatch so search code cannot read the vault.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KnQsvsU7cyqTgE7gKr9E5W"
```

---

### Task 3: Assert train/holdout disjointness in gap_report (MEDIUM)

`gap_report` scores whatever two id lists it is given. Overlapping lists score the same item in both splits, and padding holdout with correct train ids drags the gap toward zero and defeats the overfit gate. `split_ids` produces disjoint splits, so this is defense-in-depth on the public gate function, mirroring the existing rogue-preds guard in `escalate`.

**Files:**
- Modify: `ratchet/verifier.py:39` (top of `gap_report`)
- Test: `tests/test_verifier.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_verifier.py`:

```python
def test_gap_report_rejects_overlapping_splits():
    import pytest
    from ratchet.verifier import gap_report
    from ratchet.objectives.within_tol import WithinTol
    truth = {"a": "10", "b": "10", "c": "10"}
    preds = {"a": "10", "b": "10", "c": "10"}
    with pytest.raises(ValueError, match="disjoint"):
        gap_report(preds, truth, ["a", "b"], ["b", "c"], WithinTol(tol=0.5),
                   {"anomaly_at": 0.98, "overfit_gap": 0.25})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_verifier.py::test_gap_report_rejects_overlapping_splits -v`
Expected: FAIL, `DID NOT RAISE ValueError`.

- [ ] **Step 3: Write minimal implementation**

In `ratchet/verifier.py`, add as the first line of `gap_report` (before the two `score_split` calls):

```python
    overlap = set(train) & set(holdout)
    if overlap:
        raise ValueError(f"train and holdout must be disjoint; shared ids: {sorted(overlap)}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_verifier.py tests/test_loop.py -v`
Expected: PASS (new test plus existing loop/verifier tests, whose splits are disjoint).

- [ ] **Step 5: Commit**

```bash
git add ratchet/verifier.py tests/test_verifier.py
git commit -m "fix(core): gap_report rejects overlapping train/holdout splits

Defense-in-depth on the public gate: overlapping ids score an item in both
splits and shrink the gap, defeating the overfit guard. Assert disjointness.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KnQsvsU7cyqTgE7gKr9E5W"
```

---

### Task 4: Thread max_miss_rate through escalate (HIGH)

`max_miss_rate` guards the base candidate in `hill_climb` but is not passed to `escalate`'s two `run_candidate_over` calls. A systematic parse failure on the holdout shard is therefore laundered into a low holdout score and verdicted as OVERFIT at the single most decision-relevant step. The escalated `best` is exactly as known-good as hill_climb's base, so the same guard applies.

**Files:**
- Modify: `ratchet/loop.py:93` and `:101` (the two `run_candidate_over` calls in `escalate`)
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: `run_candidate_over(..., *, max_miss_rate=None)` (already exists, Task adds the kwarg at these call sites).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_loop.py` (uses the existing `_Project`, `Unparseable` imports at the top of that file):

```python
def test_escalate_halts_on_systematic_holdout_parse_failure(tmp_path):
    import pytest
    class _P(_Project):
        class _R:
            def run(self, candidate, item, policy=""):
                if item.get("holdout"):
                    raise Unparseable("model returned garbage")
                return 10
        runner = _R()
        config = type("C", (), {"guards": {"anomaly_at": 0.95, "overfit_gap": 0.10,
                                            "max_miss_rate": 0.5},
                                "salt": "t", "holdout_pct": 30})()
    items = {"a": {}, "b": {}, "h1": {"holdout": True}, "h2": {"holdout": True}}
    with pytest.raises(ValueError, match="systematic parse failure"):
        escalate(_P(), {"cid": "x", "instructions": "grade", "metrics": {}},
                 ["a", "b"], ["h1", "h2"], items,
                 {"a": "10", "b": "10", "h1": "10", "h2": "10"},
                 log_path=tmp_path / "holdout_access.log")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_loop.py::test_escalate_halts_on_systematic_holdout_parse_failure -v`
Expected: FAIL, no `ValueError` (currently the holdout misses just produce a low score and an overfit verdict).

- [ ] **Step 3: Write minimal implementation**

In `ratchet/loop.py`, change the two `run_candidate_over` calls inside `escalate`:

Line 93 (the `train_preds is None` branch):
```python
        train_preds = run_candidate_over(project, best["instructions"], train_ids, items, policy,
                                         max_miss_rate=project.config.guards.get("max_miss_rate"))
```
Line 101 (the holdout call):
```python
    holdout_preds = run_candidate_over(project, best["instructions"], holdout_ids, items, policy,
                                       max_miss_rate=project.config.guards.get("max_miss_rate"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_loop.py -v`
Expected: PASS (new test plus all existing loop tests; the existing escalate tests use no `max_miss_rate` guard, so `.get` returns None and behavior is unchanged).

- [ ] **Step 5: Commit**

```bash
git add ratchet/loop.py tests/test_loop.py
git commit -m "fix(core): apply max_miss_rate at the escalation gate

A systematic parse failure on the holdout shard was verdicted as OVERFIT instead
of halting. The escalated best is as known-good as the base, so guard it too.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KnQsvsU7cyqTgE7gKr9E5W"
```

---

### Task 5: Surface total/systematic failure on the bench and empty-attempt paths (MEDIUM)

Two gaps in the "a failing candidate still surfaces loudly" claim: (a) `bench()` calls `run_candidate_over` bare and never calls `escalate`, so a candidate that misses every item writes a silent `objective=0.0` row; (b) `run_candidate_over`'s `max_miss_rate` guard is skipped when `attempted == 0` (a fully id-mismatched items dict), so the most-broken-harness case is the one it ignores.

**Files:**
- Modify: `ratchet/loop.py` (the `max_miss_rate` guard in `run_candidate_over`, ~line 46)
- Modify: `ratchet/bench.py:25-28` (raise on a zero-prediction candidate)
- Test: `tests/test_loop.py`, `tests/test_bench.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_loop.py`:

```python
def test_max_miss_rate_halts_on_zero_attempts():
    import pytest
    class _P(_Project):
        class _R:
            def run(self, candidate, item, policy=""):
                return 10
        runner = _R()
    # ids present, but NONE overlap items -> attempted == 0 (a broken harness)
    with pytest.raises(ValueError, match="no items scored"):
        run_candidate_over(_P(), "x", ["a", "b"], {"zzz": {}}, max_miss_rate=0.5)
```

Create or add to `tests/test_bench.py`:

```python
from ratchet.adapter import Unparseable
from ratchet.bench import bench
from ratchet.objectives.within_tol import WithinTol


class _Project:
    def __init__(self):
        self.objective = WithinTol(tol=0.5)
        self.config = type("C", (), {
            "guards": {"anomaly_at": 0.95, "overfit_gap": 0.10},
            "salt": "t", "holdout_pct": 30, "version": "v1",
            "objective": type("O", (), {"name": "within_tol", "params": {}})(),
            "model": {}, "bench": {}, "project_dir": ".",
        })()
        class _R:
            def run(self, candidate, item, policy=""):
                raise Unparseable("garbage")
        self.runner = _R()


def test_bench_raises_when_a_candidate_produces_no_predictions():
    import pytest
    proj = _Project()
    with pytest.raises(ValueError, match="0/2 .* produced predictions"):
        bench(proj, ["cand"], ["a", "b"], {"a": {}, "b": {}}, {"a": "10", "b": "10"}, "v1")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_loop.py::test_max_miss_rate_halts_on_zero_attempts tests/test_bench.py::test_bench_raises_when_a_candidate_produces_no_predictions -v`
Expected: both FAIL (no ValueError raised in either).

- [ ] **Step 3: Write minimal implementation**

In `ratchet/loop.py`, replace the `max_miss_rate` guard block in `run_candidate_over` with a version that also catches the zero-attempt case:

```python
    if max_miss_rate is not None:
        if ids and attempted == 0:
            raise ValueError(f"no items scored: {len(ids)} ids share no keys with the items dict")
        if attempted and misses / attempted > max_miss_rate:
            raise ValueError(
                f"{misses}/{attempted} items unparseable "
                f"({misses / attempted:.0%} > {max_miss_rate:.0%} max_miss_rate) — "
                "systematic parse failure on a known-good candidate, halting for review")
```

In `ratchet/bench.py`, add a zero-prediction guard right after the `run_candidate_over` call (line 25):

```python
        preds = run_candidate_over(project, cand, eval_ids, items, policy)
        if eval_ids and not preds:
            raise ValueError(
                f"bench: 0/{len(eval_ids)} items produced predictions for a candidate — "
                "likely a broken runner or a misaligned items dict")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_loop.py tests/test_bench.py -v`
Expected: PASS (new tests plus existing bench/loop tests; existing tests either set no `max_miss_rate` or produce predictions, so neither guard fires).

- [ ] **Step 5: Commit**

```bash
git add ratchet/loop.py ratchet/bench.py tests/test_loop.py tests/test_bench.py
git commit -m "fix(core): surface zero-prediction and zero-attempt failures on bench/loop

bench never calls escalate, so an all-miss candidate wrote a silent 0.0 row; and
the max_miss_rate guard skipped the attempted==0 broken-harness case. Both now halt.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KnQsvsU7cyqTgE7gKr9E5W"
```

---

### Task 6: Narrow gen_preds.py to the fail-loud contract (MEDIUM — judgment call)

`gen_preds.py:52` still catches `except Exception` and demotes every failure to a miss. This was a deliberate choice (it is a human-inspected batch tool that prints a live ok/miss counter), but the red-team's point stands: a dead LM Studio mid-run yields a short-but-valid preds CSV whose tail is silently misses, and `verify` then scores that as candidate quality with no signal it was truncated. Narrowing to `except Unparseable` makes an infra fault stop the batch loudly, consistent with `adapter.py`. **Decide with Travis before implementing** — if he wants gen_preds to stay lenient, skip this task and instead print a prominent truncation warning at the end.

**Files:**
- Modify: `projects/cosmos-direction/gen_preds.py:52`
- Test: `projects/cosmos-direction/test_gen_preds.py` (new; dep-light, monkeypatches the runner)

- [ ] **Step 1: Write the failing test**

Create `projects/cosmos-direction/test_gen_preds.py`:

```python
"""gen_preds must stop loudly on a non-parse (infra) fault, not silently miss it.
Dep-light: monkeypatches ingest + runner, never calls a model."""
import sys, types, importlib
import pytest


def _install(monkeypatch, exc):
    fake_ingest = types.ModuleType("ingest")
    fake_ingest.ingest = lambda: ({"a": {"frame_path": "x", "telemetry": {}}}, {"a": "DOWNHILL"})
    class _R:
        def run(self, base, item, policy=""):
            raise exc
    fake_runner = types.ModuleType("runner")
    fake_runner.Runner = lambda: _R()
    monkeypatch.setitem(sys.modules, "ingest", fake_ingest)
    monkeypatch.setitem(sys.modules, "runner", fake_runner)


def test_gen_preds_halts_on_infra_fault(monkeypatch, tmp_path):
    from ratchet.adapter import Unparseable
    _install(monkeypatch, RuntimeError("connection refused"))
    import gen_preds; importlib.reload(gen_preds)
    with pytest.raises(RuntimeError, match="connection refused"):
        gen_preds.main(["--out", str(tmp_path / "p.csv")])


def test_gen_preds_demotes_unparseable_to_miss(monkeypatch, tmp_path):
    from ratchet.adapter import Unparseable
    _install(monkeypatch, Unparseable("no direction"))
    import gen_preds; importlib.reload(gen_preds)
    rc = gen_preds.main(["--out", str(tmp_path / "p.csv")])
    assert rc == 0  # a parse miss is tolerated and the batch completes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest projects/cosmos-direction/test_gen_preds.py -v`
Expected: `test_gen_preds_halts_on_infra_fault` FAILS (the RuntimeError is currently swallowed and `main` returns 0).

- [ ] **Step 3: Write minimal implementation**

In `projects/cosmos-direction/gen_preds.py`, add the import near the top (after the existing imports):

```python
from ratchet.adapter import Unparseable
```

Change the `except` at line 52 from `except Exception as e:` to:

```python
            except Unparseable as e:                     # a real parse miss -> omit -> counts as a miss in verify
```

(Any other exception now propagates and halts the batch, matching the loop's contract.)

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest projects/cosmos-direction/test_gen_preds.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add projects/cosmos-direction/gen_preds.py projects/cosmos-direction/test_gen_preds.py
git commit -m "fix(cosmos): gen_preds halts on infra faults, demotes only parse misses

A dead backend mid-run had produced a truncated preds CSV silently scored as
candidate quality. Narrow to except Unparseable, matching the loop contract.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KnQsvsU7cyqTgE7gKr9E5W"
```

---

## Final verification

- [ ] Run the whole suite under the dep-light venv and confirm green:
  `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest -q`
  Expected: all pass (was 59 before this plan; +~7 new tests).
- [ ] Run the toy end-to-end to confirm no core regression:
  `/Users/travis/Developer/ratchet/.venv/bin/python -m ratchet.loop_cli --project projects/toy --escalate`
  Expected: `best cid=... objective=1.0` and a gate verdict line, no traceback.

## Out of scope (follow-on plans, do NOT attempt here)

These are real red-team findings but are separate subsystems that each warrant their own plan:

1. **Regime fingerprint blind to ingest-truth and code.** Fold `sha256(sorted(ingest() (id,truth)))` plus a `logic_version` (runner/objective/mutations source) into `regime_payload`, thread it through `regime_state.enforce_regime` and the three CLIs, and wire the unused `guard_compare` into the score path. Changes every existing regime hash (a one-time, ledgered bump). Highest structural leverage; deserves a dedicated plan.
2. **Coverage-blind objective gaming.** `mae` divides by `graded`, so answering one easy item posts `mae=0.0` and wins the climb, and the overfit gap goes meaningless when coverage differs across splits. Couple coverage into the climbed scalar (or add a min-coverage gate) and into the gap. Changes objective semantics; needs its own design + plan.
3. **Author-riggable / small-n split + `.regime` hardening.** Content-derived split ids (not author-assigned strings), a rank/quantile split for stable small-n fractions, and a fail-closed `.regime` anchored in the append-only ledger. Trust-model changes; separate plan.
