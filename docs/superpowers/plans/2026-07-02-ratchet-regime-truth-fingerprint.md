# Regime Truth + Source Fingerprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the regime fingerprint cover the two comparability surfaces it is currently blind to: the DERIVED ground truth (so a relabeling constant like a gradient band moves the hash) and the runner/objective SOURCE (so a parse/scoring-logic edit moves the hash), so scores under genuinely different rules can no longer silently pool.

**Architecture:** Add two content hashes to `regime_payload`'s `frozen` block. `truth` is a hash of the `(id, label)` pairs returned by `ingest()`, which fingerprints the OUTPUT of derivation and therefore closes the item-set, item-count, and truth-derivation holes at once (it subsumes hashing `ingest.py`). `logic` is a hash of the runner (and the project objective when custom/judge) source bytes, covering the code that defines what a prediction and score MEAN. Then thread the derived truth into the enforcement path (`enforce_regime`, `record_bump`, and the four CLIs) so the fingerprint is actually computed at every entry point.

**Tech Stack:** Python 3.12+ (repo runs 3.14), pytest, the repo venv at `/Users/travis/Developer/ratchet/.venv`. Core stays stdlib + pyyaml only.

## Global Constraints

- **Core stays dependency-light.** `ratchet/` imports only stdlib + pyyaml. All core tests pass under `/Users/travis/Developer/ratchet/.venv` (no pydantic).
- **Run tests with:** `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest <path> -v` from the repo root.
- **Fingerprint direction:** over-block is safe (refuse to compare), silent-pool is the danger. When in doubt, add to the hash.
- **Backward-compat for unit tests:** `regime_payload`'s new `truth` parameter defaults to `None` and is omitted from the payload when `None`, so existing 2-arg callers keep working relative to each other. The real entry points always pass truth.
- **Match existing comment style:** terse inline comments; no em dashes (use commas/semicolons/parentheses).
- **Commit trailers:** end every commit message with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01KnQsvsU7cyqTgE7gKr9E5W`.
- **Branch:** work on `regime-fingerprint` off `master`, not on `master`.

---

### Task 1: Fingerprint the derived truth (HIGH)

Add `_truth_fingerprint(truth)` and fold it into `regime_payload` behind a new optional `truth` parameter. This one field closes three red-team findings: truth-derivation (relabeling flips labels under an unchanged config), the `ingest-full` item-set hole, and item-count changes, because it hashes ingest's OUTPUT rather than its many inputs.

**Files:**
- Modify: `ratchet/regime.py` (add `_truth_fingerprint`; extend `regime_payload` signature and `frozen`)
- Test: `tests/test_regime.py`

**Interfaces:**
- Produces: `regime_payload(config, constraints_version, truth=None) -> dict` (was 2-arg); `_truth_fingerprint(truth) -> str`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_regime.py`:

```python
def test_hash_changes_when_derived_truth_changes(tmp_path):
    # same config, DIFFERENT label values -> different regime (the truth-derivation hole)
    cfg = _cfg(tmp_path)
    a = regime_hash(regime_payload(cfg, "c1", {"id1": "DOWNHILL", "id2": "FLAT"}))
    b = regime_hash(regime_payload(cfg, "c1", {"id1": "UPHILL", "id2": "FLAT"}))
    assert a != b


def test_hash_changes_when_item_set_changes(tmp_path):
    cfg = _cfg(tmp_path)
    a = regime_hash(regime_payload(cfg, "c1", {"id1": "X"}))
    b = regime_hash(regime_payload(cfg, "c1", {"id1": "X", "id2": "X"}))  # extra item
    assert a != b


def test_truth_none_is_backward_compatible(tmp_path):
    # omitting truth must not perturb the hash relative to explicitly passing None
    cfg = _cfg(tmp_path)
    assert regime_hash(regime_payload(cfg, "c1")) == regime_hash(regime_payload(cfg, "c1", None))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_regime.py::test_hash_changes_when_derived_truth_changes -v`
Expected: FAIL with `TypeError: regime_payload() takes 2 positional arguments but 3 were given`.

- [ ] **Step 3: Write minimal implementation**

In `ratchet/regime.py`, add the helper (near `_eval_set_fingerprint`):

```python
def _truth_fingerprint(truth) -> str:
    """Content hash of the derived (id, label) ground truth. Fingerprints the OUTPUT of
    ingest(), so it closes the item-set, item-count, and truth-DERIVATION holes at once:
    change a relabeling constant (e.g. a gradient band) and every label flips, so this
    moves even when the id list and the config are byte-identical."""
    items = sorted((str(k), str(v)) for k, v in truth.items())
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()[:12]
```

Change `regime_payload`'s signature and add the field:

```python
def regime_payload(config, constraints_version, truth=None) -> dict:
    frozen = {
        "holdout_pct": config.holdout_pct,
        "guards": config.guards,
        "model": config.model,
        "eval_set": _eval_set_fingerprint(config),
    }
    # Content of the derived ground truth. Omitted when truth is not supplied (unit tests),
    # but every real entry point passes it, so a relabel or an item-set change bumps the hash.
    if truth is not None:
        frozen["truth"] = _truth_fingerprint(truth)
    env = {k: os.environ[k] for k in getattr(config, "regime_env", []) if k in os.environ}
    if env:
        frozen["env"] = env
    return {
        "salt": config.salt,
        "objective": {"name": config.objective.name, "params": config.objective.params},
        "frozen": frozen,
        "constraints_version": constraints_version,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_regime.py tests/test_regime_env.py -v`
Expected: PASS (new tests plus all existing regime tests, which call the 2-arg form and compare relatively).

- [ ] **Step 5: Commit**

```bash
git add ratchet/regime.py tests/test_regime.py
git commit -m "feat(core): fingerprint the derived truth in the regime hash

Hash the (id, label) pairs from ingest() into the payload. Fingerprints the
OUTPUT of derivation, so it closes the truth-derivation, item-set, and
item-count holes at once (relabeling under an unchanged config now bumps the hash).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KnQsvsU7cyqTgE7gKr9E5W"
```

---

### Task 2: Fingerprint the scoring/parse SOURCE (MEDIUM-HIGH)

Add `_source_fingerprint(config)` and fold it into `frozen["logic"]`. This hashes the runner (the model call + parse) and the project objective source when it is custom/judge, the code that defines what a PREDICTION and SCORE mean. `ingest.py` is deliberately NOT hashed here because Task 1's truth hash already fingerprints its output.

**Files:**
- Modify: `ratchet/regime.py` (add `_source_fingerprint`; add `frozen["logic"]`)
- Test: `tests/test_regime.py`

**Interfaces:**
- Produces: `_source_fingerprint(config) -> str`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_regime.py`:

```python
def test_logic_hash_changes_on_runner_source_edit(tmp_path):
    (tmp_path / "runner.py").write_text("class Runner:\n    def run(self,c,i,p=''): return '1'\n")
    cfg = _cfg(tmp_path); cfg.runner = "runner.py:Runner"
    a = regime_hash(regime_payload(cfg, "c1"))
    (tmp_path / "runner.py").write_text("class Runner:\n    def run(self,c,i,p=''): return '2'\n")
    b = regime_hash(regime_payload(cfg, "c1"))
    assert a != b  # a scoring/parse-logic edit bumps the regime


def test_logic_hash_is_deterministic_when_source_unchanged(tmp_path):
    (tmp_path / "runner.py").write_text("X")
    cfg = _cfg(tmp_path); cfg.runner = "runner.py:Runner"
    assert regime_hash(regime_payload(cfg, "c1")) == regime_hash(regime_payload(cfg, "c1"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_regime.py::test_logic_hash_changes_on_runner_source_edit -v`
Expected: FAIL (`a == b`; runner source is not yet in the hash).

- [ ] **Step 3: Write minimal implementation**

In `ratchet/regime.py`, add the helper:

```python
def _source_fingerprint(config) -> str:
    """Hash of the source that defines what a PREDICTION and a SCORE mean: the runner
    (model call + parse) plus the project objective when it is custom/judge. ingest.py is
    NOT hashed here because _truth_fingerprint already fingerprints its output. Conservative:
    a formatting-only edit also bumps the regime, which is the safe over-block direction."""
    refs = [config.runner]
    if config.objective.name == "custom":
        refs.append("objective.py:make_objective")
    elif config.objective.name == "judge":
        refs.append("judge.py:judge_fn")
    h = hashlib.sha256()
    for ref in refs:
        filename = ref.split(":")[0]
        p = Path(config.project_dir) / filename
        h.update(filename.encode())
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()[:12]
```

Add `logic` to `frozen` in `regime_payload` (right after the `eval_set` line):

```python
        "eval_set": _eval_set_fingerprint(config),
        "logic": _source_fingerprint(config),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_regime.py tests/test_regime_env.py -v`
Expected: PASS (new tests plus existing regime tests, which use `runner="r"` so the source file is absent and hashes deterministically to a `<missing>` fingerprint).

- [ ] **Step 5: Commit**

```bash
git add ratchet/regime.py tests/test_regime.py
git commit -m "feat(core): fingerprint runner/objective source in the regime hash

Hash the runner (and the custom/judge objective) source bytes into frozen.logic,
so a parse/scoring-logic edit that changes what a score means bumps the regime.
ingest is not hashed here; the truth fingerprint already covers its output.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KnQsvsU7cyqTgE7gKr9E5W"
```

---

### Task 3: Thread the derived truth into the enforcement path (HIGH)

Task 1's truth hash only fires when a caller passes truth. Thread it through `enforce_regime`, `record_bump`, and all four entry points so the fingerprint is computed at every real scoring command, not just in unit tests. The CLIs already call `ingest()`; this reorders them to ingest BEFORE enforcing and passes the truth in.

**Files:**
- Modify: `ratchet/regime_state.py` (`enforce_regime`, `record_bump` signatures)
- Modify: `ratchet/loop_cli.py:20-22`, `ratchet/verify.py:17-18`, `ratchet/bench_cli.py:16-17`, `ratchet/regime_cli.py:16-20`
- Test: `tests/test_regime.py`

**Interfaces:**
- Consumes: `regime_payload(config, cv, truth)` (Task 1).
- Produces: `enforce_regime(project, constraints_version, ledger_path, truth) -> str`;
  `record_bump(project, constraints_version, why, impact, author, timestamp, ledger_path, truth) -> list`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_regime.py` (add `from ratchet.regime_state import enforce_regime` near the top imports, or import inside the test):

```python
def test_enforce_regime_blocks_on_relabeled_truth(tmp_path):
    import pytest
    from ratchet.regime_state import enforce_regime
    class _P:
        def __init__(self, cfg): self.config = cfg
    cfg = _cfg(tmp_path)
    ledger = tmp_path / "regime_log.jsonl"
    # first run establishes the baseline .regime with truth A
    enforce_regime(_P(cfg), "c1", ledger, {"id1": "DOWNHILL"})
    # a relabeled truth is a different regime and must block (exit 2), no ledger rationale
    with pytest.raises(SystemExit) as ei:
        enforce_regime(_P(cfg), "c1", ledger, {"id1": "UPHILL"})
    assert ei.value.code == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_regime.py::test_enforce_regime_blocks_on_relabeled_truth -v`
Expected: FAIL with `TypeError: enforce_regime() takes 3 positional arguments but 4 were given`.

- [ ] **Step 3: Write minimal implementation**

In `ratchet/regime_state.py`, change both functions to accept and pass `truth`:

```python
def enforce_regime(project, constraints_version, ledger_path, truth):
    payload = regime_payload(project.config, constraints_version, truth)
    current = regime_hash(payload)
    state_path = Path(project.config.project_dir) / ".regime"
    if state_path.exists():
        old = json.loads(state_path.read_text())
        if regime_hash(old) != current:
            changes = diff_payload(old, payload)
            if not _covered(changes, RegimeLedger(ledger_path).entries()):
                lines = "\n".join(f"  {f}: {o!r} -> {n!r}" for f, o, n in changes)
                msg = (f"regime changed without a ledger rationale:\n{lines}\n"
                       f"Record it: python -m ratchet.regime_cli --project "
                       f"{project.config.project_dir} --why '...' --impact '...'")
                print(msg, file=sys.stderr)
                sys.exit(2)
    state_path.write_text(json.dumps(payload, sort_keys=True))
    return current


def record_bump(project, constraints_version, why, impact, author, timestamp, ledger_path, truth):
    payload = regime_payload(project.config, constraints_version, truth)
    state_path = Path(project.config.project_dir) / ".regime"
    old = json.loads(state_path.read_text()) if state_path.exists() else {}
    changes = diff_payload(old, payload)
    RegimeLedger(ledger_path).record(version=project.config.version, changed=changes,
                                     why=why, impact=impact, author=author, timestamp=timestamp)
    return changes
```

In `ratchet/loop_cli.py`, reorder lines 20-22 so ingest runs first and truth flows in:

```python
    items, truth = proj.ingest()
    enforce_regime(proj, cv, Path(args.project) / "regime_log.jsonl", truth)  # BLOCKS on silent change
    regime = regime_hash(regime_payload(proj.config, cv, truth))
```

In `ratchet/verify.py`, reorder lines 17-18:

```python
    _, truth = proj.ingest()
    truth = {k: str(v) for k, v in truth.items()}
    enforce_regime(proj, cv, Path(args.project) / "regime_log.jsonl", truth)
```

(The existing `load_column`/`preds` lines follow unchanged.)

In `ratchet/bench_cli.py`, reorder lines 16-17:

```python
    items, truth = proj.ingest()
    enforce_regime(proj, cv, Path(args.project) / "regime_log.jsonl", truth)
```

In `ratchet/regime_cli.py`, ingest and pass truth to `record_bump` (replace lines 16-20):

```python
    proj = load_project(Path(args.project))
    cv = current_version(Path(args.project) / "constraints.jsonl")
    _, truth = proj.ingest()
    changes = record_bump(proj, cv, why=args.why, impact=args.impact, author=args.author,
                          timestamp=datetime.now(timezone.utc).isoformat(),
                          ledger_path=Path(args.project) / "regime_log.jsonl", truth=truth)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_regime.py -v`
Expected: PASS (new test plus existing regime tests).

- [ ] **Step 5: Commit**

```bash
git add ratchet/regime_state.py ratchet/loop_cli.py ratchet/verify.py ratchet/bench_cli.py ratchet/regime_cli.py tests/test_regime.py
git commit -m "feat(core): thread derived truth into regime enforcement

enforce_regime/record_bump and the four CLIs now compute the fingerprint with the
ingested truth, so a relabel or item-set change blocks a run at every entry point.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KnQsvsU7cyqTgE7gKr9E5W"
```

---

### Task 4: Migrate the example projects and verify end-to-end (MEDIUM)

The payload changed, so the committed `.regime` files for `toy` and `cosmos-direction` are now stale and would block a run. Both are EXAMPLE projects with no scientific results to protect, and they are already gitignored, so the clean migration is to untrack them and let each machine regenerate its own baseline (this also fixes the tracked-but-gitignored inconsistency the red-team noted). Then confirm the fingerprint fires end-to-end.

**Files:**
- Remove from tracking: `projects/toy/.regime`, `projects/cosmos-direction/.regime`
- Modify: `tests/test_toy_e2e.py:16` (pass truth, matching real usage)

- [ ] **Step 1: Update the toy e2e to fingerprint truth**

In `tests/test_toy_e2e.py`, change line 16 from:

```python
    regime = regime_hash(regime_payload(proj.config, cv))
```
to:
```python
    regime = regime_hash(regime_payload(proj.config, cv, truth))
```

- [ ] **Step 2: Untrack the stale .regime baselines**

```bash
git rm --cached projects/toy/.regime projects/cosmos-direction/.regime
```
Expected: `rm 'projects/toy/.regime'` and `rm 'projects/cosmos-direction/.regime'`. They remain on disk (gitignored) and will be rewritten on the next run.

- [ ] **Step 3: Run the full suite**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest -q`
Expected: all pass (was 64 after Wave 1; +~6 new regime tests).

- [ ] **Step 4: Verify the toy loop regenerates its baseline and runs clean**

```bash
rm -f projects/toy/.regime
/Users/travis/Developer/ratchet/.venv/bin/python -m ratchet.loop_cli --project projects/toy --escalate
```
Expected: `best cid=... objective=1.0`, a gate verdict line, no traceback, and a fresh `projects/toy/.regime` written. Running it a SECOND time must NOT block (identical regime).

- [ ] **Step 5: Verify the truth fingerprint actually blocks a relabel (manual proof)**

```bash
# baseline is written; now force a different label set and confirm the gate blocks
/Users/travis/Developer/ratchet/.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "projects/toy")
from pathlib import Path
from ratchet.project import load_project
from ratchet.regime_state import enforce_regime
proj = load_project(Path("projects/toy"))
_, truth = proj.ingest()
relabeled = {k: ("999" if i == 0 else v) for i, (k, v) in enumerate(truth.items())}
try:
    enforce_regime(proj, "c1", Path("projects/toy/regime_log.jsonl"), relabeled)
    print("BUG: relabel did not block")
except SystemExit as e:
    print("OK: relabel blocked with exit", e.code)
PY
# restore the honest baseline so the toy project is left runnable
rm -f projects/toy/.regime
/Users/travis/Developer/ratchet/.venv/bin/python -m ratchet.loop_cli --project projects/toy --escalate >/dev/null
```
Expected: `OK: relabel blocked with exit 2`.

- [ ] **Step 6: Commit**

```bash
# Step 2's `git rm --cached` already staged the .regime removals; do NOT `git add`
# those paths (that would re-add the gitignored files). Stage only the test change.
git add tests/test_toy_e2e.py
git commit -m "chore(core): re-baseline example regimes for the fuller fingerprint

Untrack the example projects' .regime (gitignored; each machine regenerates its
own baseline) since the truth+source fields changed the payload. Toy e2e now
fingerprints truth, matching real CLI usage.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KnQsvsU7cyqTgE7gKr9E5W"
```

Verify before committing: `git status` should show the two `.regime` files as `deleted` (staged) and `tests/test_toy_e2e.py` as `modified` (staged), with the regenerated `projects/toy/.regime` NOT listed (gitignored).

---

## Final verification

- [ ] `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest -q` — all green.
- [ ] Toy `loop_cli --escalate` runs twice with no block on the second run.
- [ ] The Step-5 relabel proof prints `OK: relabel blocked with exit 2`.

## Out of scope (follow-on)

- **Wire `guard_compare` into the score path.** `score_split`/`gap_report` still take no regime and never call the existing `guard_compare`, so predictions minted under different rules pool if you call the scoring primitives directly. Closing this needs the regime stamped INTO the prediction artifact (results.py / preds CSV) and read back by `verify`, which is a distinct change with its own migration. Do it as a separate plan.
- **Add `COSMOS_VLM_MODEL` / `COSMOS_DIR` to cosmos `regime_env`** (a one-line project-config fix, folds into the env block) and reject blank `--why`/`--impact` in `regime_cli`. Small hardening; can ride along with the guard_compare plan.
- **`.regime` fail-closed + ledger-anchored baseline** (the H1/H2 enforcement-bypass findings). Separate trust-model plan.
