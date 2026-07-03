# Regime-Stamped Predictions + guard_compare Enforcement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stamp every generated predictions file with the regime it was produced under, and make `verify` refuse (exit 2) to score predictions whose stamp does not match the current regime, so a score can never silently pool predictions minted under different rules. Legacy/unstamped files are still scored, but with a strong warning that names the risk and consequence.

**Architecture:** The stamp is a single comment line `# ratchet-regime: <hash>` at the top of the preds CSV, so it travels with the file (a sidecar could get separated). `load_column` skips comment lines; a new `read_preds_regime` reads the stamp back; a new `preds_regime_gate` compares it to the current regime via the existing `guard_compare` (raise on mismatch) and returns a warning string for an unstamped file. The two writers of preds CSVs (`results.write_candidate` in core, `gen_preds.py` in the cosmos project) prepend the stamp.

**Tech Stack:** Python 3.12+ (repo runs 3.14), pytest, the repo venv at `/Users/travis/Developer/ratchet/.venv`. Core stays stdlib + pyyaml only.

## Global Constraints

- **Core stays dependency-light.** `ratchet/` imports only stdlib + pyyaml. All core tests pass under `/Users/travis/Developer/ratchet/.venv` (no pydantic).
- **Run tests with:** `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest <path> -v` from the repo root.
- **Stamp format (exact):** the first line of a stamped preds CSV is `# ratchet-regime: <12-char-hash>\n`. The prefix constant is `PREDS_REGIME_PREFIX = "# ratchet-regime: "`.
- **Preds CSVs never contain multi-line quoted fields** (they are `anon_id,value` rows), so `load_column` may filter comment lines with a line generator.
- **Legacy policy (binding):** an unstamped preds file is SCORED, not rejected, but `verify` must print a warning to stderr that states the RISK and the CONSEQUENCE (a silently wrong number that can pass a gate it should fail, or fail one it should pass).
- **Match existing comment style:** terse inline comments; no em dashes (use commas/semicolons/parentheses).
- **Commit trailers:** end every commit message with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01KnQsvsU7cyqTgE7gKr9E5W`.
- **Branch:** work on `guard-compare-preds` off `master`.

---

### Task 1: Core stamp read/write helpers + comment-skipping load_column (HIGH)

Add the stamp format helpers to `verifier.py` and make `load_column` skip comment lines so a stamped file parses cleanly.

**Files:**
- Modify: `ratchet/verifier.py` (add `PREDS_REGIME_PREFIX`, `preds_regime_header`, `read_preds_regime`; make `load_column` skip `#` lines)
- Test: `tests/test_verifier.py`

**Interfaces:**
- Produces: `PREDS_REGIME_PREFIX: str`; `preds_regime_header(regime) -> str`; `read_preds_regime(path) -> str | None`; `load_column(path, value_field=None) -> dict` (now skips comment lines).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_verifier.py`:

```python
def test_preds_regime_round_trip(tmp_path):
    from ratchet.verifier import preds_regime_header, read_preds_regime, load_column
    p = tmp_path / "preds.csv"
    p.write_text(preds_regime_header("abc123def456") + "anon_id,direction\nid1,DOWNHILL\n")
    # the stamp is readable, and load_column ignores the comment line
    assert read_preds_regime(p) == "abc123def456"
    assert load_column(p) == {"id1": "DOWNHILL"}


def test_read_preds_regime_none_when_unstamped(tmp_path):
    from ratchet.verifier import read_preds_regime, load_column
    p = tmp_path / "legacy.csv"
    p.write_text("anon_id,direction\nid1,UPHILL\n")  # no stamp
    assert read_preds_regime(p) is None
    assert load_column(p) == {"id1": "UPHILL"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_verifier.py::test_preds_regime_round_trip -v`
Expected: FAIL with `ImportError: cannot import name 'preds_regime_header'`.

- [ ] **Step 3: Write minimal implementation**

In `ratchet/verifier.py`, add near the top (after the imports):

```python
PREDS_REGIME_PREFIX = "# ratchet-regime: "


def preds_regime_header(regime) -> str:
    """The comment line that stamps a preds CSV with the regime it was generated under.
    load_column skips it; read_preds_regime reads it back for the comparability guard."""
    return f"{PREDS_REGIME_PREFIX}{regime}\n"


def read_preds_regime(path):
    """Return the regime a preds file was stamped with, or None if it is unstamped (a
    legacy or externally generated file). Stops at the first data line."""
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if line.startswith(PREDS_REGIME_PREFIX):
                return line[len(PREDS_REGIME_PREFIX):].strip()
            if line.strip() and not line.startswith("#"):
                return None
    return None
```

Change `load_column`'s reader to skip comment lines. The current body opens the file and builds `csv.DictReader(f)`; change that one line:

```python
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
```

(the rest of `load_column` is unchanged).

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_verifier.py -v`
Expected: PASS (new tests plus all existing verifier tests, whose fixtures have no comment lines).

- [ ] **Step 5: Commit**

```bash
git add ratchet/verifier.py tests/test_verifier.py
git commit -m "feat(core): preds regime-stamp helpers + comment-skipping load_column

Add PREDS_REGIME_PREFIX, preds_regime_header, read_preds_regime; load_column now
skips # comment lines so a stamped preds CSV parses cleanly.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KnQsvsU7cyqTgE7gKr9E5W"
```

---

### Task 2: The regime gate + strong legacy warning, wired into verify (HIGH)

Add `preds_regime_gate` (returns a warning for legacy, None for a match, raises `RegimeMismatch` on a mismatch) and the `LEGACY_PREDS_WARNING` text, then wire both into `verify` so a mismatched stamp exits 2 and an unstamped file scores with the warning.

**Files:**
- Modify: `ratchet/verifier.py` (add `LEGACY_PREDS_WARNING`, `preds_regime_gate`)
- Modify: `ratchet/verify.py` (capture the current regime; gate the preds; warn or exit 2)
- Test: `tests/test_verifier.py`

**Interfaces:**
- Consumes: `guard_compare`, `RegimeMismatch` (from `ratchet.regime`); `read_preds_regime` (Task 1).
- Produces: `LEGACY_PREDS_WARNING: str` (contains a `{path}` field); `preds_regime_gate(stamped, current) -> str | None` (raises `RegimeMismatch` on mismatch).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_verifier.py`:

```python
def test_preds_regime_gate_match_mismatch_and_legacy():
    import pytest
    from ratchet.verifier import preds_regime_gate, LEGACY_PREDS_WARNING
    from ratchet.regime import RegimeMismatch
    # match -> no warning, no raise
    assert preds_regime_gate("r1", "r1") is None
    # legacy (unstamped) -> a warning that names the risk and consequence
    w = preds_regime_gate(None, "r1")
    assert w is LEGACY_PREDS_WARNING
    assert "RISK" in w and ("pass" in w.lower() and "fail" in w.lower())
    # mismatch -> refuse
    with pytest.raises(RegimeMismatch):
        preds_regime_gate("r1", "r2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_verifier.py::test_preds_regime_gate_match_mismatch_and_legacy -v`
Expected: FAIL with `ImportError: cannot import name 'preds_regime_gate'`.

- [ ] **Step 3: Write minimal implementation**

In `ratchet/verifier.py`, add an import at the top:

```python
from .regime import guard_compare, RegimeMismatch
```

Then add the warning and the gate (near the other preds helpers):

```python
LEGACY_PREDS_WARNING = (
    "WARNING: predictions file {path} carries no regime stamp (legacy or externally "
    "generated). Scoring it anyway, but ratchet CANNOT confirm these predictions were "
    "produced under the current regime.\n"
    "  RISK: if they were generated under a different model, prompt, eval set, or label "
    "set, this score is comparing across incomparable rules, a silently wrong number that "
    "can PASS a gate it should fail, or FAIL one it should pass.\n"
    "  FIX: regenerate the predictions with the current gen_preds so the file is stamped, "
    "or re-run generation and scoring under one regime."
)


def preds_regime_gate(stamped, current):
    """Compare a preds file's stamped regime against the current one. Returns the legacy
    warning string when the file is unstamped, None when the stamp matches, and raises
    RegimeMismatch when they differ (the caller exits non-zero)."""
    if stamped is None:
        return LEGACY_PREDS_WARNING
    guard_compare(stamped, current)  # raises RegimeMismatch on mismatch
    return None
```

Now wire it into `ratchet/verify.py`. Add to the imports:

```python
from .verifier import (split_ids, score_split, gap_report, load_column, log_holdout_access,
                       read_preds_regime, preds_regime_gate)
from .regime import RegimeMismatch
```

Change the body so `enforce_regime`'s returned hash is captured and the preds are gated (replace the `enforce_regime(...)` and `preds = load_column(...)` lines):

```python
    current = enforce_regime(proj, cv, Path(args.project) / "regime_log.jsonl", truth)
    preds = load_column(Path(args.predictions))
    try:
        warning = preds_regime_gate(read_preds_regime(Path(args.predictions)), current)
    except RegimeMismatch as e:
        print(f"refusing to score across regimes: {e}", file=sys.stderr)
        sys.exit(2)
    if warning:
        print(warning.format(path=args.predictions), file=sys.stderr)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_verifier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ratchet/verifier.py ratchet/verify.py tests/test_verifier.py
git commit -m "feat(core): verify refuses cross-regime preds; warns on unstamped

preds_regime_gate compares a preds file's stamp to the current regime: match ->
score, mismatch -> exit 2 via guard_compare, unstamped -> score with a strong
warning naming the risk (comparing across incomparable rules) and consequence.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KnQsvsU7cyqTgE7gKr9E5W"
```

---

### Task 3: Stamp preds on write (core persistence + cosmos gen_preds) (HIGH)

Prepend the stamp when a preds CSV is written, in both writers.

**Files:**
- Modify: `ratchet/results.py` (`write_candidate` prepends the stamp)
- Modify: `projects/cosmos-direction/gen_preds.py` (compute the regime, stamp the output)
- Test: `tests/test_results.py` (create if absent), `projects/cosmos-direction/test_gen_preds.py`

**Interfaces:**
- Consumes: `preds_regime_header` (Task 1); `load_config`, `current_version`, `regime_payload`, `regime_hash`.

- [ ] **Step 1: Write the failing tests**

Create or add to `tests/test_results.py`:

```python
from ratchet.results import write_candidate
from ratchet.verifier import read_preds_regime, load_column


def test_write_candidate_stamps_the_regime(tmp_path):
    write_candidate(tmp_path, "cid1", "some instructions",
                    {"id1": "10", "id2": "8"}, {"objective": 1.0}, regime="r-abc-123")
    preds = tmp_path / "candidates" / "cid1.preds.csv"
    assert read_preds_regime(preds) == "r-abc-123"
    assert load_column(preds) == {"id1": "10", "id2": "8"}  # data still parses
```

Add to `projects/cosmos-direction/test_gen_preds.py` (extend the dep-light fake from that file; it monkeypatches `ingest` and `runner`):

```python
def test_gen_preds_stamps_output_with_a_regime(monkeypatch, tmp_path):
    from ratchet.verifier import read_preds_regime
    class _R:
        def run(self, base, item, policy=""):
            return "DOWNHILL"
    import sys, types, importlib
    fake_ingest = types.ModuleType("ingest")
    fake_ingest.ingest = lambda: ({"a": {"frame_path": "x", "telemetry": {}}}, {"a": "DOWNHILL"})
    fake_runner = types.ModuleType("runner"); fake_runner.Runner = lambda: _R()
    monkeypatch.setitem(sys.modules, "ingest", fake_ingest)
    monkeypatch.setitem(sys.modules, "runner", fake_runner)
    import gen_preds; importlib.reload(gen_preds)
    out = tmp_path / "p.csv"
    gen_preds.main(["--out", str(out)])
    assert read_preds_regime(out) is not None  # a 12-char regime hash was stamped
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_results.py projects/cosmos-direction/test_gen_preds.py -v`
Expected: both new tests FAIL (`read_preds_regime` returns None, no stamp written).

- [ ] **Step 3: Write minimal implementation**

In `ratchet/results.py`, add the import at the top:

```python
from .verifier import preds_regime_header
```

In `write_candidate`, prepend the stamp before writing the CSV header (the `with open(... "w" ...)` block):

```python
    with open(cand_dir / f"{cid}.preds.csv", "w", newline="") as fh:
        fh.write(preds_regime_header(regime))
        w = csv.writer(fh)
        w.writerow(["anon_id", "score"])
        for anon, score in preds.items():
            w.writerow([anon, score])
```

In `projects/cosmos-direction/gen_preds.py`, add the imports near the other ratchet import:

```python
from ratchet.config import load_config
from ratchet.constraints import current_version
from ratchet.regime import regime_payload, regime_hash
from ratchet.verifier import preds_regime_header
```

Compute the regime in `main` right after `items, _ = ingest.ingest()` (use the full truth for the fingerprint; `_truth_fingerprint` str-normalizes internally, so this matches verify):

```python
    items, truth = ingest.ingest()
    cfg = load_config(HERE)
    cv = current_version(HERE / "constraints.jsonl")
    regime = regime_hash(regime_payload(cfg, cv, truth))
```

(The existing `base = Path(args.base).read_text()` and the `r = runner.Runner()` lines follow.) Then stamp the output file, changing the CSV-open block to write the header line first:

```python
    with open(out, "w", newline="", encoding="utf-8") as f:
        f.write(preds_regime_header(regime))
        w = csv.writer(f)
        w.writerow(["anon_id", "direction"])
        ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest tests/test_results.py projects/cosmos-direction/test_gen_preds.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ratchet/results.py projects/cosmos-direction/gen_preds.py tests/test_results.py projects/cosmos-direction/test_gen_preds.py
git commit -m "feat: stamp the regime into written preds (loop persistence + gen_preds)

write_candidate and gen_preds now prepend the '# ratchet-regime:' stamp so verify
can prove predictions were produced under the current regime.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KnQsvsU7cyqTgE7gKr9E5W"
```

---

### Task 4: End-to-end verification (round-trip, mismatch, legacy warning)

Confirm the full loop: a stamped file scores clean, a stamp from a different regime exits 2, and an unstamped file scores with the risk/consequence warning.

**Files:**
- None (runbook); optionally append a fixture note to `tests/`.

- [ ] **Step 1: Full suite green**

Run: `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest -q`
Expected: all pass (was 70; +~5 new tests).

- [ ] **Step 2: Round-trip on the toy project (stamped -> match -> clean score)**

```bash
# regenerate the toy baseline and score a freshly stamped preds file
rm -f projects/toy/.regime
V=/Users/travis/Developer/ratchet/.venv/bin/python
$V - <<'PY'
from pathlib import Path
from ratchet.project import load_project
from ratchet.constraints import current_version
from ratchet.regime import regime_payload, regime_hash
from ratchet.results import write_candidate
proj = load_project(Path("projects/toy"))
items, truth = proj.ingest()
cv = current_version(Path("projects/toy/constraints.jsonl"))
regime = regime_hash(regime_payload(proj.config, cv, truth))
# a perfect stamped preds file for the train split
preds = {k: truth[k] for k in truth}
write_candidate(Path("/tmp/ratchet_stamp_demo"), "demo", "x", preds, {"objective": 1.0}, regime)
print("stamped preds at /tmp/ratchet_stamp_demo/candidates/demo.preds.csv with regime", regime)
PY
$V -m ratchet.verify --project projects/toy --predictions /tmp/ratchet_stamp_demo/candidates/demo.preds.csv --split train
echo "exit: $?  (expect 0, no warning on stderr)"
```
Expected: a JSON score, exit 0, no warning.

- [ ] **Step 3: Mismatch exits 2**

```bash
V=/Users/travis/Developer/ratchet/.venv/bin/python
# hand-edit the stamp to a wrong regime and confirm verify refuses
sed -i.bak 's/^# ratchet-regime: .*/# ratchet-regime: deadbeef0000/' /tmp/ratchet_stamp_demo/candidates/demo.preds.csv
$V -m ratchet.verify --project projects/toy --predictions /tmp/ratchet_stamp_demo/candidates/demo.preds.csv --split train
echo "exit: $?  (expect 2, 'refusing to score across regimes' on stderr)"
```
Expected: `refusing to score across regimes: ...` on stderr, exit 2.

- [ ] **Step 4: Legacy (unstamped) scores WITH the warning**

```bash
V=/Users/travis/Developer/ratchet/.venv/bin/python
# strip the stamp -> a legacy file
grep -v '^# ratchet-regime:' /tmp/ratchet_stamp_demo/candidates/demo.preds.csv > /tmp/ratchet_legacy.csv
$V -m ratchet.verify --project projects/toy --predictions /tmp/ratchet_legacy.csv --split train
echo "exit: $?  (expect 0, but a WARNING with RISK/consequence on stderr)"
rm -f projects/toy/.regime; $V -m ratchet.loop_cli --project projects/toy --escalate >/dev/null 2>&1  # restore baseline
```
Expected: the score prints, exit 0, and stderr shows the `WARNING ... RISK ... can PASS a gate it should fail, or FAIL one it should pass` text.

- [ ] **Step 5: Commit (if any fixture/doc changes were made; otherwise skip)**

No code changes in this task if Steps 1-4 pass as-is.

---

## Final verification

- [ ] `/Users/travis/Developer/ratchet/.venv/bin/python -m pytest -q` — all green.
- [ ] Task 4 Steps 2-4 print exit 0 / exit 2 / exit 0-with-warning respectively.

## Out of scope (follow-on)

- **Stamp + guard the bench path.** `bench` writes rows via `results.write_bench` (not per-candidate preds CSVs) and does not currently re-verify a stamped input; if bench ever consumes external preds, apply the same gate.
- **Thread the regime into `score_split`/`gap_report` directly** so ANY in-process caller (not just the `verify` CLI) is guarded, rather than only the CLI entry point. Requires passing the regime (and the preds' stamp) into the scoring primitives.
- **Add `COSMOS_VLM_MODEL` / `COSMOS_DIR` to cosmos `regime_env`** and reject blank `--why`/`--impact` in `regime_cli` (small hardening noted by the red-team).
- **`.regime` fail-closed + ledger-anchored baseline** (the enforcement-bypass findings). Separate trust-model plan.
