# Instructor in the cosmos-direction Runner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fragile regex prediction-parse in the cosmos-direction ratchet Runner with an opt-in Instructor + Pydantic schema, working across both VLM backends (Anthropic API and local LM Studio), without touching ratchet core.

**Architecture:** Instructor belongs in the project layer, not ratchet core (the core makes zero model calls). The model call + schema live in two files: `vlm.py` gains a `analyze_frame_structured()` entrypoint that owns backend dispatch, image encoding, telemetry preamble, and Instructor wrapping; the ratchet `runner.py` gains a lazily-imported `DirectionRead` schema and a branch that calls it. The legacy regex path stays the default; the structured path is gated behind `COSMOS_PARSE=structured`.

**Tech Stack:** Python 3.14, Instructor (Anthropic JSON mode + OpenAI JSON_SCHEMA mode), Pydantic v2, `anthropic` SDK, `openai` SDK (for the LM Studio OpenAI-compatible endpoint), pytest.

---

## v2 status — red-team fixes folded + IMPLEMENTED (2026-06-30)

A three-agent adversarial red-team ran against the real code + the cloned Instructor source. It cleared the Instructor-API claims (JSON mode accepts image blocks + list-form `system`; the schema block is appended *after* the `cache_control` block so the cache prefix survives) and surfaced real defects. All fixes below are IMPLEMENTED and tested; two gates remain (needs infra/data).

**Fixes folded in and done:**
1. **Determinism (was BLOCKER).** Neither backend passed `temperature`, so runs sampled at the API default despite `config.yaml` claiming `temperature: 0` — the dominant noise source, unrelated to parsing. Fix: `temperature` threaded through `analyze_frame` + `analyze_frame_structured` (default `None` = unchanged, so the *other* strava pipeline is untouched); the ratchet runner passes `temperature=0.0` in BOTH regex and structured branches. `runner.py`, `vlm.py`.
2. **Regime pooling (was BLOCKER).** `regime.py` didn't hash parse-mode/backend/mode, so structured and regex produced the same regime and would silently pool. Fix: generic `config.regime_env` allowlist (no `COSMOS_*` in core); `regime_payload` folds in declared+set env vars only. `config.yaml` declares `[COSMOS_PARSE, COSMOS_VLM_BACKEND, COSMOS_MODE]`. `regime.py`, `config.py`, `config.yaml`.
3. **Unguarded loop abort (was BLOCKER).** `loop.py:run_candidate_over` had no try/except — a parse raise (more likely under structured) aborted the whole hill-climb. Fix: per-item try/except demotes a raise to a MISS, matching `gen_preds.py`; a candidate that fails on every item still surfaces loudly via `escalate`'s 0-prediction guard. `loop.py`.
4. **LM Studio mode was wrong (caught by live probe).** The original plan used `Mode.JSON` for the local backend. LM Studio's server **rejects** `response_format={"type":"json_object"}` with HTTP 400 (verified 2026-06-30). Fix: `Mode.JSON_SCHEMA`, which LM Studio accepts (verified working on the loaded text model). `vlm.py`.
5. **Test isolation (should-fix).** The regex-default path is tested in a separate **pydantic-free** file (`test_runner_regex.py`) so ratchet's dep-light venv still covers it; structured tests (`test_runner_structured.py`, `phase2/test_vlm_structured.py`) require pydantic and run under the cosmos venv.
6. **Nits folded:** drop the cache-preservation rationale (moot — `gen_preds` sends one prompt per frame); catch `InstructorRetryException`, not `ValidationError`, at any runner-level guard (`gen_preds`/`loop`'s broad `except` already do); correct Task-5 venv path is `$COSMOS_DIR/.venv`.

**Validated live:**
- Real Anthropic structured-vision call on a real frame → validated `DirectionRead` (JSON mode + image + schema works end-to-end).
- **LM Studio + `gemma-4-31b-it-mlx@8bit` (the production VLM) + vision + `Mode.JSON_SCHEMA` on a real frame → validated `DirectionRead(direction="DOWNHILL")` through the real `vlm.analyze_frame_structured` code path (2026-06-30).** This clears the red-team's biggest "possibly DOA" risk: the local vision + structured-output path works on the exact config model.
- LM Studio `Mode.JSON_SCHEMA` also validated on a text model earlier (isolating the `json_object`→`json_schema` fix).
- Full ratchet core suite (58 tests) + structured suites (6 tests) green.

**Gate CLOSED with real data — and the verdict is: do NOT adopt structured for this project.**
Ran the parity on 60 GPS-labeled frames from prior `multiframe_*` runs (ingest derives DOWNHILL/UPHILL/FLAT from telemetry gradient sign; balanced 20/class), pixel mode, temp=0 (deterministic single run), local Gemma via the real Runner code path:

| Mode | Accuracy vs GPS truth | Hard-miss (parse failure) rate |
|---|---|---|
| regex (current) | 30/60 = 50.0% | 0/60 |
| structured (Instructor) | 27/60 = 45.0% | 0/60 |

Two findings, both decision-relevant:
1. **The parse-recovery win is zero here.** Gemma produced parseable output on 100% of frames in BOTH modes. Instructor's headline benefit (recover malformed JSON / reask) buys nothing on this model+task — there is no parse-failure problem to solve.
2. **Structured is slightly WORSE on accuracy** (45% vs 50%). The 14 regex≠structured disagreements show json_schema-constrained generation makes Gemma over-commit to DOWNHILL (FLAT and UPHILL truths collapse to DOWNHILL), where free-text+regex keeps more FLAT/UPHILL reads. This is the red-team's "changes generation modality, not just parsing" concern, measured.

Conclusion: the infra is built, correct, and validated on both backends — but the measurement says **keep `COSMOS_PARSE=regex` as the default for cosmos-direction on the local model.** The decision gate did its job: we learned this from a free 2-minute run instead of shipping it. Caveats: n=60, one prompt, pixel mode (hardest), local Gemma only — a different model or the Anthropic backend could differ, and grounded mode (answer in preamble) would show both near-ceiling. The temperature=0 fix is worth keeping regardless: it made this single run deterministic and cut the dominant hill-climb noise.

---

## Global Constraints

- **Ratchet core stays dependency-light.** `ratchet/`'s pure-Python venv (runs `verify`/`ingest`/core tests) must NOT require `instructor` or `pydantic`. All new imports (`pydantic`, `instructor`, `openai`) are lazy — imported only inside the structured code path, never at `runner.py` module top level. This mirrors the existing lazy `from phase2 import vlm` in `runner.py`.
- **Opt-in flag, regex is default.** Behavior is selected by env `COSMOS_PARSE`: unset or `regex` = legacy path (unchanged); `structured` = Instructor path. Mirrors SiteProof's `--structured`.
- **`max_retries=0` in the scoring Runner.** A ratchet scoring Runner must let a candidate's parse failures count as misses, not silently repair them (auto-retry would make a bad prompt look robust and corrupt the hill-climb). Instructor with `max_retries=0` still validates and RAISES on failure — which satisfies ratchet's fail-loud contract (`ratchet/adapter.py:10-13`).
- **JSON mode, not tool mode.** Use `instructor.Mode.ANTHROPIC_JSON` (Anthropic) and `instructor.Mode.JSON` (LM Studio). Keeps request shape closest to the current call, preserves the image `cache_control` prefix, and does not change generation modality. Tool mode is a separate future experiment.
- **Preserve the request exactly except for the response format.** Same `MODEL`, `MAX_TOKENS`, same system/telemetry preamble, same image block with `cache_control` ephemeral, same grounded-vs-pixel branch. Do not add/change `temperature` (the current `analyze_frame` sets none; matching it keeps parity — temperature is out of scope).
- **Case normalization matches the regex.** Legacy `_parse_direction` uppercases the match. The schema must uppercase before validating so a lowercase model reply is not a spurious miss.
- **Structured mode is a regime bump.** `config.yaml` notes model name is part of the regime fingerprint; parse mode changes the effective request too. Do NOT pool structured-run metrics with regex-run metrics. Record `COSMOS_PARSE=structured` alongside the run in `LOOP_LOG.md`.
- **Dep install target:** add `instructor` and `openai` to the *cosmos venv* (the one with `anthropic`+`httpx` at `$COSMOS_DIR`), never to ratchet's core venv.

**Paths (verbatim):**
- Ratchet runner: `/Users/travis/Developer/ratchet/projects/cosmos-direction/runner.py`
- VLM layer: `/Users/travis/Developer/strava-vlm-telemetry/experiments/cosmos_mtb_analysis/phase2/vlm.py`
- `$COSMOS_DIR` = `/Users/travis/Developer/strava-vlm-telemetry/experiments/cosmos_mtb_analysis`

---

### Task 1: Direction schema (project-local, lazily importable)

**Files:**
- Create: `projects/cosmos-direction/runner_structured.py`
- Test: `projects/cosmos-direction/test_runner_structured.py`

**Interfaces:**
- Produces: `DirectionRead` (Pydantic model, field `direction: Literal["DOWNHILL","UPHILL","FLAT"]`, case-normalized before validation).

- [ ] **Step 1: Write the failing test**

```python
# projects/cosmos-direction/test_runner_structured.py
import pytest
from runner_structured import DirectionRead

def test_accepts_and_uppercases():
    assert DirectionRead(direction="downhill").direction == "DOWNHILL"
    assert DirectionRead(direction=" Uphill ").direction == "UPHILL"

def test_rejects_out_of_set():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        DirectionRead(direction="SIDEWAYS")
```

- [ ] **Step 2: Run test to verify it fails**

Run (under the cosmos venv, which has pydantic): `cd projects/cosmos-direction && python -m pytest test_runner_structured.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'runner_structured'`.

- [ ] **Step 3: Write minimal implementation**

```python
# projects/cosmos-direction/runner_structured.py
"""Structured-parse schema for the cosmos-direction Runner. Imported LAZILY by
runner.py only when COSMOS_PARSE=structured, so ratchet's pure-Python venv never
needs pydantic/instructor. The schema mirrors legacy _parse_direction: it
uppercases before validating, so a lowercase model reply is not a spurious miss."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, field_validator

Direction = Literal["DOWNHILL", "UPHILL", "FLAT"]


class DirectionRead(BaseModel):
    direction: Direction

    @field_validator("direction", mode="before")
    @classmethod
    def _normalize(cls, v):
        if isinstance(v, str):
            return v.strip().upper()
        return v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd projects/cosmos-direction && python -m pytest test_runner_structured.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add projects/cosmos-direction/runner_structured.py projects/cosmos-direction/test_runner_structured.py
git commit -m "feat(cosmos-direction): DirectionRead schema for structured parse"
```

---

### Task 2: `analyze_frame_structured` — Anthropic backend

**Files:**
- Modify: `$COSMOS_DIR/phase2/vlm.py` (add function after `analyze_frame`, ~line 366)
- Test: `$COSMOS_DIR/phase2/test_vlm_structured.py`

**Interfaces:**
- Consumes: existing module globals `MODEL`, `MAX_TOKENS`, `BACKEND`, `LMSTUDIO_BASE_URL`, `LMSTUDIO_TIMEOUT_S`, helpers `_telemetry_preamble`, `_image_block`, `_lmstudio_image_data_uri`, `make_client`.
- Produces: `analyze_frame_structured(frame_path, prompt_text, response_model, *, telemetry=None, client=None, system_override=None, max_retries=0) -> response_model instance`.

- [ ] **Step 1: Write the failing test** (Anthropic path; mocks Instructor so no network)

```python
# $COSMOS_DIR/phase2/test_vlm_structured.py
import os, types
from pathlib import Path
from pydantic import BaseModel
import vlm

class _Read(BaseModel):
    direction: str

def _fake_frame(tmp_path):
    p = tmp_path / "f.jpg"; p.write_bytes(b"\xff\xd8\xff\xd9"); return p

def test_anthropic_structured_call(monkeypatch, tmp_path):
    monkeypatch.setattr(vlm, "BACKEND", "anthropic")
    captured = {}
    class _IClient:
        class messages:
            @staticmethod
            def create(**kw):
                captured.update(kw); return _Read(direction="DOWNHILL")
    fake_instructor = types.SimpleNamespace(
        from_anthropic=lambda client, mode=None: _IClient(),
        Mode=types.SimpleNamespace(ANTHROPIC_JSON="json", JSON="json"),
    )
    monkeypatch.setitem(__import__("sys").modules, "instructor", fake_instructor)
    out = vlm.analyze_frame_structured(
        _fake_frame(tmp_path), "Which direction?", _Read,
        telemetry=None, client=object(), max_retries=0)
    assert out.direction == "DOWNHILL"
    assert captured["response_model"] is _Read
    assert captured["max_retries"] == 0
    assert captured["model"] == vlm.MODEL
    # image block + system preamble were passed through
    content = captured["messages"][0]["content"]
    assert any(b.get("type") == "image" for b in content)
    assert captured["system"][0]["type"] == "text"
```

- [ ] **Step 2: Run test to verify it fails**

Run (cosmos venv): `cd $COSMOS_DIR/phase2 && python -m pytest test_vlm_structured.py::test_anthropic_structured_call -v`
Expected: FAIL with `AttributeError: module 'vlm' has no attribute 'analyze_frame_structured'`.

- [ ] **Step 3: Write minimal implementation** (add to `vlm.py`)

```python
def analyze_frame_structured(
    frame_path: Path,
    prompt_text: str,
    response_model,
    *,
    telemetry: dict[str, Any] | None = None,
    client: Any | None = None,
    system_override: str | None = None,
    max_retries: int = 0,
):
    """Structured variant of analyze_frame: returns a validated `response_model`
    instance instead of a raw Message. Same frame/telemetry/system construction
    and the same cached image prefix; Instructor (JSON mode) enforces the schema
    and re-asks up to `max_retries`. For ratchet SCORING runners pass
    max_retries=0 so a candidate's parse failures count as misses instead of
    being silently repaired. Imports instructor lazily (run-time dep only)."""
    import instructor

    system = system_override if system_override is not None else _telemetry_preamble(telemetry)

    if BACKEND == "lmstudio":
        return _analyze_frame_structured_lmstudio(
            frame_path, prompt_text, response_model,
            system=system, max_retries=max_retries)

    if client is None:
        client = make_client()
    iclient = instructor.from_anthropic(client, mode=instructor.Mode.ANTHROPIC_JSON)
    return iclient.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        max_retries=max_retries,
        response_model=response_model,
        system=[{"type": "text", "text": system}],
        messages=[{
            "role": "user",
            "content": [
                _image_block(frame_path),
                {"type": "text", "text": prompt_text},
            ],
        }],
    )
```

(The `_analyze_frame_structured_lmstudio` helper is added in Task 3; define a temporary stub that raises `NotImplementedError` so this task's test — which forces `BACKEND="anthropic"` — passes.)

```python
def _analyze_frame_structured_lmstudio(*args, **kwargs):
    raise NotImplementedError  # implemented in Task 3
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $COSMOS_DIR/phase2 && python -m pytest test_vlm_structured.py::test_anthropic_structured_call -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add phase2/vlm.py phase2/test_vlm_structured.py
git commit -m "feat(vlm): analyze_frame_structured (anthropic, JSON mode)"
```

---

### Task 3: `analyze_frame_structured` — LM Studio (OpenAI-compatible) backend

**Files:**
- Modify: `$COSMOS_DIR/phase2/vlm.py` (replace the Task 2 stub)
- Test: `$COSMOS_DIR/phase2/test_vlm_structured.py` (add a case)

**Interfaces:**
- Consumes: `MODEL`, `MAX_TOKENS`, `LMSTUDIO_BASE_URL`, `LMSTUDIO_TIMEOUT_S`, `_lmstudio_image_data_uri`.
- Produces: `_analyze_frame_structured_lmstudio(frame_path, prompt_text, response_model, *, system, max_retries) -> response_model instance`.

- [ ] **Step 1: Write the failing test**

```python
def test_lmstudio_structured_call(monkeypatch, tmp_path):
    import types, sys
    monkeypatch.setattr(vlm, "BACKEND", "lmstudio")
    captured = {}
    class _Completions:
        def create(self, **kw): captured.update(kw); return _Read(direction="FLAT")
    class _Chat:  completions = _Completions()
    class _IClient: chat = _Chat()
    fake_instructor = types.SimpleNamespace(
        from_openai=lambda oai, mode=None: _IClient(),
        Mode=types.SimpleNamespace(JSON="json", ANTHROPIC_JSON="json"))
    fake_openai = types.SimpleNamespace(OpenAI=lambda **kw: object())
    monkeypatch.setitem(sys.modules, "instructor", fake_instructor)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    out = vlm.analyze_frame_structured(
        _fake_frame(tmp_path), "Which direction?", _Read,
        system_override="sys", max_retries=0)
    assert out.direction == "FLAT"
    assert captured["response_model"] is _Read
    assert captured["max_retries"] == 0
    content = captured["messages"][1]["content"]
    assert any(b.get("type") == "image_url" for b in content)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $COSMOS_DIR/phase2 && python -m pytest test_vlm_structured.py::test_lmstudio_structured_call -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Write minimal implementation** (replace the stub)

```python
def _analyze_frame_structured_lmstudio(
    frame_path: Path,
    prompt_text: str,
    response_model,
    *,
    system: str,
    max_retries: int,
):
    """Structured parse against the local LM Studio OpenAI-compatible endpoint.
    Instructor JSON mode asks for schema-shaped JSON and validates it. Local
    models are less reliable at strict JSON than at mentioning a keyword, so a
    structured miss here is a REAL signal about the candidate (with max_retries=0
    it lowers that candidate's score, as intended for scoring runners)."""
    import instructor
    from openai import OpenAI

    oai = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio",
                 timeout=LMSTUDIO_TIMEOUT_S)
    iclient = instructor.from_openai(oai, mode=instructor.Mode.JSON)
    return iclient.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        max_retries=max_retries,
        response_model=response_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url",
                 "image_url": {"url": _lmstudio_image_data_uri(frame_path)}},
            ]},
        ],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $COSMOS_DIR/phase2 && python -m pytest test_vlm_structured.py -v`
Expected: PASS (both backend tests).

- [ ] **Step 5: Commit**

```bash
git add phase2/vlm.py phase2/test_vlm_structured.py
git commit -m "feat(vlm): analyze_frame_structured LM Studio backend"
```

---

### Task 4: Wire the Runner behind `COSMOS_PARSE=structured`

**Files:**
- Modify: `projects/cosmos-direction/runner.py` (the `Runner.run` method)
- Test: `projects/cosmos-direction/test_runner_structured.py` (add cases)

**Interfaces:**
- Consumes: `vlm.analyze_frame_structured` (Task 2/3), `DirectionRead` (Task 1).
- Produces: `Runner.run(candidate, item, policy="") -> str` — behavior unchanged when `COSMOS_PARSE` unset; structured when `COSMOS_PARSE=structured`.

- [ ] **Step 1: Write the failing test** (monkeypatch a fake `phase2.vlm`)

```python
import os, sys, types, importlib
from pathlib import Path

def _install_fake_vlm(monkeypatch, *, structured_return=None, regex_text=None):
    fake = types.ModuleType("vlm")
    fake.analyze_frame_structured = lambda *a, **k: structured_return
    fake.analyze_frame = lambda *a, **k: object()
    fake.extract_text = lambda resp: regex_text
    pkg = types.ModuleType("phase2"); pkg.vlm = fake
    monkeypatch.setitem(sys.modules, "phase2", pkg)
    monkeypatch.setitem(sys.modules, "phase2.vlm", fake)

def test_structured_path_returns_direction(monkeypatch):
    from runner_structured import DirectionRead
    _install_fake_vlm(monkeypatch, structured_return=DirectionRead(direction="UPHILL"))
    monkeypatch.setenv("COSMOS_PARSE", "structured")
    monkeypatch.setenv("COSMOS_MODE", "grounded")
    import runner; importlib.reload(runner)
    out = runner.Runner().run("cand", {"frame_path": "x.jpg", "telemetry": {}}, "")
    assert out == "UPHILL"

def test_regex_path_is_default(monkeypatch):
    _install_fake_vlm(monkeypatch, regex_text="the rider is going downhill here")
    monkeypatch.delenv("COSMOS_PARSE", raising=False)
    import runner; importlib.reload(runner)
    out = runner.Runner().run("cand", {"frame_path": "x.jpg", "telemetry": {}}, "")
    assert out == "DOWNHILL"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd projects/cosmos-direction && python -m pytest test_runner_structured.py -v`
Expected: FAIL — `test_structured_path_returns_direction` fails because `run` has no structured branch (returns a regex parse of `None` / raises).

- [ ] **Step 3: Write minimal implementation** (edit `Runner.run` in `runner.py`)

Replace the body of `run` after `prompt = ...` with:

```python
        grounded = os.environ.get("COSMOS_MODE", "pixel") == "grounded"

        if os.environ.get("COSMOS_PARSE", "regex") == "structured":
            from runner_structured import DirectionRead  # lazy: pydantic only here
            kw = ({"telemetry": item["telemetry"]} if grounded
                  else {"system_override": NEUTRAL_PREAMBLE})
            # max_retries=0: a parse failure must count against the candidate,
            # not be silently repaired (honest hill-climb).
            result = vlm.analyze_frame_structured(
                frame, prompt, DirectionRead, max_retries=0, **kw)
            return result.direction

        if grounded:
            resp = vlm.analyze_frame(frame, prompt, telemetry=item["telemetry"])
        else:
            resp = vlm.analyze_frame(frame, prompt, system_override=NEUTRAL_PREAMBLE)
        return _parse_direction(vlm.extract_text(resp))
```

(Confirm `import os` is present at the top of `runner.py` — it is.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd projects/cosmos-direction && python -m pytest test_runner_structured.py -v`
Expected: PASS (all cases).

- [ ] **Step 5: Commit**

```bash
git add projects/cosmos-direction/runner.py projects/cosmos-direction/test_runner_structured.py
git commit -m "feat(cosmos-direction): opt-in structured parse via COSMOS_PARSE"
```

---

### Task 5: Parity validation (closed loop) before adopting

This is a runbook, not code. The point is not "tests pass" — it is "does structured parsing change the measured score, and does it reduce hard parse failures?" Do this before making `structured` the default anywhere.

**Files:**
- Modify (append only): `projects/cosmos-direction/LOOP_LOG.md` (record the regime)

- [ ] **Step 1: Install deps into the cosmos venv (not ratchet's)**

```bash
# use the venv that already has anthropic + httpx
$COSMOS_DIR/../../.venv/bin/python -m pip install instructor openai   # adjust to the actual cosmos venv path
```

- [ ] **Step 2: Generate predictions on the eval set under BOTH parse modes**

Run the project's existing prediction generator once per mode (same model, same eval set), changing only `COSMOS_PARSE`:

```bash
cd projects/cosmos-direction
COSMOS_PARSE=regex      python gen_preds.py    # baseline (current behavior)
COSMOS_PARSE=structured python gen_preds.py    # structured
```

- [ ] **Step 3: Score both with ratchet `verify` and compare**

Score each candidate's preds with the normal ratchet path and compare `accuracy` and the `anomaly` flag in the resulting `*.metrics.json`. Expectation and reading:
- On the **Anthropic** backend, accuracy should match the regex baseline within noise (JSON mode does not change what Claude decides, only the wrapper). A drop means the wrapper is interfering — investigate before adopting.
- On the **local LM Studio** backend, expect a possible accuracy drop: local models mention the keyword more reliably than they emit strict JSON. If structured is worse there, that is a real limitation of local structured output, not a bug — keep regex as default for the local regime.
- Compare the **hard-failure rate** (raises / missing preds). The win for structured is fewer ambiguous responses that the regex would silently mis-key.

- [ ] **Step 4: Record the regime**

Append to `LOOP_LOG.md`: date, backend, model, `COSMOS_PARSE` value, accuracy, anomaly, and hard-failure counts for both modes. Structured and regex metrics are DIFFERENT REGIMES — never pool them into one leaderboard.

- [ ] **Step 5: Decide**

- If parity holds on the backend you run in production and hard failures drop, keep `COSMOS_PARSE=structured` as an explicit run setting (not a silent default).
- If not, leave regex as default; the structured path stays available for the backend where it helps.

---

## Self-Review notes

- **Dep isolation** (global constraint) is honored: `pydantic` is imported only inside `runner_structured.py` (lazy-imported in the structured branch); `instructor`/`openai` only inside `vlm.analyze_frame_structured`. Ratchet core `verify`/`ingest` never import them. ✔
- **Fail-loud contract**: `max_retries=0` + Instructor raising on validation failure preserves `adapter.py`'s "RAISE on unparseable" rule. ✔
- **Type consistency**: `DirectionRead.direction` is the same `Literal` set the objective compares against (`objective.py` uses uppercase `DOWNHILL/UPHILL/FLAT`); the `_normalize` validator matches legacy `.upper()`. ✔
- **Backends**: both `analyze_frame` backends (anthropic `messages.create`, LM Studio `chat.completions`) have a structured counterpart. ✔
- **Not pooled**: parse mode recorded as a regime (Task 5 step 4), consistent with `config.yaml`'s fingerprint note. ✔

## Generalization (out of scope here, noted for later)

The same shape ports to **Rubrica's `calibration/grading_loop/` Runner** — it grades exams, so its structured Runner would reuse the `GradeResult` schema + helper from `Rubrica-private/docs/plans/instructor-structured-grading.md`, again with `max_retries=0` in the scoring context. Do the Rubrica production-grading port first; the ratchet grading Runner then inherits it.
```
