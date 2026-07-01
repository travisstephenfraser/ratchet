"""Structured (Instructor) parse path. Requires pydantic; run under the cosmos venv.
Mocks the vlm entrypoint so no model is called. The pydantic-free default path is
covered separately in test_runner_regex.py so ratchet's core venv still tests it."""
import importlib
import sys
import types

import pytest
from runner_structured import DirectionRead


# --- schema ---------------------------------------------------------------
def test_schema_accepts_and_uppercases():
    assert DirectionRead(direction="downhill").direction == "DOWNHILL"
    assert DirectionRead(direction=" Uphill ").direction == "UPHILL"


def test_schema_rejects_out_of_set():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        DirectionRead(direction="SIDEWAYS")


# --- runner routing -------------------------------------------------------
def _install_fake_vlm(monkeypatch, *, structured_return):
    captured = {}
    def _afs(frame, prompt, model, **kw):
        captured["response_model"] = model
        captured.update(kw)
        return structured_return
    fake = types.ModuleType("vlm")
    fake.analyze_frame_structured = _afs
    fake.analyze_frame = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("structured path must not call analyze_frame"))
    fake.extract_text = lambda resp: "unused"
    pkg = types.ModuleType("phase2"); pkg.vlm = fake
    monkeypatch.setitem(sys.modules, "phase2", pkg)
    monkeypatch.setitem(sys.modules, "phase2.vlm", fake)
    return captured


def test_structured_path_returns_direction(monkeypatch):
    captured = _install_fake_vlm(monkeypatch, structured_return=DirectionRead(direction="UPHILL"))
    monkeypatch.setenv("COSMOS_PARSE", "structured")
    monkeypatch.setenv("COSMOS_MODE", "grounded")
    import runner; importlib.reload(runner)
    out = runner.Runner().run("cand", {"frame_path": "x.jpg", "telemetry": {"gradient_pct": -5}}, "")
    assert out == "UPHILL"
    # honest-scoring + determinism knobs are actually threaded through
    assert captured["max_retries"] == 0
    assert captured["temperature"] == 0.0
    assert captured["response_model"] is DirectionRead
    # grounded mode passes telemetry, not the neutral preamble
    assert "telemetry" in captured and "system_override" not in captured


def test_structured_pixel_mode_uses_neutral_preamble(monkeypatch):
    captured = _install_fake_vlm(monkeypatch, structured_return=DirectionRead(direction="FLAT"))
    monkeypatch.setenv("COSMOS_PARSE", "structured")
    monkeypatch.setenv("COSMOS_MODE", "pixel")
    monkeypatch.delenv("COSMOS_TEMPERATURE", raising=False)
    import runner; importlib.reload(runner)
    out = runner.Runner().run("cand", {"frame_path": "x.jpg", "telemetry": {}}, "")
    assert out == "FLAT"
    assert "system_override" in captured and "telemetry" not in captured
    assert captured["temperature"] == 0.0   # default when COSMOS_TEMPERATURE unset


def test_empty_temperature_env_omits_the_param(monkeypatch):
    # A model that rejects `temperature` must be reachable by omitting it entirely.
    captured = _install_fake_vlm(monkeypatch, structured_return=DirectionRead(direction="FLAT"))
    monkeypatch.setenv("COSMOS_PARSE", "structured")
    monkeypatch.setenv("COSMOS_TEMPERATURE", "")
    import runner; importlib.reload(runner)
    runner.Runner().run("cand", {"frame_path": "x.jpg", "telemetry": {}}, "")
    assert captured["temperature"] is None   # None => vlm does not send the param


def test_numeric_temperature_env_passes_through(monkeypatch):
    captured = _install_fake_vlm(monkeypatch, structured_return=DirectionRead(direction="FLAT"))
    monkeypatch.setenv("COSMOS_PARSE", "structured")
    monkeypatch.setenv("COSMOS_TEMPERATURE", "0.7")
    import runner; importlib.reload(runner)
    runner.Runner().run("cand", {"frame_path": "x.jpg", "telemetry": {}}, "")
    assert captured["temperature"] == 0.7
