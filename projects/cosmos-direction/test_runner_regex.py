"""Default (regex) parse path — must be testable WITHOUT pydantic/instructor, so
ratchet's dep-light venv still covers it. No pydantic import anywhere in this file.
Structured-path tests live in test_runner_structured.py (which requires pydantic)."""
import importlib
import sys
import types


def _install_fake_vlm(monkeypatch, *, regex_text):
    fake = types.ModuleType("vlm")
    fake.analyze_frame = lambda *a, **k: object()
    fake.extract_text = lambda resp: regex_text
    fake.analyze_frame_structured = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("regex path must not call analyze_frame_structured"))
    pkg = types.ModuleType("phase2"); pkg.vlm = fake
    monkeypatch.setitem(sys.modules, "phase2", pkg)
    monkeypatch.setitem(sys.modules, "phase2.vlm", fake)


def test_regex_path_is_default(monkeypatch):
    _install_fake_vlm(monkeypatch, regex_text="the rider is heading downhill here")
    monkeypatch.delenv("COSMOS_PARSE", raising=False)
    import runner; importlib.reload(runner)
    out = runner.Runner().run("cand", {"frame_path": "x.jpg", "telemetry": {}}, "")
    assert out == "DOWNHILL"


def test_regex_unparseable_raises(monkeypatch):
    _install_fake_vlm(monkeypatch, regex_text="no clear reading in this frame")
    monkeypatch.delenv("COSMOS_PARSE", raising=False)
    import runner; importlib.reload(runner)
    import pytest
    from ratchet.adapter import Unparseable
    with pytest.raises(Unparseable):
        runner.Runner().run("cand", {"frame_path": "x.jpg", "telemetry": {}}, "")
