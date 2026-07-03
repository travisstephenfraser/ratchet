"""gen_preds must stop loudly on a non-parse (infra) fault, not silently miss it.
Dep-light: monkeypatches ingest + runner, never calls a model."""
import sys
import types
import importlib
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
