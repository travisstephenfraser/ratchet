from pathlib import Path
import pytest
from ratchet.config import Config, ObjectiveCfg
from ratchet.regime import (
    regime_payload, regime_hash, diff_payload, guard_compare, RegimeMismatch, RegimeLedger,
)


def _cfg(tmp_path, tol=2.0, max_tokens=4000, eval_set="data/eval_set.txt"):
    return Config(
        project="toy", version="v1", salt="toy-v1", holdout_pct=30,
        runner="r", ingest="i", mutations="m", base_candidate="b.txt",
        objective=ObjectiveCfg("within_tol", {"tol": tol}),
        guards={"anomaly_at": 0.95, "overfit_gap": 0.10},
        search={"rounds": 6, "patience": 3},
        model={"name": "m", "temperature": 0, "max_tokens": max_tokens},
        bench={"candidates": ["a"], "eval_set": eval_set},
        project_dir=tmp_path,
    )


def test_hash_stable_and_12_chars(tmp_path):
    a = regime_hash(regime_payload(_cfg(tmp_path), "c1"))
    b = regime_hash(regime_payload(_cfg(tmp_path), "c1"))
    assert a == b and len(a) == 12


def test_hash_changes_on_frozen_param(tmp_path):
    a = regime_hash(regime_payload(_cfg(tmp_path, max_tokens=1500), "c1"))
    b = regime_hash(regime_payload(_cfg(tmp_path, max_tokens=4000), "c1"))
    assert a != b


def test_hash_changes_on_constraints_version(tmp_path):
    a = regime_hash(regime_payload(_cfg(tmp_path), "c1"))
    b = regime_hash(regime_payload(_cfg(tmp_path), "c2"))
    assert a != b


def test_hash_tracks_eval_set_CONTENTS_not_path(tmp_path):
    (tmp_path / "data").mkdir()
    es = tmp_path / "data" / "eval_set.txt"
    es.write_text("id1\nid2\n")
    a = regime_hash(regime_payload(_cfg(tmp_path), "c1"))
    es.write_text("id1\nid2\nid3\n")  # same path, different contents
    b = regime_hash(regime_payload(_cfg(tmp_path), "c1"))
    assert a != b


@pytest.mark.parametrize("bench", [{}, {"eval_set": ""}, {"eval_set": "   "}])
def test_regime_rejects_absent_or_blank_eval_set_configuration(tmp_path, bench):
    cfg = _cfg(tmp_path)
    cfg.bench = bench
    with pytest.raises(ValueError, match="must be a path or explicit null"):
        regime_payload(cfg, "c1")


def test_diff_reports_field_old_new(tmp_path):
    old = regime_payload(_cfg(tmp_path, max_tokens=1500), "c1")
    new = regime_payload(_cfg(tmp_path, max_tokens=4000), "c1")
    assert ("frozen.model.max_tokens", 1500, 4000) in diff_payload(old, new)


def test_guard_raises_on_mismatch():
    with pytest.raises(RegimeMismatch):
        guard_compare("aaaa", "bbbb")
    guard_compare("aaaa", "aaaa")


def test_ledger_roundtrip(tmp_path):
    led = RegimeLedger(tmp_path / "regime_log.jsonl")
    led.record(version="v2", changed=[("frozen.model.max_tokens", 1500, 4000)],
               why="Qwen truncated", impact="re-baseline all", author="reviewer",
               timestamp="2026-06-25T00:00:00Z", regime="abc123def456")
    assert led.entries()[0]["version"] == "v2"
    assert led.entries()[0]["changed"][0]["new"] == 4000


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


def test_hash_changes_when_evaluated_item_content_changes(tmp_path):
    cfg = _cfg(tmp_path)
    truth = {"id1": "10"}
    a = regime_hash(regime_payload(cfg, "c1", truth, {"id1": {"question": "2+2"}}))
    b = regime_hash(regime_payload(cfg, "c1", truth, {"id1": {"question": "3+3"}}))
    assert a != b


def test_item_fingerprint_rejects_non_json_content(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(TypeError, match="JSON-serializable"):
        regime_payload(cfg, "c1", {"id1": "10"}, {"id1": object()})


def test_core_scoring_fingerprint_changes_with_active_source(tmp_path, monkeypatch):
    import ratchet.regime as regime
    source = tmp_path / "active_objective.py"
    source.write_text("SCORE = 1\n")
    monkeypatch.setattr(regime, "_core_source_paths", lambda config: [source])
    cfg = _cfg(tmp_path)
    a = regime_hash(regime_payload(cfg, "c1"))
    source.write_text("SCORE = 2\n")
    b = regime_hash(regime_payload(cfg, "c1"))
    assert a != b


def test_truth_none_is_backward_compatible(tmp_path):
    # omitting truth must not perturb the hash relative to explicitly passing None
    cfg = _cfg(tmp_path)
    assert regime_hash(regime_payload(cfg, "c1")) == regime_hash(regime_payload(cfg, "c1", None))


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


def test_enforce_regime_blocks_on_relabeled_truth(tmp_path):
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


def test_enforce_regime_blocks_on_changed_item_content(tmp_path):
    from ratchet.regime_state import enforce_regime
    class _P:
        def __init__(self, cfg): self.config = cfg
    cfg = _cfg(tmp_path)
    ledger = tmp_path / "regime_log.jsonl"
    truth = {"id1": "10"}
    enforce_regime(_P(cfg), "c1", ledger, truth, {"id1": {"question": "2+2"}})
    with pytest.raises(SystemExit) as exc:
        enforce_regime(_P(cfg), "c1", ledger, truth, {"id1": {"question": "3+3"}})
    assert exc.value.code == 2


def test_ledger_entry_carries_regime(tmp_path):
    from ratchet.regime import RegimeLedger
    led = RegimeLedger(tmp_path / "log.jsonl")
    led.record(version="v1", changed=[("f", 1, 2)], why="w", impact="i",
               author="a", timestamp="t", regime="abc123def456")
    assert led.entries()[0]["regime"] == "abc123def456"


def test_ledger_corrupt_line_raises_with_line_number(tmp_path):
    from ratchet.regime import RegimeLedger
    p = tmp_path / "log.jsonl"
    p.write_text('{"version": "v1"}\nnot json{{{\n')
    with pytest.raises(ValueError, match=r"log\.jsonl:2"):
        RegimeLedger(p).entries()
