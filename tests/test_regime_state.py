import pytest
from ratchet.project import load_project
from ratchet.regime_state import enforce_regime, record_bump
from tests.test_project import _base, _cfg  # reuse fixtures


def _proj(tmp_path, max_tokens):
    _base(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "project: toy\nversion: v1\nsalt: toy-v1\nholdout_pct: 30\n"
        "runner: runner.py:Runner\ningest: ingest.py:ingest\n"
        "mutations: mutations.py:MUTATIONS\nbase_candidate: base.txt\n"
        "objective: {name: within_tol, params: {tol: 2.0}}\n"
        "guards: {anomaly_at: 0.95, overfit_gap: 0.10}\n"
        "search: {rounds: 2, patience: 1}\n"
        f"model: {{name: m, temperature: 0, max_tokens: {max_tokens}}}\n"
        "bench: {candidates: [a], eval_set: data/eval_set.txt}\n")
    return load_project(tmp_path)


def test_first_run_writes_regime(tmp_path):
    proj = _proj(tmp_path, 1500)
    h = enforce_regime(proj, "c1", tmp_path / "regime_log.jsonl")
    assert (tmp_path / ".regime").exists() and len(h) == 12


def test_silent_change_blocks_then_passes_after_bump(tmp_path):
    ledger = tmp_path / "regime_log.jsonl"
    enforce_regime(_proj(tmp_path, 1500), "c1", ledger)        # establish baseline
    proj2 = _proj(tmp_path, 4000)                              # silent frozen-param change
    with pytest.raises(SystemExit) as exc:
        enforce_regime(proj2, "c1", ledger)
    assert exc.value.code == 2
    record_bump(proj2, "c1", why="Qwen truncated", impact="re-baseline",
                author="reviewer", timestamp="2026-06-25T00:00:00Z", ledger_path=ledger)
    assert enforce_regime(proj2, "c1", ledger)                # now unblocked
