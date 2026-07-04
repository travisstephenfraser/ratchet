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
    _, truth = proj.ingest()
    h = enforce_regime(proj, "c1", tmp_path / "regime_log.jsonl", truth)
    assert (tmp_path / ".regime").exists() and len(h) == 12


def test_silent_change_blocks_then_passes_after_bump(tmp_path):
    ledger = tmp_path / "regime_log.jsonl"
    proj1 = _proj(tmp_path, 1500)
    _, truth = proj1.ingest()
    enforce_regime(proj1, "c1", ledger, truth)                # establish baseline
    proj2 = _proj(tmp_path, 4000)                             # silent frozen-param change
    with pytest.raises(SystemExit) as exc:
        enforce_regime(proj2, "c1", ledger, truth)
    assert exc.value.code == 2
    record_bump(proj2, "c1", why="Qwen truncated", impact="re-baseline",
                author="reviewer", timestamp="2026-06-25T00:00:00Z", ledger_path=ledger, truth=truth)
    assert enforce_regime(proj2, "c1", ledger, truth)         # now unblocked


def test_record_bump_rewrites_regime_and_anchors(tmp_path):
    import json as _json
    from ratchet.regime import regime_payload, regime_hash, RegimeLedger
    ledger = tmp_path / "regime_log.jsonl"
    proj1 = _proj(tmp_path, 1500)
    _, truth = proj1.ingest()
    enforce_regime(proj1, "c1", ledger, truth)
    proj2 = _proj(tmp_path, 4000)                              # frozen-param change
    record_bump(proj2, "c1", why="w", impact="i", author="a",
                timestamp="t", ledger_path=ledger, truth=truth)
    new_hash = regime_hash(regime_payload(proj2.config, "c1", truth))
    on_disk = _json.loads((tmp_path / ".regime").read_text())
    assert regime_hash(on_disk) == new_hash                    # .regime advanced in-call
    assert RegimeLedger(ledger).entries()[-1]["regime"] == new_hash


def test_record_bump_recovers_from_corrupt_regime_file(tmp_path, capsys):
    ledger = tmp_path / "regime_log.jsonl"
    proj = _proj(tmp_path, 1500)
    _, truth = proj.ingest()
    enforce_regime(proj, "c1", ledger, truth)
    (tmp_path / ".regime").write_text("GARBAGE{{{")
    record_bump(proj, "c1", why="restore", impact="re-anchor", author="a",
                timestamp="t", ledger_path=ledger, truth=truth)
    assert "corrupt" in capsys.readouterr().err.lower()        # warned, not crashed
    assert enforce_regime(proj, "c1", ledger, truth)           # clean again


def test_first_run_appends_initial_baseline_anchor(tmp_path):
    from ratchet.regime import RegimeLedger
    ledger = tmp_path / "regime_log.jsonl"
    proj = _proj(tmp_path, 1500)
    _, truth = proj.ingest()
    h = enforce_regime(proj, "c1", ledger, truth)
    entries = RegimeLedger(ledger).entries()
    assert entries[-1]["regime"] == h
    assert entries[-1]["why"] == "initial baseline"
    assert entries[-1]["author"] == "auto"


def test_missing_regime_on_established_project_fails_closed(tmp_path, capsys):
    ledger = tmp_path / "regime_log.jsonl"
    proj = _proj(tmp_path, 1500)
    _, truth = proj.ingest()
    enforce_regime(proj, "c1", ledger, truth)              # baseline + auto-anchor
    (tmp_path / ".regime").unlink()
    with pytest.raises(SystemExit) as exc:
        enforce_regime(proj, "c1", ledger, truth)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "missing" in err and "regime_cli" in err        # names the unblock


def test_corrupt_regime_fails_closed_not_traceback(tmp_path, capsys):
    ledger = tmp_path / "regime_log.jsonl"
    proj = _proj(tmp_path, 1500)
    _, truth = proj.ingest()
    enforce_regime(proj, "c1", ledger, truth)
    (tmp_path / ".regime").write_text("GARBAGE{{{")
    with pytest.raises(SystemExit) as exc:
        enforce_regime(proj, "c1", ledger, truth)
    assert exc.value.code == 2
    assert "corrupt" in capsys.readouterr().err.lower()


def test_hand_edited_regime_fails_closed(tmp_path, capsys):
    import json as _json
    ledger = tmp_path / "regime_log.jsonl"
    proj = _proj(tmp_path, 1500)
    _, truth = proj.ingest()
    enforce_regime(proj, "c1", ledger, truth)
    doctored = _json.loads((tmp_path / ".regime").read_text())
    doctored["salt"] = "evil-salt"                          # valid JSON, wrong content
    (tmp_path / ".regime").write_text(_json.dumps(doctored, sort_keys=True))
    with pytest.raises(SystemExit) as exc:
        enforce_regime(proj, "c1", ledger, truth)
    assert exc.value.code == 2
    assert "anchor" in capsys.readouterr().err.lower()


def test_corrupt_ledger_fails_closed(tmp_path, capsys):
    ledger = tmp_path / "regime_log.jsonl"
    proj = _proj(tmp_path, 1500)
    _, truth = proj.ingest()
    enforce_regime(proj, "c1", ledger, truth)
    with open(ledger, "a", encoding="utf-8") as fh:
        fh.write("corrupt{{{\n")
    with pytest.raises(SystemExit) as exc:
        enforce_regime(proj, "c1", ledger, truth)
    assert exc.value.code == 2
    assert "ledger" in capsys.readouterr().err.lower()


def test_legacy_ledger_without_anchor_self_heals(tmp_path):
    import json as _json
    from ratchet.regime import RegimeLedger
    ledger = tmp_path / "regime_log.jsonl"
    proj = _proj(tmp_path, 1500)
    _, truth = proj.ingest()
    enforce_regime(proj, "c1", ledger, truth)
    # simulate a pre-anchor ledger: strip every regime field
    entries = [{k: v for k, v in e.items() if k != "regime"}
               for e in RegimeLedger(ledger).entries()]
    ledger.write_text("\n".join(_json.dumps(e) for e in entries) + "\n")
    assert enforce_regime(proj, "c1", ledger, truth)        # no block
    assert RegimeLedger(ledger).entries()[-1]["why"] == "anchor existing baseline"


def test_covered_transition_re_anchors(tmp_path):
    # A covering bump entry WITHOUT a regime field + stale .regime (old-ratchet state):
    # enforce proceeds, advances .regime, appends the anchor entry, and the NEXT run passes.
    import json as _json
    from ratchet.regime import RegimeLedger, regime_payload, regime_hash
    ledger = tmp_path / "regime_log.jsonl"
    proj1 = _proj(tmp_path, 1500)
    _, truth = proj1.ingest()
    enforce_regime(proj1, "c1", ledger, truth)
    proj2 = _proj(tmp_path, 4000)
    record_bump(proj2, "c1", why="w", impact="i", author="a",
                timestamp="t", ledger_path=ledger, truth=truth)
    # rewind to old-ratchet state: strip regime fields, restore the stale baseline
    entries = [{k: v for k, v in e.items() if k != "regime"}
               for e in RegimeLedger(ledger).entries()]
    ledger.write_text("\n".join(_json.dumps(e) for e in entries) + "\n")
    old_payload = regime_payload(proj1.config, "c1", truth)   # proj1 holds the OLD config
    (tmp_path / ".regime").write_text(_json.dumps(old_payload, sort_keys=True))
    assert enforce_regime(proj2, "c1", ledger, truth)         # covered -> proceeds
    new_hash = regime_hash(regime_payload(proj2.config, "c1", truth))
    last = RegimeLedger(ledger).entries()[-1]
    assert last["why"] == "anchor sanctioned bump" and last["regime"] == new_hash
    assert enforce_regime(proj2, "c1", ledger, truth)         # NEXT run passes
