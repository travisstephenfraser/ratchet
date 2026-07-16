import json

import pytest

from ratchet.results import write_candidate, append_loop_log, write_bench
from ratchet.verifier import read_preds_regime, load_column


def test_write_candidate_stamps_regime(tmp_path):
    write_candidate(tmp_path, "abc1234567", "grade it",
                    {"x": "3", "y": "4"}, {"objective": 0.5, "mae": 1.0}, regime="reg123")
    meta = json.loads((tmp_path / "candidates" / "abc1234567.metrics.json").read_text())
    assert meta["regime"] == "reg123" and meta["cid"] == "abc1234567"
    preds = (tmp_path / "candidates" / "abc1234567.preds.csv").read_text()
    assert "anon_id,score" in preds and "x,3" in preds
    assert (tmp_path / "candidates" / "abc1234567.txt").read_text() == "grade it"


def test_append_loop_log(tmp_path):
    append_loop_log(tmp_path, "abc", "base", {"objective": 0.5, "mae": 1.0, "exact": 0.25})
    append_loop_log(tmp_path, "def", "r1:lenient", {"objective": 0.9, "mae": 0.2, "exact": 0.5})
    body = (tmp_path / "LOOP_LOG.md").read_text()
    assert "| abc | base |" in body          # cid in the first column of its row
    assert "| def | r1:lenient |" in body


def test_write_bench(tmp_path):
    p = write_bench(tmp_path, "reg9", [{"candidate": "good", "objective": 1.0}])
    assert p.name == "bench_reg9.json"
    assert json.loads(p.read_text())[0]["candidate"] == "good"


def test_result_writer_rejects_nonstandard_json(tmp_path):
    with pytest.raises(ValueError, match="valid JSON"):
        write_bench(tmp_path, "r1", [{"objective": float("nan")}])


def test_write_candidate_stamps_preds_regime(tmp_path):
    write_candidate(tmp_path, "cid1", "some instructions",
                    {"id1": "10", "id2": "8"}, {"objective": 1.0}, regime="r-abc-123")
    preds = tmp_path / "candidates" / "cid1.preds.csv"
    assert read_preds_regime(preds) == "r-abc-123"
    assert load_column(preds) == {"id1": "10", "id2": "8"}  # data still parses
