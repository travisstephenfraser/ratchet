import pytest
from loop_eval.verifier import split_ids, score_split, gap_report, load_column
from loop_eval.objectives.within_tol import WithinTol


def test_split_stable_and_disjoint():
    ids = [f"id{i}" for i in range(200)]
    a = split_ids(ids, "salt-x", 30)
    b = split_ids(list(reversed(ids)), "salt-x", 30)
    assert a == b
    train, holdout = a
    assert set(train).isdisjoint(holdout)
    assert len(train) + len(holdout) == 200
    assert 0.2 < len(holdout) / 200 < 0.4


def test_anomaly_direction_max():
    obj = WithinTol(tol=2.0)
    ids = ["a", "b", "c", "d"]
    truth = {i: "10" for i in ids}
    m = score_split({i: "10" for i in ids}, truth, ids, obj, anomaly_at=0.95)
    assert m["anomaly"] is True


def test_overfit_and_anomaly_surfaced_at_top_level():
    obj = WithinTol(tol=2.0)
    truth = {"a": "10", "b": "10", "c": "10", "d": "10"}
    preds = {"a": "10", "b": "10", "c": "20", "d": "20"}  # train 1.0, holdout 0.0
    r = gap_report(preds, truth, ["a", "b"], ["c", "d"], obj,
                   {"anomaly_at": 0.95, "overfit_gap": 0.10})
    assert r["gap"] == 1.0
    assert r["overfit"] is True
    assert r["anomaly"] is True  # train within_tol 1.0 > 0.95 -> surfaced at top level


def test_load_column_autodetect(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("anon_id,score\nx,3\ny,4\n")
    assert load_column(p) == {"x": "3", "y": "4"}


def test_score_split_rejects_bad_direction():
    class _BadObj:
        direction = "minimize"
        def score(self, preds, truth, ids):
            return {"objective": 0.5, "n": 1, "graded": 1}
    with pytest.raises(ValueError, match="unknown objective direction"):
        score_split({"a": "1"}, {"a": "1"}, ["a"], _BadObj(), anomaly_at=0.95)
