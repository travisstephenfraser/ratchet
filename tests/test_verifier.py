import pytest
from ratchet.verifier import split_ids, score_split, gap_report, load_column
from ratchet.objectives.within_tol import WithinTol


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


def test_score_split_slices_truth_to_ids():
    class _Snoop:
        direction = "max"
        seen_keys = None
        def score(self, preds, truth, ids):
            _Snoop.seen_keys = set(truth)
            return {"objective": 1.0}

    truth = {"a": "1", "b": "1", "HOLDOUT": "9"}
    score_split({"a": "1"}, truth, ["a", "b"], _Snoop(), anomaly_at=0.98)
    # the objective must NOT see the holdout label during a train-split scoring call
    assert _Snoop.seen_keys == {"a", "b"}
    assert "HOLDOUT" not in _Snoop.seen_keys


def test_gap_report_rejects_overlapping_splits():
    truth = {"a": "10", "b": "10", "c": "10"}
    preds = {"a": "10", "b": "10", "c": "10"}
    with pytest.raises(ValueError, match="disjoint"):
        gap_report(preds, truth, ["a", "b"], ["b", "c"], WithinTol(tol=0.5),
                   {"anomaly_at": 0.98, "overfit_gap": 0.25})


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


def test_predictions_carries_regime_and_acts_like_a_dict():
    from ratchet.verifier import Predictions
    p = Predictions({"a": "1"}, regime="r1")
    assert p == {"a": "1"} and p["a"] == "1" and len(p) == 1
    assert p.regime == "r1"
    assert Predictions().regime is None
    assert getattr({"a": "1"}, "regime", None) is None  # plain dicts stay valid inputs


def test_score_split_refuses_cross_regime_when_expected_given():
    from ratchet.verifier import Predictions
    from ratchet.regime import RegimeMismatch
    preds = Predictions({"a": "10"}, regime="r-old")
    with pytest.raises(RegimeMismatch):
        score_split(preds, {"a": "10"}, ["a"], WithinTol(tol=0.5),
                    anomaly_at=0.98, expected_regime="r-new")


def test_score_split_matching_regime_is_silent():
    import warnings
    from ratchet.verifier import Predictions
    preds = Predictions({"a": "10"}, regime="r1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = score_split(preds, {"a": "10"}, ["a"], WithinTol(tol=0.5),
                        anomaly_at=0.98, expected_regime="r1")
    assert m["objective"] == 1.0 and not caught


def test_score_split_warns_on_unstamped_when_expected_given():
    with pytest.warns(UserWarning, match="no regime stamp"):
        m = score_split({"a": "10"}, {"a": "10"}, ["a"], WithinTol(tol=0.5),
                        anomaly_at=0.98, expected_regime="r-new")
    assert m["objective"] == 1.0  # allowed-but-loud, mirrors the CLI legacy posture


def test_score_split_without_expected_never_checks():
    from ratchet.verifier import Predictions
    preds = Predictions({"a": "10"}, regime="r-old")  # mismatched stamp, no expectation
    m = score_split(preds, {"a": "10"}, ["a"], WithinTol(tol=0.5), anomaly_at=0.98)
    assert m["objective"] == 1.0  # today's behavior, zero breakage


def test_gap_report_guards_once():
    import warnings
    from ratchet.verifier import Predictions
    from ratchet.regime import RegimeMismatch
    truth = {"a": "10", "b": "10"}
    guards = {"anomaly_at": 0.98, "overfit_gap": 0.25}
    stamped = Predictions(truth, regime="r-old")
    with pytest.raises(RegimeMismatch):
        gap_report(stamped, truth, ["a"], ["b"], WithinTol(tol=0.5), guards,
                   expected_regime="r-new")
    # unstamped preds warn EXACTLY once (top of gap_report, not per internal score_split)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gap_report(dict(truth), truth, ["a"], ["b"], WithinTol(tol=0.5), guards,
                   expected_regime="r-new")
    assert len([w for w in caught if "no regime stamp" in str(w.message)]) == 1


def test_load_column_returns_stamped_predictions(tmp_path):
    from ratchet.verifier import preds_regime_header
    p = tmp_path / "preds.csv"
    p.write_text(preds_regime_header("abc123def456") + "anon_id,direction\nid1,DOWNHILL\n")
    preds = load_column(p)
    assert preds == {"id1": "DOWNHILL"}
    assert preds.regime == "abc123def456"


def test_load_column_unstamped_file_has_no_regime(tmp_path):
    p = tmp_path / "legacy.csv"
    p.write_text("anon_id,direction\nid1,UPHILL\n")
    preds = load_column(p)
    assert preds == {"id1": "UPHILL"}
    assert preds.regime is None
