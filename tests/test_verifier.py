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


@pytest.mark.parametrize("holdout_pct", [0, 100, -1, 101])
def test_split_rejects_invalid_holdout_percentage(holdout_pct):
    with pytest.raises(ValueError, match="holdout_pct"):
        split_ids(["a", "b"], "salt", holdout_pct)


@pytest.mark.parametrize("value", [True, 0, 100, 30.5, float("nan")])
def test_split_ids_revalidates_holdout_for_direct_callers(value):
    with pytest.raises(ValueError, match="holdout_pct"):
        split_ids(["a"], "salt", value)


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
                   {"anomaly_at": 0.95, "overfit_gap": 0.10,
                    "baseline": {"train": 0.0, "holdout": 0.0}})
    assert r["gap"] == 1.0
    assert r["overfit"] is True
    assert r["anomaly"] is True  # train within_tol 1.0 > 0.95 -> surfaced at top level


@pytest.mark.parametrize("overfit_gap", [
    float("nan"), float("inf"), float("-inf"), -0.1,
])
def test_gap_report_revalidates_overfit_gap_for_direct_callers(overfit_gap):
    truth = {"a": "10", "b": "10", "c": "10", "d": "10"}
    preds = {"a": "10", "b": "10", "c": "20", "d": "20"}
    guards = {
        "anomaly_at": 0.95,
        "overfit_gap": overfit_gap,
        "baseline": {"train": 0.0, "holdout": 0.0},
    }
    with pytest.raises(ValueError, match=r"guards\.overfit_gap"):
        gap_report(preds, truth, ["a", "b"], ["c", "d"],
                   WithinTol(tol=0.5), guards)


def test_holdout_only_anomaly_is_surfaced_at_top_level():
    obj = WithinTol(tol=0.5)
    truth = {"a": "10", "b": "10", "h": "10"}
    preds = {"a": "10", "h": "10"}  # train 0.5, holdout 1.0
    result = gap_report(preds, truth, ["a", "b"], ["h"], obj,
                        {"anomaly_at": 0.95, "overfit_gap": 0.10,
                         "baseline": {"train": 0.0, "holdout": 0.0}})
    assert result["train_anomaly"] is False
    assert result["holdout_anomaly"] is True
    assert result["anomaly"] is True


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


def test_score_split_rejects_scalar_objective_result():
    class _Scalar:
        direction = "max"

        def score(self, preds, truth, ids):
            return 1.0

    with pytest.raises(ValueError, match="mapping"):
        score_split({"a": "1"}, {"a": "1"}, ["a"], _Scalar(), anomaly_at=0.95)


def test_score_split_requires_objective_key():
    class _MissingObjective:
        direction = "max"

        def score(self, preds, truth, ids):
            return {"graded": 1}

    with pytest.raises(ValueError, match="requires an 'objective' key"):
        score_split({"a": "1"}, {"a": "1"}, ["a"], _MissingObjective(), anomaly_at=0.95)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "bad"])
def test_score_split_rejects_invalid_observed_objective(value):
    class Objective:
        direction = "min"

        def score(self, preds, truth, ids):
            return {"objective": value}

    with pytest.raises(ValueError, match="objective result must be a finite number"):
        score_split({"a": "1"}, {"a": "1"}, ["a"], Objective(),
                    anomaly_at=-1.0, expected_regime=None, baseline_objective=0.0)


def test_verify_cli_reports_unscorable_candidate_and_exits_two(monkeypatch, capsys, tmp_path):
    import sys
    import ratchet.verify as verify
    from ratchet.adapter import UnscorableCandidate

    config = type("Config", (), {
        "salt": "salt", "holdout_pct": 30,
        "guards": {"anomaly_at": -1.0, "baseline": {"train": 0.0}},
    })()
    project = type("Project", (), {
        "config": config,
        "objective": WithinTol(tol=0.5, climb="mae"),
        "ingest": staticmethod(lambda: ({"a": {}}, {"a": "10"})),
    })()

    monkeypatch.setattr(sys, "argv", [
        "ratchet-verify", "--project", str(tmp_path),
        "--predictions", str(tmp_path / "preds.csv"),
    ])
    monkeypatch.setattr(verify, "load_project", lambda _path: project)
    monkeypatch.setattr(verify, "current_version", lambda _path: "v1")
    monkeypatch.setattr(verify, "enforce_regime", lambda *args: "r1")
    monkeypatch.setattr(verify, "load_column", lambda _path: type(
        "Predictions", (dict,), {"regime": "r1"})())
    monkeypatch.setattr(verify, "split_ids", lambda *args: (["a"], []))
    monkeypatch.setattr(verify, "validate_baselines", lambda *args: {"train": 0.0})

    def unscorable(*args, **kwargs):
        raise UnscorableCandidate("zero predictions for MAE split")

    monkeypatch.setattr(verify, "score_split", unscorable)

    with pytest.raises(SystemExit) as exc:
        verify.main()
    assert exc.value.code == 2
    assert "unscorable candidate" in capsys.readouterr().err


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
    guards = {"anomaly_at": 0.98, "overfit_gap": 0.25,
              "baseline": {"train": 0.0, "holdout": 0.0}}
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


def test_gap_report_matching_regime_is_silent():
    import warnings
    from ratchet.verifier import Predictions
    truth = {"a": "10", "b": "10"}
    guards = {"anomaly_at": 0.98, "overfit_gap": 0.25,
              "baseline": {"train": 0.0, "holdout": 0.0}}
    preds = Predictions(truth, regime="r1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = gap_report(preds, truth, ["a"], ["b"], WithinTol(tol=0.5), guards,
                       expected_regime="r1")
    assert r["gap"] == 0.0 and not caught


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


def test_score_split_computes_split_coverage():
    m = score_split({"a": "10"}, {"a": "10", "b": "10"}, ["a", "b"],
                    WithinTol(tol=0.5), anomaly_at=0.98)
    assert m["split_coverage"] == 0.5
    assert "low_coverage" not in m                     # knob unset -> no flag key
    with pytest.raises(ValueError, match="split must not be empty"):
        score_split({}, {}, [], WithinTol(tol=0.5), anomaly_at=0.98)


def test_score_split_flags_low_coverage_when_floor_set():
    m = score_split({"a": "10"}, {"a": "10", "b": "10"}, ["a", "b"],
                    WithinTol(tol=0.5), anomaly_at=0.98, min_coverage=0.6)
    assert m["low_coverage"] is True
    ok = score_split({"a": "10", "b": "10"}, {"a": "10", "b": "10"}, ["a", "b"],
                     WithinTol(tol=0.5), anomaly_at=0.98, min_coverage=0.6)
    assert ok["low_coverage"] is False
    at_floor = score_split({"a": "10"}, {"a": "10", "b": "10"}, ["a", "b"],
                           WithinTol(tol=0.5), anomaly_at=0.98, min_coverage=0.5)
    assert at_floor["low_coverage"] is False           # floor is inclusive: == passes


def test_goodhart_one_easy_item_mae_is_flagged():
    # The hole this feature closes: 1-of-10 answered, mae computed only over the
    # graded item posts 0.0 — with the floor set, the split is flagged.
    obj = WithinTol(tol=0.5, climb="mae")
    ids = [f"i{k}" for k in range(10)]
    truth = {i: "10" for i in ids}
    m = score_split({"i0": "10"}, truth, ids, obj, anomaly_at=0.0, min_coverage=0.5)
    assert m["mae"] == 0.0 and m["low_coverage"] is True


def test_gap_report_surfaces_coverage_and_ors_the_flag():
    truth = {"a": "10", "b": "10", "c": "10", "d": "10"}
    preds = {"a": "10", "b": "10", "c": "10"}              # holdout d missing
    guards = {"anomaly_at": 1.5, "overfit_gap": 0.9, "min_coverage": 0.6,
              "baseline": {"train": 0.0, "holdout": 0.0}}
    r = gap_report(preds, truth, ["a", "b"], ["c", "d"], WithinTol(tol=0.5), guards)
    assert r["train_coverage"] == 1.0 and r["holdout_coverage"] == 0.5
    assert r["low_coverage"] is True                       # holdout below floor
    no_knob = gap_report(preds, truth, ["a", "b"], ["c", "d"], WithinTol(tol=0.5),
                         {"anomaly_at": 1.5, "overfit_gap": 0.9,
                          "baseline": {"train": 0.0, "holdout": 0.0}})
    assert "low_coverage" not in no_knob
    assert no_knob["train_coverage"] == 1.0                # informational keys always on


def test_score_split_flags_direction_aware_regression():
    max_obj = WithinTol(tol=0.5)
    max_metrics = score_split(
        {"a": "10"}, {"a": "10", "b": "10"}, ["a", "b"], max_obj,
        anomaly_at=1.5, baseline_objective=0.75,
    )
    assert max_metrics["objective"] == 0.5
    assert max_metrics["baseline_objective"] == 0.75
    assert max_metrics["objective_delta"] == -0.25
    assert max_metrics["regressed"] is True

    min_obj = WithinTol(tol=0.5, climb="mae")
    min_metrics = score_split(
        {"a": "12", "b": "12"}, {"a": "10", "b": "10"}, ["a", "b"], min_obj,
        anomaly_at=-1.0, baseline_objective=1.0,
    )
    assert min_metrics["objective"] == 2.0
    assert min_metrics["objective_delta"] == -1.0
    assert min_metrics["regressed"] is True


@pytest.mark.parametrize("baseline", [float("nan"), float("inf"), float("-inf"), True])
def test_score_split_rejects_nonfinite_or_boolean_baseline(baseline):
    with pytest.raises(ValueError, match="finite number"):
        score_split({"a": "10"}, {"a": "10"}, ["a"], WithinTol(tol=0.5),
                    anomaly_at=1.5, baseline_objective=baseline)


def test_gap_report_requires_complete_frozen_baselines():
    truth = {"a": "10", "b": "10"}
    for guards in (
        {"anomaly_at": 1.5, "overfit_gap": 0.1},
        {"anomaly_at": 1.5, "overfit_gap": 0.1, "baseline": {"train": 1.0}},
    ):
        with pytest.raises(ValueError, match="missing frozen baseline"):
            gap_report(truth, truth, ["a"], ["b"], WithinTol(tol=0.5), guards)


def test_gap_report_surfaces_regression_from_either_split():
    truth = {"a": "10", "b": "10", "c": "10", "d": "10"}
    preds = {"a": "10", "b": "10", "c": "10"}  # holdout d missing
    guards = {
        "anomaly_at": 1.5,
        "overfit_gap": 0.9,
        "baseline": {"train": 1.0, "holdout": 1.0},
    }
    result = gap_report(preds, truth, ["a", "b"], ["c", "d"],
                        WithinTol(tol=0.5), guards)
    assert result["train"]["regressed"] is False
    assert result["holdout"]["regressed"] is True
    assert result["regressed"] is True
