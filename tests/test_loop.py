import pytest

from ratchet.adapter import Unparseable, UnscorableCandidate
from ratchet.loop import cand_id, better, hill_climb, escalate, run_candidate_over
from ratchet.objectives.within_tol import WithinTol


class _Project:
    def __init__(self):
        self.objective = WithinTol(tol=0.5)
        self.config = type("C", (), {"guards": {"anomaly_at": 0.95, "overfit_gap": 0.10,
                                                   "baseline": {"train": 0.0, "holdout": 0.0}},
                                     "salt": "t", "holdout_pct": 30})()
        self.base_candidate = "grade"
        self.mutations = [("be-lenient", lambda c: c + " lenient")]

    class _R:
        def run(self, candidate, item, policy=""):
            # leniency from the candidate OR an active policy constraint
            return 10 if ("lenient" in candidate or "lenient" in policy) else 8
    runner = _R()


def test_cand_id_content_hash():
    assert cand_id("abc") == cand_id("abc") != cand_id("abd")
    assert len(cand_id("abc")) == 10


def test_better_direction_aware():
    assert better(0.9, 0.8, "max") and not better(0.8, 0.9, "max")
    assert better(1.0, 2.0, "min")


def test_none_prediction_raises():
    class _P(_Project):
        class _R:
            def run(self, c, i, policy=""):
                return None
        runner = _R()
    import pytest
    with pytest.raises(ValueError):
        run_candidate_over(_P(), "x", ["a"], {"a": {}})


def test_unparseable_demoted_to_miss():
    # A runner that raises Unparseable (the fail-loud contract) is demoted to a per-item
    # miss, not an aborted run — one bad frame must not kill the hill-climb.
    class _P(_Project):
        class _R:
            def run(self, candidate, item, policy=""):
                if item.get("boom"):
                    raise Unparseable("no direction parseable")
                return 8
        runner = _R()
    preds = run_candidate_over(_P(), "x", ["ok", "bad"], {"ok": {}, "bad": {"boom": True}})
    assert preds == {"ok": "8"}   # 'bad' dropped as a miss; no exception propagated


def test_non_parse_exception_propagates():
    # A NON-Unparseable error (transport, a harness bug) is NOT a candidate property and
    # must abort the run loudly — it may never be laundered into a miss.
    class _P(_Project):
        class _R:
            def run(self, candidate, item, policy=""):
                raise RuntimeError("connection refused")
        runner = _R()
    import pytest
    with pytest.raises(RuntimeError, match="connection refused"):
        run_candidate_over(_P(), "x", ["a"], {"a": {}})


def test_max_miss_rate_guard_halts_systematic_failure():
    # Opt-in guard: a known-good candidate missing above max_miss_rate is a broken
    # model/harness, not a few bad frames -> halt for review (ValueError).
    class _P(_Project):
        class _R:
            def run(self, candidate, item, policy=""):
                raise Unparseable("garbage")
        runner = _R()
    import pytest
    with pytest.raises(ValueError, match="systematic parse failure"):
        run_candidate_over(_P(), "x", ["a", "b"], {"a": {}, "b": {}}, max_miss_rate=0.5)


def test_max_miss_rate_off_by_default():
    # Without the guard the same all-miss candidate just returns empty preds (scores 0):
    # every existing project keeps its one-bad-frame-tolerant behavior unchanged.
    class _P(_Project):
        class _R:
            def run(self, candidate, item, policy=""):
                raise Unparseable("garbage")
        runner = _R()
    assert run_candidate_over(_P(), "x", ["a", "b"], {"a": {}, "b": {}}) == {}


def test_hill_climb_finds_mutation():
    best = hill_climb(_Project(), ["a", "b"], {"a": {}, "b": {}}, {"a": "10", "b": "10"},
                      rounds=3, patience=2, regime="r1")
    assert "lenient" in best["instructions"] and best["metrics"]["objective"] == 1.0


def test_policy_constraint_changes_prediction():
    # base candidate is strict, but an active policy makes the runner lenient -> perfect
    best = hill_climb(_Project(), ["a", "b"], {"a": {}, "b": {}}, {"a": "10", "b": "10"},
                      rounds=1, patience=1, policy="be lenient", regime="r1")
    assert best["metrics"]["objective"] == 1.0


def test_escalate_misaligned_items_raises(tmp_path):
    """Zero-coverage case: items dict shares no keys with the splits -> ValueError."""
    import pytest
    log_path = tmp_path / "holdout_access.log"
    with pytest.raises(ValueError, match="0/2 train items produced predictions"):
        escalate(_Project(), {"cid": "x", "instructions": "grade lenient", "metrics": {}},
                 ["a", "b"], ["c", "d"], {},  # empty items — zero coverage
                 {"a": "10", "b": "10", "c": "10", "d": "10"},
                 log_path=log_path, regime="r1")
    assert not log_path.exists()


def test_escalate_surfaces_unscorable_mae_for_zero_train_predictions(tmp_path):
    class _P(_Project):
        def __init__(self):
            super().__init__()
            self.objective = WithinTol(tol=0.5, climb="mae")

        class _R:
            def run(self, candidate, item, policy=""):
                if item["split"] == "train":
                    raise Unparseable("garbage")
                return 10

        runner = _R()

    log_path = tmp_path / "holdout_access.log"
    with pytest.raises(UnscorableCandidate, match="zero predictions for MAE split"):
        escalate(_P(), {"cid": "x", "instructions": "grade", "metrics": {}},
                 ["a"], ["h"], {"a": {"split": "train"}, "h": {"split": "holdout"}},
                 {"a": "10", "h": "10"}, log_path=log_path, regime="r1")
    assert not log_path.exists()


def test_escalate_scores_nonempty_train_before_holdout_access(tmp_path):
    class _P(_Project):
        def __init__(self):
            super().__init__()
            self.objective = self._Objective()
            self.runner = self._Runner()

        class _Objective:
            direction = "max"

            def score(self, preds, truth, ids):
                raise ValueError("train objective failed")

        class _Runner:
            def __init__(self):
                self.calls = []

            def run(self, candidate, item, policy=""):
                self.calls.append(item["split"])
                return 10

    project = _P()
    log_path = tmp_path / "holdout_access.log"

    with pytest.raises(ValueError, match="train objective failed"):
        escalate(project, {"cid": "x", "instructions": "grade", "metrics": {}},
                 ["a"], ["h"], {"a": {"split": "train"}, "h": {"split": "holdout"}},
                 {"a": "10", "h": "10"}, log_path=log_path, regime="r1")

    assert project.runner.calls == ["train"]
    assert not log_path.exists()


def test_escalate_reuses_its_single_train_score_in_final_report(tmp_path):
    class _P(_Project):
        def __init__(self):
            super().__init__()
            self.objective = self._Objective()

        class _Objective:
            direction = "max"

            def __init__(self):
                self.calls = []

            def score(self, preds, truth, ids):
                self.calls.append(tuple(ids))
                return {"objective": 0.8 if ids == ["a"] else 0.7}

    project = _P()
    report = escalate(
        project, {"cid": "x", "instructions": "grade", "metrics": {}},
        ["a"], ["h"], {"a": {}, "h": {}}, {"a": "10", "h": "10"},
        log_path=tmp_path / "holdout_access.log", regime="r1")

    assert project.objective.calls == [("a",), ("h",)]
    assert report["train"]["objective"] == 0.8


def test_escalate_surfaces_unscorable_mae_for_zero_holdout_predictions(tmp_path):
    class _P(_Project):
        def __init__(self):
            super().__init__()
            self.objective = WithinTol(tol=0.5, climb="mae")

        class _R:
            def run(self, candidate, item, policy=""):
                if item["split"] == "holdout":
                    raise Unparseable("garbage")
                return 10

        runner = _R()

    log_path = tmp_path / "holdout_access.log"
    with pytest.raises(UnscorableCandidate, match="zero predictions for MAE split"):
        escalate(_P(), {"cid": "x", "instructions": "grade", "metrics": {}},
                 ["a"], ["h"], {"a": {"split": "train"}, "h": {"split": "holdout"}},
                 {"a": "10", "h": "10"}, log_path=log_path, regime="r1")
    assert log_path.exists()


def test_escalate_gap_gate(tmp_path):
    gate = escalate(_Project(), {"cid": "x", "instructions": "grade lenient", "metrics": {}},
                    ["a", "b"], ["c", "d"], {"a": {}, "b": {}, "c": {}, "d": {}},
                    {"a": "10", "b": "10", "c": "10", "d": "10"},
                    log_path=tmp_path / "holdout_access.log", regime="r1")
    assert gate["overfit"] is False
    assert (tmp_path / "holdout_access.log").exists()


def test_escalate_halts_on_systematic_holdout_parse_failure(tmp_path):
    import pytest
    class _P(_Project):
        def __init__(self):
            super().__init__()
            self.config.guards["max_miss_rate"] = 0.5   # __init__ sets config; set the knob here
        class _R:
            def run(self, candidate, item, policy=""):
                if item.get("holdout"):
                    raise Unparseable("model returned garbage")
                return 10
        runner = _R()
    items = {"a": {}, "b": {}, "h1": {"holdout": True}, "h2": {"holdout": True}}
    with pytest.raises(ValueError, match="systematic parse failure"):
        escalate(_P(), {"cid": "x", "instructions": "grade", "metrics": {}},
                 ["a", "b"], ["h1", "h2"], items,
                 {"a": "10", "b": "10", "h1": "10", "h2": "10"},
                 log_path=tmp_path / "holdout_access.log", regime="r1")


def test_max_miss_rate_halts_on_zero_attempts():
    import pytest
    class _P(_Project):
        class _R:
            def run(self, candidate, item, policy=""):
                return 10
        runner = _R()
    # ids present, but NONE overlap items -> attempted == 0 (a broken harness)
    with pytest.raises(ValueError, match="no items scored"):
        run_candidate_over(_P(), "x", ["a", "b"], {"zzz": {}}, max_miss_rate=0.5)


def test_run_candidate_over_stamps_regime():
    from ratchet.loop import run_candidate_over

    class _Runner:
        def run(self, candidate, item, policy=""):
            return "1"

    class _Proj:
        runner = _Runner()

    stamped = run_candidate_over(_Proj(), "cand", ["a"], {"a": {}}, regime="r1")
    assert stamped == {"a": "1"}
    assert stamped.regime == "r1"
    unstamped = run_candidate_over(_Proj(), "cand", ["a"], {"a": {}})
    assert getattr(unstamped, "regime", None) is None


def test_eval_forwards_generated_regime_to_scoring(monkeypatch):
    import ratchet.loop as loop
    from ratchet.verifier import Predictions

    captured = {}

    def run(*args, regime=None, **kwargs):
        captured["generated"] = regime
        return Predictions({"a": "10"}, regime=regime)

    def score(*args, expected_regime, **kwargs):
        captured["expected"] = expected_regime
        return {"objective": 1.0, "anomaly": False, "split_coverage": 1.0}

    monkeypatch.setattr(loop, "run_candidate_over", run)
    monkeypatch.setattr(loop, "score_split", score)

    loop._eval(_Project(), "candidate", ["a"], {"a": {}}, {"a": "10"}, "", "r1",
               None, "base")

    assert captured == {"generated": "r1", "expected": "r1"}


@pytest.mark.parametrize("regime", [None, "", "  "])
def test_hill_climb_requires_nonblank_regime(regime):
    with pytest.raises(ValueError, match="regime"):
        hill_climb(_Project(), ["a"], {"a": {}}, {"a": "10"},
                   rounds=1, patience=1, regime=regime)


def test_escalate_regenerates_both_splits_and_forwards_one_regime(tmp_path, monkeypatch):
    import ratchet.loop as loop
    from ratchet.verifier import Predictions

    calls, captured = [], {}

    def run(project, candidate, ids, items, policy="", *, max_miss_rate=None, regime=None):
        calls.append((candidate, tuple(ids), regime))
        return Predictions({item_id: "10" for item_id in ids}, regime=regime)

    class Builder:
        def __init__(self, truth, train, holdout, objective, guards, *, expected_regime,
                     allow_unstamped=False):
            captured["expected"] = expected_regime

        def score_train(self, preds):
            captured.update(train_type=type(preds), train_stamp=preds.regime)

        def finish(self, preds):
            captured.update(type=type(preds), stamp=preds.regime)
            return {"gap": 0.0, "anomaly": False, "overfit": False, "regressed": False}

    monkeypatch.setattr(loop, "run_candidate_over", run)
    monkeypatch.setattr(loop, "_GapReportBuilder", Builder)
    best = {"cid": "winner", "instructions": "grade lenient", "metrics": {}}

    escalate(_Project(), best, ["a"], ["b"], {"a": {}, "b": {}},
             {"a": "10", "b": "10"}, tmp_path / "access.log", regime="r1")

    assert calls == [("grade lenient", ("a",), "r1"),
                     ("grade lenient", ("b",), "r1")]
    assert captured == {
        "expected": "r1",
        "train_type": Predictions,
        "train_stamp": "r1",
        "type": Predictions,
        "stamp": "r1",
    }


@pytest.mark.parametrize("regime", [None, "", "  "])
def test_escalate_requires_regime_before_holdout_log(tmp_path, regime):
    with pytest.raises(ValueError, match="regime"):
        escalate(_Project(), {"cid": "x", "instructions": "x", "metrics": {}},
                 ["a"], ["b"], {"a": {}, "b": {}}, {"a": "1", "b": "1"},
                 tmp_path / "access.log", regime=regime)
    assert not (tmp_path / "access.log").exists()


def test_escalate_has_no_cached_train_predictions_parameter():
    import inspect

    assert "train_preds" not in inspect.signature(escalate).parameters


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_run_candidate_revalidates_max_miss_rate(value):
    with pytest.raises(ValueError, match="max_miss_rate"):
        run_candidate_over(_Project(), "x", ["a"], {"a": {}}, max_miss_rate=value)


@pytest.mark.parametrize("value", [True, 0, 1.5])
def test_hill_climb_revalidates_rounds(value):
    with pytest.raises(ValueError, match="search.rounds"):
        hill_climb(_Project(), ["a"], {"a": {}}, {"a": "10"},
                   rounds=value, patience=1, regime="r1")


def test_hill_climb_skips_only_an_unscorable_mutation(monkeypatch, capsys):
    import ratchet.loop as loop
    project = _Project()
    project.mutations = [("empty", lambda _base: "empty")]
    original = loop._eval

    def fake_eval(project, candidate, *args, **kwargs):
        if candidate == "empty":
            raise UnscorableCandidate("zero predictions for MAE split")
        return original(project, candidate, *args, **kwargs)

    monkeypatch.setattr(loop, "_eval", fake_eval)
    best = loop.hill_climb(project, ["a", "b"], {"a": {}, "b": {}},
                           {"a": "10", "b": "10"}, rounds=1, patience=1,
                           regime="r1")
    assert best["instructions"] == project.base_candidate
    rejection = capsys.readouterr().err
    assert "empty" in rejection
    assert "zero predictions for MAE split" in rejection


def test_hill_climb_does_not_hide_unscorable_base(monkeypatch):
    import ratchet.loop as loop

    def fail(*args, **kwargs):
        raise UnscorableCandidate("zero predictions")

    monkeypatch.setattr(loop, "_eval", fail)
    with pytest.raises(UnscorableCandidate):
        loop.hill_climb(_Project(), ["a"], {"a": {}}, {"a": "10"},
                        rounds=1, patience=1, regime="r1")


def _mk_proj(coverage_runner, min_coverage):
    from ratchet.objectives.within_tol import WithinTol

    class _Proj:
        objective = WithinTol(tol=0.5, climb="mae")
        base_candidate = "base"
        mutations = [("drop", lambda s: s + "-mutated")]
        runner = coverage_runner
        config = type("C", (), {
            "guards": {"anomaly_at": -1.0, "overfit_gap": 0.9,
                       "min_coverage": min_coverage},
            "search": {"rounds": 1, "patience": 1},
        })()
    return _Proj()


def test_hill_climb_halts_on_low_coverage_base():
    from ratchet.loop import hill_climb

    class _R:  # answers only 1 of 4 items regardless of candidate
        def run(self, candidate, item, policy=""):
            from ratchet.adapter import Unparseable
            if item["k"] != 0:
                raise Unparseable("skip")
            return "10"

    proj = _mk_proj(_R(), min_coverage=0.5)
    ids = [f"i{k}" for k in range(4)]
    items = {i: {"k": k} for k, i in enumerate(ids)}
    truth = {i: "10" for i in ids}
    import pytest
    with pytest.raises(ValueError, match="coverage"):
        hill_climb(proj, ids, items, truth, rounds=1, patience=1, regime="r1")


def test_low_coverage_mutation_cannot_win():
    from ratchet.loop import hill_climb

    class _R:  # base answers everything imperfectly; the mutation answers ONE item
        def run(self, candidate, item, policy=""):  # perfectly (mae=0.0 -> would win)
            from ratchet.adapter import Unparseable
            if candidate.endswith("-mutated"):
                if item["k"] != 0:
                    raise Unparseable("skip")
                return "10"
            return "11"                              # base: mae=1.0, full coverage

    proj = _mk_proj(_R(), min_coverage=0.5)
    ids = [f"i{k}" for k in range(4)]
    items = {i: {"k": k} for k, i in enumerate(ids)}
    truth = {i: "10" for i in ids}
    best = hill_climb(proj, ids, items, truth, rounds=1, patience=1, regime="r1")
    assert best["instructions"] == "base"            # mae 0.0 mutation was disqualified
    assert best["metrics"]["mae"] == 1.0
