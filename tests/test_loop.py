from ratchet.adapter import Unparseable
from ratchet.loop import cand_id, better, hill_climb, escalate, run_candidate_over
from ratchet.objectives.within_tol import WithinTol


class _Project:
    def __init__(self):
        self.objective = WithinTol(tol=0.5)
        self.config = type("C", (), {"guards": {"anomaly_at": 0.95, "overfit_gap": 0.10},
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
                      rounds=3, patience=2)
    assert "lenient" in best["instructions"] and best["metrics"]["objective"] == 1.0


def test_policy_constraint_changes_prediction():
    # base candidate is strict, but an active policy makes the runner lenient -> perfect
    best = hill_climb(_Project(), ["a", "b"], {"a": {}, "b": {}}, {"a": "10", "b": "10"},
                      rounds=1, patience=1, policy="be lenient")
    assert best["metrics"]["objective"] == 1.0


def test_escalate_misaligned_items_raises(tmp_path):
    """Zero-coverage case: items dict shares no keys with the splits -> ValueError."""
    import pytest
    with pytest.raises(ValueError, match="0/2 train items produced predictions"):
        escalate(_Project(), {"cid": "x", "instructions": "grade lenient", "metrics": {}},
                 ["a", "b"], ["c", "d"], {},  # empty items — zero coverage
                 {"a": "10", "b": "10", "c": "10", "d": "10"},
                 log_path=tmp_path / "holdout_access.log")


def test_escalate_wrong_split_preds_raises(tmp_path):
    """Caller-supplied train_preds with keys outside train_ids -> ValueError."""
    import pytest
    with pytest.raises(ValueError, match="keys not in train_ids"):
        escalate(_Project(), {"cid": "x", "instructions": "grade lenient", "metrics": {}},
                 ["a", "b"], ["c", "d"],
                 {"a": {}, "b": {}, "c": {}, "d": {}},
                 {"a": "10", "b": "10", "c": "10", "d": "10"},
                 log_path=tmp_path / "holdout_access.log",
                 train_preds={"c": "10", "d": "10"})  # holdout keys passed as train_preds


def test_escalate_gap_gate(tmp_path):
    gate = escalate(_Project(), {"cid": "x", "instructions": "grade lenient", "metrics": {}},
                    ["a", "b"], ["c", "d"], {"a": {}, "b": {}, "c": {}, "d": {}},
                    {"a": "10", "b": "10", "c": "10", "d": "10"},
                    log_path=tmp_path / "holdout_access.log")
    assert gate["overfit"] is False
    assert (tmp_path / "holdout_access.log").exists()
