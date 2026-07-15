from ratchet.bench import bench, load_eval_ids
from ratchet.objectives.within_tol import WithinTol


class _Project:
    def __init__(self, tmp_path):
        self.objective = WithinTol(tol=0.5)
        self.config = type("C", (), {
            "guards": {"anomaly_at": 0.95, "overfit_gap": 0.10},
            "salt": "t", "holdout_pct": 30, "project_dir": tmp_path,
            "runner": "runner.py:Runner",  # source fingerprint reads this (missing -> deterministic)
            "objective": type("O", (), {"name": "within_tol", "params": {"tol": 0.5}})(),
            "model": {"name": "m"}, "bench": {"eval_set": "data/eval_set.txt"},
        })()

    class _R:
        def run(self, candidate, item, policy=""):
            return 10 if candidate == "good" else 5
    runner = _R()


def test_bench_ranks_same_regime(tmp_path):
    proj = _Project(tmp_path)
    rows = bench(proj, ["bad", "good"], ["a", "b"], {"a": {}, "b": {}},
                 {"a": "10", "b": "10"}, constraints_version="c1")
    assert rows[0]["candidate"] == "good" and rows[0]["objective"] == 1.0
    assert rows[1]["objective"] == 0.0
    assert rows[0]["regime"] == rows[1]["regime"]


def test_bench_regime_includes_truth_fingerprint(tmp_path):
    # The bench regime must be the same truth-inclusive hash enforce_regime baselines,
    # so a sanctioned relabel changes it instead of silently pooling bench rows.
    from ratchet.regime import regime_payload, regime_hash
    proj = _Project(tmp_path)
    truth = {"a": "10", "b": "10"}
    items = {"a": {}, "b": {}}
    rows = bench(proj, ["good"], ["a", "b"], items, truth, constraints_version="c1")
    assert rows[0]["regime"] == regime_hash(regime_payload(proj.config, "c1", truth, items))
    relabeled = bench(proj, ["good"], ["a", "b"], items,
                      {"a": "10", "b": "99"}, constraints_version="c1")
    assert relabeled[0]["regime"] != rows[0]["regime"]


def test_load_eval_ids_reads_frozen_set(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "eval_set.txt").write_text("a\nc\n")
    proj = _Project(tmp_path)
    assert load_eval_ids(proj, {"a": "1", "b": "1", "c": "1"}) == ["a", "c"]


def test_load_eval_ids_fails_when_configured_set_is_missing(tmp_path):
    proj = _Project(tmp_path)  # no eval_set file
    import pytest
    with pytest.raises(FileNotFoundError, match="configured bench eval set"):
        load_eval_ids(proj, {"a": "1", "b": "1"})


def test_load_eval_ids_uses_all_only_when_eval_set_is_explicitly_unset(tmp_path):
    proj = _Project(tmp_path)
    proj.config.bench["eval_set"] = None
    assert sorted(load_eval_ids(proj, {"a": "1", "b": "1"})) == ["a", "b"]


def test_load_eval_ids_rejects_blank_or_absent_configuration(tmp_path):
    import pytest
    proj = _Project(tmp_path)
    proj.config.bench["eval_set"] = ""
    with pytest.raises(ValueError, match="must be a path or explicit null"):
        load_eval_ids(proj, {"a": "1"})
    del proj.config.bench["eval_set"]
    with pytest.raises(ValueError, match="must be a path or explicit null"):
        load_eval_ids(proj, {"a": "1"})


def test_load_eval_ids_rejects_empty_or_unknown_configured_set(tmp_path):
    import pytest
    (tmp_path / "data").mkdir()
    eval_set = tmp_path / "data" / "eval_set.txt"
    proj = _Project(tmp_path)
    eval_set.write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        load_eval_ids(proj, {"a": "1"})
    eval_set.write_text("unknown\n")
    with pytest.raises(ValueError, match="unknown ids"):
        load_eval_ids(proj, {"a": "1"})


def test_bench_raises_when_a_candidate_produces_no_predictions(tmp_path):
    import pytest
    from ratchet.adapter import Unparseable
    proj = _Project(tmp_path)
    class _R:
        def run(self, candidate, item, policy=""):
            raise Unparseable("garbage")
    proj.runner = _R()
    with pytest.raises(ValueError, match="0/2 .* produced predictions"):
        bench(proj, ["cand"], ["a", "b"], {"a": {}, "b": {}}, {"a": "10", "b": "10"}, "c1")
