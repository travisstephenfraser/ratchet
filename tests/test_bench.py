from ratchet.bench import bench, load_eval_ids
from ratchet.objectives.within_tol import WithinTol


class _Project:
    def __init__(self, tmp_path):
        self.objective = WithinTol(tol=0.5)
        self.config = type("C", (), {
            "guards": {"anomaly_at": 0.95, "overfit_gap": 0.10},
            "salt": "t", "holdout_pct": 30, "project_dir": tmp_path,
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


def test_load_eval_ids_reads_frozen_set(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "eval_set.txt").write_text("a\nc\n")
    proj = _Project(tmp_path)
    assert load_eval_ids(proj, {"a": "1", "b": "1", "c": "1"}) == ["a", "c"]


def test_load_eval_ids_falls_back_to_all(tmp_path):
    proj = _Project(tmp_path)  # no eval_set file
    assert sorted(load_eval_ids(proj, {"a": "1", "b": "1"})) == ["a", "b"]
