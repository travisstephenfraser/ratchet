from loop_eval.project import load_project


def _base(d):
    (d / "base.txt").write_text("grade it")
    (d / "runner.py").write_text(
        "class Runner:\n    def run(self, candidate, item, policy=''):\n"
        "        return item['answer']\n")
    (d / "ingest.py").write_text("def ingest():\n    return {'a': {'answer': 10}}, {'a': '10'}\n")
    (d / "mutations.py").write_text("MUTATIONS = [('noop', lambda c: c)]\n")


def _cfg(d, objective_line):
    (d / "config.yaml").write_text(
        "project: toy\nversion: v1\nsalt: toy-v1\nholdout_pct: 30\n"
        "runner: runner.py:Runner\ningest: ingest.py:ingest\n"
        "mutations: mutations.py:MUTATIONS\nbase_candidate: base.txt\n"
        f"{objective_line}\n"
        "guards: {anomaly_at: 0.95, overfit_gap: 0.10}\n"
        "search: {rounds: 2, patience: 1}\n"
        "model: {name: stub, temperature: 0, max_tokens: 100}\n"
        "bench: {candidates: [a], eval_set: data/eval_set.txt}\n")


def test_load_within_tol(tmp_path):
    _base(tmp_path)
    _cfg(tmp_path, "objective: {name: within_tol, params: {tol: 2.0}}")
    proj = load_project(tmp_path)
    assert proj.base_candidate == "grade it"
    assert proj.objective.score({"a": "10"}, {"a": "10"}, ["a"])["objective"] == 1.0
    assert proj.runner.run("c", {"answer": 10}, "") == 10
    assert proj.mutations[0][0] == "noop"


def test_judge_injection(tmp_path):
    _base(tmp_path)
    _cfg(tmp_path, "objective: {name: judge, params: {rubric: welfare}}")
    (tmp_path / "judge.py").write_text(
        "def judge_fn(pred, rubric):\n    return 1.0 if rubric in pred else 0.0\n")
    proj = load_project(tmp_path)
    m = proj.objective.score({"a": "about welfare"}, {"a": "welfare"}, ["a"])
    assert m["objective"] == 1.0  # injected judge_fn ran
