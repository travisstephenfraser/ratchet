from loop_eval.config import load_config


def test_load_config_parses_objective_and_levers(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "project: toy\nversion: v1\nsalt: toy-v1\nholdout_pct: 30\n"
        "runner: runner.py:Runner\ningest: ingest.py:ingest\n"
        "mutations: mutations.py:MUTATIONS\nbase_candidate: base.txt\n"
        "objective: {name: within_tol, params: {tol: 2.0}}\n"
        "guards: {anomaly_at: 0.95, overfit_gap: 0.10}\n"
        "search: {rounds: 6, patience: 3}\n"
        "model: {endpoint: 'http://x', name: m, temperature: 0, max_tokens: 4000}\n"
        "bench: {candidates: [a, b], eval_set: data/eval_set.txt}\n"
    )
    cfg = load_config(tmp_path)
    assert cfg.project == "toy"
    assert cfg.salt == "toy-v1"
    assert cfg.holdout_pct == 30
    assert cfg.objective.name == "within_tol"
    assert cfg.objective.params["tol"] == 2.0
    assert not hasattr(cfg.objective, "direction")
    assert cfg.guards["overfit_gap"] == 0.10
    assert cfg.project_dir == tmp_path
