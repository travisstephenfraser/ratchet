import pytest
import yaml

from ratchet.config import load_config


def _write_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "project: toy\nversion: v1\nsalt: toy-v1\nholdout_pct: 30\n"
        "runner: runner.py:Runner\ningest: ingest.py:ingest\n"
        "mutations: mutations.py:MUTATIONS\nbase_candidate: base.txt\n"
        "objective: {name: within_tol, params: {tol: 2.0}}\n"
        "guards: {anomaly_at: 0.95, overfit_gap: 0.10}\n"
        "search: {rounds: 6, patience: 3}\n"
        "model: {endpoint: 'http://x', name: m, temperature: 0, max_tokens: 4000}\n"
        "bench: {candidates: [a, b], eval_set: data/eval_set.txt}\n"
    )
    return path


def test_load_config_parses_objective_and_levers(tmp_path):
    _write_config(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.project == "toy"
    assert cfg.salt == "toy-v1"
    assert cfg.holdout_pct == 30
    assert cfg.objective.name == "within_tol"
    assert cfg.objective.params["tol"] == 2.0
    assert not hasattr(cfg.objective, "direction")
    assert cfg.guards["overfit_gap"] == 0.10
    assert cfg.project_dir == tmp_path


@pytest.mark.parametrize("field,value", [
    ("anomaly_at", float("nan")), ("anomaly_at", float("inf")),
    ("overfit_gap", -0.1), ("min_coverage", -0.1),
    ("min_coverage", 1.1), ("max_miss_rate", 1.1),
])
def test_config_rejects_guard_values_that_disable_checks(tmp_path, field, value):
    path = _write_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["guards"][field] = value
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=rf"guards\.{field}"):
        load_config(tmp_path)


@pytest.mark.parametrize("value", [0, 100, True, 30.5])
def test_config_rejects_invalid_holdout(tmp_path, value):
    path = _write_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["holdout_pct"] = value
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="holdout_pct"):
        load_config(tmp_path)


def test_config_normalizes_accepted_numeric_strings(tmp_path):
    path = _write_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["guards"].update(anomaly_at="0.95", overfit_gap="0.1")
    raw["search"].update(rounds="6", patience="3")
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(tmp_path)
    assert cfg.guards["anomaly_at"] == 0.95
    assert cfg.guards["overfit_gap"] == 0.1
    assert cfg.search == {"rounds": 6, "patience": 3}


@pytest.mark.parametrize("section", ["objective", "guards", "search", "model", "bench"])
def test_config_rejects_non_mapping_sections(tmp_path, section):
    path = _write_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw[section] = []
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=rf"{section}.*mapping"):
        load_config(tmp_path)


def test_config_rejects_non_mapping_objective_params(tmp_path):
    path = _write_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["objective"]["params"] = []
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=r"objective\.params.*mapping"):
        load_config(tmp_path)


@pytest.mark.parametrize("value", ["PATH", ["PATH", "PATH"], ["  "], [1]])
def test_config_rejects_malformed_regime_env(tmp_path, value):
    path = _write_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["regime_env"] = value
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="regime_env"):
        load_config(tmp_path)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_config_rejects_invalid_within_tol_tolerance(tmp_path, value):
    path = _write_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["objective"]["params"]["tol"] = value
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=r"objective\.params\.tol"):
        load_config(tmp_path)


@pytest.mark.parametrize("value", ["unknown", 1])
def test_config_rejects_invalid_within_tol_climb(tmp_path, value):
    path = _write_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["objective"]["params"]["climb"] = value
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=r"objective\.params\.climb"):
        load_config(tmp_path)
