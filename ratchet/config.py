"""Load and validate a project's config.yaml into a typed Config.
Direction is intentionally NOT a config field — the Objective instance owns it."""
from dataclasses import dataclass, field
from pathlib import Path
import yaml

from .validation import finite_number, mapping, nonblank, rate, whole_number


@dataclass
class ObjectiveCfg:
    name: str
    params: dict


@dataclass
class Config:
    project: str
    version: str
    salt: str
    holdout_pct: int
    runner: str
    ingest: str
    mutations: str
    base_candidate: str
    objective: ObjectiveCfg
    guards: dict
    search: dict
    model: dict
    bench: dict
    project_dir: Path
    # Names of environment variables that alter comparability and so belong in the
    # regime fingerprint (e.g. a runner's parse mode / backend / probe mode). Declared
    # per-project in config.yaml as `regime_env: [...]`; empty by default so projects
    # that don't set it keep their existing regime hash.
    regime_env: list = field(default_factory=list)


def _validate_config(raw):
    raw = mapping(raw, "config")
    objective = mapping(raw.get("objective"), "objective")
    params = mapping(objective.get("params", {}), "objective.params")
    objective["params"] = params
    guards = mapping(raw.get("guards"), "guards")
    search = mapping(raw.get("search"), "search")
    mapping(raw.get("model"), "model")
    mapping(raw.get("bench"), "bench")

    raw["holdout_pct"] = whole_number(
        raw.get("holdout_pct"), "holdout_pct", minimum=1, maximum=99)
    guards["anomaly_at"] = finite_number(
        guards.get("anomaly_at"), "guards.anomaly_at")
    guards["overfit_gap"] = finite_number(
        guards.get("overfit_gap"), "guards.overfit_gap")
    if guards["overfit_gap"] < 0:
        raise ValueError("guards.overfit_gap must be >= 0")
    for field in ("min_coverage", "max_miss_rate"):
        if field in guards:
            guards[field] = rate(guards[field], f"guards.{field}")
    if "baseline" in guards:
        mapping(guards["baseline"], "guards.baseline")

    search["rounds"] = whole_number(search.get("rounds"), "search.rounds", minimum=1)
    search["patience"] = whole_number(search.get("patience"), "search.patience", minimum=0)
    regime_env = raw.get("regime_env", [])
    if not isinstance(regime_env, list):
        raise ValueError("regime_env must be a list of unique nonblank strings")
    raw["regime_env"] = [nonblank(value, "regime_env item") for value in regime_env]
    if len(set(raw["regime_env"])) != len(raw["regime_env"]):
        raise ValueError("regime_env must not contain duplicates")

    if objective.get("name") == "within_tol":
        params["tol"] = finite_number(params.get("tol", 2.0), "objective.params.tol")
        if params["tol"] < 0:
            raise ValueError("objective.params.tol must be >= 0")
        climb = params.get("climb", "within")
        if climb not in ("within", "mae"):
            raise ValueError("objective.params.climb must be within or mae")
        params["climb"] = climb
    return raw


def load_config(project_dir) -> Config:
    project_dir = Path(project_dir)
    raw = _validate_config(yaml.safe_load((project_dir / "config.yaml").read_text()))
    obj = raw["objective"]
    return Config(
        project=raw["project"], version=raw["version"], salt=raw["salt"],
        holdout_pct=raw["holdout_pct"], runner=raw["runner"], ingest=raw["ingest"],
        mutations=raw["mutations"], base_candidate=raw["base_candidate"],
        objective=ObjectiveCfg(obj["name"], obj.get("params", {})),
        guards=raw["guards"], search=raw["search"], model=raw["model"], bench=raw["bench"],
        project_dir=project_dir,
        regime_env=raw["regime_env"],
    )
