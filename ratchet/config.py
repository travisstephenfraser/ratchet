"""Load and validate a project's config.yaml into a typed Config.
Direction is intentionally NOT a config field — the Objective instance owns it."""
from dataclasses import dataclass, field
from pathlib import Path
import yaml


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


def load_config(project_dir) -> Config:
    project_dir = Path(project_dir)
    raw = yaml.safe_load((project_dir / "config.yaml").read_text())
    obj = raw["objective"]
    return Config(
        project=raw["project"], version=raw["version"], salt=raw["salt"],
        holdout_pct=int(raw["holdout_pct"]), runner=raw["runner"], ingest=raw["ingest"],
        mutations=raw["mutations"], base_candidate=raw["base_candidate"],
        objective=ObjectiveCfg(obj["name"], obj.get("params", {})),
        guards=raw["guards"], search=raw["search"], model=raw["model"], bench=raw["bench"],
        project_dir=project_dir,
        regime_env=list(raw.get("regime_env", [])),
    )
