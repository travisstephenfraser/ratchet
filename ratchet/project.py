"""Resolve a project directory into live adapters. judge gets its judge_fn injected
from the project's judge.py (YAML can't carry a callable); a custom objective loads
from the project's objective.py; otherwise the built-in registry is used."""
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import load_config
from .objectives import get_objective
from .objectives.judge import Judge


def _import_file(path, modname):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod  # register before exec so dataclasses/pickle resolve
    spec.loader.exec_module(mod)
    return mod


def _resolve(project_dir, ref):
    filename, attr = ref.split(":")
    modname = f"_proj_{project_dir.name}_{filename.replace('.', '_')}"
    return getattr(_import_file(project_dir / filename, modname), attr)


@dataclass
class Project:
    config: object
    runner: object
    ingest: object
    mutations: list
    base_candidate: str
    objective: object


def load_project(project_dir):
    project_dir = Path(project_dir)
    cfg = load_config(project_dir)
    name = cfg.objective.name
    if name == "judge":
        judge_fn = _resolve(project_dir, "judge.py:judge_fn")
        objective = Judge(judge_fn=judge_fn, **(cfg.objective.params or {}))
    elif name == "custom":
        objective = _resolve(project_dir, "objective.py:make_objective")(cfg.objective.params)
    else:
        objective = get_objective(name, cfg.objective.params)
    return Project(
        config=cfg,
        runner=_resolve(project_dir, cfg.runner)(),
        ingest=_resolve(project_dir, cfg.ingest),
        mutations=_resolve(project_dir, cfg.mutations),
        base_candidate=(project_dir / cfg.base_candidate).read_text(),
        objective=objective,
    )
