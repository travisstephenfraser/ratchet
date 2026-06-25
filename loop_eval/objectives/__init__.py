"""Objective registry. within_tol/prf1 take pure params; judge needs a live judge_fn
injected by the project loader (not constructible from YAML), so it is registered for
discoverability but the loader special-cases its construction."""
from .within_tol import Objective, WithinTol
from .prf1 import PRF1

_REGISTRY = {"within_tol": WithinTol}
_REGISTRY["prf1"] = PRF1


def get_objective(name, params):
    if name not in _REGISTRY:
        raise KeyError(f"unknown objective {name!r}; built-ins: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**(params or {}))
