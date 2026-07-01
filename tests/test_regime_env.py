"""Regime fingerprint must fold in project-declared env knobs (config.regime_env)
so a structured run and a regex run can never silently pool. Pure-Python; no ML deps."""
import os
from ratchet.regime import regime_payload, regime_hash


class _Cfg:
    salt = "t"
    holdout_pct = 30
    guards = {"anomaly_at": 0.98}
    model = {"name": "m"}
    project_dir = "/nonexistent"          # eval_set fingerprint falls back to "ingest-full"
    bench = {"eval_set": ""}

    class objective:
        name = "custom"
        params = {}

    def __init__(self, regime_env):
        self.regime_env = regime_env


def _clear(*names):
    for n in names:
        os.environ.pop(n, None)


def test_declared_and_set_env_changes_hash():
    _clear("COSMOS_PARSE")
    cfg = _Cfg(["COSMOS_PARSE"])
    base = regime_hash(regime_payload(cfg, "cv1"))
    os.environ["COSMOS_PARSE"] = "structured"
    try:
        structured = regime_hash(regime_payload(cfg, "cv1"))
    finally:
        _clear("COSMOS_PARSE")
    assert base != structured, "structured must be a distinct regime from regex"


def test_undeclared_env_is_ignored():
    # Setting an env var the project did NOT declare must not perturb the fingerprint.
    _clear("COSMOS_PARSE")
    cfg = _Cfg([])                         # declares nothing
    a = regime_hash(regime_payload(cfg, "cv1"))
    os.environ["COSMOS_PARSE"] = "structured"
    try:
        b = regime_hash(regime_payload(cfg, "cv1"))
    finally:
        _clear("COSMOS_PARSE")
    assert a == b


def test_unset_declared_env_does_not_perturb():
    # A declared-but-unset knob must leave the hash equal to a project that declares
    # nothing, so adding regime_env without setting it is backward-compatible.
    _clear("COSMOS_PARSE", "COSMOS_MODE")
    with_decl = regime_hash(regime_payload(_Cfg(["COSMOS_PARSE", "COSMOS_MODE"]), "cv1"))
    without = regime_hash(regime_payload(_Cfg([]), "cv1"))
    assert with_decl == without
