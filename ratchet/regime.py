"""First-class versioning. A regime is the fingerprint of everything that affects
whether two results can be compared. The eval set is tracked by CONTENT, not path,
so swapping which exams are benched changes the regime. Direction is derivable from
the objective (name+params), so it is not hashed separately."""
import hashlib
import json
import os
from pathlib import Path


class RegimeMismatch(Exception):
    pass


def _eval_set_fingerprint(config):
    p = Path(config.project_dir) / config.bench.get("eval_set", "")
    if config.bench.get("eval_set") and p.exists():
        return hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    return "ingest-full"


def regime_payload(config, constraints_version) -> dict:
    frozen = {
        "holdout_pct": config.holdout_pct,
        "guards": config.guards,
        "model": config.model,
        "eval_set": _eval_set_fingerprint(config),
    }
    # Environment knobs a project declares as regime-affecting (config.regime_env).
    # Folded in only when declared AND set, so projects that don't use it keep their
    # existing hash, and an unset knob doesn't perturb the fingerprint. This is what
    # makes e.g. COSMOS_PARSE=structured a distinct, un-poolable regime.
    env = {k: os.environ[k] for k in getattr(config, "regime_env", []) if k in os.environ}
    if env:
        frozen["env"] = env
    return {
        "salt": config.salt,
        "objective": {"name": config.objective.name, "params": config.objective.params},
        "frozen": frozen,
        "constraints_version": constraints_version,
    }


def regime_hash(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        out.update(_flatten(v, key)) if isinstance(v, dict) else out.__setitem__(key, v)
    return out


def diff_payload(old, new):
    fo, fn = _flatten(old), _flatten(new)
    keys = sorted(set(fo) | set(fn))
    return [(k, fo.get(k), fn.get(k)) for k in keys if fo.get(k) != fn.get(k)]


def guard_compare(regime_a, regime_b):
    if regime_a != regime_b:
        raise RegimeMismatch(
            f"refusing to compare across regimes: {regime_a} != {regime_b}. "
            f"Re-run under one regime, or bump the version with a ledger rationale.")


class RegimeLedger:
    def __init__(self, path):
        self.path = Path(path)

    def record(self, *, version, changed, why, impact, author, timestamp):
        entry = {"version": version, "timestamp": timestamp, "author": author,
                 "why": why, "impact": impact,
                 "changed": [{"field": f, "old": o, "new": n} for f, o, n in changed]}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def entries(self):
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
