"""First-class versioning. A regime is the fingerprint of everything that affects
whether two results can be compared. The eval set is tracked by CONTENT, not path,
so swapping which exams are benched changes the regime. Direction is derivable from
the objective (name+params), so it is not hashed separately."""
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

from . import __version__


CORE_SENTINELS = {"__init__.py", "regime.py", "verifier.py", "loop.py", "bench.py"}


class RegimeMismatch(Exception):
    pass


def _eval_set_fingerprint(config):
    if "eval_set" not in config.bench:
        raise ValueError("bench.eval_set must be a path or explicit null")
    configured = config.bench["eval_set"]
    if configured is None:
        return "ingest-full"
    if not isinstance(configured, str) or not configured.strip():
        raise ValueError("bench.eval_set must be a path or explicit null")
    p = Path(config.project_dir) / configured
    if p.exists():
        return hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    return "ingest-full"


def _truth_fingerprint(truth) -> str:
    """Content hash of the derived (id, label) ground truth. Fingerprints the OUTPUT of
    ingest(), so it closes the label-set, item-count, and truth-DERIVATION holes at once:
    change a relabeling constant (e.g. a gradient band) and every label flips, so this
    moves even when the id list and the config are byte-identical."""
    items = sorted((str(k), str(v)) for k, v in truth.items())
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()[:12]


def _items_fingerprint(items) -> str:
    """Canonical content hash of the actual inputs presented to the runner."""
    normalized = sorted(((str(k), v) for k, v in items.items()), key=lambda pair: pair[0])
    try:
        blob = json.dumps(normalized, sort_keys=True, ensure_ascii=False,
                          allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as e:
        raise TypeError(
            "ingested items must be JSON-serializable for a stable regime fingerprint; "
            "convert custom objects to deterministic dict/list/scalar values in ingest()"
        ) from e
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _core_root():
    return Path(__file__).parent


def _core_source_paths():
    root = _core_root()
    return sorted(root.rglob("*.py"), key=lambda p: p.relative_to(root).as_posix())


def _framed_update(digest, data):
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)


def _core_fingerprint() -> str:
    root = _core_root()
    paths = _core_source_paths()
    names = {p.relative_to(root).as_posix() for p in paths}
    missing = CORE_SENTINELS - names
    if missing:
        raise RuntimeError(
            "Ratchet requires a sourceful package; missing core Python sources: "
            f"{sorted(missing)}")
    digest = hashlib.sha256()
    _framed_update(digest, __version__.encode())
    for path in paths:
        _framed_update(digest, path.relative_to(root).as_posix().encode())
        _framed_update(digest, path.read_bytes())
    return digest.hexdigest()[:12]


def _source_fingerprint(config) -> str:
    """Hash of the source that defines what a PREDICTION and a SCORE mean: the runner
    (model call + parse) plus the project objective when it is custom/judge. ingest.py is
    NOT hashed here because truth and item fingerprints cover its outputs. Conservative:
    a formatting-only edit also bumps the regime, which is the safe over-block direction."""
    refs = [config.runner]
    if config.objective.name == "custom":
        refs.append("objective.py:make_objective")
    elif config.objective.name == "judge":
        refs.append("judge.py:judge_fn")
    h = hashlib.sha256()
    for ref in refs:
        filename = ref.split(":")[0]
        p = Path(config.project_dir) / filename
        h.update(filename.encode())
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()[:12]


def regime_payload(config, constraints_version, truth=None, items=None) -> dict:
    frozen = {
        "holdout_pct": config.holdout_pct,
        "guards": config.guards,
        "model": config.model,
        "eval_set": _eval_set_fingerprint(config),
        "logic": _source_fingerprint(config),
        "scoring_core": {
            "version": __version__,
            "logic": _core_fingerprint(),
            "runtime": {
                "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                "pyyaml": yaml.__version__,
            },
        },
    }
    # Content of the derived ground truth. Omitted when truth is not supplied (unit tests),
    # but every real entry point passes it, so a relabel or an item-set change bumps the hash.
    if truth is not None:
        frozen["truth"] = _truth_fingerprint(truth)
    if items is not None:
        frozen["items"] = _items_fingerprint(items)
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

    def record(self, *, version, changed, why, impact, author, timestamp, regime):
        entry = {"version": version, "timestamp": timestamp, "author": author,
                 "why": why, "impact": impact, "regime": regime,
                 "changed": [{"field": f, "old": o, "new": n} for f, o, n in changed]}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def entries(self):
        if not self.path.exists():
            return []
        out = []
        for lineno, line in enumerate(self.path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError as e:
                raise ValueError(f"{self.path}:{lineno}: corrupt ledger line ({e})") from e
        return out
