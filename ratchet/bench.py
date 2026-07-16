"""Frozen-param bench: evaluate fixed candidates under one regime with no search.
A configured eval-set file is required and validated; an explicit null selects all
ingested ids. Every row carries the same truth-, item-, and scoring-aware regime hash."""
from collections.abc import Mapping
from pathlib import Path

from .verifier import score_split
from .loop import run_candidate_over
from .regime import regime_payload, regime_hash
from . import results


class BenchInputError(ValueError):
    pass


def _materialize(value, field):
    if value is None or isinstance(value, (str, bytes, Mapping)):
        raise BenchInputError(f"{field} must be an iterable, not {type(value).__name__}")
    try:
        return list(value)
    except TypeError as exc:
        raise BenchInputError(f"{field} must be an iterable") from exc


def normalize_candidates(value):
    candidates = _materialize(value, "bench.candidates")
    if not candidates:
        raise BenchInputError("bench requires at least one candidate")
    if any(not isinstance(candidate, str) or not candidate.strip() for candidate in candidates):
        raise BenchInputError("bench candidates must be nonblank strings")
    return candidates


def normalize_eval_ids(value, truth):
    ids = _materialize(value, "bench eval ids")
    if not ids:
        raise BenchInputError("bench eval set must not be empty")
    if any(not isinstance(item_id, str) or not item_id.strip() for item_id in ids):
        raise BenchInputError("bench eval ids must be nonblank strings")
    seen, duplicates = set(), set()
    for item_id in ids:
        if item_id in seen:
            duplicates.add(item_id)
        seen.add(item_id)
    if duplicates:
        raise BenchInputError(f"bench eval set contains duplicate ids: {sorted(duplicates)}")
    unknown = sorted(item_id for item_id in ids if item_id not in truth)
    if unknown:
        raise BenchInputError(f"bench eval set contains unknown ids: {unknown}")
    return ids


def load_eval_ids(project, truth):
    if "eval_set" not in project.config.bench:
        raise BenchInputError("bench.eval_set must be a path or explicit null")
    configured = project.config.bench["eval_set"]
    if configured is None:
        return normalize_eval_ids(list(truth), truth)
    if not isinstance(configured, str) or not configured.strip():
        raise BenchInputError("bench.eval_set must be a path or explicit null")
    p = Path(project.config.project_dir) / configured
    if not p.exists():
        raise BenchInputError(f"configured bench eval set does not exist: {p}")
    try:
        wanted = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    except OSError as exc:
        raise BenchInputError(f"could not read configured bench eval set: {p}") from exc
    return normalize_eval_ids(wanted, truth)


def bench(project, candidates, eval_ids, items, truth, constraints_version, policy="", out_dir=None):
    candidates = normalize_candidates(candidates)
    eval_ids = normalize_eval_ids(eval_ids, truth)
    regime = regime_hash(regime_payload(project.config, constraints_version, truth, items))
    rows = []
    for cand in candidates:
        preds = run_candidate_over(project, cand, eval_ids, items, policy, regime=regime)
        m = score_split(preds, truth, eval_ids, project.objective,
                        project.config.guards["anomaly_at"],
                        expected_regime=regime,
                        min_coverage=project.config.guards.get("min_coverage"))
        if eval_ids and not preds:
            raise ValueError(
                f"bench: 0/{len(eval_ids)} items produced predictions for a candidate — "
                "likely a broken runner or a misaligned items dict")
        rows.append({"candidate": cand, "objective": m["objective"], "metrics": m, "regime": regime})
    rows.sort(key=lambda r: r["objective"], reverse=(project.objective.direction == "max"))
    if out_dir is not None:
        results.write_bench(out_dir, regime, rows)
    return rows
