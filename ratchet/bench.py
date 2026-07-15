"""Frozen-param bench: evaluate fixed candidates under one regime with no search.
A configured eval-set file is required and validated; an explicit null selects all
ingested ids. Every row carries the same truth-, item-, and scoring-aware regime hash."""
from pathlib import Path

from .verifier import score_split
from .loop import run_candidate_over
from .regime import regime_payload, regime_hash
from . import results


def load_eval_ids(project, truth):
    if "eval_set" not in project.config.bench:
        raise ValueError("bench.eval_set must be a path or explicit null")
    configured = project.config.bench["eval_set"]
    if configured is None:
        return list(truth)
    if not isinstance(configured, str) or not configured.strip():
        raise ValueError("bench.eval_set must be a path or explicit null")
    p = Path(project.config.project_dir) / configured
    if not p.exists():
        raise FileNotFoundError(f"configured bench eval set does not exist: {p}")
    wanted = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    if not wanted:
        raise ValueError(f"configured bench eval set is empty: {p}")
    unknown = [i for i in wanted if i not in truth]
    if unknown:
        raise ValueError(f"configured bench eval set contains unknown ids: {unknown}")
    return wanted


def bench(project, candidates, eval_ids, items, truth, constraints_version, policy="", out_dir=None):
    if not eval_ids:
        raise ValueError("bench eval set must not be empty")
    regime = regime_hash(regime_payload(project.config, constraints_version, truth, items))
    rows = []
    for cand in candidates:
        preds = run_candidate_over(project, cand, eval_ids, items, policy, regime=regime)
        if eval_ids and not preds:
            raise ValueError(
                f"bench: 0/{len(eval_ids)} items produced predictions for a candidate — "
                "likely a broken runner or a misaligned items dict")
        m = score_split(preds, truth, eval_ids, project.objective,
                        project.config.guards["anomaly_at"],
                        min_coverage=project.config.guards.get("min_coverage"))
        rows.append({"candidate": cand, "objective": m["objective"], "metrics": m, "regime": regime})
    rows.sort(key=lambda r: r["objective"], reverse=(project.objective.direction == "max"))
    if out_dir is not None:
        results.write_bench(out_dir, regime, rows)
    return rows
