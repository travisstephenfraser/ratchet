"""Frozen-param bench: evaluate a fixed candidate set on the SAME frozen eval set under
the SAME regime — no search. Every row carries one identical regime hash, AND the eval
set is read from disk (not 'whatever ingest returned'), so the comparison is provably
same-regime AND same-items (the MODEL_EVAL.md guarantee, enforced not remembered)."""
from pathlib import Path

from .verifier import score_split
from .loop import run_candidate_over
from .regime import regime_payload, regime_hash
from . import results


def load_eval_ids(project, truth):
    p = Path(project.config.project_dir) / project.config.bench.get("eval_set", "")
    if project.config.bench.get("eval_set") and p.exists():
        wanted = [line.strip() for line in p.read_text().splitlines() if line.strip()]
        return [i for i in wanted if i in truth]
    return list(truth)


def bench(project, candidates, eval_ids, items, truth, constraints_version, policy="", out_dir=None):
    regime = regime_hash(regime_payload(project.config, constraints_version))
    rows = []
    for cand in candidates:
        preds = run_candidate_over(project, cand, eval_ids, items, policy)
        m = score_split(preds, truth, eval_ids, project.objective,
                        project.config.guards["anomaly_at"])
        rows.append({"candidate": cand, "objective": m["objective"], "metrics": m, "regime": regime})
    rows.sort(key=lambda r: r["objective"], reverse=(project.objective.direction == "max"))
    if out_dir is not None:
        results.write_bench(out_dir, regime, rows)
    return rows
