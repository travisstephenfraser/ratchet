"""The escalation loop. Greedy hill-climb over the project's mutations on the TRAIN
split only; the active constraints `policy` is threaded into every grading call; each
candidate is persisted with its regime stamp; on plateau the best is escalated to the
holdout gate. Works for any Runner + Objective."""
import hashlib

from .verifier import score_split, gap_report, log_holdout_access
from . import results


def cand_id(instructions):
    return hashlib.sha256(instructions.encode()).hexdigest()[:10]


def better(a, b, direction):
    return a > b + 1e-9 if direction == "max" else a < b - 1e-9


def run_candidate_over(project, candidate, ids, items, policy=""):
    preds = {}
    for i in ids:
        if i in items:
            p = project.runner.run(candidate, items[i], policy)
            if p is None:
                raise ValueError(f"runner returned None for {i} (fail-loud: parse must raise)")
            preds[i] = str(p)
    return preds


def _eval(project, candidate, ids, items, truth, policy, regime, out_dir, label):
    preds = run_candidate_over(project, candidate, ids, items, policy)
    m = score_split(preds, truth, ids, project.objective, project.config.guards["anomaly_at"])
    cid = cand_id(candidate)
    if out_dir is not None:
        results.write_candidate(out_dir, cid, candidate, preds, m, regime)
        results.append_loop_log(out_dir, cid, label, m)
    return {"cid": cid, "instructions": candidate, "metrics": m}


def hill_climb(project, train_ids, items, truth, rounds, patience,
               policy="", regime="", out_dir=None):
    best = _eval(project, project.base_candidate, train_ids, items, truth, policy, regime, out_dir, "base")
    direction = project.objective.direction
    seen, stale = {best["cid"]}, 0
    for r in range(rounds):
        if stale >= patience:
            break
        improved = False
        for name, transform in project.mutations:
            cand = transform(best["instructions"])
            cid = cand_id(cand)
            if cid in seen:
                continue
            seen.add(cid)
            m = _eval(project, cand, train_ids, items, truth, policy, regime, out_dir, f"r{r+1}:{name}")
            if better(m["metrics"]["objective"], best["metrics"]["objective"], direction):
                best, improved = m, True
        stale = 0 if improved else stale + 1
    return best


def escalate(project, best, train_ids, holdout_ids, items, truth, log_path,
             policy="", train_preds=None):
    log_holdout_access(log_path, "escalation_gate", best["cid"])
    if train_preds is None:
        train_preds = run_candidate_over(project, best["instructions"], train_ids, items, policy)
    else:
        # Caller-supplied train_preds must belong to the train split — catch wrong-split preds.
        rogue = set(train_preds) - set(train_ids)
        if rogue:
            raise ValueError(
                f"escalate: train_preds contains keys not in train_ids: {sorted(rogue)}"
            )
    holdout_preds = run_candidate_over(project, best["instructions"], holdout_ids, items, policy)
    # Guard against gross misalignment: non-empty id list but zero predictions produced.
    if train_ids and not train_preds:
        raise ValueError(
            f"escalate: 0/{len(train_ids)} train items produced predictions — "
            "likely a misaligned items dict"
        )
    if holdout_ids and not holdout_preds:
        raise ValueError(
            f"escalate: 0/{len(holdout_ids)} holdout items produced predictions — "
            "likely a misaligned items dict"
        )
    preds = {**train_preds, **holdout_preds}
    return gap_report(preds, truth, train_ids, holdout_ids, project.objective, project.config.guards)
