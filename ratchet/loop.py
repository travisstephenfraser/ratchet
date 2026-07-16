"""The escalation loop. Greedy hill-climb over the project's mutations on the TRAIN
split only; the active constraints `policy` is threaded into every grading call; each
candidate is persisted with its regime stamp; on plateau the best is escalated to the
holdout regression/anomaly/coverage/overfit gate. Works for any Runner + Objective."""
import hashlib
import sys

from .adapter import Unparseable, UnscorableCandidate
from .verifier import score_split, gap_report, log_holdout_access, Predictions
from .validation import nonblank, rate, whole_number
from . import results


def cand_id(instructions):
    return hashlib.sha256(instructions.encode()).hexdigest()[:10]


def better(a, b, direction):
    return a > b + 1e-9 if direction == "max" else a < b - 1e-9


def run_candidate_over(project, candidate, ids, items, policy="", *, max_miss_rate=None,
                       regime=None):
    max_miss_rate = None if max_miss_rate is None else rate(max_miss_rate, "guards.max_miss_rate")
    preds, attempted, misses = {}, 0, 0
    for i in ids:
        if i not in items:
            continue
        attempted += 1
        # Only Unparseable is demoted to a per-item MISS: that failure is a property of
        # the CANDIDATE (an unparseable reply), so one bad frame must not abort the whole
        # hill-climb — the objective counts the absent id against the candidate. Every
        # OTHER exception (transport, a harness bug) is NOT the candidate's fault and
        # propagates loudly, halting the run. See adapter.Unparseable.
        try:
            p = project.runner.run(candidate, items[i], policy)
        except Unparseable as e:
            print(f"  miss {i}: {e}", file=sys.stderr)
            misses += 1
            continue
        if p is None:
            raise ValueError(f"runner returned None for {i} (fail-loud: parse must raise Unparseable)")
        preds[i] = str(p)
    # Opt-in systematic-failure guard (guards.max_miss_rate; off by default). A KNOWN-GOOD
    # candidate missing more than this rate is not "a few bad frames"; it is a broken
    # model/harness emitting Unparseable in bulk, which escalate()'s 0-pred guard only
    # catches at 100%. Applied to the base during the climb and to the winner at the
    # escalate gate; mid-climb mutations are unguarded, so a merely-bad mutation still
    # just scores low instead of aborting the run.
    if max_miss_rate is not None:
        if ids and attempted == 0:
            raise ValueError(f"no items scored: {len(ids)} ids share no keys with the items dict")
        if attempted and misses / attempted > max_miss_rate:
            raise ValueError(
                f"{misses}/{attempted} items unparseable "
                f"({misses / attempted:.0%} > {max_miss_rate:.0%} max_miss_rate) — "
                "systematic parse failure on a known-good candidate, halting for review")
    return Predictions(preds, regime=regime)


def _eval(project, candidate, ids, items, truth, policy, regime, out_dir, label, max_miss_rate=None):
    regime = nonblank(regime, "regime")
    preds = run_candidate_over(project, candidate, ids, items, policy,
                               max_miss_rate=max_miss_rate, regime=regime)
    m = score_split(preds, truth, ids, project.objective, project.config.guards["anomaly_at"],
                    expected_regime=regime,
                    min_coverage=project.config.guards.get("min_coverage"))
    cid = cand_id(candidate)
    if out_dir is not None:
        results.write_candidate(out_dir, cid, candidate, preds, m, regime)
        results.append_loop_log(out_dir, cid, label, m)
    return {"cid": cid, "instructions": candidate, "metrics": m}


def hill_climb(project, train_ids, items, truth, rounds, patience,
               policy="", regime="", out_dir=None):
    rounds = whole_number(rounds, "search.rounds", minimum=1)
    patience = whole_number(patience, "search.patience", minimum=0)
    regime = nonblank(regime, "regime")
    # Guard the base only: it is the known-good reference, so a high miss rate there means
    # a broken model/harness (halt), whereas a bad mutation is allowed to just score low.
    best = _eval(project, project.base_candidate, train_ids, items, truth, policy, regime, out_dir, "base",
                 max_miss_rate=project.config.guards.get("max_miss_rate"))
    if best["metrics"].get("low_coverage"):
        raise ValueError(
            f"base candidate coverage {best['metrics']['split_coverage']:.0%} is below "
            f"guards.min_coverage — a known-good candidate under the floor means a broken "
            f"model/harness, halting for review")
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
            try:
                m = _eval(project, cand, train_ids, items, truth, policy, regime, out_dir,
                          f"r{r+1}:{name}")
            except UnscorableCandidate:
                continue
            if m["metrics"].get("low_coverage"):
                continue  # a low-coverage mutation may not win on a coverage-blind scalar
            if better(m["metrics"]["objective"], best["metrics"]["objective"], direction):
                best, improved = m, True
        stale = 0 if improved else stale + 1
    return best


def escalate(project, best, train_ids, holdout_ids, items, truth, log_path,
             policy="", *, regime):
    regime = nonblank(regime, "regime")
    train_preds = run_candidate_over(
        project, best["instructions"], train_ids, items, policy,
        max_miss_rate=project.config.guards.get("max_miss_rate"), regime=regime)
    if train_ids and not train_preds:
        raise ValueError(f"escalate: 0/{len(train_ids)} train items produced predictions")
    log_holdout_access(log_path, "escalation_gate", best["cid"])
    holdout_preds = run_candidate_over(project, best["instructions"], holdout_ids, items, policy,
                                       max_miss_rate=project.config.guards.get("max_miss_rate"),
                                       regime=regime)
    if holdout_ids and not holdout_preds:
        raise ValueError(f"escalate: 0/{len(holdout_ids)} holdout items produced predictions")
    preds = Predictions({**train_preds, **holdout_preds}, regime=regime)
    return gap_report(preds, truth, train_ids, holdout_ids, project.objective,
                      project.config.guards, expected_regime=regime)
