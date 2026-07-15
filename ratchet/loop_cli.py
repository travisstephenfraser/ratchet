import argparse
import sys
from pathlib import Path
from .project import load_project
from .verifier import split_ids, score_split, log_holdout_access, validate_baselines
from .loop import hill_climb, escalate, run_candidate_over
from .constraints import load_constraints, current_version
from .regime import regime_payload, regime_hash
from .regime_state import enforce_regime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--escalate", action="store_true")
    ap.add_argument("--establish-baseline", action="store_true")
    args = ap.parse_args()
    proj = load_project(Path(args.project))
    if args.escalate and not args.establish_baseline:
        try:
            validate_baselines(proj.config.guards)
        except ValueError as e:
            print(f"{e}; set finite numeric values in guards.baseline and record the regime change",
                  file=sys.stderr)
            sys.exit(2)
    cpath = Path(args.project) / "constraints.jsonl"
    policy, cv = load_constraints(cpath), current_version(cpath)
    items, truth = proj.ingest()
    train, holdout = split_ids(list(truth), proj.config.salt, proj.config.holdout_pct)
    if args.establish_baseline:
        max_miss = proj.config.guards.get("max_miss_rate")
        train_preds = run_candidate_over(proj, proj.base_candidate, train, items, policy,
                                         max_miss_rate=max_miss)
        log_holdout_access(Path(args.project) / "holdout_access.log",
                           "establish_baseline", "base_candidate")
        holdout_preds = run_candidate_over(proj, proj.base_candidate, holdout, items, policy,
                                           max_miss_rate=max_miss)
        min_cov = proj.config.guards.get("min_coverage")
        train_metrics = score_split(train_preds, truth, train, proj.objective,
                                    proj.config.guards["anomaly_at"], min_coverage=min_cov)
        holdout_metrics = score_split(holdout_preds, truth, holdout, proj.objective,
                                      proj.config.guards["anomaly_at"], min_coverage=min_cov)
        if train_metrics.get("low_coverage") or holdout_metrics.get("low_coverage"):
            print("cannot establish baseline below guards.min_coverage", file=sys.stderr)
            sys.exit(2)
        print("baseline:")
        print(f"  train: {train_metrics['objective']!r}")
        print(f"  holdout: {holdout_metrics['objective']!r}")
        return
    enforce_regime(proj, cv, Path(args.project) / "regime_log.jsonl", truth, items)  # BLOCKS on silent change
    regime = regime_hash(regime_payload(proj.config, cv, truth, items))
    rounds = args.rounds or proj.config.search["rounds"]
    best = hill_climb(proj, train, items, truth, rounds=rounds,
                      patience=proj.config.search["patience"], policy=policy,
                      regime=regime, out_dir=Path(args.project))
    print(f"best cid={best['cid']} objective={best['metrics']['objective']}")
    if args.escalate:
        gate = escalate(proj, best, train, holdout, items, truth,
                        log_path=Path(args.project) / "holdout_access.log", policy=policy,
                        regime=regime)
        verdict = ("ANOMALY — reject" if gate["anomaly"]
                   else "REGRESSION — reject" if gate.get("regressed")
                   else "LOW COVERAGE — reject" if gate.get("low_coverage")
                   else "OVERFIT — reject" if gate["overfit"] else "generalizes — pass")
        print(f"gate gap={gate['gap']:.3f} anomaly={gate['anomaly']} -> {verdict}")


if __name__ == "__main__":
    main()
