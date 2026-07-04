import argparse
from pathlib import Path
from .project import load_project
from .verifier import split_ids
from .loop import hill_climb, escalate
from .constraints import load_constraints, current_version
from .regime import regime_payload, regime_hash
from .regime_state import enforce_regime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--escalate", action="store_true")
    args = ap.parse_args()
    proj = load_project(Path(args.project))
    cpath = Path(args.project) / "constraints.jsonl"
    policy, cv = load_constraints(cpath), current_version(cpath)
    items, truth = proj.ingest()
    enforce_regime(proj, cv, Path(args.project) / "regime_log.jsonl", truth)  # BLOCKS on silent change
    regime = regime_hash(regime_payload(proj.config, cv, truth))
    train, holdout = split_ids(list(truth), proj.config.salt, proj.config.holdout_pct)
    rounds = args.rounds or proj.config.search["rounds"]
    best = hill_climb(proj, train, items, truth, rounds=rounds,
                      patience=proj.config.search["patience"], policy=policy,
                      regime=regime, out_dir=Path(args.project))
    print(f"best cid={best['cid']} objective={best['metrics']['objective']}")
    if args.escalate:
        gate = escalate(proj, best, train, holdout, items, truth,
                        log_path=Path(args.project) / "holdout_access.log", policy=policy,
                        regime=regime)
        verdict = ("LOW COVERAGE — reject" if gate.get("low_coverage")
                   else "OVERFIT — reject" if gate["overfit"] else "generalizes — pass")
        print(f"gate gap={gate['gap']:.3f} anomaly={gate['anomaly']} -> {verdict}")


if __name__ == "__main__":
    main()
