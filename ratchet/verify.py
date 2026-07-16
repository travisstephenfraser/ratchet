import argparse, sys
from pathlib import Path
from .project import load_project
from .verifier import (split_ids, score_split, gap_report, load_column, log_holdout_access,
                       preds_regime_gate, validate_baselines)
from .constraints import current_version
from .regime_state import enforce_regime
from .regime import RegimeMismatch
from .adapter import UnscorableCandidate
from .results import _json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--split", choices=["train", "holdout", "gap"], default="train")
    args = ap.parse_args()
    proj = load_project(Path(args.project))
    cv = current_version(Path(args.project) / "constraints.jsonl")
    items, truth = proj.ingest()
    truth = {k: str(v) for k, v in truth.items()}
    current = enforce_regime(proj, cv, Path(args.project) / "regime_log.jsonl", truth, items)
    preds = load_column(Path(args.predictions))
    try:
        warning = preds_regime_gate(preds.regime, current)
    except RegimeMismatch as e:
        print(f"refusing to score across regimes: {e}", file=sys.stderr)
        sys.exit(2)
    if warning:
        print(warning.format(path=args.predictions), file=sys.stderr)
    train, holdout = split_ids(list(truth), proj.config.salt, proj.config.holdout_pct)
    guards = proj.config.guards
    required = ("train", "holdout") if args.split == "gap" else (args.split,)
    try:
        baseline = validate_baselines(guards, required)
    except ValueError as e:
        print(f"{e}; set finite numeric values in guards.baseline and record the regime change",
              file=sys.stderr)
        sys.exit(2)
    if args.split != "train":
        log_holdout_access(Path(args.project) / "holdout_access.log", "verify_cli", args.predictions)
    try:
        if args.split == "train":
            result = score_split(preds, truth, train, proj.objective, guards["anomaly_at"],
                                 min_coverage=guards.get("min_coverage"),
                                 baseline_objective=baseline.get("train"))
        elif args.split == "holdout":
            result = score_split(preds, truth, holdout, proj.objective, guards["anomaly_at"],
                                 min_coverage=guards.get("min_coverage"),
                                 baseline_objective=baseline.get("holdout"))
        else:
            result = gap_report(preds, truth, train, holdout, proj.objective, guards)
    except UnscorableCandidate as exc:
        print(f"unscorable candidate: {exc}", file=sys.stderr)
        sys.exit(2)
    print(_json(result))
    if (result.get("anomaly") or result.get("overfit") or result.get("low_coverage")
            or result.get("regressed")):
        sys.exit(2)


if __name__ == "__main__":
    main()
