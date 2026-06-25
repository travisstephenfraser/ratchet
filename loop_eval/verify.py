import argparse, json, sys
from pathlib import Path
from .project import load_project
from .verifier import split_ids, score_split, gap_report, load_column, log_holdout_access
from .constraints import current_version
from .regime_state import enforce_regime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--split", choices=["train", "holdout", "gap"], default="train")
    args = ap.parse_args()
    proj = load_project(Path(args.project))
    cv = current_version(Path(args.project) / "constraints.jsonl")
    enforce_regime(proj, cv, Path(args.project) / "regime_log.jsonl")
    _, truth = proj.ingest()
    truth = {k: str(v) for k, v in truth.items()}
    preds = load_column(Path(args.predictions))
    train, holdout = split_ids(list(truth), proj.config.salt, proj.config.holdout_pct)
    guards = proj.config.guards
    if args.split != "train":
        log_holdout_access(Path(args.project) / "holdout_access.log", "verify_cli", args.predictions)
    if args.split == "train":
        result = score_split(preds, truth, train, proj.objective, guards["anomaly_at"])
    elif args.split == "holdout":
        result = score_split(preds, truth, holdout, proj.objective, guards["anomaly_at"])
    else:
        result = gap_report(preds, truth, train, holdout, proj.objective, guards)
    json.dump(result, sys.stdout, indent=2, default=str)
    print()
    if result.get("anomaly") or result.get("overfit"):
        sys.exit(2)


if __name__ == "__main__":
    main()
