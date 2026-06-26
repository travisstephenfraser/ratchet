import argparse
from datetime import datetime, timezone
from pathlib import Path
from .constraints import review, ConstraintsLedger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--consolidate", metavar="WHY")
    ap.add_argument("--author", default="author")
    args = ap.parse_args()
    cpath = Path(args.project) / "constraints.jsonl"
    report = review(cpath)
    if args.review or args.consolidate:
        print("constraints: clean" if not report else "\n".join(report))
    if args.consolidate:
        ts = datetime.now(timezone.utc).isoformat()
        ConstraintsLedger(Path(args.project) / "constraints_log.jsonl").record(
            version="manual", changed=report or ["(no flags)"], why=args.consolidate,
            author=args.author, timestamp=ts)
        print("recorded consolidation in constraints_log.jsonl")


if __name__ == "__main__":
    main()
