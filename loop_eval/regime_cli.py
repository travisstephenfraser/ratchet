import argparse
from datetime import datetime, timezone
from pathlib import Path
from .project import load_project
from .constraints import current_version
from .regime_state import record_bump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--why", required=True)
    ap.add_argument("--impact", default="")
    ap.add_argument("--author", default="travis")
    args = ap.parse_args()
    proj = load_project(Path(args.project))
    cv = current_version(Path(args.project) / "constraints.jsonl")
    changes = record_bump(proj, cv, why=args.why, impact=args.impact, author=args.author,
                          timestamp=datetime.now(timezone.utc).isoformat(),
                          ledger_path=Path(args.project) / "regime_log.jsonl")
    print(f"recorded {len(changes)} change(s) to regime_log.jsonl")
    for f, o, n in changes:
        print(f"  {f}: {o!r} -> {n!r}")


if __name__ == "__main__":
    main()
