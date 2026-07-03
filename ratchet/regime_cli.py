import argparse
from datetime import datetime, timezone
from pathlib import Path
from .project import load_project
from .constraints import current_version
from .regime_state import record_bump


def _nonblank(value):
    # The ledger rationale is the audit trail for a sanctioned bump; --why "" defeats it.
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be blank")
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--why", required=True, type=_nonblank)
    ap.add_argument("--impact", required=True, type=_nonblank)
    ap.add_argument("--author", default="author")
    args = ap.parse_args()
    proj = load_project(Path(args.project))
    cv = current_version(Path(args.project) / "constraints.jsonl")
    _, truth = proj.ingest()
    changes = record_bump(proj, cv, why=args.why, impact=args.impact, author=args.author,
                          timestamp=datetime.now(timezone.utc).isoformat(),
                          ledger_path=Path(args.project) / "regime_log.jsonl", truth=truth)
    print(f"recorded {len(changes)} change(s) to regime_log.jsonl")
    for f, o, n in changes:
        print(f"  {f}: {o!r} -> {n!r}")


if __name__ == "__main__":
    main()
