import argparse
import sys
from pathlib import Path
from .project import load_project
from .bench import (
    BenchInputError,
    bench,
    load_eval_ids,
    normalize_candidates,
)
from .constraints import load_constraints, current_version
from .regime_state import enforce_regime
from .results import _json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    args = ap.parse_args()
    proj = load_project(Path(args.project))
    cpath = Path(args.project) / "constraints.jsonl"
    policy, cv = load_constraints(cpath), current_version(cpath)
    items, truth = proj.ingest()
    try:
        candidates = normalize_candidates(proj.config.bench.get("candidates"))
        eval_ids = load_eval_ids(proj, truth)
    except BenchInputError as exc:
        print(f"bench input error: {exc}", file=sys.stderr)
        sys.exit(2)
    enforce_regime(proj, cv, Path(args.project) / "regime_log.jsonl", truth, items)
    rows = bench(proj, candidates, eval_ids, items, truth, cv,
                 policy=policy, out_dir=Path(args.project))
    print(_json([{"candidate": r["candidate"], "objective": r["objective"],
                  "regime": r["regime"]} for r in rows]))


if __name__ == "__main__":
    main()
