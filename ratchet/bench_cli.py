import argparse
from pathlib import Path
from .project import load_project
from .bench import bench, load_eval_ids
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
    enforce_regime(proj, cv, Path(args.project) / "regime_log.jsonl", truth, items)
    eval_ids = load_eval_ids(proj, truth)
    rows = bench(proj, proj.config.bench["candidates"], eval_ids, items, truth, cv,
                 policy=policy, out_dir=Path(args.project))
    print(_json([{"candidate": r["candidate"], "objective": r["objective"],
                  "regime": r["regime"]} for r in rows]))


if __name__ == "__main__":
    main()
