"""Persist results to disk, every one stamped with the regime that produced it — the
substrate the regime guard reads to detect a change, and the traceability the spec
requires. Mirrors loop-eng/harness.py's candidates/*.{txt,preds.csv,metrics.json} +
LOOP_LOG.md."""
import csv
import json
from pathlib import Path


def write_candidate(out_dir, cid, instructions, preds, metrics, regime):
    cand_dir = Path(out_dir) / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    (cand_dir / f"{cid}.txt").write_text(instructions)
    with open(cand_dir / f"{cid}.preds.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["anon_id", "score"])
        for anon, score in preds.items():
            w.writerow([anon, score])
    (cand_dir / f"{cid}.metrics.json").write_text(
        json.dumps({**metrics, "regime": regime, "cid": cid}, indent=2, default=str))
    return cand_dir / f"{cid}.metrics.json"


def append_loop_log(out_dir, cid, label, metrics):
    path = Path(out_dir) / "LOOP_LOG.md"
    if not path.exists():
        path.write_text("# LOOP_LOG\n\n| cid | label | objective | mae | exact |\n"
                        "|---|---|---|---|---|\n")
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
    with open(path, "a") as fh:
        fh.write(f"| {cid} | {label} | {fmt(metrics.get('objective'))} | "
                 f"{fmt(metrics.get('mae'))} | {fmt(metrics.get('exact'))} |\n")


def write_bench(out_dir, regime, rows):
    path = Path(out_dir) / f"bench_{regime}.json"
    path.write_text(json.dumps(rows, indent=2, default=str))
    return path
