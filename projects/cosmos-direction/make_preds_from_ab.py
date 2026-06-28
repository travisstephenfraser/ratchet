"""Convert a phase2/ab_pixel_only.py result JSON into a ratchet predictions CSV, so
`verify` can score real model reads against telemetry truth with NO new model calls.

The anon_id matches ingest.py's key ("<run_name>/<frame_dir>"). Usage:
  python make_preds_from_ab.py <ab_result.json> [pixel_only|grounded] > data/preds.csv
"""
import csv
import json
import sys
from pathlib import Path

ab = json.loads(Path(sys.argv[1]).read_text())
arm = sys.argv[2] if len(sys.argv) > 2 else "pixel_only"
run = Path(ab["results_dir"]).name

w = csv.writer(sys.stdout)
w.writerow(["anon_id", "direction"])
for fr in ab["frames"]:
    d = fr[arm].get("direction")
    w.writerow([f"{run}/{fr['frame']}", d if d is not None else ""])
