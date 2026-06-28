"""ingest(): telemetry-derived direction ground truth for the cosmos trail VLM.

Truth is the GPS gradient sign stored in each analyzed frame's telemetry.json --
free, objective labels with no hand-labeling: DOWNHILL / UPHILL / FLAT. This is the
whole reason ratchet fits here: the ground truth comes from the synchronized GPS, not
a human labeling session. Items carry the frame path + telemetry so the runner can
score either grounded or pixel-only.

Source runs: COSMOS_RUNS (colon-separated multiframe result dirs) or the default
laguna run. Pure stdlib -- `verify` and `ingest` need no model dependencies, which is
what lets the gate run in CI / under ratchet's own venv.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Default bench: the laguna_ta multiframe run (6 frames, one clean descent). Grow the
# bench by pointing COSMOS_RUNS at more run dirs (hammertime, luge, old_camp2_chunk...)
# -- more frames = more statistical weight on the gate.
DEFAULT_RUNS = [
    "/Users/travis/Developer/strava-vlm-telemetry/experiments/cosmos_mtb_analysis"
    "/phase2/results/multiframe_20260622_090322",
]

# |grade| < band => FLAT. Matches phase2/ab_pixel_only.py so labels are consistent
# with the A/B that motivated this project.
GRADE_BAND_PCT = 3.0


def _direction(grade: float) -> str:
    if grade <= -GRADE_BAND_PCT:
        return "DOWNHILL"
    if grade >= GRADE_BAND_PCT:
        return "UPHILL"
    return "FLAT"


def ingest():
    runs = os.environ.get("COSMOS_RUNS")
    run_dirs = [Path(p) for p in runs.split(":")] if runs else [Path(p) for p in DEFAULT_RUNS]

    items, truth = {}, {}
    for run in run_dirs:
        if not run.exists():
            continue
        for sub in sorted(run.iterdir()):
            tj = sub / "telemetry.json"
            if not (sub.is_dir() and tj.exists()):
                continue
            t = json.loads(tj.read_text())
            grade = t.get("gradient_pct", 0.0)
            anon = f"{run.name}/{sub.name}"          # stable, unique across runs
            items[anon] = {
                "frame_path": str(sub / "frame.jpg"),
                "telemetry": t,
                "gradient_pct": grade,
            }
            truth[anon] = _direction(grade)

    if not truth:
        raise RuntimeError(
            "no telemetry.json frames found; point COSMOS_RUNS at one or more "
            "phase2/results/multiframe_* dirs")
    return items, truth
