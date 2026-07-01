"""ingest(): telemetry-derived direction ground truth for the cosmos trail VLM.

Truth is the GPS gradient sign stored in each analyzed frame's telemetry.json --
free, objective labels with no hand-labeling: DOWNHILL / UPHILL / FLAT. This is the
whole reason ratchet fits here: the ground truth comes from the synchronized GPS, not
a human labeling session. Items carry the frame path + telemetry so the runner can
score either grounded or pixel-only.

Source: every `multiframe_*` run under COSMOS_RESULTS_ROOT (or an explicit
colon-separated COSMOS_RUNS list). Frames are DEDUPED by (clip, frame-second): the same
trail second analyzed in two runs is one item, taken from the densest run, so near-
identical frames can't split across train/holdout and leak. Pure stdlib -- `verify`
and `ingest` need no model dependencies, which is what lets the gate run in CI.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

DEFAULT_RESULTS_ROOT = (
    "/Users/travis/Developer/strava-vlm-telemetry/experiments/cosmos_mtb_analysis"
    "/phase2/results")

# |grade| < band => FLAT. Matches phase2/ab_pixel_only.py so labels are consistent
# with the A/B that motivated this project.
GRADE_BAND_PCT = 3.0


def _direction(grade: float) -> str:
    if grade <= -GRADE_BAND_PCT:
        return "DOWNHILL"
    if grade >= GRADE_BAND_PCT:
        return "UPHILL"
    return "FLAT"


def _clip_id(run: Path):
    """Clip id from summary.json, or None if the run predates clip_id (can't dedup
    cleanly, so it's skipped rather than polluting clip grouping)."""
    try:
        return json.loads((run / "summary.json").read_text()).get("clip_id")
    except (OSError, json.JSONDecodeError):
        return None


def _runs():
    explicit = os.environ.get("COSMOS_RUNS")
    if explicit:
        return [Path(p) for p in explicit.split(":") if p]
    root = Path(os.environ.get("COSMOS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    return sorted(root.glob("multiframe_*"))


def ingest():
    # Collect (run, clip, usable frame dirs); process densest runs first so they win
    # the (clip, second) dedup -- a dense run is the better-sampled source for a clip.
    collected = []
    for run in _runs():
        if not run.is_dir():
            continue
        clip = _clip_id(run)
        if clip is None:        # run predates clip_id; can't dedup cleanly -> skip
            continue
        frames = [s for s in sorted(run.iterdir())
                  if s.is_dir() and (s / "telemetry.json").exists() and (s / "frame.jpg").exists()]
        if frames:
            collected.append((run, clip, frames))
    collected.sort(key=lambda c: len(c[2]), reverse=True)

    items, truth, seen = {}, {}, set()
    for run, clip, frames in collected:
        for sub in frames:
            t = json.loads((sub / "telemetry.json").read_text())
            sec = t.get("raw_video_sec")
            key = (clip, sec)
            if key in seen:
                continue
            seen.add(key)
            anon = f"{clip}/sec{int(sec):04d}" if sec is not None else f"{clip}/{sub.name}"
            items[anon] = {
                "frame_path": str(sub / "frame.jpg"),
                "telemetry": t,
                "gradient_pct": t.get("gradient_pct", 0.0),
            }
            truth[anon] = _direction(t.get("gradient_pct", 0.0))

    if not truth:
        raise RuntimeError(
            "no telemetry.json frames found; set COSMOS_RESULTS_ROOT or COSMOS_RUNS")

    # COSMOS_MAX_PER_CLIP: evenly subsample to N frames per clip for a fast, balanced
    # dev bench during prompt iteration (loop). Deterministic. Unset = full bench (gate).
    cap = int(os.environ.get("COSMOS_MAX_PER_CLIP", "0"))
    if cap > 0:
        by_clip = {}
        for k in sorted(truth):
            by_clip.setdefault(k.split("/")[0], []).append(k)
        keep = set()
        for ks in by_clip.values():
            step = max(1, len(ks) // cap)
            keep.update(ks[::step][:cap])
        items = {k: items[k] for k in keep}
        truth = {k: truth[k] for k in keep}
    return items, truth
