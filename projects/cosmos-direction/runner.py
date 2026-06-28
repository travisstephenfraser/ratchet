"""Runner.run(): apply one direction prompt to one frame via the cosmos VLM and return
the direction string. The single place that calls the model. FAILS LOUD when the read
can't be parsed (per the ratchet contract -- a silent miss would corrupt scoring).

Mode (env COSMOS_MODE):
  "pixel"    (default) -- telemetry WITHHELD, neutral preamble: the honest perception
             probe. A near-perfect score here is the cue-leak signature (anomaly guard).
  "grounded"           -- telemetry injected (the production config): the compliance
             gate. Tells you whether the model still OBEYS the telemetry anchor after a
             model/prompt swap; a drop here is the regression that matters in production.

Requires the cosmos venv (anthropic/httpx) AT RUN TIME; imported lazily so that ratchet's
pure-Python `verify`/`ingest`/tests never need those deps. Generate predictions under the
cosmos venv; score them with `verify` under ratchet's venv.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

COSMOS_DIR = Path(os.environ.get(
    "COSMOS_DIR",
    "/Users/travis/Developer/strava-vlm-telemetry/experiments/cosmos_mtb_analysis"))

# Neutral: withholds telemetry WITHOUT listing visual cues. A cue-listing preamble
# leaks the answer (Gemma 6/6 -> 3/6 once neutralized); keep this honest.
NEUTRAL_PREAMBLE = (
    "You are looking at a single frame from a mountain bike chest-mounted camera "
    "(first-person view). No telemetry is available. Answer using only what the image "
    "shows. Commit to a best estimate; do not refuse."
)


def _parse_direction(text: str) -> str:
    m = re.search(r"\b(DOWNHILL|UPHILL|FLAT)\b", text, re.I)
    if not m:
        raise ValueError(f"no direction parseable from model response: {text[:200]!r}")
    return m.group(1).upper()


class Runner:
    def run(self, candidate, item, policy=""):
        if str(COSMOS_DIR) not in sys.path:
            sys.path.insert(0, str(COSMOS_DIR))
        from phase2 import vlm  # lazy: needs the cosmos venv

        frame = Path(item["frame_path"])
        prompt = f"{policy}\n\n{candidate}".strip() if policy else candidate
        if os.environ.get("COSMOS_MODE", "pixel") == "grounded":
            resp = vlm.analyze_frame(frame, prompt, telemetry=item["telemetry"])
        else:
            resp = vlm.analyze_frame(frame, prompt, system_override=NEUTRAL_PREAMBLE)
        return _parse_direction(vlm.extract_text(resp))
