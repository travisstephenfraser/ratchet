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


def _temperature() -> float | None:
    """Sampling temperature for the scoring call, from env COSMOS_TEMPERATURE.

    Default 0.0 (greedy: lowest sampling noise for a fair hill-climb; note greedy
    reduces but does not guarantee determinism on MoE/GPU kernels). Set
    COSMOS_TEMPERATURE="" to OMIT the parameter entirely -- required for models that
    reject it (e.g. reasoning models that fix sampling internally). A numeric value is
    passed through. Declared in config.yaml regime_env so it is part of the fingerprint."""
    raw = os.environ.get("COSMOS_TEMPERATURE")
    if raw is None:
        return 0.0
    raw = raw.strip()
    if raw == "":
        return None
    return float(raw)


class Runner:
    def run(self, candidate, item, policy=""):
        if str(COSMOS_DIR) not in sys.path:
            sys.path.insert(0, str(COSMOS_DIR))
        from phase2 import vlm  # lazy: needs the cosmos venv

        frame = Path(item["frame_path"])
        prompt = f"{policy}\n\n{candidate}".strip() if policy else candidate
        grounded = os.environ.get("COSMOS_MODE", "pixel") == "grounded"
        # temperature default 0.0 (greedy) so the hill-climb isn't swamped by sampling
        # noise -- vlm otherwise runs at the API default (~1.0). Env-configurable
        # (COSMOS_TEMPERATURE) because a growing class of models reject the parameter;
        # temp=None omits it. Applied to BOTH paths so a mode swap stays comparable.
        temp = _temperature()
        mode_kw = ({"telemetry": item["telemetry"]} if grounded
                   else {"system_override": NEUTRAL_PREAMBLE})

        if os.environ.get("COSMOS_PARSE", "regex") == "structured":
            from runner_structured import DirectionRead  # lazy: pydantic only here
            # max_retries=0: a parse failure must count against the candidate, not be
            # silently repaired. The raise is demoted to a miss by gen_preds / loop.
            result = vlm.analyze_frame_structured(
                frame, prompt, DirectionRead, temperature=temp, max_retries=0, **mode_kw)
            return result.direction

        resp = vlm.analyze_frame(frame, prompt, temperature=temp, **mode_kw)
        return _parse_direction(vlm.extract_text(resp))
