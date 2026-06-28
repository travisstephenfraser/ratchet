"""Candidate transforms for the OPTIONAL hill-climb.

NOTE ON SCOPE: the primary use of this project is `verify` -- a regression gate on
telemetry-derived truth -- NOT `loop`. Improving the PIXEL-ONLY prompt optimizes a
capability that never ships (telemetry is load-bearing in production) and actively
rewards cue-leaking prompts, which is the artifact we caught (Gemma 6/6 -> 3/6). If you
do run `loop`, keep mutations STRUCTURAL (a field/criterion), never a NEVER/CRITICAL nag
-- ban-lists backfire on capable models. The one legitimate loop use is a one-off
"pixel-only ceiling" probe; the anomaly guard will flag a cue-leak win.
"""

MUTATIONS = [
    ("add-confidence-field",
     lambda c: c + "\nAlso output CONFIDENCE: <HIGH|MEDIUM|LOW>."),
    # ANTI-PATTERN (do not copy): ("shout", lambda c: c + " NEVER say UPHILL on a descent!!")
]
