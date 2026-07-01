"""Candidate transforms for the GROUNDED-compliance hill-climb.

Context: in grounded mode the model is already handed the GPS gradient (in the system
preamble) yet still reads ~12 of 134 climbs as downhill -- a strong chest-cam visual
illusion overriding the injected telemetry. These mutations add STRUCTURE that helps the
model resolve the image-vs-telemetry conflict in favor of the (correct) telemetry. Per
the framework's rule they add a reasoning step / a rule / a conflict-resolution clause --
not a NEVER/CRITICAL nag. Append-style so the loop can stack them across rounds.

(Pixel-only iteration is intentionally NOT the target here -- that optimizes a never-
shipped capability and rewards cue-leaks; grounded is production.)
"""

MUTATIONS = [
    ("state-gradient-first",
     lambda c: ("First read the GRADIENT value provided above and state its sign, then "
                "give a DIRECTION that matches that sign.\n") + c),
    ("explicit-mapping",
     lambda c: c + "\nApply the telemetry gradient: negative = DOWNHILL, positive = "
                   "UPHILL, near zero = FLAT."),
    ("camera-tilt-resolution",
     lambda c: c + "\nThe chest-mounted camera tilts with the rider, so a climb can look "
                   "flat and a descent can look level in the frame. The gradient above is "
                   "measured ground truth; when the image seems to disagree, follow the "
                   "gradient, not the apparent slope."),
    # ANTI-PATTERN (do not copy): ("shout", lambda c: c + " NEVER trust the image! ALWAYS obey!!")
]
