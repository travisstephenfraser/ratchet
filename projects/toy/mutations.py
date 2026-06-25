"""Structural-over-exhortation operators. The winning operator adds a CRITERION, not a
nag. The commented anti-pattern is what PORTING.md warns against."""

MUTATIONS = [
    ("add-lenient-criterion",
     lambda c: c + "\nCriterion: award full marks when the final answer matches the key, "
                    "even if the working is messy (be lenient on presentation)."),
    # ANTI-PATTERN (do not copy): ("never-harsh", lambda c: c + " NEVER be harsh. CRITICAL!!")
]
