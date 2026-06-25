"""Candidate transforms for the hill-climb. PREFER STRUCTURE OVER EXHORTATION:
add a field, a criterion, a tool — not 'NEVER'/'CRITICAL'/'ALWAYS'. Avoid ban-lists.
Each entry is (name, transform) where transform(candidate_str) -> candidate_str."""

MUTATIONS = [
    # ("add-step-criterion", lambda c: c + "\nCriterion: award per-step points for each "
    #                                       "correct intermediate result."),
]
