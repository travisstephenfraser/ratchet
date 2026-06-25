"""40 deterministic synthetic exams; half 'messy' (the strict grader under-credits these)."""


def ingest():
    items, truth = {}, {}
    for i in range(40):
        anon = f"toy{i:03d}"
        items[anon] = {"true": 10, "messy": (i % 2 == 0)}
        truth[anon] = "10"
    return items, truth
