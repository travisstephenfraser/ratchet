"""Custom objective: 3-class direction accuracy (exact match of DOWNHILL/UPHILL/FLAT).

prf1 measures only descent detection (positive=DOWNHILL); for prompt iteration we want
the whole picture -- a fix that turns a climb-read-as-downhill into UPHILL should count,
and so should UPHILL<->FLAT corrections. Accuracy captures every direction error.

Missing predictions are misses (counted in the denominator), consistent with the rest
of ratchet. Loaded via config `objective: {name: custom}` -> make_objective.
"""
from ratchet.objectives.within_tol import Objective


class Accuracy(Objective):
    direction = "max"

    def score(self, preds, truth, ids):
        n = len(ids)
        graded = sum(1 for i in ids if i in preds)
        correct = sum(1 for i in ids if preds.get(i) == truth[i])
        acc = correct / n if n else 0.0
        return {"n": n, "graded": graded, "correct": correct,
                "accuracy": acc, "objective": acc}


def make_objective(params):
    return Accuracy()
