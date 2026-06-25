"""Numeric-closeness objective — ports scorer.py's within-tolerance logic, plus the
harness.py lesson that within-rate ceilings on easy questions (no gradient), so MAE
is the climb signal there. `climb` selects which scalar the loop climbs. Missing
predictions are misses: denominator is the full id list."""
import statistics


class Objective:
    direction = "max"

    def score(self, preds, truth, ids):
        raise NotImplementedError


class WithinTol(Objective):
    def __init__(self, tol=2.0, climb="within"):
        self.tol = float(tol)
        if climb not in ("within", "mae"):
            raise ValueError(f"climb must be within|mae, got {climb!r}")
        self.climb = climb
        self.direction = "max" if climb == "within" else "min"

    def score(self, preds, truth, ids):
        n = len(ids)
        deltas = [float(preds[i]) - float(truth[i]) for i in ids if i in preds]
        abs_deltas = [abs(d) for d in deltas]
        graded = len(deltas)
        within = sum(1 for d in abs_deltas if d <= self.tol) / n if n else 0.0
        mae = statistics.mean(abs_deltas) if abs_deltas else None
        objective = within if self.climb == "within" else (mae if mae is not None else float("inf"))
        return {
            "n": n, "graded": graded, "coverage": graded / n if n else 0.0,
            "mae": mae, "mean_delta": statistics.mean(deltas) if deltas else None,
            "exact": sum(1 for d in deltas if d == 0) / n if n else 0.0,
            "within_tol": within, "tol": self.tol, "objective": objective,
        }
