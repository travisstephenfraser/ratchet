"""The scorer. Split discipline (stable hash split, holdout vault, missing-as-miss)
and anti-leak guards (anomaly, overfit) are objective-agnostic and direction-aware.
The anomaly (verifier-leak) flag is surfaced at the top level of the gap report and
re-checked at the escalation gate, so a leak is loud, not buried."""
import csv
import hashlib
import math
import warnings
from datetime import datetime, timezone

from .regime import guard_compare, RegimeMismatch
from .validation import finite_number, mapping, nonblank, rate, whole_number


PREDS_REGIME_PREFIX = "# ratchet-regime: "


def preds_regime_header(regime) -> str:
    """The comment line that stamps a preds CSV with the regime it was generated under.
    load_column skips it; read_preds_regime reads it back for the comparability guard."""
    return f"{PREDS_REGIME_PREFIX}{regime}\n"


def read_preds_regime(path):
    """Return the regime a preds file was stamped with, or None if it is unstamped (a
    legacy or externally generated file). Stops at the first data line."""
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if line.startswith(PREDS_REGIME_PREFIX):
                return line[len(PREDS_REGIME_PREFIX):].strip()
            if line.strip() and not line.startswith("#"):
                return None
    return None


class Predictions(dict):
    """Prediction map carrying the regime it was produced under (None = unstamped).
    A dict subclass so every existing consumer and plain-dict caller keeps working;
    the stamp travels WITH the data, like the CSV comment line it mirrors.
    Standard dict copy/merge operations (.copy(), dict(p), {**p}) return a plain dict and DROP the stamp; re-wrap with Predictions(..., regime=...) after copying."""
    def __init__(self, data=None, *, regime=None):
        super().__init__(data or {})
        self.regime = regime


UNSTAMPED_PREDS_WARNING = (
    "predictions carry no regime stamp; scoring anyway, but ratchet cannot confirm they "
    "were produced under expected regime {expected!r}.\n"
    "  RISK: a cross-regime score is a silently wrong number that can PASS a gate it "
    "should fail, or FAIL one it should pass.\n"
    "  FIX: produce predictions via run_candidate_over(..., regime=...) or load_column() "
    "on a stamped file so provenance travels with them."
)


def check_expected_regime(preds, expected_regime, *, allow_unstamped=False):
    """In-process comparability guard. Raises RegimeMismatch when the preds' stamp and
    the expectation differ or is missing; an explicit compatibility override makes a
    missing stamp allowed-but-loud. Silent when the caller passes no expectation."""
    if expected_regime is None:
        return
    expected_regime = nonblank(expected_regime, "expected_regime")
    stamped = getattr(preds, "regime", None)
    if stamped is None:
        if not allow_unstamped:
            raise RegimeMismatch("predictions carry no regime stamp")
        warnings.warn(UNSTAMPED_PREDS_WARNING.format(expected=expected_regime), stacklevel=3)
        return
    guard_compare(stamped, expected_regime)  # raises RegimeMismatch on mismatch


def load_column(path, value_field=None):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        fields = reader.fieldnames or []
        if value_field is None:
            rest = [c for c in fields if c != "anon_id"]
            if not rest:
                raise ValueError(f"{path}: no value column (fields={fields})")
            value_field = rest[0]
        rows = {row["anon_id"]: row[value_field] for row in reader if row["anon_id"]}
    return Predictions(rows, regime=read_preds_regime(path))


def split_ids(ids, salt, holdout_pct):
    holdout_pct = whole_number(holdout_pct, "holdout_pct", minimum=1, maximum=99)
    train, holdout = [], []
    for anon in sorted(ids):
        bucket = int(hashlib.sha256(f"{salt}:{anon}".encode()).hexdigest()[:8], 16) % 100
        (holdout if bucket < holdout_pct else train).append(anon)
    return train, holdout


def _finite_baseline(value):
    if isinstance(value, bool):
        raise ValueError("frozen baseline objectives must be finite numbers, not booleans")
    try:
        value = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError("frozen baseline objectives must be finite numbers") from e
    if not math.isfinite(value):
        raise ValueError("frozen baseline objectives must be finite numbers")
    return value


def validate_baselines(guards, splits=("train", "holdout")):
    baseline = guards.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("missing frozen baseline objectives in guards.baseline")
    missing = [split for split in splits if split not in baseline]
    if missing:
        raise ValueError("missing frozen baseline objective for " + ", ".join(missing))
    return {split: _finite_baseline(baseline[split]) for split in splits}


def score_split(preds, truth, ids, objective, anomaly_at, *, expected_regime,
                allow_unstamped=False, min_coverage=None, baseline_objective=None):
    if not ids:
        raise ValueError("split must not be empty; add more labeled items or change the split salt")
    check_expected_regime(preds, expected_regime, allow_unstamped=allow_unstamped)
    if objective.direction not in ("max", "min"):
        raise ValueError(f"unknown objective direction: {objective.direction!r}")
    anomaly_at = finite_number(anomaly_at, "guards.anomaly_at")
    if min_coverage is not None:
        min_coverage = rate(min_coverage, "guards.min_coverage")
    # hand the objective only the labels for the ids being scored, never the whole vault
    scoped_truth = {item_id: truth[item_id] for item_id in ids if item_id in truth}
    base = mapping(objective.score(preds, scoped_truth, ids), "objective result")
    if "objective" not in base:
        raise ValueError("objective result requires an 'objective' key")
    value = finite_number(base["objective"], "objective result")
    base = {**base, "objective": value}
    anomaly = value > anomaly_at if objective.direction == "max" else value < anomaly_at
    # Coverage is computed by the CORE from (preds, ids) — never read from the
    # objective's report — so a coverage-blind objective (mae) cannot Goodhart it.
    coverage = sum(1 for item_id in ids if item_id in preds) / len(ids)
    out = {**base, "anomaly": anomaly, "split_coverage": coverage}
    if min_coverage is not None:
        out["low_coverage"] = coverage < min_coverage
    if baseline_objective is not None:
        baseline = finite_number(baseline_objective, "frozen baseline objective")
        delta = value - baseline if objective.direction == "max" else baseline - value
        delta = finite_number(delta, "objective delta")
        out.update(baseline_objective=baseline, objective_delta=delta,
                   regressed=delta < -1e-9)
    return out


class _GapReportBuilder:
    """Private two-phase scorer used when holdout access must follow train scoring."""

    def __init__(self, truth, train, holdout, objective, guards, *, expected_regime,
                 allow_unstamped=False):
        self.expected_regime = nonblank(expected_regime, "expected_regime")
        self.allow_unstamped = allow_unstamped
        self.truth = truth
        self.train = train
        self.holdout = holdout
        self.objective = objective
        self.guards = mapping(guards, "guards")
        overlap = set(train) & set(holdout)
        if overlap:
            raise ValueError(f"train and holdout must be disjoint; shared ids: {sorted(overlap)}")
        self.overfit_gap = finite_number(
            self.guards.get("overfit_gap"), "guards.overfit_gap")
        if self.overfit_gap < 0:
            raise ValueError("guards.overfit_gap must be >= 0")
        self.min_coverage = self.guards.get("min_coverage")
        self.baselines = validate_baselines(self.guards)
        self._checked_predictions = set()
        self._train_values = None
        self._train_result = None

    def _check_predictions(self, preds):
        identity = id(preds)
        if identity not in self._checked_predictions:
            check_expected_regime(
                preds, self.expected_regime, allow_unstamped=self.allow_unstamped)
            self._checked_predictions.add(identity)

    def score_train(self, preds):
        if self._train_result is not None:
            raise RuntimeError("train split has already been scored")
        self._check_predictions(preds)
        self._train_values = {
            item_id: preds[item_id] for item_id in self.train if item_id in preds
        }
        self._train_result = score_split(
            preds, self.truth, self.train, self.objective, self.guards["anomaly_at"],
            expected_regime=None, min_coverage=self.min_coverage,
            baseline_objective=self.baselines["train"])

    def finish(self, preds):
        if self._train_result is None:
            raise RuntimeError("train split must be scored before the gap report is finished")
        self._check_predictions(preds)
        train_values = {
            item_id: preds[item_id] for item_id in self.train if item_id in preds
        }
        if train_values != self._train_values:
            raise ValueError("train predictions changed after train scoring")
        holdout_result = score_split(
            preds, self.truth, self.holdout, self.objective, self.guards["anomaly_at"],
            expected_regime=None, min_coverage=self.min_coverage,
            baseline_objective=self.baselines["holdout"])
        train_result = self._train_result
        gap = (train_result["objective"] - holdout_result["objective"]
               if self.objective.direction == "max"
               else holdout_result["objective"] - train_result["objective"])
        gap = finite_number(gap, "observed train/holdout gap")
        out = {
            "train": train_result, "holdout": holdout_result, "gap": gap,
            "overfit": gap > self.overfit_gap,
            "anomaly": train_result["anomaly"] or holdout_result["anomaly"],
            "train_anomaly": train_result["anomaly"],
            "holdout_anomaly": holdout_result["anomaly"],
            "train_coverage": train_result["split_coverage"],
            "holdout_coverage": holdout_result["split_coverage"],
            "regressed": train_result["regressed"] or holdout_result["regressed"],
        }
        if self.min_coverage is not None:
            out["low_coverage"] = (
                train_result["low_coverage"] or holdout_result["low_coverage"])
        return out


def gap_report(preds, truth, train, holdout, objective, guards, *, expected_regime,
               allow_unstamped=False):
    builder = _GapReportBuilder(
        truth, train, holdout, objective, guards, expected_regime=expected_regime,
        allow_unstamped=allow_unstamped)
    builder.score_train(preds)
    return builder.finish(preds)


def log_holdout_access(log_path, caller, predictions_path):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()}\t{caller}\t{predictions_path}\n")
