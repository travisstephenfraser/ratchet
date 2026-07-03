"""The scorer. Split discipline (stable hash split, holdout vault, missing-as-miss)
and anti-leak guards (anomaly, overfit) are objective-agnostic and direction-aware.
The anomaly (verifier-leak) flag is surfaced at the top level of the gap report and
re-checked at the escalation gate, so a leak is loud, not buried."""
import csv
import hashlib
import warnings
from datetime import datetime, timezone

from .regime import guard_compare, RegimeMismatch


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


LEGACY_PREDS_WARNING = (
    "WARNING: predictions file {path} carries no regime stamp (legacy or externally "
    "generated). Scoring it anyway, but ratchet CANNOT confirm these predictions were "
    "produced under the current regime.\n"
    "  RISK: if they were generated under a different model, prompt, eval set, or label "
    "set, this score is comparing across incomparable rules, a silently wrong number that "
    "can PASS a gate it should fail, or FAIL one it should pass.\n"
    "  FIX: regenerate the predictions with the current gen_preds so the file is stamped, "
    "or re-run generation and scoring under one regime."
)


def preds_regime_gate(stamped, current):
    """Compare a preds file's stamped regime against the current one. Returns the legacy
    warning string when the file is unstamped, None when the stamp matches, and raises
    RegimeMismatch when they differ (the caller exits non-zero)."""
    if stamped is None:
        return LEGACY_PREDS_WARNING
    guard_compare(stamped, current)  # raises RegimeMismatch on mismatch
    return None


class Predictions(dict):
    """Prediction map carrying the regime it was produced under (None = unstamped).
    A dict subclass so every existing consumer and plain-dict caller keeps working;
    the stamp travels WITH the data, like the CSV comment line it mirrors."""
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


def check_expected_regime(preds, expected_regime):
    """In-process comparability guard. Raises RegimeMismatch when the preds' stamp and
    the expectation differ; warns (allowed-but-loud, the CLI legacy posture) when the
    preds are unstamped; silent when the caller passes no expectation."""
    if expected_regime is None:
        return
    stamped = getattr(preds, "regime", None)
    if stamped is None:
        warnings.warn(UNSTAMPED_PREDS_WARNING.format(expected=expected_regime))
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
        return {row["anon_id"]: row[value_field] for row in reader if row["anon_id"]}


def split_ids(ids, salt, holdout_pct):
    train, holdout = [], []
    for anon in sorted(ids):
        bucket = int(hashlib.sha256(f"{salt}:{anon}".encode()).hexdigest()[:8], 16) % 100
        (holdout if bucket < holdout_pct else train).append(anon)
    return train, holdout


def score_split(preds, truth, ids, objective, anomaly_at, *, expected_regime=None):
    check_expected_regime(preds, expected_regime)
    if objective.direction not in ("max", "min"):
        raise ValueError(f"unknown objective direction: {objective.direction!r}")
    # hand the objective only the labels for the ids being scored, never the whole vault
    scoped_truth = {i: truth[i] for i in ids if i in truth}
    base = objective.score(preds, scoped_truth, ids)
    val = base["objective"]
    anomaly = (val > anomaly_at) if objective.direction == "max" else (val < anomaly_at)
    return {**base, "anomaly": anomaly}


def gap_report(preds, truth, train, holdout, objective, guards, *, expected_regime=None):
    overlap = set(train) & set(holdout)
    if overlap:
        raise ValueError(f"train and holdout must be disjoint; shared ids: {sorted(overlap)}")
    check_expected_regime(preds, expected_regime)
    tr = score_split(preds, truth, train, objective, guards["anomaly_at"])
    ho = score_split(preds, truth, holdout, objective, guards["anomaly_at"])
    gap = (tr["objective"] - ho["objective"]) if objective.direction == "max" \
        else (ho["objective"] - tr["objective"])
    return {"train": tr, "holdout": ho, "gap": gap,
            "overfit": gap > guards["overfit_gap"], "anomaly": tr["anomaly"]}


def log_holdout_access(log_path, caller, predictions_path):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()}\t{caller}\t{predictions_path}\n")
