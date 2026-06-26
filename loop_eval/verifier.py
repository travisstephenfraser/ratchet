"""The scorer. Split discipline (stable hash split, holdout vault, missing-as-miss)
and anti-leak guards (anomaly, overfit) are objective-agnostic and direction-aware.
The anomaly (verifier-leak) flag is surfaced at the top level of the gap report and
re-checked at the escalation gate, so a leak is loud, not buried."""
import csv
import hashlib
from datetime import datetime, timezone


def load_column(path, value_field=None):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
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


def score_split(preds, truth, ids, objective, anomaly_at):
    if objective.direction not in ("max", "min"):
        raise ValueError(f"unknown objective direction: {objective.direction!r}")
    base = objective.score(preds, truth, ids)
    val = base["objective"]
    anomaly = (val > anomaly_at) if objective.direction == "max" else (val < anomaly_at)
    return {**base, "anomaly": anomaly}


def gap_report(preds, truth, train, holdout, objective, guards):
    tr = score_split(preds, truth, train, objective, guards["anomaly_at"])
    ho = score_split(preds, truth, holdout, objective, guards["anomaly_at"])
    gap = (tr["objective"] - ho["objective"]) if objective.direction == "max" \
        else (ho["objective"] - tr["objective"])
    return {"train": tr, "holdout": ho, "gap": gap,
            "overfit": gap > guards["overfit_gap"], "anomaly": tr["anomaly"]}


def log_holdout_access(log_path, caller, predictions_path):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()}\t{caller}\t{predictions_path}\n")
