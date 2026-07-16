import pytest

from ratchet.adapter import UnscorableCandidate
from ratchet.objectives import get_objective
from ratchet.objectives.within_tol import WithinTol


def test_within_counts_missing_as_miss():
    obj = WithinTol(tol=2.0)
    truth = {"a": "10", "b": "10", "c": "10", "d": "10"}
    preds = {"a": "10", "b": "11", "c": "13"}  # d missing; c is 3 off
    m = obj.score(preds, truth, ["a", "b", "c", "d"])
    assert m["n"] == 4 and m["graded"] == 3
    assert m["objective"] == 0.5  # a,b within 2 -> 2/4
    assert m["exact"] == 0.25
    assert round(m["mae"], 4) == round((0 + 1 + 3) / 3, 4)
    assert obj.direction == "max"


def test_climb_mae_flips_objective_and_direction():
    obj = WithinTol(tol=2.0, climb="mae")
    truth = {"a": "10", "b": "10"}
    preds = {"a": "10", "b": "12"}  # mae 1.0
    m = obj.score(preds, truth, ["a", "b"])
    assert obj.direction == "min"
    assert m["objective"] == 1.0          # objective is MAE now
    assert m["within_tol"] == 1.0         # still reported for visibility


def test_mae_with_zero_predictions_is_explicitly_unscorable():
    with pytest.raises(UnscorableCandidate, match="zero predictions"):
        WithinTol(tol=0.5, climb="mae").score({}, {"a": "10"}, ["a"])


def test_mae_with_one_prediction_keeps_existing_semantics():
    result = WithinTol(tol=0.5, climb="mae").score(
        {"a": "10"}, {"a": "10", "b": "10"}, ["a", "b"])
    assert result["objective"] == 0.0
    assert result["coverage"] == 0.5


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_within_tol_rejects_nonfinite_inputs(value):
    objective = WithinTol(tol=0.5)
    with pytest.raises(ValueError, match="finite"):
        objective.score({"a": value}, {"a": "10"}, ["a"])
    with pytest.raises(ValueError, match="finite"):
        objective.score({"a": "10"}, {"a": value}, ["a"])


def test_registry():
    assert isinstance(get_objective("within_tol", {"tol": 1.0}), WithinTol)
    with pytest.raises(KeyError):
        get_objective("nope", {})
