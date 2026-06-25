from loop_eval.objectives import get_objective
from loop_eval.objectives.prf1 import PRF1


def test_f1_simple():
    obj = PRF1(positive_label="1")
    truth = {"a": "1", "b": "1", "c": "0", "d": "1"}
    preds = {"a": "1", "b": "0", "c": "0", "d": "1"}  # tp=2 fn=1 fp=0
    m = obj.score(preds, truth, ["a", "b", "c", "d"])
    assert m["precision"] == 1.0
    assert round(m["recall"], 4) == round(2 / 3, 4)
    assert round(m["f1"], 4) == round(2 * 1.0 * (2 / 3) / (1.0 + 2 / 3), 4)
    assert m["objective"] == m["f1"]
    assert obj.direction == "max"


def test_missing_pred_is_negative():
    obj = PRF1(positive_label="1")
    m = obj.score({"a": "1"}, {"a": "1", "b": "1"}, ["a", "b"])
    assert round(m["recall"], 4) == 0.5


def test_registry():
    assert isinstance(get_objective("prf1", {"positive_label": "1"}), PRF1)
