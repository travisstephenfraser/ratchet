from loop_eval.objectives.judge import Judge


def test_judge_means_injected_scores():
    obj = Judge(judge_fn=lambda pred, rubric: 1.0 if rubric in pred else 0.0, rubric="welfare")
    truth = {"a": "welfare", "b": "welfare", "c": "welfare"}
    preds = {"a": "discusses welfare loss", "b": "off topic"}  # c missing -> 0
    m = obj.score(preds, truth, ["a", "b", "c"])
    assert round(m["mean_judge"], 4) == round(1 / 3, 4)
    assert m["objective"] == m["mean_judge"]
    assert obj.direction == "max"


def test_missing_pred_scores_zero():
    obj = Judge(judge_fn=lambda p, r: 1.0, rubric="x")
    m = obj.score({"a": "yes"}, {"a": "x", "b": "x"}, ["a", "b"])
    assert round(m["mean_judge"], 4) == 0.5


def test_default_judge_raises_if_called():
    import pytest
    with pytest.raises(RuntimeError):
        Judge().score({"a": "x"}, {"a": "x"}, ["a"])
