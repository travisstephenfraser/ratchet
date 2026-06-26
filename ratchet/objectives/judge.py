"""LLM-judge objective for open-ended generation. judge_fn is injected by the project
loader (it can't be expressed in YAML), so the core never hits a network in tests.
Missing predictions score 0 (a non-answer is the worst answer)."""
from .within_tol import Objective


def _no_judge(pred, rubric):
    raise RuntimeError("Judge needs a judge_fn injected (project must provide judge.py:judge_fn)")


class Judge(Objective):
    direction = "max"

    def __init__(self, judge_fn=None, rubric=""):
        self.judge_fn = judge_fn or _no_judge
        self.rubric = rubric

    def score(self, preds, truth, ids):
        scores = [float(self.judge_fn(preds[i], truth.get(i, self.rubric))) if i in preds else 0.0
                  for i in ids]
        mean = sum(scores) / len(scores) if scores else 0.0
        return {"n": len(ids), "graded": sum(1 for i in ids if i in preds),
                "mean_judge": mean, "objective": mean}
