"""Binary precision/recall/F1 for classification / extraction. A missing prediction
is the negative label, so failing to predict a positive item is a recall miss."""
from .within_tol import Objective


class PRF1(Objective):
    direction = "max"

    def __init__(self, positive_label="1"):
        self.positive = str(positive_label)

    def score(self, preds, truth, ids):
        tp = fp = fn = 0
        for i in ids:
            t = str(truth[i]) == self.positive
            p = str(preds.get(i, "")) == self.positive
            if p and t:
                tp += 1
            elif p and not t:
                fp += 1
            elif not p and t:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {"n": len(ids), "graded": sum(1 for i in ids if i in preds),
                "precision": precision, "recall": recall, "f1": f1, "objective": f1}
