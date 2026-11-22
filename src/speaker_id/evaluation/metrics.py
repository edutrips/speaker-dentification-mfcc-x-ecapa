"""Class-ordered metrics with recording-level confusion matrix."""

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(
    truth: Sequence[str], predictions: Sequence[str], classes: list[str]
) -> dict[str, object]:
    if not truth or len(truth) != len(predictions):
        raise ValueError("Truth and predictions must be non-empty and aligned")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=classes, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(classes)
    }
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(truth, predictions, labels=classes).tolist(),
    }
