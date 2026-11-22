"""Fixed, reproducible MFCC + scaler + RBF SVM baseline."""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    decision_function_shape="ovr",
                    probability=False,
                ),
            ),
        ]
    )


def decision_scores(model: Pipeline, x: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Return one uncalibrated decision score per class and segment."""

    classes = [str(value) for value in model.classes_]
    scores = np.asarray(model.decision_function(x), dtype=np.float64)
    if len(classes) == 2 and scores.ndim == 1:
        scores = np.column_stack((-scores, scores))
    if scores.shape != (len(x), len(classes)) or not np.isfinite(scores).all():
        raise ValueError("Classifier decision scores are invalid")
    return classes, scores


def mean_recording_scores(model: Pipeline, x: np.ndarray) -> tuple[str, dict[str, float]]:
    """Predict a source recording from the mean of its segment decision scores."""

    if len(x) == 0:
        raise ValueError("A recording needs at least one segment")
    classes, scores = decision_scores(model, x)
    means = scores.mean(axis=0)
    return classes[int(np.argmax(means))], dict(zip(classes, means.tolist(), strict=True))
