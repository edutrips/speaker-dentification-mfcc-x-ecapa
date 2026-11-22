"""Recording-level open-set evaluation for already-produced predictions."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from speaker_id.datasets import is_unknown_speaker_id
from speaker_id.evaluation.metrics import classification_metrics
from speaker_id.open_set.protocol import ACTIVE_KNOWN_SPEAKER_IDS, OPTIONAL_KNOWN_SPEAKER_IDS

from .rejection import UNKNOWN_LABEL


@dataclass(frozen=True, slots=True)
class OpenSetEvaluation:
    """Recording-level open-set metrics in an explicit class order."""

    recording_count: int
    known_count: int
    unknown_count: int
    known_identification_accuracy: float
    unknown_rejection_rate: float
    far: float
    frr: float
    known_to_known_misidentification_count: int
    known_to_known_misidentification_rate: float
    overall_accuracy: float
    macro_f1: float
    class_order: tuple[str, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    per_class: dict[str, dict[str, float | int]]


def evaluation_label_for_speaker(speaker_id: str) -> str:
    """Collapse Uxx identities to UNKNOWN; reject optional V05 for this protocol."""

    if not isinstance(speaker_id, str) or not speaker_id:
        raise ValueError("speaker_id must be a non-empty string")
    if speaker_id in ACTIVE_KNOWN_SPEAKER_IDS or speaker_id == UNKNOWN_LABEL:
        return speaker_id
    if speaker_id in OPTIONAL_KNOWN_SPEAKER_IDS:
        raise ValueError(f"{speaker_id} is optional, not enrolled as active-known")
    if is_unknown_speaker_id(speaker_id):
        return UNKNOWN_LABEL
    raise ValueError(f"speaker_id is not eligible for this open-set evaluation: {speaker_id}")


def _normalize_labels(labels: Sequence[str], name: str) -> list[str]:
    if isinstance(labels, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty sequence of labels")
    try:
        result = list(labels)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-empty sequence of labels") from exc
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(label, str) or not label for label in result):
        raise ValueError(f"{name} must contain non-empty string labels")
    return result


def evaluate_open_set(
    y_true: Sequence[str], y_pred: Sequence[str], known_classes: Sequence[str]
) -> OpenSetEvaluation:
    """Evaluate one final prediction per recording; input labels must already be collapsed.

    ``known_classes`` order is preserved and ``UNKNOWN`` is appended to define
    the metrics and confusion-matrix order. Predictions must be final labels;
    threshold selection and segment aggregation are outside this evaluator.
    """

    truth = _normalize_labels(y_true, "y_true")
    predictions = _normalize_labels(y_pred, "y_pred")
    if len(truth) != len(predictions):
        raise ValueError("y_true and y_pred must have equal lengths")
    classes = _normalize_labels(known_classes, "known_classes")
    if len(set(classes)) != len(classes):
        raise ValueError("known_classes must not contain duplicates")
    if UNKNOWN_LABEL in classes:
        raise ValueError("UNKNOWN is an output label and cannot be a known class")
    optional_in_classes = set(classes).intersection(OPTIONAL_KNOWN_SPEAKER_IDS)
    if optional_in_classes:
        raise ValueError(
            "optional supported speakers cannot be evaluated as active-known: "
            f"{sorted(optional_in_classes)}"
        )
    if set(classes) != set(ACTIVE_KNOWN_SPEAKER_IDS):
        raise ValueError(
            "known_classes must contain exactly the active-known speaker set: "
            f"{ACTIVE_KNOWN_SPEAKER_IDS}"
        )

    label_set = set(classes) | {UNKNOWN_LABEL}
    invalid_truth = set(truth) - label_set
    invalid_predictions = set(predictions) - label_set
    if invalid_truth:
        raise ValueError(
            "y_true contains labels outside known_classes + UNKNOWN: "
            f"{sorted(invalid_truth)}"
        )
    if invalid_predictions:
        raise ValueError(
            "y_pred contains labels outside known_classes + UNKNOWN: "
            f"{sorted(invalid_predictions)}"
        )

    known_count = sum(label != UNKNOWN_LABEL for label in truth)
    unknown_count = len(truth) - known_count
    if known_count == 0:
        raise ValueError("open-set evaluation requires at least one known recording")
    if unknown_count == 0:
        raise ValueError("open-set evaluation requires at least one UNKNOWN recording")

    correct_known = sum(
        true_label != UNKNOWN_LABEL and true_label == predicted_label
        for true_label, predicted_label in zip(truth, predictions, strict=True)
    )
    false_rejects = sum(
        true_label != UNKNOWN_LABEL and predicted_label == UNKNOWN_LABEL
        for true_label, predicted_label in zip(truth, predictions, strict=True)
    )
    known_to_known_misidentifications = sum(
        true_label != UNKNOWN_LABEL
        and predicted_label != UNKNOWN_LABEL
        and true_label != predicted_label
        for true_label, predicted_label in zip(truth, predictions, strict=True)
    )
    false_accepts = sum(
        true_label == UNKNOWN_LABEL and predicted_label != UNKNOWN_LABEL
        for true_label, predicted_label in zip(truth, predictions, strict=True)
    )
    correct_unknown_rejections = sum(
        true_label == UNKNOWN_LABEL and predicted_label == UNKNOWN_LABEL
        for true_label, predicted_label in zip(truth, predictions, strict=True)
    )

    known_accuracy = correct_known / known_count
    unknown_rejection_rate = correct_unknown_rejections / unknown_count
    far = false_accepts / unknown_count
    frr = false_rejects / known_count
    misidentification_rate = known_to_known_misidentifications / known_count
    class_order = (*classes, UNKNOWN_LABEL)
    metrics = classification_metrics(truth, predictions, list(class_order))
    confusion = tuple(tuple(int(value) for value in row) for row in metrics["confusion_matrix"])
    macro_f1 = float(metrics["macro_f1"])
    overall_accuracy = float(metrics["accuracy"])

    bounded_metrics = (
        known_accuracy,
        unknown_rejection_rate,
        far,
        frr,
        misidentification_rate,
        overall_accuracy,
        macro_f1,
    )
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in bounded_metrics):
        raise ArithmeticError("open-set metrics must be finite values between zero and one")
    if not math.isclose(unknown_rejection_rate, 1.0 - far, rel_tol=0.0, abs_tol=1e-12):
        raise ArithmeticError("unknown rejection rate must equal one minus FAR")

    return OpenSetEvaluation(
        recording_count=len(truth),
        known_count=known_count,
        unknown_count=unknown_count,
        known_identification_accuracy=known_accuracy,
        unknown_rejection_rate=unknown_rejection_rate,
        far=far,
        frr=frr,
        known_to_known_misidentification_count=known_to_known_misidentifications,
        known_to_known_misidentification_rate=misidentification_rate,
        overall_accuracy=overall_accuracy,
        macro_f1=macro_f1,
        class_order=class_order,
        confusion_matrix=confusion,
        per_class=metrics["per_class"],
    )
