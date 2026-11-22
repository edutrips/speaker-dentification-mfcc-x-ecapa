"""Known-speaker-only cross-session validation for persisted SVM models."""

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from speaker_id.evaluation.metrics import classification_metrics
from speaker_id.models import mean_recording_scores
from speaker_id.open_set.protocol import ACTIVE_KNOWN_SPEAKER_IDS

METADATA_FIELDS = (
    "segment_id",
    "recording_id",
    "speaker_id",
    "session_id",
    "text_id",
    "split",
    "start_seconds",
    "end_seconds",
)
VALIDATION_TEXTS = tuple(f"T{number:02d}" for number in range(6, 11))
PREDICTION_FIELDS = (
    "recording_id",
    "true_speaker",
    "text_id",
    "predicted_speaker",
    "correct",
    "top_decision_score",
    "second_best_decision_score",
    "score_margin",
)


def validate_known_validation_rows(
    labels: np.ndarray,
    segment_ids: list[str],
    metadata: list[dict[str, str]],
) -> None:
    """Enforce the complete known V01-V04, S02, T06-T10 validation scope."""

    if labels.ndim != 1 or len(labels) != len(segment_ids) or len(labels) != len(metadata):
        raise ValueError("Validation labels, segment IDs, and metadata must be aligned")
    if [row.get("segment_id") for row in metadata] != segment_ids:
        raise ValueError("Validation segment IDs or their order differ from metadata")
    if [row.get("speaker_id") for row in metadata] != labels.tolist():
        raise ValueError("Validation labels differ from metadata")
    if any(any(not row.get(field) for field in METADATA_FIELDS) for row in metadata):
        raise ValueError("Validation metadata is missing required values")

    identities: dict[str, tuple[str, str, str, str]] = {}
    for row in metadata:
        if row["speaker_id"] not in ACTIVE_KNOWN_SPEAKER_IDS:
            raise ValueError(f"Ineligible validation speaker: {row['speaker_id']}")
        if row["session_id"] != "S02" or row["split"] != "validation":
            raise ValueError("Validation rows must belong only to S02/validation")
        if row["text_id"] not in VALIDATION_TEXTS:
            raise ValueError(f"Ineligible validation text: {row['text_id']}")
        identity = (row["speaker_id"], row["session_id"], row["text_id"], row["split"])
        previous = identities.setdefault(row["recording_id"], identity)
        if previous != identity:
            raise ValueError(f"Conflicting metadata for recording {row['recording_id']}")

    expected_pairs = {
        (speaker, text) for speaker in ACTIVE_KNOWN_SPEAKER_IDS for text in VALIDATION_TEXTS
    }
    observed_pairs = {(identity[0], identity[2]) for identity in identities.values()}
    if len(identities) != 20 or observed_pairs != expected_pairs:
        raise ValueError("Validation must contain exactly 20 V01-V04/T06-T10 recordings")
    if Counter(identity[0] for identity in identities.values()) != Counter(
        {speaker: 5 for speaker in ACTIVE_KNOWN_SPEAKER_IDS}
    ):
        raise ValueError("Validation must contain exactly five recordings per known speaker")


def validate_aligned_representations(
    mfcc_x: np.ndarray,
    mfcc_y: np.ndarray,
    mfcc_metadata: list[dict[str, str]],
    ecapa_x: np.ndarray,
    ecapa_y: np.ndarray,
    ecapa_segment_ids: np.ndarray,
    ecapa_metadata: list[dict[str, str]],
) -> dict[str, Any]:
    """Validate row alignment and scope before any trained model is loaded."""

    mfcc_ids = [row.get("segment_id", "") for row in mfcc_metadata]
    ecapa_ids = ecapa_segment_ids.tolist()
    lengths = (
        len(mfcc_x), len(mfcc_y), len(mfcc_ids), len(mfcc_metadata),
        len(ecapa_x), len(ecapa_y), len(ecapa_ids), len(ecapa_metadata),
    )
    if len(set(lengths)) != 1:
        raise ValueError("MFCC and ECAPA validation row counts differ")
    if mfcc_x.ndim != 2 or ecapa_x.ndim != 2:
        raise ValueError("Validation representations must be two-dimensional")
    if not np.isfinite(mfcc_x).all() or not np.isfinite(ecapa_x).all():
        raise ValueError("Validation representations contain non-finite values")
    if mfcc_ids != ecapa_ids:
        raise ValueError("MFCC and ECAPA segment IDs or their order differ")
    if mfcc_y.tolist() != ecapa_y.tolist():
        raise ValueError("MFCC and ECAPA labels differ")
    for field in METADATA_FIELDS:
        if [row.get(field) for row in mfcc_metadata] != [row.get(field) for row in ecapa_metadata]:
            raise ValueError(f"MFCC and ECAPA metadata differ for {field}")

    validate_known_validation_rows(mfcc_y, mfcc_ids, mfcc_metadata)
    recordings = list(dict.fromkeys(row["recording_id"] for row in mfcc_metadata))
    return {
        "rows": len(mfcc_ids),
        "recordings": recordings,
        "classes": list(ACTIVE_KNOWN_SPEAKER_IDS),
        "texts": list(VALIDATION_TEXTS),
    }


def evaluate_known_validation(
    x: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, str]],
    classes: list[str],
    model: Any,
    representation: str,
    model_sha256: str,
    validation_dataset_sha256: str,
) -> dict[str, Any]:
    """Score segments with the persisted model and evaluate mean scores per recording."""

    segment_ids = [row.get("segment_id", "") for row in metadata]
    validate_known_validation_rows(labels, segment_ids, metadata)
    if tuple(classes) != tuple(model.classes_.tolist()):
        raise ValueError("Supplied model classes do not match the model's class order")
    if len(classes) != len(set(classes)) or set(classes) != set(ACTIVE_KNOWN_SPEAKER_IDS):
        raise ValueError("Model classes must be exactly the four active known speakers")
    if "UNKNOWN" in classes:
        raise ValueError("UNKNOWN cannot be a class in known-only validation")
    if not representation or not model_sha256 or not validation_dataset_sha256:
        raise ValueError("Representation and source hashes are required")
    if x.ndim != 2 or len(x) != len(metadata) or not np.isfinite(x).all():
        raise ValueError("Validation matrix is invalid or misaligned")

    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        grouped.setdefault(row["recording_id"], []).append(index)

    records: list[dict[str, Any]] = []
    truth: list[str] = []
    predictions: list[str] = []
    for recording_id, indices in grouped.items():
        helper_prediction, score_mapping = mean_recording_scores(model, x[indices])
        if list(score_mapping) != classes:
            raise ValueError("Decision-score class order differs from model class order")
        mean_scores = np.asarray([score_mapping[label] for label in classes], dtype=np.float64)
        top_indices = np.argsort(-mean_scores, kind="stable")
        top_index, runner_index = int(top_indices[0]), int(top_indices[1])
        predicted = classes[top_index]
        if helper_prediction != predicted:
            raise ValueError("Recording helper prediction disagrees with mean-score argmax")
        first = metadata[indices[0]]
        is_correct = predicted == first["speaker_id"]
        truth.append(first["speaker_id"])
        predictions.append(predicted)
        records.append(
            {
                "recording_id": recording_id,
                "true_speaker": first["speaker_id"],
                "text_id": first["text_id"],
                "predicted_speaker": predicted,
                "correct": "YES" if is_correct else "NO",
                "top_decision_score": float(mean_scores[top_index]),
                "second_best_decision_score": float(mean_scores[runner_index]),
                "score_margin": float(mean_scores[top_index] - mean_scores[runner_index]),
            }
        )

    if len(records) != 20:
        raise ValueError(f"Expected 20 recording-level predictions, received {len(records)}")
    metrics = classification_metrics(truth, predictions, classes)
    per_speaker = {
        speaker: {
            "correct": sum(
                row["true_speaker"] == speaker and row["predicted_speaker"] == speaker
                for row in records
            ),
            "total": sum(row["true_speaker"] == speaker for row in records),
        }
        for speaker in classes
    }
    metrics_payload = {
        "evaluation_type": "known_validation",
        "training_session": "S01",
        "validation_session": "S02",
        "classes": classes,
        "recording_count": len(records),
        "segment_count": len(metadata),
        "representation": representation,
        "model_artifact_sha256": model_sha256,
        "validation_dataset_sha256": validation_dataset_sha256,
        "aggregation": "mean decision scores per recording",
        "decision_rule": "argmax known class",
        "threshold_used": None,
        "score_semantics": "SVM decision-function scores; not probabilities; no softmax",
        "accuracy": metrics["accuracy"],
        "known_identification_accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class": metrics["per_class"],
        "per_speaker_recording_accuracy": per_speaker,
        "confusion_matrix": metrics["confusion_matrix"],
        "confusion_matrix_rows": "true",
        "confusion_matrix_columns": "predicted",
        "class_order": classes,
        "open_set_metrics_computed": False,
        "unknown_predictions_generated": False,
    }
    return {"metrics": metrics_payload, "predictions": records}


def compare_known_validation(mfcc: dict[str, Any], ecapa: dict[str, Any]) -> dict[str, Any]:
    """Compare the same recordings pairwise without inferential statistics."""

    mfcc_rows, ecapa_rows = mfcc["predictions"], ecapa["predictions"]
    keys = ("recording_id", "true_speaker", "text_id")
    if len(mfcc_rows) != 20 or len(ecapa_rows) != 20:
        raise ValueError("Paired comparison requires exactly 20 recordings per representation")
    if any(tuple(left[key] for key in keys) != tuple(right[key] for key in keys)
           for left, right in zip(mfcc_rows, ecapa_rows, strict=True)):
        raise ValueError("MFCC and ECAPA predictions do not describe the same ordered recordings")
    if mfcc["metrics"]["class_order"] != ecapa["metrics"]["class_order"]:
        raise ValueError("MFCC and ECAPA class orders differ")

    both_correct: list[str] = []
    mfcc_only: list[str] = []
    ecapa_only: list[str] = []
    both_wrong: list[str] = []
    for left, right in zip(mfcc_rows, ecapa_rows, strict=True):
        left_ok, right_ok = left["correct"] == "YES", right["correct"] == "YES"
        target = (
            both_correct if left_ok and right_ok else
            mfcc_only if left_ok else
            ecapa_only if right_ok else both_wrong
        )
        target.append(left["recording_id"])
    return {
        "comparison_type": "paired known-speaker validation",
        "recording_count": 20,
        "both_correct": len(both_correct),
        "mfcc_only_correct": len(mfcc_only),
        "ecapa_only_correct": len(ecapa_only),
        "both_wrong": len(both_wrong),
        "discordant_recording_ids": {
            "mfcc_correct_ecapa_wrong": mfcc_only,
            "mfcc_wrong_ecapa_correct": ecapa_only,
        },
        "score_magnitude_comparison_allowed": False,
        "significance_test_performed": False,
    }


def persist_known_validation_results(
    output_dir: Path, mfcc: dict[str, Any], ecapa: dict[str, Any], comparison: dict[str, Any]
) -> None:
    """Write prediction tables and JSON metrics into the isolated result namespace."""

    for name, result in (("mfcc", mfcc), ("ecapa", ecapa)):
        persist_single_known_validation_result(output_dir, name, result)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def persist_single_known_validation_result(
    output_dir: Path, name: str, result: dict[str, Any]
) -> None:
    """Persist one representation's prediction table and metrics after its audit."""

    if name not in {"mfcc", "ecapa"}:
        raise ValueError("Representation name must be mfcc or ecapa")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{name}_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(result["predictions"])
    (output_dir / f"{name}_metrics.json").write_text(
        json.dumps(result["metrics"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
