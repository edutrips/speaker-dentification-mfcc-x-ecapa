"""Validation-only recording-score extraction and open-set calibration integration."""

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from speaker_id.models import mean_recording_scores

from .protocol import (
    ACTIVE_KNOWN_SPEAKER_IDS,
    UNKNOWN_VALIDATION_SPEAKER_IDS,
    VALIDATION_SESSION,
    VALIDATION_SPLIT,
    VALIDATION_TEXT_IDS,
)
from .rejection import (
    UNKNOWN_LABEL,
    ThresholdCalibration,
    calibrate_threshold,
    open_set_decision,
)

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
REPRESENTATION_DIMENSIONS = {"mfcc": 120, "ecapa": 192}
RECORDING_SCORE_FIELDS = (
    "recording_id",
    "true_speaker",
    "text_id",
    "group",
    "top_class",
    "top_score",
    "second_best_score",
    "score_margin",
    *(f"mean_score_{speaker}" for speaker in ACTIVE_KNOWN_SPEAKER_IDS),
    "threshold",
    "accepted",
    "open_set_prediction",
    "outcome",
)


@dataclass(frozen=True, slots=True)
class RecordingScores:
    """Ordered recording-level score row plus source metadata."""

    values: dict[str, Any]
    mean_scores: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ValidationCalibration:
    """One representation's independently calibrated validation results."""

    representation: str
    classes: tuple[str, ...]
    calibration: ThresholdCalibration
    rows: tuple[dict[str, Any], ...]
    known_top_scores: tuple[float, ...]
    unknown_top_scores: tuple[float, ...]
    known_segment_count: int
    unknown_segment_count: int
    model_sha256: str
    known_dataset_sha256: str
    unknown_dataset_sha256: str


def validate_validation_dataset(
    x: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, str]],
    *,
    group: str,
    expected_dimension: int,
) -> dict[str, Any]:
    """Require exact known/unknown S02 validation coverage and aligned finite rows."""

    if group not in {"known", "unknown"}:
        raise ValueError("group must be 'known' or 'unknown'")
    expected_speakers = (
        ACTIVE_KNOWN_SPEAKER_IDS if group == "known" else UNKNOWN_VALIDATION_SPEAKER_IDS
    )
    expected_recordings = 20 if group == "known" else 15
    if (
        not isinstance(x, np.ndarray)
        or x.ndim != 2
        or x.dtype != np.float32
        or x.shape[1] != expected_dimension
        or not np.isfinite(x).all()
        or not isinstance(labels, np.ndarray)
        or labels.ndim != 1
        or len(x) != len(labels)
        or len(x) != len(metadata)
        or len(x) == 0
    ):
        raise ValueError(f"{group} validation feature matrix is invalid or misaligned")
    if any(any(not row.get(field) for field in METADATA_FIELDS) for row in metadata):
        raise ValueError(f"{group} validation metadata is incomplete")
    if [row["speaker_id"] for row in metadata] != labels.tolist():
        raise ValueError(f"{group} validation labels differ from metadata")
    if len({row["segment_id"] for row in metadata}) != len(metadata):
        raise ValueError(f"{group} validation has duplicate segment IDs")

    recordings: dict[str, tuple[str, str, str, str]] = {}
    for row in metadata:
        if row["speaker_id"] not in expected_speakers:
            raise ValueError(f"Ineligible {group} validation speaker: {row['speaker_id']}")
        if row["session_id"] != VALIDATION_SESSION or row["split"] != VALIDATION_SPLIT:
            raise ValueError(f"{group} validation rows must be S02/validation")
        if row["text_id"] not in VALIDATION_TEXT_IDS:
            raise ValueError(f"Ineligible {group} validation text: {row['text_id']}")
        record_contract = (
            row["speaker_id"], row["session_id"], row["text_id"], row["split"]
        )
        previous = recordings.setdefault(row["recording_id"], record_contract)
        if previous != record_contract:
            raise ValueError(f"Conflicting metadata for {row['recording_id']}")

    expected_pairs = {
        (speaker, text)
        for speaker in expected_speakers
        for text in VALIDATION_TEXT_IDS
    }
    observed_pairs = {(identity[0], identity[2]) for identity in recordings.values()}
    if (
        len(recordings) != expected_recordings
        or observed_pairs != expected_pairs
        or Counter(identity[0] for identity in recordings.values())
        != Counter({speaker: 5 for speaker in expected_speakers})
    ):
        raise ValueError(f"{group} validation does not have exact recording/text coverage")
    return {
        "recording_count": len(recordings),
        "segment_count": len(metadata),
        "speakers": list(expected_speakers),
        "sessions": sorted({row["session_id"] for row in metadata}),
        "texts": sorted({row["text_id"] for row in metadata}),
        "recording_order": list(recordings),
        "segment_ids": [row["segment_id"] for row in metadata],
        "labels": labels.tolist(),
        "metadata": [tuple(row[field] for field in METADATA_FIELDS) for row in metadata],
    }


def validate_representation_alignment(
    left: tuple[np.ndarray, np.ndarray, list[dict[str, str]]],
    right: tuple[np.ndarray, np.ndarray, list[dict[str, str]]],
    *,
    group: str,
) -> dict[str, Any]:
    """Compare identity/order metadata only; representation vectors are never compared."""

    left_x, left_y, left_metadata = left
    right_x, right_y, right_metadata = right
    if len(left_x) != len(right_x) or left_y.tolist() != right_y.tolist():
        raise ValueError(f"{group} MFCC/ECAPA row counts or labels differ")
    for field in METADATA_FIELDS:
        if [row[field] for row in left_metadata] != [row[field] for row in right_metadata]:
            raise ValueError(f"{group} MFCC/ECAPA metadata differs for {field}")
    return {
        "group": group,
        "segments": len(left_x),
        "recording_order": list(dict.fromkeys(row["recording_id"] for row in left_metadata)),
        "aligned": True,
    }


def extract_recording_scores(
    model: Any,
    x: np.ndarray,
    metadata: list[dict[str, str]],
    *,
    group: str,
    classes: Sequence[str],
) -> tuple[RecordingScores, ...]:
    """Aggregate each recording using the project's shared mean-score helper."""

    class_order = tuple(classes)
    if group not in {"known", "unknown"}:
        raise ValueError("group must be 'known' or 'unknown'")
    if tuple(str(value) for value in model.classes_) != class_order:
        raise ValueError("Supplied classes do not preserve the model class order")
    if set(class_order) != set(ACTIVE_KNOWN_SPEAKER_IDS) or UNKNOWN_LABEL in class_order:
        raise ValueError("The fitted model must contain exactly the four known classes")
    if x.ndim != 2 or len(x) != len(metadata) or not np.isfinite(x).all():
        raise ValueError("Validation matrix is invalid or misaligned")

    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        grouped.setdefault(row["recording_id"], []).append(index)

    rows: list[RecordingScores] = []
    for recording_id, positions in grouped.items():
        first = metadata[positions[0]]
        if any(
            (row["speaker_id"], row["text_id"], row["session_id"], row["split"])
            != (first["speaker_id"], first["text_id"], first["session_id"], first["split"])
            for row in (metadata[index] for index in positions)
        ):
            raise ValueError(f"Recording {recording_id} contains mixed metadata")
        top_class, score_mapping = mean_recording_scores(model, x[positions])
        if tuple(score_mapping) != class_order:
            raise ValueError("Recording score class order differs from fitted model")
        scores = tuple(float(score_mapping[label]) for label in class_order)
        if not all(np.isfinite(score) for score in scores):
            raise ValueError(f"Recording {recording_id} produced non-finite mean scores")
        ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        top_index, second_index = ranked[:2]
        if top_class != class_order[top_index]:
            raise ValueError("Shared recording-score helper disagrees with score argmax")
        rows.append(
            RecordingScores(
                values={
                    "recording_id": recording_id,
                    "true_speaker": first["speaker_id"],
                    "text_id": first["text_id"],
                    "group": group,
                    "top_class": class_order[top_index],
                    "top_score": scores[top_index],
                    "second_best_score": scores[second_index],
                    "score_margin": scores[top_index] - scores[second_index],
                    **{
                        f"mean_score_{label}": score
                        for label, score in zip(class_order, scores, strict=True)
                    },
                },
                mean_scores=scores,
            )
        )
    return tuple(rows)


def calibrate_validation_scores(
    representation: str,
    classes: Sequence[str],
    known_rows: Sequence[RecordingScores],
    unknown_rows: Sequence[RecordingScores],
    *,
    known_segment_count: int,
    unknown_segment_count: int,
    model_sha256: str,
    known_dataset_sha256: str,
    unknown_dataset_sha256: str,
) -> ValidationCalibration:
    """Calibrate one representation and apply its threshold to these same rows."""

    if representation not in REPRESENTATION_DIMENSIONS:
        raise ValueError(f"Unsupported representation: {representation}")
    if len(known_rows) != 20 or len(unknown_rows) != 15:
        raise ValueError("Calibration requires 20 known and 15 unknown recordings")
    if any(row.values["group"] != "known" for row in known_rows) or any(
        row.values["group"] != "unknown" for row in unknown_rows
    ):
        raise ValueError("Known and unknown score groups must remain separate")
    known_ids = {row.values["recording_id"] for row in known_rows}
    unknown_ids = {row.values["recording_id"] for row in unknown_rows}
    if len(known_ids) != len(known_rows) or len(unknown_ids) != len(unknown_rows):
        raise ValueError("Duplicate recording-level calibration rows")
    if known_ids & unknown_ids:
        raise ValueError("Known and unknown validation recording IDs overlap")

    known_top_scores = tuple(float(row.values["top_score"]) for row in known_rows)
    unknown_top_scores = tuple(float(row.values["top_score"]) for row in unknown_rows)
    calibration = calibrate_threshold(known_top_scores, unknown_top_scores)
    result_rows: list[dict[str, Any]] = []
    class_order = tuple(classes)
    for row in (*known_rows, *unknown_rows):
        decision = open_set_decision(row.mean_scores, class_order, calibration.threshold)
        if (
            decision.top_class != row.values["top_class"]
            or decision.top_score != row.values["top_score"]
        ):
            raise ValueError("Open-set decision disagrees with recording-level score row")
        if row.values["group"] == "known":
            outcome = (
                "false_reject" if not decision.accepted else
                "correct_known" if decision.prediction == row.values["true_speaker"] else
                "known_to_known_misidentification"
            )
        else:
            outcome = "false_accept" if decision.accepted else "correct_unknown_rejection"
        result_rows.append(
            {
                **row.values,
                "threshold": decision.threshold,
                "accepted": decision.accepted,
                "open_set_prediction": decision.prediction,
                "outcome": outcome,
            }
        )
    return ValidationCalibration(
        representation=representation,
        classes=class_order,
        calibration=calibration,
        rows=tuple(result_rows),
        known_top_scores=known_top_scores,
        unknown_top_scores=unknown_top_scores,
        known_segment_count=known_segment_count,
        unknown_segment_count=unknown_segment_count,
        model_sha256=model_sha256,
        known_dataset_sha256=known_dataset_sha256,
        unknown_dataset_sha256=unknown_dataset_sha256,
    )


def persist_validation_calibration(
    result: ValidationCalibration,
    output_dir: Path,
    *,
    known_summary: Mapping[str, Any],
    unknown_summary: Mapping[str, Any],
) -> None:
    """Write deterministic calibration JSON, complete curve, and 35 decision rows."""

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite calibration artifacts: {output_dir}")
    output_dir.mkdir(parents=True)
    calibration = result.calibration
    curve = [
        {
            "threshold": point.threshold,
            "far": point.far,
            "frr": point.frr,
            "objective": point.objective,
        }
        for point in calibration.candidates
    ]
    correct_outcomes = {"correct_known", "correct_unknown_rejection"}
    errors = [row for row in result.rows if row["outcome"] not in correct_outcomes]
    payload = {
        "artifact_type": "validation_threshold_calibration",
        "representation": result.representation,
        "threshold": calibration.threshold,
        "far": calibration.far,
        "frr": calibration.frr,
        "objective": calibration.objective,
        "known_count": calibration.known_count,
        "unknown_count": calibration.unknown_count,
        "candidate_count": calibration.candidate_count,
        "class_order": list(result.classes),
        "model_contract": {
            "pipeline": ["StandardScaler", "SVC"],
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": "balanced",
            "decision_function_shape": "ovr",
            "probability": False,
        },
        "aggregation": "mean SVM decision-function scores per recording",
        "score_semantics": (
            "SVM decision-function scores; not probabilities; no softmax or normalization"
        ),
        "objective_definition": "0.5 * FAR + 0.5 * FRR",
        "threshold_tie_break": "highest threshold among exact objective ties",
        "decision_inequality": "top_score >= threshold accepts the top known class",
        "training_session": "S01",
        "validation_session": VALIDATION_SESSION,
        "validation_split": VALIDATION_SPLIT,
        "validation_texts": list(VALIDATION_TEXT_IDS),
        "known_speakers": list(ACTIVE_KNOWN_SPEAKER_IDS),
        "unknown_validation_speakers": list(UNKNOWN_VALIDATION_SPEAKER_IDS),
        "known_recording_count": calibration.known_count,
        "unknown_recording_count": calibration.unknown_count,
        "known_segment_count": result.known_segment_count,
        "unknown_segment_count": result.unknown_segment_count,
        "known_dataset": dict(known_summary),
        "unknown_dataset": dict(unknown_summary),
        "model_artifact_sha256": result.model_sha256,
        "known_dataset_sha256": result.known_dataset_sha256,
        "unknown_dataset_sha256": result.unknown_dataset_sha256,
        "validation_calibration_performance": {
            "correct_known_accepted": sum(row["outcome"] == "correct_known" for row in result.rows),
            "false_rejects": sum(row["outcome"] == "false_reject" for row in result.rows),
            "known_to_known_misidentifications": sum(
                row["outcome"] == "known_to_known_misidentification" for row in result.rows
            ),
            "correct_unknown_rejections": sum(
                row["outcome"] == "correct_unknown_rejection" for row in result.rows
            ),
            "false_accepts": sum(row["outcome"] == "false_accept" for row in result.rows),
            "error_recording_ids": [row["recording_id"] for row in errors],
            "is_independent_test_performance": False,
        },
    }
    (output_dir / "calibration.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "threshold_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("threshold", "far", "frr", "objective"))
        writer.writeheader()
        writer.writerows(curve)
    with (output_dir / "recording_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RECORDING_SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(result.rows)
