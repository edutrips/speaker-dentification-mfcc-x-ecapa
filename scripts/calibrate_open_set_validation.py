"""Calibrate representation-specific rejection thresholds on S02 validation only."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from scripts.evaluate_open_set_known_validation import (
    _dataset_sha256,
    _load_fitted_model,
    _sha256,
)

from speaker_id.features.ecapa_storage import load_ecapa_features
from speaker_id.features.storage import load_features
from speaker_id.open_set.calibration import (
    calibrate_validation_scores,
    extract_recording_scores,
    persist_validation_calibration,
    validate_representation_alignment,
    validate_validation_dataset,
)
from speaker_id.open_set.protocol import ACTIVE_KNOWN_SPEAKER_IDS

ROOT = Path(__file__).resolve().parents[1]
OPEN_SET = ROOT / "artifacts/open_set"
OUTPUT_ROOT = OPEN_SET / "calibration"


def _load_representation_data(representation: str) -> dict[str, Any]:
    mfcc_known_dir = OPEN_SET / "mfcc/validation_known"
    mfcc_unknown_dir = OPEN_SET / "mfcc/validation_unknown"
    ecapa_known_dir = OPEN_SET / "ecapa/validation_known"
    ecapa_unknown_dir = OPEN_SET / "ecapa/validation_unknown"

    mfcc_known_x, mfcc_known_y, mfcc_known_meta, mfcc_known_config = load_features(
        mfcc_known_dir
    )
    mfcc_unknown_x, mfcc_unknown_y, mfcc_unknown_meta, mfcc_unknown_config = load_features(
        mfcc_unknown_dir
    )
    ecapa_known_x, ecapa_known_y, _known_ids, ecapa_known_meta, ecapa_known_config = (
        load_ecapa_features(ecapa_known_dir)
    )
    ecapa_unknown_x, ecapa_unknown_y, _unknown_ids, ecapa_unknown_meta, ecapa_unknown_config = (
        load_ecapa_features(ecapa_unknown_dir)
    )

    known_alignment = validate_representation_alignment(
        (mfcc_known_x, mfcc_known_y, mfcc_known_meta),
        (ecapa_known_x, ecapa_known_y, ecapa_known_meta),
        group="known",
    )
    unknown_alignment = validate_representation_alignment(
        (mfcc_unknown_x, mfcc_unknown_y, mfcc_unknown_meta),
        (ecapa_unknown_x, ecapa_unknown_y, ecapa_unknown_meta),
        group="unknown",
    )
    known_summary = validate_validation_dataset(
        mfcc_known_x,
        mfcc_known_y,
        mfcc_known_meta,
        group="known",
        expected_dimension=120,
    )
    unknown_summary = validate_validation_dataset(
        mfcc_unknown_x,
        mfcc_unknown_y,
        mfcc_unknown_meta,
        group="unknown",
        expected_dimension=120,
    )
    ecapa_known_summary = validate_validation_dataset(
        ecapa_known_x,
        ecapa_known_y,
        ecapa_known_meta,
        group="known",
        expected_dimension=192,
    )
    ecapa_unknown_summary = validate_validation_dataset(
        ecapa_unknown_x,
        ecapa_unknown_y,
        ecapa_unknown_meta,
        group="unknown",
        expected_dimension=192,
    )
    if mfcc_known_config.get("n_mfcc") != 20 or mfcc_known_config != mfcc_unknown_config:
        raise ValueError("Known and unknown MFCC configurations differ from canonical MFCC")
    if ecapa_known_config != ecapa_unknown_config:
        raise ValueError("Known and unknown ECAPA configurations differ")

    if (
        set(known_summary["recording_order"]) & set(unknown_summary["recording_order"])
        or set(known_summary["segment_ids"]) & set(unknown_summary["segment_ids"])
    ):
        raise ValueError("Known and unknown validation datasets overlap")

    if representation == "mfcc":
        return {
            "known_x": mfcc_known_x,
            "known_metadata": mfcc_known_meta,
            "unknown_x": mfcc_unknown_x,
            "unknown_metadata": mfcc_unknown_meta,
            "known_summary": known_summary,
            "unknown_summary": unknown_summary,
            "known_dir": mfcc_known_dir,
            "unknown_dir": mfcc_unknown_dir,
            "model_path": OPEN_SET / "mfcc/model/model.joblib",
            "alignment": {"known": known_alignment, "unknown": unknown_alignment},
            "dimension": 120,
        }
    if representation == "ecapa":
        return {
            "known_x": ecapa_known_x,
            "known_metadata": ecapa_known_meta,
            "unknown_x": ecapa_unknown_x,
            "unknown_metadata": ecapa_unknown_meta,
            "known_summary": ecapa_known_summary,
            "unknown_summary": ecapa_unknown_summary,
            "known_dir": ecapa_known_dir,
            "unknown_dir": ecapa_unknown_dir,
            "model_path": OPEN_SET / "ecapa/model/model.joblib",
            "alignment": {"known": known_alignment, "unknown": unknown_alignment},
            "dimension": 192,
        }
    raise ValueError(f"Unsupported representation: {representation}")


def _check_output_paths() -> None:
    for representation in ("mfcc", "ecapa"):
        output = OUTPUT_ROOT / representation
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite calibration output: {output}")


def _run_representation(representation: str, data: dict[str, Any]):
    model, classes = _load_fitted_model(data["model_path"])
    if classes != list(ACTIVE_KNOWN_SPEAKER_IDS):
        raise ValueError(f"Unexpected {representation} class order: {classes}")
    svm = model.named_steps["svm"]
    expected_svc_parameters = {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "class_weight": "balanced",
        "decision_function_shape": "ovr",
        "probability": False,
    }
    actual_svc_parameters = svm.get_params(deep=False)
    if any(
        actual_svc_parameters.get(key) != value
        for key, value in expected_svc_parameters.items()
    ):
        raise ValueError(f"{representation} SVC parameters differ from frozen protocol")
    if int(model.n_features_in_) != data["dimension"]:
        raise ValueError(
            f"{representation} model input dimension {model.n_features_in_} "
            f"does not match validation data {data['dimension']}"
        )
    known_rows = extract_recording_scores(
        model,
        data["known_x"],
        data["known_metadata"],
        group="known",
        classes=classes,
    )
    unknown_rows = extract_recording_scores(
        model,
        data["unknown_x"],
        data["unknown_metadata"],
        group="unknown",
        classes=classes,
    )
    return calibrate_validation_scores(
        representation,
        classes,
        known_rows,
        unknown_rows,
        known_segment_count=len(data["known_x"]),
        unknown_segment_count=len(data["unknown_x"]),
        model_sha256=_sha256(data["model_path"]),
        known_dataset_sha256=_dataset_sha256(data["known_dir"]),
        unknown_dataset_sha256=_dataset_sha256(data["unknown_dir"]),
    )


def _score_summary(scores: tuple[float, ...]) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def main() -> None:
    _check_output_paths()
    # Validate all S02 datasets and cross-representation alignment before model loading.
    mfcc_data = _load_representation_data("mfcc")
    ecapa_data = _load_representation_data("ecapa")
    mfcc_result = _run_representation("mfcc", mfcc_data)
    ecapa_result = _run_representation("ecapa", ecapa_data)

    for result, data in ((mfcc_result, mfcc_data), (ecapa_result, ecapa_data)):
        persist_validation_calibration(
            result,
            OUTPUT_ROOT / result.representation,
            known_summary={
                **data["known_summary"],
                "top_score_summary": _score_summary(result.known_top_scores),
            },
            unknown_summary={
                **data["unknown_summary"],
                "top_score_summary": _score_summary(result.unknown_top_scores),
            },
        )

    report = {}
    for result, data in ((mfcc_result, mfcc_data), (ecapa_result, ecapa_data)):
        cal = result.calibration
        outcomes = {
            outcome: sum(row["outcome"] == outcome for row in result.rows)
            for outcome in (
                "correct_known",
                "false_reject",
                "known_to_known_misidentification",
                "correct_unknown_rejection",
                "false_accept",
            )
        }
        report[result.representation] = {
            "threshold": cal.threshold,
            "far": cal.far,
            "frr": cal.frr,
            "objective": cal.objective,
            "candidate_count": cal.candidate_count,
            "known_top_scores": _score_summary(result.known_top_scores),
            "unknown_top_scores": _score_summary(result.unknown_top_scores),
            "outcomes": outcomes,
            "error_recording_ids": [
                row["recording_id"]
                for row in result.rows
                if row["outcome"] not in {"correct_known", "correct_unknown_rejection"}
            ],
            "model_sha256": result.model_sha256,
            "known_dataset_sha256": result.known_dataset_sha256,
            "unknown_dataset_sha256": result.unknown_dataset_sha256,
            "alignment": data["alignment"],
            "artifact_path": str((OUTPUT_ROOT / result.representation).relative_to(ROOT)),
        }
    print(json.dumps(report, indent=2, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
