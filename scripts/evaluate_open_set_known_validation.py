"""Evaluate persisted S01 models on stored, known-only S02 representations."""

import hashlib
import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from speaker_id.evaluation.known_validation import (
    compare_known_validation,
    evaluate_known_validation,
    persist_known_validation_results,
    persist_single_known_validation_result,
    validate_aligned_representations,
)
from speaker_id.features.ecapa_storage import load_ecapa_features
from speaker_id.features.storage import load_features

ROOT = Path(__file__).resolve().parents[1]
OPEN_SET = ROOT / "artifacts/open_set"
OUTPUT = OPEN_SET / "evaluation/known_validation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.iterdir() if item.is_file()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_fitted_model(path: Path) -> tuple[Pipeline, list[str]]:
    model = joblib.load(path)
    if not isinstance(model, Pipeline) or tuple(name for name, _ in model.steps) != (
        "scaler", "svm"
    ):
        raise TypeError(f"Unexpected model pipeline at {path}")
    scaler, svm = model.named_steps["scaler"], model.named_steps["svm"]
    if not isinstance(scaler, StandardScaler) or not isinstance(svm, SVC):
        raise TypeError(f"Expected fitted StandardScaler and SVC at {path}")
    check_is_fitted(scaler)
    check_is_fitted(svm)
    if svm.probability is not False:
        raise ValueError("SVC must use probability=False")
    classes = [str(value) for value in model.classes_]
    if len(classes) != 4 or set(classes) != {"V01", "V02", "V03", "V04"}:
        raise ValueError(f"Unexpected model classes: {classes}")
    return model, classes


def main() -> None:
    mfcc_dir = OPEN_SET / "mfcc/validation_known"
    ecapa_dir = OPEN_SET / "ecapa/validation_known"
    mfcc_x, mfcc_y, mfcc_metadata, _ = load_features(mfcc_dir)
    ecapa_x, ecapa_y, ecapa_ids, ecapa_metadata, _ = load_ecapa_features(ecapa_dir)
    alignment = validate_aligned_representations(
        mfcc_x,
        mfcc_y,
        mfcc_metadata,
        ecapa_x,
        ecapa_y,
        ecapa_ids,
        ecapa_metadata,
    )

    # Models are loaded only after both persisted validation datasets pass all gates.
    mfcc_model, mfcc_classes = _load_fitted_model(OPEN_SET / "mfcc/model/model.joblib")
    ecapa_model, ecapa_classes = _load_fitted_model(OPEN_SET / "ecapa/model/model.joblib")
    if mfcc_classes != ecapa_classes:
        raise ValueError("MFCC and ECAPA model class orders differ")
    mfcc_result = evaluate_known_validation(
        mfcc_x,
        mfcc_y,
        mfcc_metadata,
        mfcc_classes,
        mfcc_model,
        "mfcc",
        _sha256(OPEN_SET / "mfcc/model/model.joblib"),
        _dataset_sha256(mfcc_dir),
    )
    mfcc_metrics = mfcc_result["metrics"]
    if (
        mfcc_metrics["recording_count"] != 20
        or mfcc_metrics["threshold_used"] is not None
        or mfcc_metrics["unknown_predictions_generated"]
        or len(mfcc_result["predictions"]) != 20
    ):
        raise ValueError("MFCC audit failed: expected 20 recording-level predictions")
    persist_single_known_validation_result(OUTPUT, "mfcc", mfcc_result)

    ecapa_result = evaluate_known_validation(
        ecapa_x,
        ecapa_y,
        ecapa_metadata,
        ecapa_classes,
        ecapa_model,
        "ecapa",
        _sha256(OPEN_SET / "ecapa/model/model.joblib"),
        _dataset_sha256(ecapa_dir),
    )
    ecapa_metrics = ecapa_result["metrics"]
    if (
        ecapa_metrics["recording_count"] != 20
        or ecapa_metrics["threshold_used"] is not None
        or ecapa_metrics["unknown_predictions_generated"]
        or len(ecapa_result["predictions"]) != 20
    ):
        raise ValueError("ECAPA audit failed: expected 20 recording-level predictions")
    persist_single_known_validation_result(OUTPUT, "ecapa", ecapa_result)

    comparison = compare_known_validation(mfcc_result, ecapa_result)
    persist_known_validation_results(OUTPUT, mfcc_result, ecapa_result, comparison)
    print(
        json.dumps(
            {
                "alignment": alignment,
                "mfcc": mfcc_result["metrics"],
                "ecapa": ecapa_result["metrics"],
                "comparison": comparison,
                "output": str(OUTPUT.relative_to(ROOT)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
