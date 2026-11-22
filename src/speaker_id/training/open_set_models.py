"""Train and persist the frozen MFCC/ECAPA open-set known-speaker models."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from speaker_id.features.ecapa_storage import load_ecapa_features
from speaker_id.features.storage import load_features
from speaker_id.models import decision_scores, make_model
from speaker_id.open_set.protocol import (
    ACTIVE_KNOWN_SPEAKER_IDS,
    OPTIONAL_KNOWN_SPEAKER_IDS,
    TRAINING_SESSION,
    TRAINING_SPLIT,
    TRAINING_TEXT_IDS,
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
EXPECTED_SVC_PARAMETERS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "class_weight": "balanced",
    "decision_function_shape": "ovr",
    "probability": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_training_data(
    mfcc_x: np.ndarray,
    mfcc_y: np.ndarray,
    mfcc_metadata: list[dict[str, str]],
    ecapa_x: np.ndarray,
    ecapa_y: np.ndarray,
    ecapa_segment_ids: np.ndarray,
    ecapa_metadata: list[dict[str, str]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Reject misalignment and any data outside the frozen S01 training scope."""

    mfcc_ids = [row["segment_id"] for row in mfcc_metadata]
    ecapa_ids = ecapa_segment_ids.tolist()
    if not (
        len(mfcc_x)
        == len(mfcc_y)
        == len(mfcc_metadata)
        == len(ecapa_x)
        == len(ecapa_y)
        == len(ecapa_ids)
        == len(ecapa_metadata)
    ):
        raise ValueError("MFCC and ECAPA training row counts differ")
    if mfcc_ids != ecapa_ids:
        raise ValueError("MFCC and ECAPA segment IDs or their order differ")
    if mfcc_y.tolist() != ecapa_y.tolist():
        raise ValueError("MFCC and ECAPA labels differ")
    for field in METADATA_FIELDS:
        if [row[field] for row in mfcc_metadata] != [row[field] for row in ecapa_metadata]:
            raise ValueError(f"MFCC and ECAPA metadata differ for {field}")

    labels = set(mfcc_y.tolist())
    if labels != set(ACTIVE_KNOWN_SPEAKER_IDS):
        raise ValueError(
            "Training labels must contain exactly active-known speakers "
            f"{ACTIVE_KNOWN_SPEAKER_IDS}; received {sorted(labels)}"
        )
    sessions = {row["session_id"] for row in mfcc_metadata}
    splits = {row["split"] for row in mfcc_metadata}
    texts = {row["text_id"] for row in mfcc_metadata}
    if sessions != {TRAINING_SESSION}:
        raise ValueError(f"Ineligible training sessions: {sorted(sessions)}")
    if splits != {TRAINING_SPLIT}:
        raise ValueError(f"Ineligible training splits: {sorted(splits)}")
    if texts != set(TRAINING_TEXT_IDS):
        raise ValueError(f"Training texts must be exactly {TRAINING_TEXT_IDS}")

    recording_rows: dict[str, tuple[str, str, str, str]] = {}
    for row in mfcc_metadata:
        identity = (row["speaker_id"], row["session_id"], row["text_id"], row["split"])
        previous = recording_rows.setdefault(row["recording_id"], identity)
        if previous != identity:
            raise ValueError(f"Conflicting metadata for recording {row['recording_id']}")
    recording_ids = list(recording_rows)
    expected_recordings = provenance.get("source_recordings")
    if (
        not isinstance(expected_recordings, list)
        or recording_ids != expected_recordings
        or provenance.get("recording_count") != len(recording_ids)
    ):
        raise ValueError("Training recordings do not match training protocol provenance")
    expected_pairs = {
        (speaker, text) for speaker in ACTIVE_KNOWN_SPEAKER_IDS for text in TRAINING_TEXT_IDS
    }
    if {(speaker, text) for speaker, _, text, _ in recording_rows.values()} != expected_pairs:
        raise ValueError("Training recordings must cover every active speaker and T01-T05")

    return {
        "rows": int(len(mfcc_y)),
        "recording_ids": recording_ids,
        "recordings": len(recording_ids),
        "class_counts": dict(sorted(Counter(mfcc_y.tolist()).items())),
        "segment_ids": mfcc_ids,
        "labels": mfcc_y.tolist(),
        "metadata": mfcc_metadata,
    }


def _validate_model_factory(model: Any) -> Pipeline:
    if not isinstance(model, Pipeline):
        raise TypeError("make_model() must return an sklearn Pipeline")
    if tuple(name for name, _ in model.steps) != ("scaler", "svm"):
        raise ValueError("Model pipeline must contain scaler then svm")
    if not isinstance(model.named_steps["scaler"], StandardScaler):
        raise TypeError("Pipeline scaler must be StandardScaler")
    if not isinstance(model.named_steps["svm"], SVC):
        raise TypeError("Pipeline classifier must be SVC")
    actual = {key: model.named_steps["svm"].get_params()[key] for key in EXPECTED_SVC_PARAMETERS}
    if actual != EXPECTED_SVC_PARAMETERS:
        raise ValueError(f"SVC factory parameters changed: {actual}")
    return model


def _model_config(
    representation: str,
    dimension: int,
    classes: list[str],
    summary: dict[str, Any],
    segmentation: dict[str, Any],
    protocol_path: Path,
    dataset_dir: Path,
    data_filename: str,
    metadata_filename: str,
    config_filename: str,
    ecapa_model_source: str | None = None,
) -> dict[str, Any]:
    artifact_representation = (
        "ecapa" if representation == "ecapa_embedding" else representation
    )
    config: dict[str, Any] = {
        "representation_type": representation,
        "representation_dimension": dimension,
        "classifier_type": "sklearn.pipeline.Pipeline(StandardScaler, SVC)",
        "classifier_parameters": dict(EXPECTED_SVC_PARAMETERS),
        "scaler_parameters": {"with_mean": True, "with_std": True},
        "classes": classes,
        "training_rows": summary["rows"],
        "training_recordings": summary["recordings"],
        "training_session": TRAINING_SESSION,
        "training_texts": list(TRAINING_TEXT_IDS),
        "training_split": TRAINING_SPLIT,
        "active_known_speakers": list(ACTIVE_KNOWN_SPEAKER_IDS),
        "optional_supported_known": list(OPTIONAL_KNOWN_SPEAKER_IDS),
        "class_distribution": summary["class_counts"],
        "training_dataset": {
            "artifact": f"artifacts/open_set/{artifact_representation}/training/{data_filename}",
            "sha256": _sha256(dataset_dir / data_filename),
            "metadata_sha256": _sha256(dataset_dir / metadata_filename),
            "feature_config_sha256": _sha256(dataset_dir / config_filename),
            "provenance_artifact": "artifacts/open_set/training_protocol.json",
            "provenance_sha256": _sha256(protocol_path),
        },
        "segmentation": segmentation,
        "score_semantics": "SVM decision-function scores; not probabilities; no softmax",
        "threshold_calibrated": False,
    }
    if representation == "ecapa_embedding":
        config["embedding_model_source"] = ecapa_model_source
        config["ecapa_frozen"] = True
    return config


def train_open_set_models(
    mfcc_training_dir: Path,
    ecapa_training_dir: Path,
    protocol_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Fit exactly two factory models on aligned training-only stored features."""

    mfcc_x, mfcc_y, mfcc_metadata, mfcc_feature_config = load_features(mfcc_training_dir)
    ecapa_x, ecapa_y, ecapa_ids, ecapa_metadata, ecapa_config = load_ecapa_features(
        ecapa_training_dir
    )
    provenance = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise ValueError("Training protocol provenance must be a JSON object")
    if (
        provenance.get("active_known_speakers") != list(ACTIVE_KNOWN_SPEAKER_IDS)
        or provenance.get("optional_supported_known") != list(OPTIONAL_KNOWN_SPEAKER_IDS)
        or provenance.get("training_session") != TRAINING_SESSION
        or provenance.get("training_split") != TRAINING_SPLIT
        or provenance.get("training_texts") != list(TRAINING_TEXT_IDS)
    ):
        raise ValueError("Training protocol provenance differs from the frozen eligibility policy")
    summary = _validate_training_data(
        mfcc_x,
        mfcc_y,
        mfcc_metadata,
        ecapa_x,
        ecapa_y,
        ecapa_ids,
        ecapa_metadata,
        provenance,
    )
    if (
        mfcc_x.shape[1] != provenance["mfcc"]["feature_dimension"]
        or ecapa_x.shape[1] != provenance["ecapa"]["embedding_dimension"]
        or mfcc_feature_config != provenance["mfcc"]["config"]
        or ecapa_config.model_source != provenance["ecapa"]["model_source"]
        or provenance["ecapa"]["frozen"] is not True
    ):
        raise ValueError("Feature configurations differ from frozen provenance")

    mfcc_model_dir = output_root / "mfcc" / "model"
    ecapa_model_dir = output_root / "ecapa" / "model"
    if mfcc_model_dir.exists() or ecapa_model_dir.exists():
        raise FileExistsError("Refusing to overwrite an existing open-set model directory")

    mfcc_model = _validate_model_factory(make_model())
    ecapa_model = _validate_model_factory(make_model())
    if mfcc_model is ecapa_model:
        raise ValueError("make_model() must return a fresh object for each representation")
    mfcc_model.fit(mfcc_x, mfcc_y)
    ecapa_model.fit(ecapa_x, ecapa_y)

    model_specs = (
        (
            "mfcc",
            mfcc_model,
            mfcc_x,
            mfcc_training_dir,
            "features.npz",
            "feature_metadata.csv",
            "feature_config.json",
            mfcc_feature_config,
            None,
        ),
        (
            "ecapa",
            ecapa_model,
            ecapa_x,
            ecapa_training_dir,
            "embeddings.npz",
            "embedding_metadata.csv",
            "embedding_config.json",
            ecapa_config,
            ecapa_config.model_source,
        ),
    )
    results: dict[str, Any] = {}
    for (
        name,
        model,
        x,
        training_dir,
        data_filename,
        metadata_filename,
        config_filename,
        feature_config,
        model_source,
    ) in model_specs:
        class_order = [str(value) for value in model.named_steps["svm"].classes_]
        if set(class_order) != set(ACTIVE_KNOWN_SPEAKER_IDS):
            raise ValueError(f"Fitted {name} classifier has unexpected classes: {class_order}")
        model_dir = output_root / name / "model"
        model_dir.mkdir(parents=True)
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)
        reloaded = joblib.load(model_path)
        if (
            not isinstance(reloaded, Pipeline)
            or tuple(step for step, _ in reloaded.steps) != ("scaler", "svm")
            or not isinstance(reloaded.named_steps["scaler"], StandardScaler)
            or not isinstance(reloaded.named_steps["svm"], SVC)
            or [str(value) for value in reloaded.named_steps["svm"].classes_] != class_order
        ):
            raise ValueError(f"Serialized {name} model failed reload validation")
        prediction = str(reloaded.predict(x[:1])[0])
        score_classes, scores = decision_scores(reloaded, x[:1])
        if score_classes != class_order or scores.shape != (1, len(class_order)):
            raise ValueError(f"Serialized {name} model failed decision score smoke test")
        scaler_fitted = (
            hasattr(reloaded.named_steps["scaler"], "mean_")
            and hasattr(reloaded.named_steps["scaler"], "scale_")
        )
        svm_fitted = (
            hasattr(reloaded.named_steps["svm"], "classes_")
            and hasattr(reloaded.named_steps["svm"], "support_vectors_")
        )
        if not scaler_fitted or not svm_fitted:
            raise ValueError(f"Serialized {name} model is not fitted")
        segmentation = (
            feature_config.to_dict()
            if hasattr(feature_config, "to_dict")
            else {
                key: feature_config[key]
                for key in (
                    "sample_rate_hz",
                    "segment_duration_seconds",
                    "segment_hop_seconds",
                    "minimum_final_segment_seconds",
                )
            }
        )
        # Keep only segmentation fields in the shared model-config section.
        segmentation = {
            key: segmentation[key]
            for key in (
                "sample_rate_hz",
                "segment_duration_seconds",
                "segment_hop_seconds",
                "minimum_final_segment_seconds",
            )
        }
        config = _model_config(
            "mfcc" if name == "mfcc" else "ecapa_embedding",
            int(x.shape[1]),
            class_order,
            summary,
            segmentation,
            protocol_path,
            training_dir,
            data_filename,
            metadata_filename,
            config_filename,
            model_source,
        )
        (model_dir / "model_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        results[name] = {
            "model_path": model_path,
            "model_size_bytes": model_path.stat().st_size,
            "dimension": int(x.shape[1]),
            "classes": class_order,
            "fit_rows": int(len(x)),
            "scaler_fitted": scaler_fitted,
            "svm_fitted": svm_fitted,
            "support_vectors": int(reloaded.named_steps["svm"].support_vectors_.shape[0]),
            "prediction": prediction,
            "true_label": str(mfcc_y[0]),
            "segment_id": mfcc_metadata[0]["segment_id"],
            "scores": dict(zip(score_classes, scores[0].tolist(), strict=True)),
            "config": config,
        }

    return {
        "summary": summary,
        "mfcc": results["mfcc"],
        "ecapa": results["ecapa"],
        "no_threshold_calibrated": True,
        "no_open_set_evaluation": True,
    }
