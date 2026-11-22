"""Build known-only S02 validation MFCC and ECAPA datasets."""

import argparse
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from speaker_id.datasets import RecordingRecord, load_manifest
from speaker_id.features import (
    EcapaFeatureConfig,
    load_ecapa_features,
    load_mfcc_config,
)
from speaker_id.features.pipeline import features_for_recording
from speaker_id.features.storage import load_features, save_features
from speaker_id.features.types import FeatureRecord
from speaker_id.open_set.protocol import (
    ACTIVE_KNOWN_SPEAKER_IDS,
    TRAINING_SESSION,
    UNKNOWN_VALIDATION_SPEAKER_IDS,
    VALIDATION_KNOWN_TEXT_IDS,
    VALIDATION_SESSION,
    VALIDATION_SPLIT,
    select_open_set_validation_known_records,
    select_open_set_validation_unknown_records,
)

if TYPE_CHECKING:
    from speaker_id.features.ecapa import EcapaEncoder

ROOT = Path(__file__).resolve().parents[1]
MFCC_OUTPUT = Path("artifacts/open_set/mfcc/validation_known")
ECAPA_OUTPUT = Path("artifacts/open_set/ecapa/validation_known")
VALIDATION_TARGETS = {
    "known": {
        "speakers": ACTIVE_KNOWN_SPEAKER_IDS,
        "count": 20,
        "artifact_type": "validation_known",
        "mfcc_output": MFCC_OUTPUT,
        "ecapa_output": ECAPA_OUTPUT,
        "selector": select_open_set_validation_known_records,
    },
    "unknown": {
        "speakers": UNKNOWN_VALIDATION_SPEAKER_IDS,
        "count": 15,
        "artifact_type": "validation_unknown",
        "mfcc_output": Path("artifacts/open_set/mfcc/validation_unknown"),
        "ecapa_output": Path("artifacts/open_set/ecapa/validation_unknown"),
        "selector": select_open_set_validation_unknown_records,
    },
}
PROVENANCE_FILENAME = "validation_provenance.json"
SEGMENT_METADATA_FIELDS = (
    "segment_id",
    "recording_id",
    "speaker_id",
    "session_id",
    "text_id",
    "split",
    "start_seconds",
    "end_seconds",
)
CLIPPING_SUSPECT_FRACTION = 0.001
LEVEL_OUTLIER_DB = 12.0


def select_and_audit_validation_records(
    records: Iterable[RecordingRecord], *, target: str = "known"
) -> list[RecordingRecord]:
    """Select a validation target and require its exact speaker/text coverage."""

    if target not in VALIDATION_TARGETS:
        raise ValueError(f"Unsupported validation target: {target!r}")
    contract = VALIDATION_TARGETS[target]
    selected = contract["selector"](records)
    expected = {
        (speaker, text)
        for speaker in contract["speakers"]
        for text in VALIDATION_KNOWN_TEXT_IDS
    }
    observed = Counter((record.speaker_id, record.text_id) for record in selected)
    if (
        observed != Counter({pair: 1 for pair in expected})
        or len(selected) != contract["count"]
        or any(
            record.session_id != VALIDATION_SESSION or record.split != VALIDATION_SPLIT
            for record in selected
        )
    ):
        target_label = contract["artifact_type"]
        raise ValueError(
            f"{target_label} selection must contain exactly one S02/validation "
            "record for every expected speaker and T06-T10"
        )
    return selected


def audio_level_metrics(samples: np.ndarray) -> dict[str, float | None]:
    """Compute read-only peak, RMS, dBFS, and clipping fraction for canonical audio."""

    if (
        not isinstance(samples, np.ndarray)
        or samples.ndim != 1
        or samples.dtype != np.float32
        or samples.size == 0
        or not np.isfinite(samples).all()
    ):
        raise ValueError("samples must be non-empty, finite, one-dimensional float32 audio")
    absolute = np.abs(samples)
    peak = float(np.max(absolute))
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return {
        "peak": peak,
        "rms": rms,
        "peak_dbfs": 20.0 * math.log10(peak) if peak > 0.0 else None,
        "rms_dbfs": 20.0 * math.log10(rms) if rms > 0.0 else None,
        "clipping_fraction": float(np.mean(absolute >= 0.99)),
    }


def audio_level_diagnostic_flags(
    measurements: Iterable[dict[str, Any]],
    *,
    level_outlier_db: float = LEVEL_OUTLIER_DB,
    clipping_fraction_threshold: float = CLIPPING_SUSPECT_FRACTION,
) -> list[dict[str, Any]]:
    """Flag within-speaker/session level outliers and relevant clipping only."""

    items = list(measurements)
    if not math.isfinite(level_outlier_db) or level_outlier_db <= 0.0:
        raise ValueError("level_outlier_db must be positive and finite")
    if (
        not math.isfinite(clipping_fraction_threshold)
        or not 0.0 < clipping_fraction_threshold <= 1.0
    ):
        raise ValueError("clipping_fraction_threshold must be in (0, 1]")
    groups: dict[tuple[str, str], list[float]] = {}
    for item in items:
        level = item.get("rms_dbfs")
        if level is not None:
            if not math.isfinite(float(level)):
                raise ValueError("rms_dbfs values must be finite or None")
            groups.setdefault((item["speaker_id"], item["session_id"]), []).append(float(level))

    medians = {key: float(np.median(values)) for key, values in groups.items()}
    flags = []
    for item in items:
        key = (item["speaker_id"], item["session_id"])
        level = item.get("rms_dbfs")
        median = medians.get(key)
        if level is not None and median is not None:
            delta = float(level) - median
            if delta < -level_outlier_db:
                flags.append(
                    {
                        "recording_id": item["recording_id"],
                        "reason": "LOW_LEVEL_SUSPECT",
                        "measured_value": float(level),
                        "reference_value": median,
                    }
                )
            elif delta > level_outlier_db:
                flags.append(
                    {
                        "recording_id": item["recording_id"],
                        "reason": "HIGH_LEVEL_SUSPECT",
                        "measured_value": float(level),
                        "reference_value": median,
                    }
                )
        clipping = float(item.get("clipping_fraction", 0.0))
        if not math.isfinite(clipping) or not 0.0 <= clipping <= 1.0:
            raise ValueError("clipping_fraction must be finite and in [0, 1]")
        if clipping >= clipping_fraction_threshold:
            flags.append(
                {
                    "recording_id": item["recording_id"],
                    "reason": "CLIPPING_SUSPECT",
                    "measured_value": clipping,
                    "reference_value": clipping_fraction_threshold,
                }
            )
    return flags


def _require_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing validation artifact path: {path}")


def _write_validation_provenance(
    output_dir: Path,
    records: list[RecordingRecord],
    representation_type: str,
    dimension: int,
    config: Any,
    *,
    target: str = "known",
    embedding_source: str | None = None,
) -> None:
    contract = VALIDATION_TARGETS[target]
    provenance: dict[str, Any] = {
        "artifact_type": contract["artifact_type"],
        "validation_session": VALIDATION_SESSION,
        "validation_split": VALIDATION_SPLIT,
        "validation_texts": list(VALIDATION_KNOWN_TEXT_IDS),
        "expected_speakers": list(contract["speakers"]),
        "selected_speakers": sorted({record.speaker_id for record in records}),
        "selected_recording_ids": [record.recording_id for record in records],
        "recording_count": len(records),
        "sample_rate_hz": config.sample_rate_hz,
        "segment_duration_seconds": config.segment_duration_seconds,
        "segment_hop_seconds": config.segment_hop_seconds,
        "minimum_tail_seconds": config.minimum_final_segment_seconds,
        "representation_type": representation_type,
        "representation_dimension": dimension,
    }
    if target == "known":
        # Preserve the existing key used by known-validation consumers.
        provenance["active_known_speakers"] = list(ACTIVE_KNOWN_SPEAKER_IDS)
    if representation_type == "ecapa_embedding":
        provenance["embedding_source"] = embedding_source
        provenance["ecapa_frozen"] = True
    (output_dir / PROVENANCE_FILENAME).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _audit_validation_rows(
    x: np.ndarray,
    y: np.ndarray,
    segment_ids: list[str],
    metadata: list[dict[str, str]],
    records: list[RecordingRecord],
    *,
    target: str = "known",
    expected_dimension: int | None = None,
) -> dict[str, Any]:
    if target not in VALIDATION_TARGETS:
        raise ValueError(f"Unsupported validation target: {target!r}")
    expected_speakers = set(VALIDATION_TARGETS[target]["speakers"])
    expected_recording_ids = [record.recording_id for record in records]
    if (
        x.ndim != 2
        or x.dtype != np.float32
        or not np.isfinite(x).all()
        or len(x) == 0
        or len(x) != len(y)
        or len(x) != len(segment_ids)
        or len(x) != len(metadata)
        or len(set(segment_ids)) != len(segment_ids)
        or segment_ids != [row["segment_id"] for row in metadata]
        or y.tolist() != [row["speaker_id"] for row in metadata]
        or (expected_dimension is not None and x.shape[1] != expected_dimension)
    ):
        raise ValueError("Validation feature rows are malformed, misaligned, or non-finite")
    recording_order = list(dict.fromkeys(row["recording_id"] for row in metadata))
    if recording_order != expected_recording_ids:
        raise ValueError("Validation recording order differs from selected manifest order")
    if any(
        row["speaker_id"] not in expected_speakers
        or row["session_id"] != VALIDATION_SESSION
        or row["split"] != VALIDATION_SPLIT
        or row["text_id"] not in VALIDATION_KNOWN_TEXT_IDS
        for row in metadata
    ):
        raise ValueError("Validation features contain ineligible labels or metadata")
    return {
        "recordings": len(recording_order),
        "segments": int(len(x)),
        "dimension": int(x.shape[1]),
        "dtype": str(x.dtype),
        "finite": bool(np.isfinite(x).all()),
        "recording_order": recording_order,
        "segment_ids": segment_ids,
        "labels": y.tolist(),
        "metadata": [tuple(row[field] for field in SEGMENT_METADATA_FIELDS) for row in metadata],
        "segments_per_speaker": dict(sorted(Counter(y.tolist()).items())),
        "segments_per_recording": dict(
            sorted(Counter(row["recording_id"] for row in metadata).items())
        ),
    }


def build_mfcc_validation_dataset(
    records: list[RecordingRecord],
    dataset_root: Path,
    output_dir: Path,
    config,
    *,
    target: str = "known",
) -> dict[str, Any]:
    """Extract and persist MFCC only for the explicitly selected validation target."""

    selected = select_and_audit_validation_records(records, target=target)
    _require_new_output(output_dir)
    feature_records: list[FeatureRecord] = []
    for recording in selected:
        rows = features_for_recording(dataset_root / recording.relative_path, recording, config)
        if not rows:
            raise ValueError(f"No MFCC validation segments for {recording.recording_id}")
        feature_records.extend(rows)
    save_features(output_dir, feature_records, config)
    x, y, metadata, stored_config = load_features(output_dir)
    if stored_config != config.to_dict():
        raise ValueError("MFCC validation config changed during artifact persistence")
    summary = _audit_validation_rows(
        x,
        y,
        [row["segment_id"] for row in metadata],
        metadata,
        selected,
        target=target,
        expected_dimension=120,
    )
    _write_validation_provenance(
        output_dir, selected, "mfcc", summary["dimension"], config, target=target
    )
    summary["metadata"] = metadata
    summary["config"] = stored_config
    return summary


def build_ecapa_validation_dataset(
    records: list[RecordingRecord],
    dataset_root: Path,
    output_dir: Path,
    segment_config,
    encoder: "EcapaEncoder",
    *,
    target: str = "known",
    ecapa_builder: Callable[..., EcapaFeatureConfig] | None = None,
) -> dict[str, Any]:
    """Encode one explicit validation target with one injected frozen encoder."""

    selected = select_and_audit_validation_records(records, target=target)
    _require_new_output(output_dir)
    if ecapa_builder is None:
        if __package__:
            from scripts.build_ecapa_embeddings import build_ecapa_feature_dataset
        else:
            from build_ecapa_embeddings import build_ecapa_feature_dataset

        ecapa_builder = build_ecapa_feature_dataset
    stored_config = ecapa_builder(selected, dataset_root, output_dir, segment_config, encoder)
    x, y, segment_ids, metadata, loaded_config = load_ecapa_features(output_dir)
    if loaded_config != stored_config or loaded_config.model_source != encoder.source:
        raise ValueError("ECAPA validation config/source changed during persistence")
    summary = _audit_validation_rows(
        x,
        y,
        segment_ids.tolist(),
        metadata,
        selected,
        target=target,
    )
    if x.shape[1] != loaded_config.embedding_dimension:
        raise ValueError("ECAPA validation dimension differs from stored config")
    _write_validation_provenance(
        output_dir,
        selected,
        "ecapa_embedding",
        summary["dimension"],
        loaded_config,
        target=target,
        embedding_source=loaded_config.model_source,
    )
    summary["metadata"] = metadata
    summary["config"] = loaded_config.to_dict()
    return summary


def audit_validation_config(training_dir: Path, config_path: Path):
    """Require the current MFCC segmentation/extraction config to match OPEN-02B."""

    _, _, _, training_config = load_features(training_dir)
    config = load_mfcc_config(config_path)
    expected = {
        "sample_rate_hz": 16_000,
        "segment_duration_seconds": 3.0,
        "segment_hop_seconds": 1.5,
        "minimum_final_segment_seconds": 1.5,
        "n_mfcc": 20,
        "n_mels": 40,
        "delta_orders": [1, 2],
        "temporal_aggregation": "mean_std",
    }
    if config.to_dict() != training_config or any(
        training_config.get(field) != value for field, value in expected.items()
    ) or config.feature_dimension != 120:
        raise ValueError("MFCC validation config is not exactly equivalent to OPEN-02B training")
    return config


def audit_representation_alignment(mfcc: dict[str, Any], ecapa: dict[str, Any]) -> None:
    """Require row-for-row identity between the persisted validation representations."""

    fields = ("recording_order", "segment_ids", "labels", "metadata")
    different = [field for field in fields if mfcc[field] != ecapa[field]]
    if mfcc["segments"] != ecapa["segments"] or different:
        raise ValueError(f"MFCC/ECAPA validation rows are misaligned: {different}")


def audit_train_validation_isolation(
    training_mfcc_dir: Path,
    validation_mfcc_dir: Path,
    training_ecapa_dir: Path | None = None,
    validation_ecapa_dir: Path | None = None,
):
    """Verify training and validation recording/segment namespaces are disjoint."""

    _, _, training_metadata, _ = load_features(training_mfcc_dir)
    _, _, validation_metadata, _ = load_features(validation_mfcc_dir)
    recording_overlap = {
        row["recording_id"] for row in training_metadata
    }.intersection(row["recording_id"] for row in validation_metadata)
    segment_overlap = {row["segment_id"] for row in training_metadata}.intersection(
        row["segment_id"] for row in validation_metadata
    )
    training_sessions = sorted({row["session_id"] for row in training_metadata})
    validation_sessions = sorted({row["session_id"] for row in validation_metadata})
    checks = {
        "mfcc_recording_overlap_count": len(recording_overlap),
        "mfcc_segment_overlap_count": len(segment_overlap),
        "mfcc_training_sessions": training_sessions,
        "mfcc_validation_sessions": validation_sessions,
    }
    if training_ecapa_dir is not None or validation_ecapa_dir is not None:
        if training_ecapa_dir is None or validation_ecapa_dir is None:
            raise ValueError("both ECAPA train and validation directories must be supplied")
        _, _, train_ids, train_meta, _ = load_ecapa_features(training_ecapa_dir)
        _, _, validation_ids, validation_meta, _ = load_ecapa_features(validation_ecapa_dir)
        ecapa_recording_overlap = {
            row["recording_id"] for row in train_meta
        }.intersection(row["recording_id"] for row in validation_meta)
        ecapa_segment_overlap = set(train_ids.tolist()).intersection(validation_ids.tolist())
        ecapa_training_sessions = sorted({row["session_id"] for row in train_meta})
        ecapa_validation_sessions = sorted({row["session_id"] for row in validation_meta})
        checks.update(
            {
                "ecapa_recording_overlap_count": len(ecapa_recording_overlap),
                "ecapa_segment_overlap_count": len(ecapa_segment_overlap),
                "ecapa_training_sessions": ecapa_training_sessions,
                "ecapa_validation_sessions": ecapa_validation_sessions,
            }
        )
    overlap_counts = [value for key, value in checks.items() if "overlap_count" in key]
    session_checks = [
        value
        for key, value in checks.items()
        if key.endswith("training_sessions")
    ]
    validation_session_checks = [
        value
        for key, value in checks.items()
        if key.endswith("validation_sessions")
    ]
    if (
        any(overlap_counts)
        or any(value != [TRAINING_SESSION] for value in session_checks)
        or any(value != [VALIDATION_SESSION] for value in validation_session_checks)
    ):
        raise ValueError("Training/validation datasets overlap or have incorrect session metadata")
    return {
        **checks,
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=tuple(VALIDATION_TARGETS), default="known")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    contract = VALIDATION_TARGETS[args.target]
    records = load_manifest(ROOT / "data/metadata/recordings.csv")
    selected = select_and_audit_validation_records(records, target=args.target)
    mfcc_output = ROOT / contract["mfcc_output"]
    ecapa_output = ROOT / contract["ecapa_output"]
    _require_new_output(mfcc_output)
    _require_new_output(ecapa_output)
    config = audit_validation_config(
        ROOT / "artifacts/open_set/mfcc/training", ROOT / "configs/mfcc.yaml"
    )
    mfcc_summary = build_mfcc_validation_dataset(
        records, ROOT, mfcc_output, config, target=args.target
    )

    # Import and construct the pretrained frozen encoder only after MFCC auditing passes.
    from speaker_id.features import EcapaEncoder

    encoder = EcapaEncoder.from_pretrained(device=args.device)
    ecapa_summary = build_ecapa_validation_dataset(
        records, ROOT, ecapa_output, config, encoder, target=args.target
    )
    audit_representation_alignment(mfcc_summary, ecapa_summary)
    isolation = audit_train_validation_isolation(
        ROOT / "artifacts/open_set/mfcc/training",
        mfcc_output,
        ROOT / "artifacts/open_set/ecapa/training",
        ecapa_output,
    )
    known_mfcc_path = ROOT / MFCC_OUTPUT
    known_ecapa_path = ROOT / ECAPA_OUTPUT
    cross_validation_checks: dict[str, int] = {}
    if args.target == "unknown" and known_mfcc_path.exists():
        _, _, known_metadata, _ = load_features(known_mfcc_path)
        unknown_recording_ids = set(mfcc_summary["recording_order"])
        cross_validation_checks["known_mfcc_recording_overlap_count"] = len(
            unknown_recording_ids & {row["recording_id"] for row in known_metadata}
        )
        cross_validation_checks["known_mfcc_segment_overlap_count"] = len(
            set(mfcc_summary["segment_ids"])
            & {row["segment_id"] for row in known_metadata}
        )
        if known_ecapa_path.exists():
            _, _, known_ids, known_ecapa_metadata, _ = load_ecapa_features(known_ecapa_path)
            cross_validation_checks["known_ecapa_recording_overlap_count"] = len(
                unknown_recording_ids & {row["recording_id"] for row in known_ecapa_metadata}
            )
            cross_validation_checks["known_ecapa_segment_overlap_count"] = len(
                set(ecapa_summary["segment_ids"]) & set(known_ids.tolist())
            )
        if any(cross_validation_checks.values()):
            raise ValueError("Unknown/known validation artifacts overlap")
    report = {
        "selected_recordings": [record.recording_id for record in selected],
        "mfcc": {key: value for key, value in mfcc_summary.items() if key != "metadata"},
        "ecapa": {key: value for key, value in ecapa_summary.items() if key != "metadata"},
        "representation_alignment": "PASS",
        "train_validation_isolation": isolation,
        "known_validation_isolation": cross_validation_checks,
        "target": args.target,
        "no_model_inference": True,
        "no_threshold_calibration": True,
        "no_s03_feature_build": True,
    }
    print(json.dumps(report, indent=2, default=str, sort_keys=True))


if __name__ == "__main__":
    main()
