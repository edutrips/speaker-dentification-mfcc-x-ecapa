"""Build aligned MFCC and ECAPA artifacts for the open-set training split only."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if __package__:
    from scripts.build_ecapa_embeddings import build_ecapa_feature_dataset
else:
    from build_ecapa_embeddings import build_ecapa_feature_dataset

from speaker_id.datasets import RecordingRecord, load_manifest
from speaker_id.features import (
    EcapaFeatureConfig,
    load_ecapa_features,
    load_mfcc_config,
)
from speaker_id.features.pipeline import features_for_recording
from speaker_id.features.storage import load_features, save_features
from speaker_id.open_set.protocol import (
    ACTIVE_KNOWN_SPEAKER_IDS,
    OPTIONAL_KNOWN_SPEAKER_IDS,
    TRAINING_SESSION,
    TRAINING_SPLIT,
    TRAINING_TEXT_IDS,
    is_open_set_training_record,
    select_open_set_training_records,
)

if TYPE_CHECKING:
    from speaker_id.features.ecapa import EcapaEncoder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/metadata/recordings.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/open_set"
MFCC_OUTPUT = Path("mfcc/training")
ECAPA_OUTPUT = Path("ecapa/training")
PROVENANCE_FILE = "training_protocol.json"
MFCC_METADATA_FIELDS = (
    "segment_id",
    "recording_id",
    "speaker_id",
    "session_id",
    "text_id",
    "split",
    "start_seconds",
    "end_seconds",
)
SUMMARY_DETAIL_FIELDS = {"recording_order", "segment_ids", "labels", "metadata"}


def select_and_audit_training_records(
    manifest_path: Path = DEFAULT_MANIFEST,
) -> list[RecordingRecord]:
    """Select solely through the protocol helper and enforce current completeness."""

    records = select_open_set_training_records(load_manifest(manifest_path))
    expected_pairs = {
        (speaker_id, text_id)
        for speaker_id in ACTIVE_KNOWN_SPEAKER_IDS
        for text_id in TRAINING_TEXT_IDS
    }
    actual_pairs = Counter((record.speaker_id, record.text_id) for record in records)
    if actual_pairs != Counter({pair: 1 for pair in expected_pairs}):
        raise ValueError(
            "Training selection must contain exactly one S01/T01-T05 record "
            "per active speaker"
        )
    if any(
        not is_open_set_training_record(record)
        or record.session_id != TRAINING_SESSION
        or record.split != TRAINING_SPLIT
        for record in records
    ):
        raise ValueError("Training selector returned a record outside the frozen protocol")
    return records


def _require_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing open-set artifact path: {path}")


def build_mfcc_training_artifacts(
    records: list[RecordingRecord],
    dataset_root: Path,
    output_dir: Path,
    config,
) -> dict[str, object]:
    """Extract and persist MFCC vectors for already-selected recordings."""

    _require_new_output(output_dir)
    feature_records = []
    for recording in records:
        rows = features_for_recording(dataset_root / recording.relative_path, recording, config)
        if not rows:
            raise ValueError(f"No MFCC segments for {recording.recording_id}")
        feature_records.extend(rows)
    save_features(output_dir, feature_records, config)
    x, y, metadata, stored_config = load_features(output_dir)
    _audit_mfcc_output(records, x, y, metadata, stored_config, config)
    return _dataset_summary(records, x, y, metadata)


def _audit_mfcc_output(records, x, y, metadata, stored_config, config) -> None:
    segment_ids = [row["segment_id"] for row in metadata]
    recording_order = list(dict.fromkeys(row["recording_id"] for row in metadata))
    if (
        x.ndim != 2
        or x.dtype.name != "float32"
        or not np.isfinite(x).all()
        or x.shape[1] != config.feature_dimension
        or config.feature_dimension != 120
        or len(x) != len(y)
        or len(x) != len(metadata)
        or y.tolist() != [row["speaker_id"] for row in metadata]
        or len(set(segment_ids)) != len(segment_ids)
        or recording_order != [record.recording_id for record in records]
        or stored_config != config.to_dict()
    ):
        raise ValueError("MFCC training artifact failed structural or provenance audit")


def build_ecapa_training_artifacts(
    records: list[RecordingRecord],
    dataset_root: Path,
    output_dir: Path,
    config,
    encoder: "EcapaEncoder",
) -> tuple[EcapaFeatureConfig, dict[str, object]]:
    """Encode selected recordings with the caller's single shared encoder."""

    _require_new_output(output_dir)
    ecapa_config = build_ecapa_feature_dataset(
        records,
        dataset_root,
        output_dir,
        config,
        encoder,
    )
    x, y, segment_ids, metadata, stored_config = load_ecapa_features(output_dir)
    if (
        x.ndim != 2
        or x.dtype.name != "float32"
        or not np.isfinite(x).all()
        or x.shape[1] != ecapa_config.embedding_dimension
        or len(x) != len(y)
        or len(x) != len(segment_ids)
        or len(x) != len(metadata)
        or y.tolist() != [row["speaker_id"] for row in metadata]
        or segment_ids.tolist() != [row["segment_id"] for row in metadata]
        or len(set(segment_ids.tolist())) != len(segment_ids)
        or stored_config != ecapa_config
    ):
        raise ValueError("ECAPA training artifact failed structural or provenance audit")
    return ecapa_config, _dataset_summary(records, x, y, metadata)


def _dataset_summary(records, x, y, metadata) -> dict[str, object]:
    return {
        "recordings": len({record.recording_id for record in records}),
        "rows": int(x.shape[0]),
        "dimension": int(x.shape[1]),
        "dtype": str(x.dtype),
        "finite": bool(np.isfinite(x).all()),
        "segments_per_speaker": dict(
            sorted(Counter(row["speaker_id"] for row in metadata).items())
        ),
        "recording_order": list(dict.fromkeys(row["recording_id"] for row in metadata)),
        "segment_ids": [row["segment_id"] for row in metadata],
        "labels": y.tolist(),
        "metadata": [tuple(row[field] for field in MFCC_METADATA_FIELDS) for row in metadata],
    }


def _audit_representation_alignment(mfcc: dict[str, object], ecapa: dict[str, object]) -> None:
    comparisons = (
        "recording_order",
        "segment_ids",
        "labels",
        "metadata",
    )
    mismatched = [key for key in comparisons if mfcc[key] != ecapa[key]]
    if mismatched:
        raise ValueError(f"MFCC/ECAPA training metadata is misaligned: {mismatched}")


def _write_protocol_provenance(
    output_root: Path,
    records: list[RecordingRecord],
    config,
    ecapa_config: EcapaFeatureConfig,
) -> None:
    provenance = {
        "selection_policy": "speaker_id.open_set.protocol.select_open_set_training_records",
        "active_known_speakers": list(ACTIVE_KNOWN_SPEAKER_IDS),
        "optional_supported_known": list(OPTIONAL_KNOWN_SPEAKER_IDS),
        "training_session": TRAINING_SESSION,
        "training_split": TRAINING_SPLIT,
        "training_texts": list(TRAINING_TEXT_IDS),
        "source_recordings": [record.recording_id for record in records],
        "recording_count": len(records),
        "segmentation": {
            "sample_rate_hz": config.sample_rate_hz,
            "segment_duration_seconds": config.segment_duration_seconds,
            "segment_hop_seconds": config.segment_hop_seconds,
            "minimum_final_segment_seconds": config.minimum_final_segment_seconds,
        },
        "mfcc": {
            "representation_type": "mfcc",
            "feature_dimension": config.feature_dimension,
            "config": config.to_dict(),
        },
        "ecapa": {
            "representation_type": "ecapa_embedding",
            "model_source": ecapa_config.model_source,
            "embedding_dimension": ecapa_config.embedding_dimension,
            "frozen": True,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / PROVENANCE_FILE
    _require_new_output(path)
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("representation", choices=("mfcc", "ecapa"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    args = _argument_parser().parse_args()
    records = select_and_audit_training_records(args.manifest)
    config = load_mfcc_config(ROOT / "configs/mfcc.yaml")
    if args.representation == "mfcc":
        summary = build_mfcc_training_artifacts(
            records,
            args.dataset_root,
            args.output_root / MFCC_OUTPUT,
            config,
        )
        print(json.dumps(summary, indent=2))
        return 0

    mfcc_x, mfcc_y, mfcc_metadata, mfcc_stored_config = load_features(
        args.output_root / MFCC_OUTPUT
    )
    _audit_mfcc_output(records, mfcc_x, mfcc_y, mfcc_metadata, mfcc_stored_config, config)
    mfcc_summary = _dataset_summary(records, mfcc_x, mfcc_y, mfcc_metadata)
    ecapa_output = args.output_root / ECAPA_OUTPUT
    _require_new_output(ecapa_output)
    _require_new_output(args.output_root / PROVENANCE_FILE)
    from speaker_id.features import EcapaEncoder

    encoder = EcapaEncoder.from_pretrained(device=args.device)
    ecapa_config, ecapa_summary = build_ecapa_training_artifacts(
        records,
        args.dataset_root,
        ecapa_output,
        config,
        encoder,
    )
    _audit_representation_alignment(mfcc_summary, ecapa_summary)
    _write_protocol_provenance(args.output_root, records, config, ecapa_config)
    print(
        json.dumps(
            {
                "mfcc": {
                    key: value
                    for key, value in mfcc_summary.items()
                    if key not in SUMMARY_DETAIL_FIELDS
                },
                "ecapa": {
                    key: value
                    for key, value in ecapa_summary.items()
                    if key not in SUMMARY_DETAIL_FIELDS
                },
                "alignment": "PASS",
                "ecapa_model_source": ecapa_config.model_source,
                "ecapa_frozen": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
