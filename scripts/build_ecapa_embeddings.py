"""Build ECAPA embedding artifacts from the accepted recordings manifest."""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from speaker_id.datasets import RecordingRecord, load_manifest
from speaker_id.features import (
    EcapaFeatureConfig,
    load_mfcc_config,
    save_ecapa_features,
)
from speaker_id.features.config import MfccConfig
from speaker_id.features.ecapa_pipeline import ecapa_features_for_recording
from speaker_id.features.types import FeatureRecord

if TYPE_CHECKING:
    from speaker_id.features.ecapa import EcapaEncoder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/metadata/recordings.csv"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/ecapa_svm"
DEFAULT_DATASET_ROOT = ROOT
DEFAULT_SEGMENT_CONFIG = ROOT / "configs/mfcc.yaml"


def build_ecapa_feature_dataset(
    recordings: list[RecordingRecord],
    dataset_root: Path,
    output_dir: Path,
    segment_config: MfccConfig,
    encoder: "EcapaEncoder",
) -> EcapaFeatureConfig:
    """Build and persist all records using the caller-provided encoder instance."""

    if not recordings:
        raise ValueError("The manifest contains no recordings to process")

    feature_records: list[FeatureRecord] = []
    for recording in recordings:
        try:
            rows = ecapa_features_for_recording(
                dataset_root,
                recording,
                segment_config,
                encoder,
            )
            if not rows:
                raise ValueError("recording produced no segments")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build ECAPA features for recording {recording.recording_id}"
            ) from exc
        feature_records.extend(rows)

    embedding_dimension = feature_records[0].feature_vector.size
    for record in feature_records[1:]:
        if record.feature_vector.shape != (embedding_dimension,):
            raise ValueError(
                f"Inconsistent ECAPA embedding dimension at {record.segment_id}: "
                f"expected {embedding_dimension}, got {record.feature_vector.shape}"
            )

    # MfccConfig is temporarily the shared source of segmentation parameters.
    config = EcapaFeatureConfig(
        model_source=encoder.source,
        embedding_dimension=embedding_dimension,
        sample_rate_hz=segment_config.sample_rate_hz,
        segment_duration_seconds=segment_config.segment_duration_seconds,
        segment_hop_seconds=segment_config.segment_hop_seconds,
        minimum_final_segment_seconds=segment_config.minimum_final_segment_seconds,
    )
    save_ecapa_features(output_dir, feature_records, config)
    return config


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    args = _argument_parser().parse_args()
    recordings = load_manifest(args.manifest)
    if not recordings:
        raise ValueError(f"Manifest contains no recordings: {args.manifest}")

    # Segmentation settings remain shared with the current MFCC configuration for now.
    segment_config = load_mfcc_config(DEFAULT_SEGMENT_CONFIG)
    from speaker_id.features import EcapaEncoder

    encoder = EcapaEncoder.from_pretrained(device=args.device)
    config = build_ecapa_feature_dataset(
        recordings,
        args.dataset_root,
        args.output_dir,
        segment_config,
        encoder,
    )
    print(f"Recordings: {len(recordings)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Embedding dimension: {config.embedding_dimension}")
    print("ECAPA FEATURE BUILD PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
