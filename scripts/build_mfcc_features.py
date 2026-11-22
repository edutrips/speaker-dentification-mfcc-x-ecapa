"""Build and audit local MFCC artifacts for accepted manifest recordings."""

from pathlib import Path

from speaker_id.datasets import load_manifest
from speaker_id.features import features_for_recording, load_mfcc_config
from speaker_id.features.storage import load_features, save_features

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/mfcc_svm"


def main() -> int:
    config = load_mfcc_config(ROOT / "configs/mfcc.yaml")
    source_records = load_manifest(ROOT / "data/metadata/recordings.csv")
    features = []
    for record in source_records:
        rows = features_for_recording(ROOT / record.relative_path, record, config)
        if not rows:
            raise ValueError(f"No accepted segments for {record.recording_id}")
        features.extend(rows)
    audit = save_features(OUTPUT, features, config)
    x, y, metadata, stored_config = load_features(OUTPUT)
    if (
        audit["source_recordings"] != len(source_records)
        or len(metadata) != len(features)
        or x.shape[1] != config.feature_dimension
        or len(set(y)) == 0
        or stored_config != config.to_dict()
    ):
        raise ValueError("Feature artifact verification failed")
    for key, value in audit.items():
        print(f"{key}: {value}")
    print("FEATURE PIPELINE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
