"""Local feature artifact serialization and structural audit."""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from speaker_id.datasets import is_valid_speaker_id

from .config import MfccConfig
from .types import FeatureRecord

FEATURE_FIELDS = (
    "segment_id",
    "recording_id",
    "speaker_id",
    "session_id",
    "text_id",
    "split",
    "start_seconds",
    "end_seconds",
)


def save_features(
    directory: Path, records: list[FeatureRecord], config: MfccConfig
) -> dict[str, object]:
    """Persist feature rows in matching NPZ and CSV order and audit them."""

    if not records:
        raise ValueError("No feature records were produced")
    x = np.stack([record.feature_vector for record in records])
    y = np.asarray([record.speaker_id for record in records])
    segment_ids = np.asarray([record.segment_id for record in records])
    if x.shape[1] != config.feature_dimension or not np.isfinite(x).all():
        raise ValueError("Feature matrix has invalid dimension or non-finite values")
    if len(set(segment_ids)) != len(records):
        raise ValueError("Duplicate segment IDs")
    parent_metadata: dict[str, tuple[str, str, str, str]] = {}
    for record in records:
        if not is_valid_speaker_id(record.speaker_id):
            raise ValueError(f"Invalid speaker label: {record.speaker_id}")
        if not record.segment_id.startswith(f"{record.recording_id}_SEG"):
            raise ValueError(f"Segment has inconsistent parent: {record.segment_id}")
        identity = (
            record.speaker_id,
            record.session_id,
            record.text_id,
            record.split,
        )
        previous = parent_metadata.setdefault(record.recording_id, identity)
        if previous != identity:
            raise ValueError(f"Conflicting parent metadata: {record.recording_id}")
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "features.npz", X=x, y=y, segment_ids=segment_ids)
    with (directory / "feature_metadata.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: getattr(record, field) for field in FEATURE_FIELDS})
    (directory / "feature_config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return {
        "source_recordings": len({record.recording_id for record in records}),
        "segments": len(records),
        "segments_per_speaker": dict(
            sorted(Counter(record.speaker_id for record in records).items())
        ),
        "segments_per_recording": dict(
            sorted(Counter(record.recording_id for record in records).items())
        ),
        "segments_per_text": dict(sorted(Counter(record.text_id for record in records).items())),
        "feature_dimension": x.shape[1],
        "nan_count": int(np.isnan(x).sum()),
        "inf_count": int(np.isinf(x).sum()),
    }


def load_features(
    directory: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]], dict[str, object]]:
    """Load persisted features and verify CSV/NPZ row alignment."""

    with np.load(directory / "features.npz", allow_pickle=False) as data:
        x, y, ids = data["X"], data["y"], data["segment_ids"]
    with (directory / "feature_metadata.csv").open(newline="", encoding="utf-8") as source:
        metadata = list(csv.DictReader(source))
    config = json.loads((directory / "feature_config.json").read_text(encoding="utf-8"))
    if (
        x.ndim != 2
        or len(x) != len(y)
        or len(x) != len(metadata)
        or [row["segment_id"] for row in metadata] != ids.tolist()
        or [row["speaker_id"] for row in metadata] != y.tolist()
        or not np.isfinite(x).all()
    ):
        raise ValueError("Feature artifact rows are inconsistent or non-finite")
    return x, y, metadata, config
