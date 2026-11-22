"""Persistence and validation for ECAPA embedding feature records."""

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .ecapa_constants import DEFAULT_ECAPA_SOURCE
from .types import FeatureRecord

REPRESENTATION_TYPE = "ecapa_embedding"
SAMPLE_RATE_HZ = 16_000
ARTIFACT_FILES = ("embeddings.npz", "embedding_metadata.csv", "embedding_config.json")
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


@dataclass(frozen=True, slots=True)
class EcapaFeatureConfig:
    """Small, serializable description of an ECAPA embedding collection."""

    embedding_dimension: int
    segment_duration_seconds: float
    segment_hop_seconds: float
    minimum_final_segment_seconds: float
    model_source: str = DEFAULT_ECAPA_SOURCE
    sample_rate_hz: int = SAMPLE_RATE_HZ
    representation_type: str = REPRESENTATION_TYPE

    def __post_init__(self) -> None:
        if self.representation_type != REPRESENTATION_TYPE:
            raise ValueError(f"representation_type must be {REPRESENTATION_TYPE!r}")
        if not isinstance(self.model_source, str) or not self.model_source.strip():
            raise ValueError("model_source must be a non-empty string")
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        if self.sample_rate_hz != SAMPLE_RATE_HZ:
            raise ValueError(f"sample_rate_hz must be {SAMPLE_RATE_HZ}")
        durations = (
            self.segment_duration_seconds,
            self.segment_hop_seconds,
            self.minimum_final_segment_seconds,
        )
        if not all(math.isfinite(value) and value > 0 for value in durations):
            raise ValueError("segment durations and hop must be positive and finite")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible fields in stable declaration order."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "EcapaFeatureConfig":
        """Validate and restore the persisted configuration mapping."""

        try:
            return cls(
                embedding_dimension=int(values["embedding_dimension"]),
                segment_duration_seconds=float(values["segment_duration_seconds"]),
                segment_hop_seconds=float(values["segment_hop_seconds"]),
                minimum_final_segment_seconds=float(values["minimum_final_segment_seconds"]),
                model_source=str(values["model_source"]),
                sample_rate_hz=int(values["sample_rate_hz"]),
                representation_type=str(values["representation_type"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid ECAPA embedding config: {exc}") from exc


def _validate_records(
    records: list[FeatureRecord], config: EcapaFeatureConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("No ECAPA feature records were provided")

    segment_ids = [record.segment_id for record in records]
    if any(not isinstance(segment_id, str) or not segment_id for segment_id in segment_ids):
        raise ValueError("Every record must have a non-empty segment_id")
    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError("Duplicate segment IDs")

    vectors: list[np.ndarray] = []
    expected_dimension: int | None = None
    for record in records:
        vector = record.feature_vector
        if not isinstance(vector, np.ndarray):
            raise ValueError(f"Feature vector for {record.segment_id} must be a NumPy array")
        if vector.ndim != 1:
            raise ValueError(f"Feature vector for {record.segment_id} must be one-dimensional")
        if vector.size == 0:
            raise ValueError(f"Feature vector for {record.segment_id} must not be empty")
        if vector.dtype.kind != "f":
            raise ValueError(f"Feature vector for {record.segment_id} must be floating-point")
        if not np.isfinite(vector).all():
            raise ValueError(f"Feature vector for {record.segment_id} must be finite")
        if expected_dimension is None:
            expected_dimension = vector.size
        elif vector.shape != (expected_dimension,):
            raise ValueError(
                f"Inconsistent embedding dimension for {record.segment_id}: "
                f"expected {expected_dimension}, got {vector.shape}"
            )
        vectors.append(vector)

        for field in METADATA_FIELDS[:6]:
            value = getattr(record, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a non-empty string for {record.segment_id}")
        if (
            not math.isfinite(record.start_seconds)
            or not math.isfinite(record.end_seconds)
            or record.start_seconds < 0
            or record.end_seconds <= record.start_seconds
        ):
            raise ValueError(f"Invalid segment time interval for {record.segment_id}")

    if expected_dimension != config.embedding_dimension:
        raise ValueError(
            "Configured embedding_dimension does not match records: "
            f"{config.embedding_dimension} != {expected_dimension}"
        )

    x = np.stack(vectors)
    y = np.asarray([record.speaker_id for record in records], dtype=np.str_)
    ids = np.asarray(segment_ids, dtype=np.str_)
    return x, y, ids


def save_ecapa_features(
    directory: Path,
    records: list[FeatureRecord],
    config: EcapaFeatureConfig,
) -> None:
    """Validate and persist ECAPA vectors, labels, ordered metadata, and config."""

    x, y, segment_ids = _validate_records(records, config)
    directory.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(directory / "embeddings.npz", X=x, y=y, segment_ids=segment_ids)
    with (directory / "embedding_metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: getattr(record, field) for field in METADATA_FIELDS})
    (directory / "embedding_config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_ecapa_features(
    directory: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]], EcapaFeatureConfig]:
    """Load the ECAPA artifacts and validate array/metadata row alignment."""

    with np.load(directory / "embeddings.npz", allow_pickle=False) as data:
        x, y, segment_ids = data["X"], data["y"], data["segment_ids"]
    with (directory / "embedding_metadata.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not set(METADATA_FIELDS).issubset(reader.fieldnames):
            raise ValueError("ECAPA metadata CSV is missing required columns")
        metadata = list(reader)
    raw_config = json.loads(
        (directory / "embedding_config.json").read_text(encoding="utf-8")
    )
    if not isinstance(raw_config, dict):
        raise ValueError("ECAPA embedding config must be a JSON object")
    config = EcapaFeatureConfig.from_dict(raw_config)

    if (
        x.ndim != 2
        or x.shape[0] == 0
        or x.dtype.kind != "f"
        or y.ndim != 1
        or segment_ids.ndim != 1
        or len(x) != len(y)
        or len(x) != len(segment_ids)
        or len(x) != len(metadata)
        or x.shape[1] != config.embedding_dimension
        or not np.isfinite(x).all()
    ):
        raise ValueError("ECAPA embedding artifact shapes, dimension, or values are invalid")

    id_list = segment_ids.tolist()
    if (
        len(set(id_list)) != len(id_list)
        or [row["segment_id"] for row in metadata] != id_list
        or [row["speaker_id"] for row in metadata] != y.tolist()
    ):
        raise ValueError("ECAPA embedding rows are misaligned or contain duplicate segment IDs")
    return x, y, segment_ids, metadata, config
