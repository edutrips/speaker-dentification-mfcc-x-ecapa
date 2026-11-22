"""Small, dependency-free reader for the flat M4 MFCC YAML contract."""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MfccConfig:
    sample_rate_hz: int
    segment_duration_seconds: float
    segment_hop_seconds: float
    minimum_final_segment_seconds: float
    frame_length_ms: float
    frame_hop_ms: float
    n_mels: int
    n_mfcc: int
    delta_orders: tuple[int, int]
    temporal_aggregation: str

    @property
    def feature_dimension(self) -> int:
        return self.n_mfcc * (1 + len(self.delta_orders)) * 2

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["delta_orders"] = list(self.delta_orders)
        return result


def load_mfcc_config(path: Path) -> MfccConfig:
    """Parse the project's deliberately flat YAML without adding a YAML dependency."""

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            key, separator, value = line.partition(":")
            if not separator or key.strip() in values:
                raise ValueError(f"Invalid or repeated config key: {line}")
            values[key.strip()] = value.strip()
    expected = set(MfccConfig.__dataclass_fields__)
    if set(values) != expected:
        raise ValueError(f"MFCC config keys differ: {sorted(set(values) ^ expected)}")
    delta_orders = tuple(
        int(item.strip()) for item in values["delta_orders"].strip("[]").split(",")
    )
    config = MfccConfig(
        sample_rate_hz=int(values["sample_rate_hz"]),
        segment_duration_seconds=float(values["segment_duration_seconds"]),
        segment_hop_seconds=float(values["segment_hop_seconds"]),
        minimum_final_segment_seconds=float(values["minimum_final_segment_seconds"]),
        frame_length_ms=float(values["frame_length_ms"]),
        frame_hop_ms=float(values["frame_hop_ms"]),
        n_mels=int(values["n_mels"]),
        n_mfcc=int(values["n_mfcc"]),
        delta_orders=delta_orders,
        temporal_aggregation=values["temporal_aggregation"],
    )
    if (
        config.sample_rate_hz != 16_000
        or config.segment_duration_seconds <= 0
        or not 0 < config.segment_hop_seconds <= config.segment_duration_seconds
        or not 0 < config.minimum_final_segment_seconds <= config.segment_duration_seconds
        or config.frame_length_ms <= 0
        or config.frame_hop_ms <= 0
        or config.n_mels < config.n_mfcc
        or config.n_mfcc <= 0
        or config.delta_orders != (1, 2)
        or config.temporal_aggregation != "mean_std"
    ):
        raise ValueError("Unsupported MFCC configuration")
    return config


def config_from_dict(values: dict[str, object]) -> MfccConfig:
    """Restore the exact config persisted alongside a model."""

    return MfccConfig(
        sample_rate_hz=int(values["sample_rate_hz"]),
        segment_duration_seconds=float(values["segment_duration_seconds"]),
        segment_hop_seconds=float(values["segment_hop_seconds"]),
        minimum_final_segment_seconds=float(values["minimum_final_segment_seconds"]),
        frame_length_ms=float(values["frame_length_ms"]),
        frame_hop_ms=float(values["frame_hop_ms"]),
        n_mels=int(values["n_mels"]),
        n_mfcc=int(values["n_mfcc"]),
        delta_orders=tuple(int(value) for value in values["delta_orders"]),
        temporal_aggregation=str(values["temporal_aggregation"]),
    )
