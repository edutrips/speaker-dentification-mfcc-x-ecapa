"""Immutable types used by the recordings manifest."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecordingRecord:
    """Metadata for one recording, independent of the WAV file's existence."""

    recording_id: str
    speaker_id: str
    session_id: str
    text_id: str
    split: str
    relative_path: str
    source_sample_rate_hz: int
    source_channels: int
    duration_seconds: float
    device_label: str | None = None
    environment_label: str | None = None
    notes: str | None = None
