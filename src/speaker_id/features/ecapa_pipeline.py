"""Build in-memory ECAPA feature records for one source recording."""

from pathlib import Path

import numpy as np

from speaker_id.audio import load_audio
from speaker_id.datasets import RecordingRecord

from .config import MfccConfig
from .ecapa import EcapaEncoder
from .segmentation import segment_audio
from .types import FeatureRecord


def ecapa_features_for_recording(
    root: Path,
    recording: RecordingRecord,
    config: MfccConfig,
    encoder: EcapaEncoder,
) -> list[FeatureRecord]:
    """Load, segment, and encode one recording using the injected ECAPA encoder."""

    audio = load_audio(root / recording.relative_path)
    if audio.sample_rate != config.sample_rate_hz:
        raise ValueError("Canonical sample rate differs from segmentation config")

    segments = segment_audio(audio.samples, recording, config)
    records: list[FeatureRecord] = []
    embedding_dimension: int | None = None
    for segment in segments:
        embedding = encoder.encode(segment.samples, sample_rate=audio.sample_rate)
        if not isinstance(embedding, np.ndarray) or embedding.ndim != 1 or embedding.size == 0:
            raise ValueError(f"Encoder returned an invalid vector for {segment.segment_id}")
        if embedding_dimension is None:
            embedding_dimension = embedding.size
        elif embedding.shape != (embedding_dimension,):
            raise ValueError(
                f"Inconsistent embedding dimension for {segment.segment_id}: "
                f"expected {embedding_dimension}, got {embedding.shape}"
            )

        records.append(
            FeatureRecord(
                segment_id=segment.segment_id,
                recording_id=segment.recording_id,
                speaker_id=segment.speaker_id,
                session_id=segment.session_id,
                text_id=segment.text_id,
                split=segment.split,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                feature_vector=embedding,
            )
        )
    return records
