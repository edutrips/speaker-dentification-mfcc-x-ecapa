"""Shared source-recording to fixed-feature pipeline."""

from pathlib import Path

from speaker_id.audio import load_audio
from speaker_id.datasets import RecordingRecord

from .config import MfccConfig
from .mfcc import extract_mfcc_vector
from .segmentation import segment_audio
from .types import FeatureRecord


def features_for_recording(
    path: Path, recording: RecordingRecord, config: MfccConfig
) -> list[FeatureRecord]:
    audio = load_audio(path)
    if audio.sample_rate != config.sample_rate_hz:
        raise ValueError("Canonical sample rate differs from MFCC config")
    segments = segment_audio(audio.samples, recording, config)
    return [
        FeatureRecord(
            segment_id=segment.segment_id,
            recording_id=segment.recording_id,
            speaker_id=segment.speaker_id,
            session_id=segment.session_id,
            text_id=segment.text_id,
            split=segment.split,
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
            feature_vector=extract_mfcc_vector(segment.samples, config),
        )
        for segment in segments
    ]
