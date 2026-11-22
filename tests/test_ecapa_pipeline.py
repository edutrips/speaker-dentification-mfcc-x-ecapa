"""Offline tests for the per-recording ECAPA feature pipeline."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from speaker_id.datasets import RecordingRecord
from speaker_id.features import ecapa_features_for_recording, load_mfcc_config


class FakeEncoder:
    def __init__(self, dimensions: tuple[int, ...] = (4,)) -> None:
        self.dimensions = dimensions
        self.calls: list[tuple[np.ndarray, int]] = []

    def encode(self, samples: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
        self.calls.append((samples.copy(), sample_rate))
        dimension_index = min(len(self.calls) - 1, len(self.dimensions) - 1)
        dimension = self.dimensions[dimension_index]
        return np.full(dimension, len(self.calls), dtype=np.float32)


@pytest.fixture
def config():
    return load_mfcc_config(Path("configs/mfcc.yaml"))


@pytest.fixture
def recording() -> RecordingRecord:
    return RecordingRecord(
        "V01_S01_T01_R999",
        "V01",
        "S01",
        "T01",
        "train",
        "data/raw/V01/S01/V01_S01_T01_R999.wav",
        16_000,
        1,
        7.5,
    )


def test_ecapa_pipeline_preserves_order_metadata_and_reuses_encoder(
    tmp_path: Path, config, recording: RecordingRecord
) -> None:
    wav_path = tmp_path / recording.relative_path
    wav_path.parent.mkdir(parents=True)
    waveform = np.linspace(-0.5, 0.5, 7 * config.sample_rate_hz + config.sample_rate_hz // 2)
    sf.write(wav_path, waveform.astype(np.float32), config.sample_rate_hz)
    encoder = FakeEncoder()

    records = ecapa_features_for_recording(tmp_path, recording, config, encoder)

    assert len(records) == 4
    assert len(encoder.calls) == len(records)
    assert all(sample_rate == 16_000 for _samples, sample_rate in encoder.calls)
    assert [record.segment_id for record in records] == [
        f"{recording.recording_id}_SEG{index:03d}" for index in range(4)
    ]
    assert [record.start_seconds for record in records] == [0.0, 1.5, 3.0, 4.5]
    assert [record.end_seconds for record in records] == [3.0, 4.5, 6.0, 7.5]
    for record in records:
        assert record.recording_id == recording.recording_id
        assert record.speaker_id == recording.speaker_id
        assert record.session_id == recording.session_id
        assert record.text_id == recording.text_id
        assert record.split == recording.split
        assert record.feature_vector.shape == (4,)
        assert record.feature_vector.ndim == 1
        assert np.isfinite(record.feature_vector).all()
    assert [record.feature_vector[0] for record in records] == [1, 2, 3, 4]


def test_ecapa_pipeline_rejects_inconsistent_embedding_dimensions(
    tmp_path: Path, config, recording: RecordingRecord
) -> None:
    wav_path = tmp_path / recording.relative_path
    wav_path.parent.mkdir(parents=True)
    sf.write(
        wav_path,
        np.zeros(5 * config.sample_rate_hz, dtype=np.float32),
        config.sample_rate_hz,
    )
    encoder = FakeEncoder(dimensions=(4, 5))

    with pytest.raises(ValueError, match="Inconsistent embedding dimension"):
        ecapa_features_for_recording(tmp_path, recording, config, encoder)
