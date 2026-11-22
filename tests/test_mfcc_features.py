"""Focused checks for the shared classical feature path."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from speaker_id.datasets import RecordingRecord
from speaker_id.features import (
    extract_mfcc_vector,
    features_for_recording,
    load_mfcc_config,
    segment_audio,
)
from speaker_id.features.storage import load_features, save_features
from speaker_id.features.types import FeatureRecord


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
        3.0,
    )


def test_exact_and_overlapping_segments(config, recording: RecordingRecord) -> None:
    exact = segment_audio(np.zeros(48_000, dtype=np.float32), recording, config)
    assert len(exact) == 1
    assert exact[0].segment_id == "V01_S01_T01_R999_SEG000"
    longer = segment_audio(np.zeros(72_000, dtype=np.float32), recording, config)
    assert [(row.start_seconds, row.end_seconds) for row in longer] == [(0, 3), (1.5, 4.5)]
    assert all(row.recording_id == recording.recording_id for row in longer)
    assert all(row.speaker_id == "V01" and row.text_id == "T01" for row in longer)


def test_partial_tail_threshold_and_padding_do_not_mutate_source(config, recording) -> None:
    short = np.ones(23_999, dtype=np.float32)
    assert segment_audio(short, recording, config) == []
    source = np.ones(24_000, dtype=np.float32)
    partial = segment_audio(source, recording, config)
    assert len(partial) == 1
    assert partial[0].end_seconds == 1.5
    assert partial[0].samples.size == 48_000
    assert np.all(partial[0].samples[24_000:] == 0)
    assert np.all(source == 1)


def test_mfcc_vector_is_fixed_finite_deterministic_including_silence(config) -> None:
    signal = np.sin(2 * np.pi * 220 * np.arange(48_000) / 16_000).astype(np.float32)
    first = extract_mfcc_vector(signal, config)
    second = extract_mfcc_vector(signal, config)
    assert first.shape == (config.feature_dimension,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(extract_mfcc_vector(np.zeros(48_000, dtype=np.float32), config)).all()


def test_feature_pipeline_inherits_parent_metadata(
    tmp_path: Path, config, recording: RecordingRecord
) -> None:
    path = tmp_path / "sample.wav"
    sf.write(path, np.zeros(48_000, dtype=np.float32), 16_000)
    first = features_for_recording(path, recording, config)
    second = features_for_recording(path, recording, config)
    assert len(first) == 1
    assert first[0].recording_id == recording.recording_id
    assert first[0].speaker_id == recording.speaker_id
    assert first[0].feature_vector.size == config.feature_dimension
    np.testing.assert_array_equal(first[0].feature_vector, second[0].feature_vector)


def test_feature_audit_rejects_invalid_labels_and_parent_links(
    tmp_path: Path, config, recording: RecordingRecord
) -> None:
    path = tmp_path / "sample.wav"
    sf.write(path, np.zeros(48_000, dtype=np.float32), 16_000)
    feature = features_for_recording(path, recording, config)[0]
    for label in ("U00", "X01"):
        with pytest.raises(ValueError, match="Invalid speaker label"):
            save_features(
                tmp_path / f"invalid_label_{label}",
                [replace(feature, speaker_id=label)],
                config,
            )
    with pytest.raises(ValueError, match="inconsistent parent"):
        save_features(tmp_path / "invalid_parent", [replace(feature, recording_id="other")], config)


def test_mfcc_storage_round_trip_accepts_known_and_generic_unknown_ids(
    tmp_path: Path, config
) -> None:
    records = [
        FeatureRecord(
            f"{speaker_id}_{session_id}_{text_id}_R999_SEG000",
            f"{speaker_id}_{session_id}_{text_id}_R999",
            speaker_id,
            session_id,
            text_id,
            split,
            0.0,
            3.0,
            np.full(config.feature_dimension, index, dtype=np.float32),
        )
        for index, (speaker_id, session_id, text_id, split) in enumerate(
            [
                ("V01", "S01", "T01", "train"),
                ("U01", "S02", "T06", "validation"),
                ("U99", "S03", "T11", "test"),
            ],
            start=1,
        )
    ]
    output = tmp_path / "mfcc-features"

    save_features(output, records, config)
    x, y, metadata, stored_config = load_features(output)

    assert x.shape == (3, config.feature_dimension)
    assert y.tolist() == ["V01", "U01", "U99"]
    assert [row["speaker_id"] for row in metadata] == y.tolist()
    assert [row["segment_id"] for row in metadata] == [record.segment_id for record in records]
    assert [row["split"] for row in metadata] == ["train", "validation", "test"]
    assert stored_config == config.to_dict()


def test_mfcc_feature_pipeline_and_storage_process_unknown_recording(
    tmp_path: Path, config
) -> None:
    identifier = "U01_S02_T06_R999"
    recording = RecordingRecord(
        identifier,
        "U01",
        "S02",
        "T06",
        "validation",
        f"data/raw/U01/S02/{identifier}.wav",
        16_000,
        1,
        3.0,
    )
    wav_path = tmp_path / "synthetic.wav"
    sf.write(wav_path, np.zeros(48_000, dtype=np.float32), 16_000)

    features = features_for_recording(wav_path, recording, config)
    output = tmp_path / "mfcc-unknown"
    save_features(output, features, config)
    _x, y, metadata, _stored_config = load_features(output)

    assert y.tolist() == ["U01"]
    assert metadata[0]["segment_id"] == f"{identifier}_SEG000"
    assert metadata[0]["session_id"] == "S02"
    assert metadata[0]["text_id"] == "T06"
    assert metadata[0]["split"] == "validation"
