from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from speaker_id.audio import (
    AudioFileNotFoundError,
    InvalidAudioError,
    UnsupportedAudioFormatError,
    load_audio,
)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    sf.write(path, samples, sample_rate)


def test_loads_mono_16khz_wav(tmp_path: Path) -> None:
    samples = np.linspace(-0.25, 0.25, 1600, dtype=np.float32)
    path = tmp_path / "mono.wav"
    write_wav(path, samples, 16_000)

    audio = load_audio(path)

    assert audio.samples.ndim == 1
    assert audio.samples.dtype == np.float32
    assert audio.sample_rate == 16_000
    assert audio.source_sample_rate == 16_000
    assert audio.source_channels == 1
    assert audio.duration_seconds == pytest.approx(0.1, abs=1 / 16_000)


def test_downmixes_stereo_by_channel_mean(tmp_path: Path) -> None:
    left = np.array([0.2, -0.2, 0.4], dtype=np.float32)
    right = np.array([0.6, 0.4, -0.4], dtype=np.float32)
    path = tmp_path / "stereo.wav"
    write_wav(path, np.column_stack((left, right)), 16_000)

    audio = load_audio(path)

    assert audio.source_channels == 2
    assert audio.samples.ndim == 1
    np.testing.assert_allclose(audio.samples, (left + right) / 2, atol=4e-5)


@pytest.mark.parametrize("source_sample_rate", [48_000, 44_100])
def test_resamples_to_16khz_and_preserves_duration(
    tmp_path: Path, source_sample_rate: int
) -> None:
    duration_seconds = 0.25
    sample_count = int(source_sample_rate * duration_seconds)
    signal = np.sin(
        2 * np.pi * 440 * np.arange(sample_count, dtype=np.float64) / source_sample_rate
    ).astype(np.float32)
    path = tmp_path / f"source_{source_sample_rate}.wav"
    write_wav(path, signal, source_sample_rate)

    audio = load_audio(path)

    assert audio.sample_rate == 16_000
    assert audio.samples.dtype == np.float32
    assert audio.samples.ndim == 1
    assert audio.samples.size == pytest.approx(4_000, abs=2)
    assert audio.duration_seconds == pytest.approx(duration_seconds, abs=1 / 16_000)
    assert np.isfinite(audio.samples).all()


def test_missing_file_raises_controlled_error(tmp_path: Path) -> None:
    with pytest.raises(AudioFileNotFoundError, match="does not exist"):
        load_audio(tmp_path / "missing.wav")


def test_non_wav_extension_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "audio.mp3"
    path.write_bytes(b"not audio")

    with pytest.raises(UnsupportedAudioFormatError, match="only WAV"):
        load_audio(path)


def test_empty_wav_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    write_wav(path, np.empty(0, dtype=np.float32), 16_000)

    with pytest.raises(InvalidAudioError, match="no samples"):
        load_audio(path)


def test_non_finite_samples_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "nan.wav"
    sf.write(path, np.array([np.nan], dtype=np.float32), 16_000, subtype="FLOAT")

    with pytest.raises(InvalidAudioError, match="non-finite"):
        load_audio(path)


def test_loader_does_not_peak_normalize(tmp_path: Path) -> None:
    samples = np.array([0.1, -0.05, 0.02], dtype=np.float32)
    path = tmp_path / "low_amplitude.wav"
    write_wav(path, samples, 16_000)

    audio = load_audio(path)

    assert np.max(np.abs(audio.samples)) == pytest.approx(0.1, abs=4e-5)
