"""Minimal WAV loader and canonicalizer for milestone M1."""

from math import gcd
from os import PathLike
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .errors import (
    AudioFileNotFoundError,
    InvalidAudioError,
    UnsupportedAudioFormatError,
)
from .types import AudioData

TARGET_SAMPLE_RATE_HZ = 16_000


def load_audio(path: str | PathLike[str]) -> AudioData:
    """Load a WAV file and return it in the project's canonical format.

    The loader performs only validation, channel averaging, and sample-rate
    conversion. It does not normalize, filter, denoise, or segment the signal.
    """

    audio_path = Path(path)
    if audio_path.suffix.lower() != ".wav":
        raise UnsupportedAudioFormatError(
            f"Unsupported audio format for '{audio_path}'; only WAV is accepted"
        )
    if not audio_path.is_file():
        raise AudioFileNotFoundError(f"Audio file does not exist: '{audio_path}'")

    try:
        source_samples, source_sample_rate = sf.read(
            audio_path,
            dtype="float64",
            always_2d=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise InvalidAudioError(f"Could not read valid WAV audio from '{audio_path}'") from exc

    if not isinstance(source_sample_rate, (int, np.integer)) or source_sample_rate <= 0:
        raise InvalidAudioError(
            f"Invalid source sample rate in '{audio_path}': {source_sample_rate!r}"
        )
    if source_samples.ndim != 2 or source_samples.shape[1] < 1:
        raise InvalidAudioError(f"Audio must contain at least one channel: '{audio_path}'")
    if source_samples.shape[0] == 0:
        raise InvalidAudioError(f"Audio file contains no samples: '{audio_path}'")
    if not np.isfinite(source_samples).all():
        raise InvalidAudioError(f"Audio contains non-finite samples: '{audio_path}'")

    source_sample_rate = int(source_sample_rate)
    source_channels = int(source_samples.shape[1])
    mono_samples = source_samples.mean(axis=1, dtype=np.float64)

    if source_sample_rate != TARGET_SAMPLE_RATE_HZ:
        divisor = gcd(source_sample_rate, TARGET_SAMPLE_RATE_HZ)
        up = TARGET_SAMPLE_RATE_HZ // divisor
        down = source_sample_rate // divisor
        mono_samples = resample_poly(mono_samples, up=up, down=down)

    samples = np.asarray(mono_samples, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0:
        raise InvalidAudioError(f"Canonical audio contains no usable samples: '{audio_path}'")
    if not np.isfinite(samples).all():
        raise InvalidAudioError(f"Canonical audio contains non-finite samples: '{audio_path}'")

    return AudioData(
        samples=samples,
        sample_rate=TARGET_SAMPLE_RATE_HZ,
        source_sample_rate=source_sample_rate,
        source_channels=source_channels,
        duration_seconds=samples.size / TARGET_SAMPLE_RATE_HZ,
    )
