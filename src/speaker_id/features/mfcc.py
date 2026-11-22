"""MFCC, delta, delta-delta, and temporal statistics."""

import librosa
import numpy as np

from .config import MfccConfig


def extract_mfcc_vector(samples: np.ndarray, config: MfccConfig) -> np.ndarray:
    """Return a fixed, finite vector of coefficient means and standard deviations."""

    expected_samples = round(config.segment_duration_seconds * config.sample_rate_hz)
    if samples.ndim != 1 or samples.dtype != np.float32 or samples.size != expected_samples:
        raise ValueError("MFCC input must be one canonical fixed-duration float32 segment")
    if not np.isfinite(samples).all():
        raise ValueError("MFCC input contains non-finite samples")
    n_fft = round(config.frame_length_ms * config.sample_rate_hz / 1000)
    hop_length = round(config.frame_hop_ms * config.sample_rate_hz / 1000)
    coefficients = librosa.feature.mfcc(
        y=samples,
        sr=config.sample_rate_hz,
        n_mfcc=config.n_mfcc,
        n_mels=config.n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        center=False,
    )
    frames = np.concatenate(
        (
            coefficients,
            librosa.feature.delta(coefficients, order=1),
            librosa.feature.delta(coefficients, order=2),
        ),
        axis=0,
    )
    vector = np.concatenate((frames.mean(axis=1), frames.std(axis=1))).astype(np.float32)
    if vector.size != config.feature_dimension or not np.isfinite(vector).all():
        raise ValueError("MFCC feature vector has invalid dimension or non-finite values")
    return vector
