"""Data structures for canonical audio."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AudioData:
    """Canonical audio samples and provenance metadata."""

    samples: np.ndarray
    sample_rate: int
    source_sample_rate: int
    source_channels: int
    duration_seconds: float
