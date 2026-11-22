"""Immutable metadata for source segments and their MFCC features."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AudioSegment:
    segment_id: str
    recording_id: str
    speaker_id: str
    session_id: str
    text_id: str
    split: str
    start_seconds: float
    end_seconds: float
    samples: np.ndarray


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    segment_id: str
    recording_id: str
    speaker_id: str
    session_id: str
    text_id: str
    split: str
    start_seconds: float
    end_seconds: float
    feature_vector: np.ndarray
