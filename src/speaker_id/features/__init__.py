"""Canonical segmentation and MFCC feature extraction."""

from .config import MfccConfig, config_from_dict, load_mfcc_config
from .ecapa_constants import DEFAULT_ECAPA_SOURCE
from .ecapa_storage import (
    EcapaFeatureConfig,
    load_ecapa_features,
    save_ecapa_features,
)
from .mfcc import extract_mfcc_vector
from .pipeline import features_for_recording
from .segmentation import segment_audio
from .types import AudioSegment, FeatureRecord

__all__ = [
    "AudioSegment",
    "DEFAULT_ECAPA_SOURCE",
    "EcapaEncoder",
    "EcapaFeatureConfig",
    "FeatureRecord",
    "MfccConfig",
    "config_from_dict",
    "extract_mfcc_vector",
    "ecapa_features_for_recording",
    "features_for_recording",
    "load_mfcc_config",
    "load_ecapa_features",
    "segment_audio",
    "save_ecapa_features",
]


def __getattr__(name: str):
    """Load neural ECAPA APIs only when requested by a caller."""

    if name == "EcapaEncoder":
        from .ecapa import EcapaEncoder

        return EcapaEncoder
    if name == "ecapa_features_for_recording":
        from .ecapa_pipeline import ecapa_features_for_recording

        return ecapa_features_for_recording
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
