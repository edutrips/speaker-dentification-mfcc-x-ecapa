"""Dataset manifest contract for milestone M2."""

from .errors import ManifestError, ManifestValidationError
from .manifest import (
    KNOWN_SPEAKER_IDS,
    MANIFEST_FIELDS,
    REQUIRED_FIELDS,
    SESSION_TO_SPLIT,
    SPEAKER_IDS,
    is_known_speaker_id,
    is_unknown_speaker_id,
    is_valid_speaker_id,
    load_manifest,
    validate_manifest,
    validate_record,
    write_manifest,
)
from .prompts import PromptRecord, load_prompts, prompts_for_session, validate_prompts
from .types import RecordingRecord

__all__ = [
    "MANIFEST_FIELDS",
    "ManifestError",
    "ManifestValidationError",
    "KNOWN_SPEAKER_IDS",
    "REQUIRED_FIELDS",
    "RecordingRecord",
    "PromptRecord",
    "SESSION_TO_SPLIT",
    "SPEAKER_IDS",
    "is_known_speaker_id",
    "is_unknown_speaker_id",
    "is_valid_speaker_id",
    "load_manifest",
    "load_prompts",
    "prompts_for_session",
    "validate_manifest",
    "validate_record",
    "validate_prompts",
    "write_manifest",
]
