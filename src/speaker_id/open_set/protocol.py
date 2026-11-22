"""Frozen data protocol for the current open-set speaker experiment."""

from collections.abc import Iterable

from speaker_id.datasets import KNOWN_SPEAKER_IDS, RecordingRecord

SUPPORTED_KNOWN_SPEAKER_IDS = KNOWN_SPEAKER_IDS
ACTIVE_KNOWN_SPEAKER_IDS = ("V01", "V02", "V03", "V04")
OPTIONAL_KNOWN_SPEAKER_IDS = ("V05",)

UNKNOWN_VALIDATION_SPEAKER_IDS = ("U01", "U02", "U03")
UNKNOWN_TEST_SPEAKER_IDS = ("U04", "U05", "U06")

TRAINING_SESSION = "S01"
VALIDATION_SESSION = "S02"
TEST_SESSION = "S03"

TRAINING_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"

TRAINING_TEXT_IDS = tuple(f"T{number:02d}" for number in range(1, 6))
VALIDATION_TEXT_IDS = tuple(f"T{number:02d}" for number in range(6, 11))
TEST_TEXT_IDS = tuple(f"T{number:02d}" for number in range(11, 16))
VALIDATION_KNOWN_TEXT_IDS = tuple(f"T{number:02d}" for number in range(6, 11))


def is_open_set_training_record(record: RecordingRecord) -> bool:
    """Return whether a recording belongs to the current known-speaker train split."""

    return (
        record.speaker_id in ACTIVE_KNOWN_SPEAKER_IDS
        and record.session_id == TRAINING_SESSION
        and record.split == TRAINING_SPLIT
        and record.text_id in TRAINING_TEXT_IDS
    )


def select_open_set_training_records(
    records: Iterable[RecordingRecord],
) -> list[RecordingRecord]:
    """Select eligible recordings, preserving input order and record instances."""

    return [record for record in records if is_open_set_training_record(record)]


def is_open_set_validation_known_record(record: RecordingRecord) -> bool:
    """Return whether a recording is active-known S02 validation data."""

    return (
        record.speaker_id in ACTIVE_KNOWN_SPEAKER_IDS
        and record.session_id == VALIDATION_SESSION
        and record.split == VALIDATION_SPLIT
        and record.text_id in VALIDATION_KNOWN_TEXT_IDS
    )


def select_open_set_validation_known_records(
    records: Iterable[RecordingRecord],
) -> list[RecordingRecord]:
    """Select eligible known-validation recordings, preserving input order."""

    return [record for record in records if is_open_set_validation_known_record(record)]


def is_open_set_validation_unknown_record(record: RecordingRecord) -> bool:
    """Return whether a recording belongs to the frozen unknown-validation split."""

    return (
        record.speaker_id in UNKNOWN_VALIDATION_SPEAKER_IDS
        and record.session_id == VALIDATION_SESSION
        and record.split == VALIDATION_SPLIT
        and record.text_id in VALIDATION_TEXT_IDS
    )


def select_open_set_validation_unknown_records(
    records: Iterable[RecordingRecord],
) -> list[RecordingRecord]:
    """Select eligible unknown-validation recordings, preserving input order."""

    return [record for record in records if is_open_set_validation_unknown_record(record)]
