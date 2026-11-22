"""Validation and CSV I/O for the recordings manifest."""

import csv
import math
import re
from collections.abc import Iterable
from numbers import Real
from os import PathLike
from pathlib import Path, PurePosixPath, PureWindowsPath

from .errors import ManifestError, ManifestValidationError
from .types import RecordingRecord

SPEAKER_IDS = ("V01", "V02", "V03", "V04", "V05")
KNOWN_SPEAKER_IDS = SPEAKER_IDS
SESSION_TO_SPLIT = {
    "S01": "train",
    "S02": "validation",
    "S03": "test",
}
RECORDING_ID_PATTERN = re.compile(r"^([VU]\d{2})_(S\d{2})_(T\d{2})_(R\d{3})$")
UNKNOWN_SPEAKER_ID_PATTERN = re.compile(r"^U(?:0[1-9]|[1-9]\d)$")
TEXT_ID_PATTERN = re.compile(r"^T\d{2}$")

REQUIRED_FIELDS = (
    "recording_id",
    "speaker_id",
    "session_id",
    "text_id",
    "split",
    "relative_path",
    "source_sample_rate_hz",
    "source_channels",
    "duration_seconds",
)
OPTIONAL_FIELDS = ("device_label", "environment_label", "notes")
MANIFEST_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS


def is_known_speaker_id(speaker_id: str) -> bool:
    """Return whether an ID belongs to the configured known speaker classes."""

    return speaker_id in KNOWN_SPEAKER_IDS


def is_unknown_speaker_id(speaker_id: str) -> bool:
    """Return whether an ID follows the generic U01–U99 unknown identity contract."""

    return isinstance(speaker_id, str) and UNKNOWN_SPEAKER_ID_PATTERN.fullmatch(
        speaker_id
    ) is not None


def is_valid_speaker_id(speaker_id: str) -> bool:
    """Return whether an ID is a configured known or structurally valid unknown speaker."""

    return is_known_speaker_id(speaker_id) or is_unknown_speaker_id(speaker_id)


def validate_record(record: RecordingRecord) -> None:
    """Validate one record against the M2 manifest contract."""

    if not isinstance(record, RecordingRecord):
        raise ManifestValidationError("record must be a RecordingRecord")

    _require_text(record.recording_id, "recording_id")
    _require_text(record.speaker_id, "speaker_id")
    _require_text(record.session_id, "session_id")
    _require_text(record.text_id, "text_id")
    _require_text(record.split, "split")
    _require_text(record.relative_path, "relative_path")

    if not is_valid_speaker_id(record.speaker_id):
        raise ManifestValidationError(f"Invalid speaker_id: {record.speaker_id!r}")
    if record.session_id not in SESSION_TO_SPLIT:
        raise ManifestValidationError(f"Invalid session_id: {record.session_id!r}")
    expected_split = SESSION_TO_SPLIT[record.session_id]
    if record.split != expected_split:
        raise ManifestValidationError(
            f"session_id {record.session_id!r} requires split {expected_split!r}, "
            f"got {record.split!r}"
        )
    if TEXT_ID_PATTERN.fullmatch(record.text_id) is None:
        raise ManifestValidationError(f"Invalid text_id: {record.text_id!r}")

    recording_match = RECORDING_ID_PATTERN.fullmatch(record.recording_id)
    if recording_match is None:
        raise ManifestValidationError(f"Invalid recording_id: {record.recording_id!r}")
    recording_speaker, recording_session, recording_text, _ = recording_match.groups()
    if (
        recording_speaker != record.speaker_id
        or recording_session != record.session_id
        or recording_text != record.text_id
    ):
        raise ManifestValidationError(
            "recording_id is inconsistent with speaker_id, session_id, or text_id"
        )

    _validate_relative_path(record.relative_path, record)
    _validate_positive_integer(record.source_sample_rate_hz, "source_sample_rate_hz")
    _validate_positive_integer(record.source_channels, "source_channels")
    _validate_positive_duration(record.duration_seconds)
    for field_name in OPTIONAL_FIELDS:
        value = getattr(record, field_name)
        if value is not None and not isinstance(value, str):
            raise ManifestValidationError(f"{field_name} must be a string or None")


def validate_manifest(records: Iterable[RecordingRecord]) -> None:
    """Validate records and reject duplicate recording IDs."""

    seen_ids: set[str] = set()
    for record in records:
        validate_record(record)
        if record.recording_id in seen_ids:
            raise ManifestValidationError(f"Duplicate recording_id: {record.recording_id!r}")
        seen_ids.add(record.recording_id)


def load_manifest(path: str | PathLike[str]) -> list[RecordingRecord]:
    """Load and structurally validate a recordings CSV.

    The WAV files referenced by the records are intentionally not required to
    exist for this operation.
    """

    manifest_path = Path(path)
    try:
        with manifest_path.open("r", encoding="utf-8", newline="") as manifest_file:
            reader = csv.DictReader(manifest_file)
            _validate_header(reader.fieldnames)
            records = [
                _record_from_row(row, row_number) for row_number, row in enumerate(reader, 2)
            ]
    except OSError as exc:
        raise ManifestError(f"Could not read manifest '{manifest_path}'") from exc
    except csv.Error as exc:
        raise ManifestValidationError(f"Invalid CSV manifest '{manifest_path}'") from exc

    validate_manifest(records)
    return records


def write_manifest(path: str | PathLike[str], records: Iterable[RecordingRecord]) -> None:
    """Validate and write records to a CSV manifest."""

    records_list = list(records)
    validate_manifest(records_list)
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in records_list:
            writer.writerow(
                {field_name: getattr(record, field_name) or "" for field_name in MANIFEST_FIELDS}
            )


def _require_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ManifestValidationError(f"{field_name} must be a non-empty string")


def _validate_relative_path(relative_path: str, record: RecordingRecord) -> None:
    if (
        relative_path.startswith("/")
        or PurePosixPath(relative_path).is_absolute()
        or PureWindowsPath(relative_path).is_absolute()
    ):
        raise ManifestValidationError("relative_path must be relative")
    if "\\" in relative_path:
        raise ManifestValidationError("relative_path must use POSIX separators")

    path = PurePosixPath(relative_path)
    if ".." in path.parts:
        raise ManifestValidationError("relative_path must not contain path traversal")
    if path.suffix != ".wav":
        raise ManifestValidationError("relative_path must end with .wav")

    expected = PurePosixPath(
        "data/raw",
        record.speaker_id,
        record.session_id,
        f"{record.recording_id}.wav",
    )
    if path != expected:
        raise ManifestValidationError(
            f"relative_path must be '{expected.as_posix()}', got {relative_path!r}"
        )


def _validate_positive_integer(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestValidationError(f"{field_name} must be a positive integer")


def _validate_positive_duration(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ManifestValidationError("duration_seconds must be a positive finite float")
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise ManifestValidationError("duration_seconds must be a positive finite float")


def _validate_header(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ManifestValidationError("Manifest CSV is missing its header")
    if len(fieldnames) != len(set(fieldnames)):
        raise ManifestValidationError("Manifest CSV header contains duplicate fields")
    missing = [field for field in REQUIRED_FIELDS if field not in fieldnames]
    unknown = [field for field in fieldnames if field not in MANIFEST_FIELDS]
    if missing:
        raise ManifestValidationError(f"Manifest CSV is missing fields: {', '.join(missing)}")
    if unknown:
        raise ManifestValidationError(f"Manifest CSV contains unknown fields: {', '.join(unknown)}")


def _record_from_row(row: dict[str, str | None], row_number: int) -> RecordingRecord:
    if None in row:
        raise ManifestValidationError(f"Manifest row {row_number} contains extra columns")

    try:
        source_sample_rate_hz = int(_required_csv_value(row, "source_sample_rate_hz"))
        source_channels = int(_required_csv_value(row, "source_channels"))
        duration_seconds = float(_required_csv_value(row, "duration_seconds"))
    except ValueError as exc:
        raise ManifestValidationError(
            f"Invalid numeric value on manifest row {row_number}"
        ) from exc

    return RecordingRecord(
        recording_id=_required_csv_value(row, "recording_id"),
        speaker_id=_required_csv_value(row, "speaker_id"),
        session_id=_required_csv_value(row, "session_id"),
        text_id=_required_csv_value(row, "text_id"),
        split=_required_csv_value(row, "split"),
        relative_path=_required_csv_value(row, "relative_path"),
        source_sample_rate_hz=source_sample_rate_hz,
        source_channels=source_channels,
        duration_seconds=duration_seconds,
        device_label=_optional_csv_value(row.get("device_label")),
        environment_label=_optional_csv_value(row.get("environment_label")),
        notes=_optional_csv_value(row.get("notes")),
    )


def _required_csv_value(row: dict[str, str | None], field_name: str) -> str:
    value = row.get(field_name)
    if value is None or value == "":
        raise ManifestValidationError(f"Manifest field {field_name!r} is empty")
    return value


def _optional_csv_value(value: str | None) -> str | None:
    return None if value in (None, "") else value
