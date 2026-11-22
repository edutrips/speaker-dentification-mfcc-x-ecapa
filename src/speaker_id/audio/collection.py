"""Persistence helpers for accepted and rejected collection attempts."""

import csv
import os
import re
import shutil
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

from speaker_id.datasets import (
    MANIFEST_FIELDS,
    RecordingRecord,
    load_manifest,
    validate_record,
    write_manifest,
)

from .errors import RecordingError
from .recorder import RecordedAudio

ATTEMPT_PATTERN = re.compile(r"_R(\d{3})(?:\.rejected)?\.wav$")
BACKUP_EXTRA_FIELDS = (
    "archived_at_utc",
    "replacement_recording_id",
    "archive_reason",
    "original_relative_path",
    "backup_relative_path",
)
BACKUP_MANIFEST_FIELDS = MANIFEST_FIELDS + BACKUP_EXTRA_FIELDS


def recording_id(speaker_id: str, session_id: str, text_id: str, attempt: int) -> str:
    """Build the deterministic recording identifier."""

    if not 1 <= attempt <= 999:
        raise RecordingError("attempt must be between 1 and 999")
    return f"{speaker_id}_{session_id}_{text_id}_R{attempt:03d}"


def next_attempt_number(
    project_root: str | Path, speaker_id: str, session_id: str, text_id: str
) -> int:
    """Return the next attempt after all accepted, rejected, staged, and archived takes."""

    root = Path(project_root)
    prefix = f"{speaker_id}_{session_id}_{text_id}_R"
    used_attempts: set[int] = set()

    manifest_path = root / "data/metadata/recordings.csv"
    if manifest_path.exists():
        used_attempts.update(
            _attempts_from_ids(
                record.recording_id
                for record in load_manifest(manifest_path)
                if record.speaker_id == speaker_id
                and record.session_id == session_id
                and record.text_id == text_id
            )
        )

    for directory, pattern in (
        (root / "data/raw" / speaker_id / session_id, f"{prefix}*.wav"),
        (root / "data/rejected" / speaker_id / session_id, f"{prefix}*.rejected.wav"),
        (root / "data/backup/recordings" / speaker_id / session_id, f"{prefix}*.wav"),
        (root / "data/staging/recordings" / speaker_id / session_id, f"{prefix}*.wav"),
    ):
        if directory.exists():
            for path in directory.glob(pattern):
                match = ATTEMPT_PATTERN.search(path.name)
                if match:
                    used_attempts.add(int(match.group(1)))

    backup_manifest = root / "data/backup/recordings.csv"
    if backup_manifest.exists():
        with backup_manifest.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            used_attempts.update(
                _attempts_from_ids(
                    row["recording_id"]
                    for row in reader
                    if row.get("speaker_id") == speaker_id
                    and row.get("session_id") == session_id
                    and row.get("text_id") == text_id
                )
            )

    attempt = max(used_attempts, default=0) + 1
    while attempt in used_attempts:
        attempt += 1
    if attempt > 999:
        raise RecordingError("No unused Rnnn attempt number remains")
    return attempt


def accepted_recording_exists(
    project_root: str | Path, speaker_id: str, session_id: str, text_id: str
) -> bool:
    """Return whether the manifest already accepts a recording for this prompt."""

    return active_recording_for_slot(project_root, speaker_id, session_id, text_id) is not None


def active_recording_for_slot(
    project_root: str | Path, speaker_id: str, session_id: str, text_id: str
) -> RecordingRecord | None:
    """Return the single valid active take, rejecting manifest/raw inconsistencies."""

    root = Path(project_root)
    manifest_path = root / "data/metadata/recordings.csv"
    records = load_manifest(manifest_path) if manifest_path.exists() else []
    matches = [
        record
        for record in records
        if (record.speaker_id, record.session_id, record.text_id)
        == (speaker_id, session_id, text_id)
    ]
    if len(matches) > 1:
        raise RecordingError(
            f"Multiple active manifest rows exist for {speaker_id}/{session_id}/{text_id}"
        )

    raw_directory = root / "data/raw" / speaker_id / session_id
    prefix = f"{speaker_id}_{session_id}_{text_id}_R"
    raw_files = sorted(path for path in raw_directory.glob(f"{prefix}*.wav") if path.is_file())
    if not matches:
        if raw_files:
            raise RecordingError(
                f"Raw WAV exists without an active manifest row for "
                f"{speaker_id}/{session_id}/{text_id}: {raw_files[0].name}"
            )
        return None

    record = matches[0]
    active_path = root / record.relative_path
    if not active_path.is_file():
        raise RecordingError(
            f"Manifest points to a missing active WAV for {speaker_id}/{session_id}/{text_id}: "
            f"{record.relative_path}"
        )
    if raw_files != [active_path]:
        raise RecordingError(
            f"Raw WAV files do not match the single manifest entry for "
            f"{speaker_id}/{session_id}/{text_id}"
        )
    try:
        info = sf.info(active_path)
        if info.frames <= 0:
            raise RecordingError(f"Active WAV is empty: {record.relative_path}")
    except (OSError, RuntimeError, ValueError) as exc:
        raise RecordingError(f"Active WAV is unreadable: {record.relative_path}") from exc
    return record


def persist_candidate_recording(
    project_root: str | Path, record: RecordingRecord, audio: RecordedAudio
) -> Path:
    """Persist a replacement candidate outside raw while leaving the active take intact."""

    _validate_audio_record(record, audio)
    root = Path(project_root)
    candidate = (
        root
        / "data/staging/recordings"
        / record.speaker_id
        / record.session_id
        / f"{record.recording_id}.wav"
    )
    _write_wav_exclusive(candidate, audio)
    return candidate


def replace_accepted_recording(
    project_root: str | Path,
    previous: RecordingRecord,
    replacement: RecordingRecord,
    candidate_path: str | Path,
) -> tuple[Path, Path]:
    """Promote an accepted candidate and archive the prior active take transactionally."""

    root = Path(project_root)
    validate_record(previous)
    validate_record(replacement)
    slot = (previous.speaker_id, previous.session_id, previous.text_id)
    if (replacement.speaker_id, replacement.session_id, replacement.text_id) != slot:
        raise RecordingError("Replacement recording must target the same logical slot")
    if previous.recording_id == replacement.recording_id:
        raise RecordingError("A replacement must use a new recording_id")

    active = active_recording_for_slot(root, *slot)
    if active != previous:
        raise RecordingError("Active recording changed before replacement promotion")

    candidate = Path(candidate_path)
    expected_candidate = (
        root
        / "data/staging/recordings"
        / replacement.speaker_id
        / replacement.session_id
        / f"{replacement.recording_id}.wav"
    )
    if candidate != expected_candidate or not candidate.is_file():
        raise RecordingError("Replacement candidate is missing or outside the staging namespace")

    manifest_path = root / "data/metadata/recordings.csv"
    original_records = load_manifest(manifest_path)
    active_rows = [row for row in original_records if row == previous]
    if len(active_rows) != 1:
        raise RecordingError("Expected exactly one old active manifest row")
    if any(row.recording_id == replacement.recording_id for row in original_records):
        raise RecordingError("Replacement recording_id already appears in the active manifest")

    replacement_path = root / replacement.relative_path
    rejected_path = (
        root
        / "data/rejected"
        / replacement.speaker_id
        / replacement.session_id
        / f"{replacement.recording_id}.rejected.wav"
    )
    backup_path = (
        root
        / "data/backup/recordings"
        / previous.speaker_id
        / previous.session_id
        / f"{previous.recording_id}.wav"
    )
    if replacement_path.exists() or rejected_path.exists():
        raise RecordingError(f"Replacement attempt already exists: {replacement.recording_id}")
    if backup_path.exists():
        raise RecordingError(f"Backup collision; refusing to overwrite: {backup_path}")

    backup_manifest_path = root / "data/backup/recordings.csv"
    backup_manifest_existed = backup_manifest_path.exists()
    backup_rows = _read_backup_manifest(backup_manifest_path)
    if any(row["recording_id"] == previous.recording_id for row in backup_rows):
        raise RecordingError(f"Backup manifest already contains {previous.recording_id}")

    old_path = root / previous.relative_path
    backup_created = False
    replacement_promoted = False
    try:
        _copy_file_exclusive(old_path, backup_path)
        backup_created = True
        if _sha256_file(old_path) != _sha256_file(backup_path):
            raise RecordingError("Backup verification failed; active recording was not changed")

        _link_without_overwrite(candidate, replacement_path)
        replacement_promoted = True

        backup_record = _backup_row(previous, replacement, root, backup_path)
        _write_backup_manifest_atomically(
            backup_manifest_path, [*backup_rows, backup_record]
        )
        new_records = [row for row in original_records if row != previous]
        new_records.append(replacement)
        _write_manifest_atomically(manifest_path, new_records)
        old_path.unlink()
    except BaseException:
        if not old_path.exists() and backup_created and backup_path.exists():
            try:
                _link_without_overwrite(backup_path, old_path)
            except BaseException:
                # Keep the archived copy and promoted candidate for manual recovery.
                raise
        rollback_ok = True
        replacement_state = _manifest_record_state(manifest_path, replacement)
        previous_state = _manifest_record_state(manifest_path, previous)
        if replacement_state is None or previous_state is None:
            rollback_ok = False
        elif replacement_state:
            try:
                _write_manifest_atomically(manifest_path, original_records)
            except BaseException:
                rollback_ok = False
        elif not previous_state:
            rollback_ok = False
        if rollback_ok:
            backup_state = _backup_manifest_record_state(
                backup_manifest_path, previous.recording_id
            )
            if backup_state is None:
                rollback_ok = False
            elif backup_state:
                try:
                    _write_backup_manifest_atomically(backup_manifest_path, backup_rows)
                except BaseException:
                    rollback_ok = False
                else:
                    if not backup_manifest_existed and not backup_rows:
                        backup_manifest_path.unlink(missing_ok=True)
            if rollback_ok and replacement_promoted:
                replacement_path.unlink(missing_ok=True)
            if rollback_ok and backup_created:
                backup_path.unlink(missing_ok=True)
        # Keep candidate staging as a recoverable copy if any transaction step failed.
        raise

    # The new active row and verified backup are committed before candidate cleanup.
    candidate.unlink(missing_ok=True)
    return replacement_path, backup_path


def persist_accepted_recording(
    project_root: str | Path, record: RecordingRecord, audio: RecordedAudio
) -> Path:
    """Write an accepted PCM_16 WAV and atomically update the manifest."""

    _validate_audio_record(record, audio)

    root = Path(project_root)
    accepted_path = root / record.relative_path
    rejected_path = (
        root
        / "data/rejected"
        / record.speaker_id
        / record.session_id
        / f"{record.recording_id}.rejected.wav"
    )
    if accepted_path.exists() or rejected_path.exists():
        raise RecordingError(f"Attempt already exists: {record.recording_id}")

    manifest_path = root / "data/metadata/recordings.csv"
    existing_records = load_manifest(manifest_path) if manifest_path.exists() else []
    slot_records = [
        row
        for row in existing_records
        if (row.speaker_id, row.session_id, row.text_id)
        == (record.speaker_id, record.session_id, record.text_id)
    ]
    if slot_records:
        raise RecordingError("An active recording already exists for this logical slot")
    raw_directory = root / "data/raw" / record.speaker_id / record.session_id
    prefix = f"{record.speaker_id}_{record.session_id}_{record.text_id}_R"
    if any(raw_directory.glob(f"{prefix}*.wav")):
        raise RecordingError("Raw WAV exists without an active manifest row for this slot")

    candidate = _candidate_path(root, record)
    _write_wav_exclusive(candidate, audio)
    promoted = False
    try:
        _link_without_overwrite(candidate, accepted_path)
        promoted = True
        _write_manifest_atomically(manifest_path, [*existing_records, record])
    except BaseException:
        if promoted and _manifest_record_state(manifest_path, record) is False:
            accepted_path.unlink(missing_ok=True)
        raise
    candidate.unlink(missing_ok=True)
    return accepted_path


def persist_rejected_recording(
    project_root: str | Path, record: RecordingRecord, audio: RecordedAudio
) -> Path:
    """Write a rejected PCM_16 WAV outside the experimental manifest."""

    validate_record(record)
    root = Path(project_root)
    rejected_path = (
        root
        / "data/rejected"
        / record.speaker_id
        / record.session_id
        / f"{record.recording_id}.rejected.wav"
    )
    if rejected_path.exists():
        raise RecordingError(f"Attempt already exists: {record.recording_id}")
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(rejected_path, audio.samples, audio.sample_rate, format="WAV", subtype="PCM_16")
    return rejected_path


def _attempts_from_ids(recording_ids: Iterable[str]) -> set[int]:
    attempts: set[int] = set()
    for identifier in recording_ids:
        match = ATTEMPT_PATTERN.search(f"{identifier}.wav")
        if match:
            attempts.add(int(match.group(1)))
    return attempts


def _write_manifest_atomically(path: Path, records: list[RecordingRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        write_manifest(temporary_path, records)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_audio_record(record: RecordingRecord, audio: RecordedAudio) -> None:
    validate_record(record)
    if record.source_sample_rate_hz != audio.sample_rate:
        raise RecordingError("Manifest sample rate does not match captured audio")
    if record.source_channels != audio.channels:
        raise RecordingError("Manifest channel count does not match captured audio")


def _candidate_path(root: Path, record: RecordingRecord) -> Path:
    return (
        root
        / "data/staging/recordings"
        / record.speaker_id
        / record.session_id
        / f"{record.recording_id}.wav"
    )


def _write_wav_exclusive(path: Path, audio: RecordedAudio) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb"):
            pass
    except FileExistsError as exc:
        raise RecordingError(f"Recording candidate already exists: {path}") from exc
    try:
        sf.write(path, audio.samples, audio.sample_rate, format="WAV", subtype="PCM_16")
        info = sf.info(path)
        if info.frames <= 0:
            raise RecordingError(f"Written candidate is empty: {path}")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _link_without_overwrite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise RecordingError(f"Refusing to overwrite existing recording: {destination}") from exc


def _copy_file_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        input_file = source.open("rb")
        try:
            output_file = destination.open("xb")
            created = True
        except BaseException:
            input_file.close()
            raise
        with input_file, output_file:
            shutil.copyfileobj(input_file, output_file)
            output_file.flush()
            os.fsync(output_file.fileno())
    except BaseException:
        if created and destination.exists():
            destination.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_backup_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != BACKUP_MANIFEST_FIELDS:
            raise RecordingError("Backup recordings manifest has an unexpected header")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise RecordingError("Backup recordings manifest contains malformed rows")
    return rows


def _backup_row(
    record: RecordingRecord,
    replacement: RecordingRecord,
    root: Path,
    backup_path: Path,
) -> dict[str, str]:
    values = {field: getattr(record, field) or "" for field in MANIFEST_FIELDS}
    values.update(
        {
            "archived_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "replacement_recording_id": replacement.recording_id,
            "archive_reason": "re_recorded",
            "original_relative_path": record.relative_path,
            "backup_relative_path": backup_path.relative_to(root).as_posix(),
        }
    )
    return values


def _write_backup_manifest_atomically(
    path: Path, rows: list[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            newline="",
            encoding="utf-8",
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.DictWriter(
                temporary_file,
                fieldnames=BACKUP_MANIFEST_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _manifest_record_state(path: Path, record: RecordingRecord) -> bool | None:
    if not path.exists():
        return None
    try:
        return record in load_manifest(path)
    except BaseException:
        return None


def _backup_manifest_record_state(path: Path, recording_id: str) -> bool | None:
    if not path.exists():
        return False
    try:
        return any(
            row["recording_id"] == recording_id for row in _read_backup_manifest(path)
        )
    except BaseException:
        return None
