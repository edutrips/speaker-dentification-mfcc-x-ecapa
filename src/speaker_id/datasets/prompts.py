"""CSV prompt loading and validation for guided collection."""

import csv
import re
from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from types import MappingProxyType

from .errors import ManifestError, ManifestValidationError

PROMPT_FIELDS = ("text_id", "session_id", "text")
PROMPT_SESSION_RANGES = MappingProxyType(
    {
        "S01": range(1, 6),
        "S02": range(6, 11),
        "S03": range(11, 16),
    }
)
TEXT_ID_PATTERN = re.compile(r"^T\d{2}$")
SESSION_IDS = tuple(PROMPT_SESSION_RANGES)


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """Immutable prompt metadata used by the guided collection CLI."""

    text_id: str
    session_id: str
    text: str


def load_prompts(path: str | PathLike[str]) -> list[PromptRecord]:
    """Load and validate prompts from a CSV file."""

    prompt_path = Path(path)
    try:
        with prompt_path.open("r", encoding="utf-8", newline="") as prompt_file:
            reader = csv.DictReader(prompt_file)
            if reader.fieldnames != list(PROMPT_FIELDS):
                raise ManifestValidationError(
                    f"Prompt CSV header must be: {','.join(PROMPT_FIELDS)}"
                )
            prompts = []
            for row_number, row in enumerate(reader, 2):
                if None in row:
                    raise ManifestValidationError(
                        f"Prompt row {row_number} contains extra columns"
                    )
                values = [row.get(field) for field in PROMPT_FIELDS]
                if any(value is None or not value.strip() for value in values):
                    raise ManifestValidationError(f"Prompt row {row_number} has an empty field")
                prompts.append(
                    PromptRecord(
                        text_id=values[0],
                        session_id=values[1],
                        text=values[2],
                    )
                )
    except OSError as exc:
        raise ManifestError(f"Could not read prompts '{prompt_path}'") from exc
    except csv.Error as exc:
        raise ManifestValidationError(f"Invalid prompt CSV '{prompt_path}'") from exc

    validate_prompts(prompts)
    return prompts


def validate_prompts(prompts: Iterable[PromptRecord]) -> None:
    """Validate prompt IDs, session assignment, text, and uniqueness."""

    seen_ids: set[str] = set()
    for prompt in prompts:
        if not isinstance(prompt, PromptRecord):
            raise ManifestValidationError("prompt must be a PromptRecord")
        if TEXT_ID_PATTERN.fullmatch(prompt.text_id) is None:
            raise ManifestValidationError(f"Invalid text_id: {prompt.text_id!r}")
        if prompt.session_id not in SESSION_IDS:
            raise ManifestValidationError(f"Invalid session_id: {prompt.session_id!r}")
        try:
            text_number = int(prompt.text_id[1:])
        except ValueError as exc:
            raise ManifestValidationError(f"Invalid text_id: {prompt.text_id!r}") from exc
        if text_number not in PROMPT_SESSION_RANGES[prompt.session_id]:
            raise ManifestValidationError(
                f"{prompt.text_id} is not assigned to {prompt.session_id}"
            )
        if not isinstance(prompt.text, str) or not prompt.text.strip():
            raise ManifestValidationError(f"Prompt text is empty for {prompt.text_id!r}")
        if prompt.text_id in seen_ids:
            raise ManifestValidationError(f"Duplicate text_id: {prompt.text_id!r}")
        seen_ids.add(prompt.text_id)


def prompts_for_session(
    prompts: Iterable[PromptRecord], session_id: str
) -> list[PromptRecord]:
    """Return prompts for one session in text ID order."""

    if session_id not in SESSION_IDS:
        raise ManifestValidationError(f"Invalid session_id: {session_id!r}")
    selected = [prompt for prompt in prompts if prompt.session_id == session_id]
    return sorted(selected, key=lambda prompt: prompt.text_id)
