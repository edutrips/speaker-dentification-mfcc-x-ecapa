"""Source-recording-safe leave-one-text-id-out folds."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TextFold:
    held_out_text: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    audit: dict[str, object]


def text_folds(metadata: list[dict[str, str]]) -> list[TextFold]:
    """Build folds and fail if any recording/text leaks across a boundary."""

    if not metadata:
        raise ValueError("No feature metadata to evaluate")
    sessions = {row["session_id"] for row in metadata}
    if sessions != {"S01"}:
        raise ValueError("Exploratory same-session evaluation requires S01-only features")
    all_speakers = {row["speaker_id"] for row in metadata}
    if len(all_speakers) < 2:
        raise ValueError("At least two speakers are required")
    folds: list[TextFold] = []
    for text_id in sorted({row["text_id"] for row in metadata}):
        train = np.array([i for i, row in enumerate(metadata) if row["text_id"] != text_id])
        test = np.array([i for i, row in enumerate(metadata) if row["text_id"] == text_id])
        train_rows = [metadata[i] for i in train]
        test_rows = [metadata[i] for i in test]
        train_recordings = {row["recording_id"] for row in train_rows}
        test_recordings = {row["recording_id"] for row in test_rows}
        train_texts = {row["text_id"] for row in train_rows}
        test_texts = {row["text_id"] for row in test_rows}
        train_speakers = {row["speaker_id"] for row in train_rows}
        test_speakers = {row["speaker_id"] for row in test_rows}
        audit: dict[str, object] = {
            "held_out_text": text_id,
            "train_source_recordings": len(train_recordings),
            "test_source_recordings": len(test_recordings),
            "train_speakers": sorted(train_speakers),
            "test_speakers": sorted(test_speakers),
            "recording_overlap_count": len(train_recordings & test_recordings),
            "text_overlap_count": len(train_texts & test_texts),
        }
        if (
            not len(train)
            or not len(test)
            or train_speakers != all_speakers
            or test_speakers != all_speakers
            or audit["recording_overlap_count"] != 0
            or audit["text_overlap_count"] != 0
        ):
            raise ValueError(f"Invalid or leaking fold: {audit}")
        folds.append(TextFold(text_id, train, test, audit))
    if len(folds) < 2:
        raise ValueError("At least two text folds are required")
    return folds
