"""Deterministic recording-preserving segmentation."""

import numpy as np

from speaker_id.datasets import RecordingRecord

from .config import MfccConfig
from .types import AudioSegment


def segment_audio(
    samples: np.ndarray, recording: RecordingRecord | None, config: MfccConfig
) -> list[AudioSegment]:
    """Create overlapping full windows and at most one non-overlapping padded tail.

    The tail starts where the last full window ends. A recording shorter than
    one window is a single padded tail only if it meets the minimum duration.
    """

    if samples.ndim != 1 or samples.dtype != np.float32:
        raise ValueError("Expected canonical mono float32 samples")
    rate = config.sample_rate_hz
    window = round(config.segment_duration_seconds * rate)
    hop = round(config.segment_hop_seconds * rate)
    minimum = round(config.minimum_final_segment_seconds * rate)
    if min(window, hop, minimum) <= 0:
        raise ValueError("Segment lengths must be positive")

    spans: list[tuple[int, int]] = []
    start = 0
    while start + window <= samples.size:
        spans.append((start, start + window))
        start += hop
    tail_start = spans[-1][1] if spans else 0
    if samples.size - tail_start >= minimum:
        spans.append((tail_start, samples.size))

    segments: list[AudioSegment] = []
    for index, (begin, end) in enumerate(spans):
        segment_samples = np.zeros(window, dtype=np.float32)
        segment_samples[: end - begin] = samples[begin:end]
        recording_id = recording.recording_id if recording is not None else "INPUT"
        segments.append(
            AudioSegment(
                segment_id=f"{recording_id}_SEG{index:03d}",
                recording_id=recording_id,
                speaker_id=recording.speaker_id if recording is not None else "",
                session_id=recording.session_id if recording is not None else "",
                text_id=recording.text_id if recording is not None else "",
                split=recording.split if recording is not None else "",
                start_seconds=begin / rate,
                end_seconds=end / rate,
                samples=segment_samples,
            )
        )
    return segments
