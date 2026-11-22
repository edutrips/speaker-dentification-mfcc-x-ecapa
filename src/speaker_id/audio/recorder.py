"""Hardware-independent microphone capture primitives and diagnostics."""

from dataclasses import dataclass
from threading import Event
from time import monotonic

import numpy as np

from .backend import get_sounddevice
from .devices import get_default_input_device, get_input_device
from .errors import RecordingError

DEFAULT_MAX_DURATION_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class RecordedAudio:
    """Raw captured samples and acquisition metadata.

    Mono samples are one-dimensional; multi-channel samples use
    ``(n_frames, channels)``. The samples remain unnormalized float32 values
    until they are written as PCM_16 by the collection persistence layer.
    """

    samples: np.ndarray
    sample_rate: int
    channels: int
    duration_seconds: float
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class RecordingDiagnostics:
    """Basic, non-rejecting diagnostics for one capture attempt."""

    duration_seconds: float
    sample_rate: int
    channels: int
    peak_abs_amplitude: float
    rms: float

    @property
    def duration_warning(self) -> str | None:
        if self.duration_seconds < 5:
            return "duration is below 5 seconds"
        if self.duration_seconds > 30:
            return "duration is above 30 seconds"
        return None

    @property
    def clipping_warning(self) -> bool:
        return self.peak_abs_amplitude >= 0.999


def record_until_stop(
    *,
    device: int | None = None,
    sample_rate: float | None = None,
    channels: int = 1,
    stop_event: Event | None = None,
    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
) -> RecordedAudio:
    """Capture until ``stop_event`` is set or the safety limit is reached.

    No terminal input or output is performed here. The caller owns the stop
    event, which keeps the function usable by a CLI and by mocked tests.
    """

    if channels <= 0:
        raise RecordingError("channels must be positive")
    if max_duration_seconds <= 0:
        raise RecordingError("max_duration_seconds must be positive")

    selected_device = (
        get_input_device(device) if device is not None else get_default_input_device()
    )
    sd = get_sounddevice()
    capture_rate = float(sample_rate or selected_device.default_samplerate)
    if capture_rate <= 0:
        raise RecordingError("sample_rate must be positive")

    event = stop_event or Event()
    chunks: list[np.ndarray] = []

    def callback(indata: np.ndarray, _frames: int, _time: object, _status: object) -> None:
        chunks.append(indata.copy())

    timed_out = False
    try:
        with sd.InputStream(
            samplerate=capture_rate,
            device=selected_device.index,
            channels=channels,
            dtype="float32",
            callback=callback,
        ):
            deadline = monotonic() + max_duration_seconds
            while not event.wait(0.05):
                if monotonic() >= deadline:
                    timed_out = True
                    event.set()
                    break
    except (OSError, RuntimeError, sd.PortAudioError) as exc:
        raise RecordingError(f"Microphone capture failed: {exc}") from exc

    if chunks:
        samples = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        if channels == 1:
            samples = samples[:, 0]
    elif channels == 1:
        samples = np.empty(0, dtype=np.float32)
    else:
        samples = np.empty((0, channels), dtype=np.float32)

    frame_count = samples.shape[0] if samples.ndim > 1 else samples.size
    return RecordedAudio(
        samples=samples,
        sample_rate=int(round(capture_rate)),
        channels=channels,
        duration_seconds=frame_count / capture_rate,
        timed_out=timed_out,
    )


def compute_diagnostics(audio: RecordedAudio) -> RecordingDiagnostics:
    """Calculate basic diagnostics without changing or rejecting samples."""

    samples = np.asarray(audio.samples, dtype=np.float64)
    if samples.size:
        peak = float(np.max(np.abs(samples)))
        rms = float(np.sqrt(np.mean(np.square(samples))))
    else:
        peak = 0.0
        rms = 0.0
    return RecordingDiagnostics(
        duration_seconds=audio.duration_seconds,
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        peak_abs_amplitude=peak,
        rms=rms,
    )
