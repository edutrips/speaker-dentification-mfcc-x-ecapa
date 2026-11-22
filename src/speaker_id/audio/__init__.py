"""Public audio APIs with hardware capture imports kept lazy."""

from .errors import (
    AudioDeviceError,
    AudioError,
    AudioFileNotFoundError,
    InvalidAudioError,
    NoInputDeviceError,
    RecordingError,
    UnsupportedAudioFormatError,
)
from .loader import TARGET_SAMPLE_RATE_HZ, load_audio
from .types import AudioData

_LAZY_EXPORTS = {
    "DEFAULT_MAX_DURATION_SECONDS",
    "InputDevice",
    "RecordedAudio",
    "RecordingDiagnostics",
    "active_recording_for_slot",
    "accepted_recording_exists",
    "compute_diagnostics",
    "get_default_input_device",
    "get_input_device",
    "list_input_devices",
    "next_attempt_number",
    "persist_accepted_recording",
    "persist_candidate_recording",
    "persist_rejected_recording",
    "record_until_stop",
    "replace_accepted_recording",
    "recording_id",
}


def __getattr__(name: str) -> object:
    """Load capture-related modules only when their APIs are requested."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .collection import (
        accepted_recording_exists,
        active_recording_for_slot,
        next_attempt_number,
        persist_accepted_recording,
        persist_candidate_recording,
        persist_rejected_recording,
        recording_id,
        replace_accepted_recording,
    )
    from .devices import InputDevice, get_default_input_device, get_input_device, list_input_devices
    from .recorder import (
        DEFAULT_MAX_DURATION_SECONDS,
        RecordedAudio,
        RecordingDiagnostics,
        compute_diagnostics,
        record_until_stop,
    )

    exports = {
        "DEFAULT_MAX_DURATION_SECONDS": DEFAULT_MAX_DURATION_SECONDS,
        "InputDevice": InputDevice,
        "RecordedAudio": RecordedAudio,
        "RecordingDiagnostics": RecordingDiagnostics,
        "active_recording_for_slot": active_recording_for_slot,
        "accepted_recording_exists": accepted_recording_exists,
        "compute_diagnostics": compute_diagnostics,
        "get_default_input_device": get_default_input_device,
        "get_input_device": get_input_device,
        "list_input_devices": list_input_devices,
        "next_attempt_number": next_attempt_number,
        "persist_accepted_recording": persist_accepted_recording,
        "persist_candidate_recording": persist_candidate_recording,
        "persist_rejected_recording": persist_rejected_recording,
        "record_until_stop": record_until_stop,
        "recording_id": recording_id,
        "replace_accepted_recording": replace_accepted_recording,
    }
    globals().update(exports)
    return exports[name]


__all__ = [
    "AudioData",
    "AudioDeviceError",
    "AudioError",
    "AudioFileNotFoundError",
    "DEFAULT_MAX_DURATION_SECONDS",
    "InputDevice",
    "InvalidAudioError",
    "NoInputDeviceError",
    "RecordedAudio",
    "RecordingDiagnostics",
    "RecordingError",
    "TARGET_SAMPLE_RATE_HZ",
    "UnsupportedAudioFormatError",
    "accepted_recording_exists",
    "compute_diagnostics",
    "get_default_input_device",
    "get_input_device",
    "list_input_devices",
    "load_audio",
    "next_attempt_number",
    "persist_accepted_recording",
    "persist_rejected_recording",
    "record_until_stop",
    "recording_id",
]
