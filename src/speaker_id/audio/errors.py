"""Exceptions raised by the public audio loading API."""


class AudioError(Exception):
    """Base class for controlled audio contract errors."""


class AudioFileNotFoundError(AudioError):
    """Raised when the requested audio path does not refer to a file."""


class UnsupportedAudioFormatError(AudioError):
    """Raised when the input extension is not supported by this milestone."""


class InvalidAudioError(AudioError):
    """Raised when a WAV cannot satisfy the canonical audio contract."""


class AudioDeviceError(AudioError):
    """Raised when input devices cannot be queried or used."""


class NoInputDeviceError(AudioDeviceError):
    """Raised when no usable input device is available."""


class RecordingError(AudioError):
    """Raised when a capture or recording persistence operation fails."""
