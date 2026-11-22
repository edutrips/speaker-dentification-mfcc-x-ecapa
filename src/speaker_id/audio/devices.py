"""Input-device discovery independent of the collection CLI."""

from dataclasses import dataclass
from math import isfinite
from typing import Any

from .backend import get_sounddevice
from .errors import AudioDeviceError, NoInputDeviceError


@dataclass(frozen=True, slots=True)
class InputDevice:
    """The non-sensitive input-device fields needed for collection."""

    index: int
    name: str
    max_input_channels: int
    default_samplerate: float


def list_input_devices() -> list[InputDevice]:
    """Return devices that expose at least one input channel."""

    sd = get_sounddevice()
    try:
        raw_devices = sd.query_devices()
    except (OSError, RuntimeError, sd.PortAudioError) as exc:
        raise AudioDeviceError(f"Could not query audio devices: {exc}") from exc

    if isinstance(raw_devices, dict):
        raw_devices = [raw_devices]

    devices: list[InputDevice] = []
    for index, info in enumerate(raw_devices):
        max_input_channels = int(info.get("max_input_channels", 0))
        if max_input_channels <= 0:
            continue
        default_samplerate = float(info.get("default_samplerate", 0.0))
        if not isfinite(default_samplerate) or default_samplerate <= 0:
            raise AudioDeviceError(f"Invalid default sample rate for input device {index}")
        devices.append(
            InputDevice(
                index=index,
                name=str(info.get("name", f"Input device {index}")),
                max_input_channels=max_input_channels,
                default_samplerate=default_samplerate,
            )
        )
    return devices


def get_input_device(index: int) -> InputDevice:
    """Return a specific input device or raise a controlled error."""

    for device in list_input_devices():
        if device.index == index:
            return device
    raise NoInputDeviceError(f"No input device with index {index}")


def get_default_input_device() -> InputDevice:
    """Return the configured default input device."""

    sd = get_sounddevice()
    devices = list_input_devices()
    if not devices:
        raise NoInputDeviceError("No input device with input channels is available")

    try:
        configured_default = sd.default.device
    except (OSError, RuntimeError, sd.PortAudioError) as exc:
        raise AudioDeviceError(f"Could not query the default input device: {exc}") from exc

    if hasattr(configured_default, "__getitem__"):
        default_index = configured_default[0]
    else:
        default_index = configured_default
    if default_index is not None and int(default_index) >= 0:
        for device in devices:
            if device.index == int(default_index):
                return device

    try:
        default_info: Any = sd.query_devices(kind="input")
    except (OSError, RuntimeError, sd.PortAudioError) as exc:
        raise AudioDeviceError(f"Could not query the default input device: {exc}") from exc
    default_name = str(default_info.get("name", ""))
    matching_devices = [device for device in devices if device.name == default_name]
    if len(matching_devices) == 1:
        return matching_devices[0]
    if len(devices) == 1:
        return devices[0]
    raise NoInputDeviceError("The default input device could not be identified")
