"""Lazy access to the optional-at-import-time PortAudio binding."""

def get_sounddevice() -> object:
    """Return the SoundDevice module when a hardware operation is requested."""

    import sounddevice

    return sounddevice
