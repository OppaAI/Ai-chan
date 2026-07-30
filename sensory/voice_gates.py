"""Backward-compatible re-exports — prefer sensory.listen. """
from sensory.listen import (  # noqa: F401
    barge_in_always_on,
    barge_in_enabled,
    speaker_verify_gate,
)


def install_listen_s0_hooks(listen=None) -> None:
    """No-op: gates and ASR text fixes live in listen.py."""
    return None
