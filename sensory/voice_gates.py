"""Backward-compatible re-exports from sensory.listen.

Prefer: from sensory.listen import barge_in_enabled, barge_in_always_on, speaker_verify_gate
"""
from sensory.listen import (  # noqa: F401
    barge_in_always_on,
    barge_in_enabled,
    speaker_verify_gate,
)


def install_listen_hooks(listen=None) -> None:
    """No-op: barge / speaker / ASR text fixes are native in listen.py."""
    return None


def install_listen_s0_hooks(listen=None) -> None:
    """Compat alias for install_listen_hooks."""
    return None
