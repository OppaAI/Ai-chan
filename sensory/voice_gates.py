"""Backward-compatible re-exports from sensory.listen_native."""
from sensory.listen_native import (  # noqa: F401
    barge_in_always_on,
    barge_in_enabled,
    bind_native_gates,
    speaker_verify_gate,
)


def install_listen_hooks(listen=None) -> None:
    """Compat: bind native gates (idempotent)."""
    bind_native_gates()


def install_listen_s0_hooks(listen=None) -> None:
    """Compat alias."""
    bind_native_gates()
