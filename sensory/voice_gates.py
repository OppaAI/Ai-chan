"""Shared barge / speaker gate flags for listen + webui.

Logic that uses these flags lives in sensory.listen (no runtime monkeypatch).
"""
from __future__ import annotations

import os


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def barge_in_enabled() -> bool:
    """Master barge-in switch (browser + Jetson)."""
    return _env_bool("BARGE_IN_ENABLED", "0")


def barge_in_always_on() -> bool:
    """Local Silero monitor outside TTS wait — only if barge_in_enabled()."""
    return barge_in_enabled() and _env_bool("BARGE_IN_ALWAYS_ON", "0")


def speaker_verify_enabled() -> bool:
    return _env_bool("SPEAKER_VERIFY_ENABLED", "0")


def speaker_verify_gate() -> bool:
    """When True with verify active, failed match drops the utterance."""
    return _env_bool("SPEAKER_VERIFY_GATE", "0")


def install_listen_s0_hooks(listen=None) -> None:
    """Deprecated no-op kept for import compatibility."""
    return None
