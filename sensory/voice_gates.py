"""Shared ASR/voice gate flags + S0/S2 hooks for AikoListen."""
from __future__ import annotations

import os
import time
from typing import Any

_LOG = None
_HOOKS_INSTALLED = False


def _log():
    global _LOG
    if _LOG is None:
        import logging
        _LOG = logging.getLogger(__name__)
    return _LOG


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


def install_listen_s0_hooks(listen: Any = None) -> None:
    """Patch AikoListen barge + speaker-gate + S2 ASR correct (idempotent)."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return
    try:
        from sensory import listen as mod
    except Exception as e:
        _log().debug("S0 hooks deferred: %s", e)
        return

    cls = mod.AikoListen

    _orig_trigger = cls.trigger_barge_in

    def trigger_barge_in(self) -> None:
        if not barge_in_enabled():
            return
        _orig_trigger(self)

    cls.trigger_barge_in = trigger_barge_in

    _orig_listen = cls.listen

    def listen_wrapped(
        self,
        status_callback=None,
        wait_fn=None,
        speak=None,
        chunk_source=None,
        vad_presegmented: bool = False,
    ):
        if speak is not None and speak.is_playing() and not barge_in_enabled():
            if status_callback:
                try:
                    status_callback("__WAITING__")
                except Exception:
                    pass
            while speak.is_playing():
                time.sleep(0.05)
            speak = None

        text, info = _orig_listen(
            self,
            status_callback=status_callback,
            wait_fn=wait_fn,
            speak=speak,
            chunk_source=chunk_source,
            vad_presegmented=vad_presegmented,
        )
        if (
            speaker_verify_gate()
            and self.speaker_verify_active()
            and info is not None
            and info.get("verified") is False
        ):
            _log().info("[gate] speaker verify failed (score=%s) — dropping utterance",
                        info.get("speaker_score"))
            return "", info
        return text, info

    cls.listen = listen_wrapped

    _orig_loop = cls._barge_in_loop

    def _barge_in_loop(self) -> None:
        if not barge_in_enabled() and not barge_in_always_on():
            while getattr(self, "_barge_in_active", False):
                time.sleep(0.5)
            return
        _orig_loop(self)

    cls._barge_in_loop = _barge_in_loop

    # S2: post-ASR name/phrase corrections
    try:
        from sensory.asr_correct import install_asr_correct_hooks
        install_asr_correct_hooks()
    except Exception:
        _log().exception("S2 ASR correct hooks failed")

    _HOOKS_INSTALLED = True
    _log().info("ASR S0/S2 hooks installed (barge / speaker gate / asr_correct)")

    if listen is not None:
        pass
