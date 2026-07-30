"""
Native barge / speaker / ASR-text gates for AikoListen.

Bound once at boot (see system/wakeup.py). No external monkeypatch module
and no reliance on install_listen_hooks. Flags and correct_asr_text live here
so listen.py stays the capture/ASR core.
"""
from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from typing import Any

_BOUND = False


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def barge_in_enabled() -> bool:
    """Master barge-in switch (browser + Jetson)."""
    return _env_bool("BARGE_IN_ENABLED", "0")


def barge_in_always_on() -> bool:
    """Local Silero monitor outside TTS wait — only if barge_in_enabled()."""
    return barge_in_enabled() and _env_bool("BARGE_IN_ALWAYS_ON", "0")


def speaker_verify_gate() -> bool:
    """When True with verify active, failed match drops the utterance."""
    return _env_bool("SPEAKER_VERIFY_GATE", "0")


_DEFAULT_ASR_PAIRS: tuple[tuple[str, str], ...] = (
    ("hey aiko", "hey Aiko"),
    ("hey iko", "hey Aiko"),
    ("hey eco", "hey Aiko"),
    ("hey ecko", "hey Aiko"),
    ("hey echo", "hey Aiko"),
    ("hey ico", "hey Aiko"),
    ("hey aico", "hey Aiko"),
    ("hi aiko", "hi Aiko"),
    ("hi iko", "hi Aiko"),
    ("aiko", "Aiko"),
    ("oppaai", "OppaAI"),
    ("oppa ai", "OppaAI"),
    ("op ai", "OppaAI"),
    ("oppa a i", "OppaAI"),
    ("opper ai", "OppaAI"),
    ("opa ai", "OppaAI"),
)


def _parse_asr_user_map(raw: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for part in (raw or "").split("|"):
        part = part.strip()
        if not part or "->" not in part:
            continue
        src, dst = part.split("->", 1)
        src, dst = src.strip(), dst.strip()
        if src and dst:
            out.append((src.lower(), dst))
    return out


@lru_cache(maxsize=4)
def _pairs_cached(user_raw: str) -> tuple[tuple[str, str], ...]:
    user = _parse_asr_user_map(user_raw)
    seen: set[str] = set()
    merged: list[tuple[str, str]] = []
    for src, dst in list(user) + list(_DEFAULT_ASR_PAIRS):
        key = src.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append((src, dst))
    merged.sort(key=lambda p: len(p[0]), reverse=True)
    return tuple(merged)


def correction_pairs() -> tuple[tuple[str, str], ...]:
    return _pairs_cached(os.getenv("ASR_CORRECTIONS", "").strip())


def correct_asr_text(text: str) -> str:
    """Apply name/phrase corrections; preserves non-matched regions."""
    if not text or not text.strip():
        return text
    out = text
    for src, dst in correction_pairs():
        pattern = re.compile(rf"(?i)(?<!\w){re.escape(src)}(?!\w)")
        out = pattern.sub(dst, out)
    return out


def align_listen_defaults(mod: Any) -> None:
    """Re-bind module-level fallbacks to match config/sensory.yaml."""
    mod.SILENCE_CHUNKS = int(os.getenv("LISTEN_SILENCE_CHUNKS", "66"))
    mod.BARGE_IN_THRESHOLD = float(os.getenv("BARGE_IN_THRESHOLD", "0.95"))
    mod.BARGE_IN_CONFIRM = int(os.getenv("BARGE_IN_CONFIRM_CHUNKS", "4"))
    mod.ACTIVATION_TIMEOUT_S = float(os.getenv("ACTIVATION_TIMEOUT_S", "3600"))


def bind_native_gates(mod: Any = None) -> None:
    """Wrap AikoListen methods with barge / speaker / ASR-text gates (idempotent)."""
    global _BOUND
    if _BOUND:
        return
    if mod is None:
        from sensory import listen as mod

    align_listen_defaults(mod)
    cls = mod.AikoListen
    log = mod.log

    _orig_trigger = cls.trigger_barge_in

    def trigger_barge_in(self) -> None:
        if not barge_in_enabled():
            return
        _orig_trigger(self)

    cls.trigger_barge_in = trigger_barge_in

    _orig_loop = cls._barge_in_loop

    def _barge_in_loop(self) -> None:
        if not barge_in_enabled():
            while getattr(self, "_barge_in_active", False):
                time.sleep(0.5)
            return
        _orig_loop(self)

    cls._barge_in_loop = _barge_in_loop

    _orig_listen = cls.listen

    def listen(
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
            wait_fn = None  # already waited out TTS; avoid double wait_fn

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
            log.info(
                "[gate] speaker verify failed (score=%s) — dropping utterance",
                info.get("speaker_score"),
            )
            return "", info
        return text, info

    cls.listen = listen

    _orig_transcribe = cls._transcribe

    def _transcribe(self, audio):
        text = _orig_transcribe(self, audio)
        try:
            return correct_asr_text(text)
        except Exception:
            return text

    cls._transcribe = _transcribe

    _BOUND = True
    log.info("native voice gates bound (barge / speaker / ASR text correct)")
