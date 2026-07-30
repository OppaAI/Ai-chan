"""
sensory/asr_correct.py

S2: lightweight post-ASR text fixes for names / common SenseVoice mangling.
No model finetune — ordered phrase replacements (longest first).

Configure via ASR_CORRECTIONS env/yaml:
  "op ai->OppaAI|oppa ai->OppaAI|hey iko->hey Aiko|..."

Built-in defaults cover Aiko / OppaAI; user map is applied on top (wins on same key).
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

# phrase (lowercase) -> replacement (display form)
_DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (
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


def _parse_user_map(raw: str) -> list[tuple[str, str]]:
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
    user = _parse_user_map(user_raw)
    # User entries first so they override defaults when sources collide
    seen: set[str] = set()
    merged: list[tuple[str, str]] = []
    for src, dst in list(user) + list(_DEFAULT_PAIRS):
        key = src.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append((src, dst))
    # Longest source first so "hey aiko" wins over "aiko"
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
        # Word-boundary-ish match, case-insensitive
        pattern = re.compile(rf"(?i)(?<!\w){re.escape(src)}(?!\w)")
        out = pattern.sub(dst, out)
    return out


def install_asr_correct_hooks() -> None:
    """Wrap AikoListen._transcribe to run correct_asr_text (idempotent)."""
    from sensory import listen as mod

    cls = mod.AikoListen
    if getattr(cls, "_s2_asr_correct_installed", False):
        return

    orig = cls._transcribe

    def _transcribe(self, audio):
        text = orig(self, audio)
        try:
            return correct_asr_text(text)
        except Exception:
            return text

    cls._transcribe = _transcribe
    cls._s2_asr_correct_installed = True
