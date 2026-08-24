"""Shared public-social identity helpers (Threads, Bluesky, ...).

Sources (no hardcoding in yaml):
  - AI name / phrase trigger: AI_NAME from config/system.yaml (via env after load)
  - Platform handle: .env only (THREADS_USERNAME, BLUESKY_HANDLE, ...)
  - Owner display name: <USER_SPACE_ROOT>/<user_id>/profile/USER.md
"""

from __future__ import annotations

import re

try:
    from social.services import env
except ModuleNotFoundError:
    from . import env  # type: ignore

_NAME_LINE_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?(?:name|what to call them|display name|call me)(?:\*\*)?\s*[:\-]\s*(.+?)\s*$"
)


def ai_name() -> str:
    return (env("AI_NAME") or "Aiko").strip() or "Aiko"


def reply_trigger_phrase(platform: str = "") -> str:
    """Default public trigger: Hi {AI_NAME}. Optional per-platform override env."""
    key = f"{platform.upper()}_REPLY_TRIGGER" if platform else ""
    if key:
        override = env(key, "").strip()
        if override:
            return override
    # Generic override used by Bluesky path historically
    override = env("SOCIAL_REPLY_TRIGGER", "").strip()
    if override:
        return override
    return f"Hi {ai_name()}"


def platform_username(env_key: str) -> str:
    """Handle from .env only; strip leading @."""
    return env(env_key, "").strip().lstrip("@")


def mention_trigger(env_key: str) -> str:
    """Mention form: @ + username from .env. Empty if username unset."""
    user = platform_username(env_key)
    return f"@{user}" if user else ""


def owner_display_name() -> str:
    """Prefer Name from profile/USER.md; fall back to current_display_name()."""
    try:
        from system.userspace import user_profile_path, current_display_name

        path = user_profile_path()
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            match = _NAME_LINE_RE.search(text)
            if match:
                name = match.group(1).strip().strip("*`\"'")
                if name:
                    return name
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("#"):
                    name = line.lstrip("#").strip()
                    if name:
                        return name
        return current_display_name()
    except Exception:
        return env("CURRENT_DISPLAY_NAME") or env("AIKO_USER_ID") or "owner"


def is_trigger(text: str, *, phrase: str, mention: str) -> bool:
    body = str(text or "")
    return phrase.casefold() in body.casefold() or bool(
        mention and mention.casefold() in body.casefold()
    )
