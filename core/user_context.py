"""Helpers for per-user runtime paths and identifiers."""

from __future__ import annotations

import os
import re
from pathlib import Path

_DEFAULT_USER_ID = "OppaAI"
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def current_user_id() -> str:
    """Return the active runtime user id from OAuth/session or local env."""
    return os.getenv("AIKO_USER_ID") or os.getenv("USER_ID") or _DEFAULT_USER_ID


def normalize_user_id(provider: str | None, user_id: object) -> str:
    """Create a filesystem-safe, provider-scoped id for OAuth identities."""
    provider_part = _SAFE_RE.sub("_", str(provider or "local")).strip("._-") or "local"
    user_part = _SAFE_RE.sub("_", str(user_id or _DEFAULT_USER_ID)).strip("._-") or _DEFAULT_USER_ID
    return f"{provider_part}_{user_part}"


def user_state_dir(user_id: str | None = None) -> Path:
    """Root directory for user-private mutable state."""
    root = Path(os.getenv("AIKO_USER_STATE_ROOT", str(Path.home() / ".aiko" / "users"))).expanduser()
    uid = _SAFE_RE.sub("_", user_id or current_user_id()).strip("._-") or _DEFAULT_USER_ID
    return root / uid


def user_state_path(filename: str, user_id: str | None = None) -> Path:
    return user_state_dir(user_id) / filename


def user_workspace_root(user_id: str | None = None) -> Path:
    """Workspace root isolated by user unless WORKSPACE_ROOT explicitly overrides."""
    if os.getenv("WORKSPACE_ROOT"):
        return Path(os.environ["WORKSPACE_ROOT"]).expanduser().resolve()
    return (user_state_dir(user_id) / "workspace").resolve()


def user_profile_path(user_id: str | None = None) -> Path:
    """Per-user editable profile/bio markdown path."""
    if os.getenv("USER_PROFILE_PATH"):
        return Path(os.environ["USER_PROFILE_PATH"]).expanduser().resolve()
    return user_state_path("user.md", user_id).resolve()
