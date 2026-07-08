"""Helpers for per-user runtime paths and identifiers."""

from __future__ import annotations

import contextvars
import os
import re
from pathlib import Path

_DEFAULT_USER_ID = "OppaAI"
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_CURRENT_USER_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("aiko_current_user_id", default=None)


def set_current_user_id(user_id: str | None) -> contextvars.Token[str | None]:
    """Set the request-local active user id and return a token for reset()."""
    return _CURRENT_USER_ID.set(user_id)


def reset_current_user_id(token: contextvars.Token[str | None]) -> None:
    """Reset the request-local active user id using a token from set_current_user_id()."""
    _CURRENT_USER_ID.reset(token)


def current_user_id() -> str:
    """Return the active runtime user id from OAuth/session or local env."""
    return _CURRENT_USER_ID.get() or os.getenv("AIKO_USER_ID") or os.getenv("USER_ID") or _DEFAULT_USER_ID


def normalize_user_id(provider: str | None, user_id: object) -> str:
    """Create a filesystem-safe, provider-scoped id for OAuth identities."""
    provider_part = _SAFE_RE.sub("_", str(provider or "local")).strip("._-") or "local"
    user_part = _SAFE_RE.sub("_", str(user_id or _DEFAULT_USER_ID)).strip("._-") or _DEFAULT_USER_ID
    return f"{provider_part}_{user_part}"


def user_state_dir(user_id: str | None = None) -> Path:
    """Root directory for user-private mutable state.

    Defaults to ~/.aiko/<user_id>, but keeps existing installs on the legacy
    ~/.aiko/users/<user_id> layout when that directory already exists.
    """
    root_value = os.getenv("AIKO_USER_STATE_ROOT") or str(Path.home() / ".aiko")
    root = Path(root_value).expanduser()
    uid = _SAFE_RE.sub("_", user_id or current_user_id()).strip("._-") or _DEFAULT_USER_ID
    state_dir = root / uid
    legacy_dir = root / "users" / uid
    if not os.getenv("AIKO_USER_STATE_ROOT") and legacy_dir.exists() and not state_dir.exists():
        return legacy_dir
    return state_dir


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
