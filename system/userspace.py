"""
system/userspace.py

Helpers for per-user runtime paths and identifiers.

This module provides utilities for managing per-user state in a multi-user
environment. All user-specific data is stored under <USER_STATE_ROOT>/<user_id>/ with
subdirectories:

  memory/         — SQLite memory DB, embeddings, consolidation state
  profile/        — USER.md profile/bio markdown  
  workspace/      — user workspace (code, projects)
  social/weekly/  — weekly social draft bundles (images, posts)
  logs/           — per-user log files

Key functions:
  - current_user_id()     — get the active user ID from session or env
  - user_state_dir()      — resolve <USER_STATE_ROOT>/<user_id> for a user
  - user_state_path()     — resolve a file path under user state
  - user_workspace_root() — resolve workspace root for a user
  - user_profile_path()   — resolve profile path (defaults to profile/USER.md)
  - set_current_user_id() / reset_current_user_id() — per-request user context

The multi-user design allows running multiple Aiko instances (e.g., for
different team members) on the same machine, each with their own isolated
state, memories, and configurations.
"""

from __future__ import annotations

import contextvars
import json
from pathlib import Path
import re
import tempfile

_DEFAULT_USER_ID = "guest"
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_DOTDOT_RE = re.compile(r"\.{2,}")
_CURRENT_USER_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("aiko_current_user_id", default=None)
_CURRENT_DISPLAY_NAME: contextvars.ContextVar[str | None] = contextvars.ContextVar("aiko_current_display_name", default=None)


def set_current_user_id(user_id: str | None) -> contextvars.Token[str | None]:
    """Set the request-local active user id and return a token for reset()."""
    return _CURRENT_USER_ID.set(user_id)


def reset_current_user_id(token: contextvars.Token[str | None]) -> None:
    """Reset the request-local active user id using a token from set_current_user_id()."""
    _CURRENT_USER_ID.reset(token)


def current_user_id() -> str:
    """Return the active runtime user id from OAuth/session or local env."""
    return _CURRENT_USER_ID.get() or os.getenv("AIKO_USER_ID") or _DEFAULT_USER_ID


def set_current_display_name(name: str | None) -> contextvars.Token[str | None]:
    """Set the request-local display name (e.g. GitHub login) and return a token."""
    return _CURRENT_DISPLAY_NAME.set(name)


def reset_current_display_name(token: contextvars.Token[str | None]) -> None:
    """Reset the display name context var using a token from set_current_display_name()."""
    _CURRENT_DISPLAY_NAME.reset(token)


def _display_name_cache_path(user_id: str) -> Path:
    """Cache of the last-known display name for a user_id, resolved at
    login. Lives under the user's own state dir (not profile/USER.md,
    which is user-editable bio content) — this is a runtime bridge value
    so the scheduler daemon thread and a fresh boot (neither of which has
    the per-request contextvar) can recover the login-resolved name."""
    return user_state_path("runtime/display_name.json", user_id)


def remember_display_name(user_id: str, display_name: str) -> None:
    """Cache a display name resolved at login (GitHub/Patreon handle)."""
    if not display_name:
        return
    path = _display_name_cache_path(user_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"display_name": display_name}), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        import logging
        logging.getLogger(__name__).warning("Failed to cache display name: %s", e)


def _load_cached_display_name(user_id: str) -> str | None:
    path = _display_name_cache_path(user_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("display_name")
    except Exception:
        return None


def current_display_name() -> str:
    """Return the user's display name (e.g. GitHub login).
    Order: request-local contextvar -> cached login value under
    USER_STATE_ROOT -> env var -> raw user_id as last resort."""
    name = _CURRENT_DISPLAY_NAME.get()
    if name:
        return name
    uid = current_user_id()
    cached = _load_cached_display_name(uid)
    if cached:
        return cached
    return os.getenv("AIKO_DISPLAY_NAME") or uid


def normalize_user_id(provider: str | None, user_id: object) -> str:
    """Create a filesystem-safe, provider-scoped id for OAuth identities."""
    provider_part = _SAFE_RE.sub("_", str(provider or "local")).strip("._-") or "local"
    provider_part = _DOTDOT_RE.sub("_", provider_part)
    user_part = _SAFE_RE.sub("_", str(user_id or _DEFAULT_USER_ID)).strip("._-") or _DEFAULT_USER_ID
    user_part = _DOTDOT_RE.sub("_", user_part)
    return f"{provider_part}_{user_part}"


def _user_state_root_value() -> str:
    """Return the configured root for per-user mutable state.

    USER_STATE_ROOT is the canonical name. AIKO_USER_STATE_ROOT and the older
    USER_SPACE_ROOT are accepted as compatibility aliases so deployments and
    docs that used those names still point Aiko at the same per-user files.
    """
    return (
        os.getenv("USER_STATE_ROOT")
        or os.getenv("AIKO_USER_STATE_ROOT")
        or os.getenv("USER_SPACE_ROOT")
        or str(Path.home() / ".aiko")
    )


def user_state_dir(user_id: str | None = None) -> Path:
    """Root directory for user-private mutable state.

    Resolves to <USER_STATE_ROOT>/<user_id>. For a real authenticated
    user_id, creates it (locked to owner-only) if missing. For the guest
    sentinel (no one authenticated yet), returns the path WITHOUT creating
    it — callers doing existence checks (e.g. profile lookup) correctly
    see nothing there, and no stray folder is left on disk before login.
    """
    root = Path(_user_state_root_value()).expanduser()
    uid = user_id or current_user_id()
    uid = _SAFE_RE.sub("_", uid).strip("._-") or _DEFAULT_USER_ID
    uid = _DOTDOT_RE.sub("_", uid)
    path = root / uid

    if uid == _DEFAULT_USER_ID:
        return path  # no mkdir — nothing on disk for an unauthenticated guest

    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def user_state_path(filename: str, user_id: str | None = None) -> Path:
    return user_state_dir(user_id) / filename


def user_workspace_root(user_id: str | None = None) -> Path:
    """Workspace root isolated by user unless WORKSPACE_ROOT explicitly overrides."""
    if os.getenv("WORKSPACE_ROOT"):
        return Path(os.environ["WORKSPACE_ROOT"]).expanduser().resolve()
    return (user_state_dir(user_id) / "workspace").resolve()


def user_profile_path(user_id: str | None = None) -> Path:
    """Per-user editable profile/bio markdown path.

    Defaults to <USER_STATE_ROOT>/<user_id>/profile/USER.md. The profile stores
    user-provided biographical information, preferences, and identity
    details that Aiko can use to personalize responses.
    """
    if os.getenv("USER_PROFILE_PATH"):
        return Path(os.environ["USER_PROFILE_PATH"]).expanduser().resolve()
    return user_state_path("profile/USER.md", user_id).resolve()
