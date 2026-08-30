"""
system/userspace.py
 
Helpers for per-user runtime paths and identifiers.
 
This module provides utilities for managing per-user state in a multi-user
environment. All user-specific data is stored under <USER_SPACE_ROOT>/<user_id>/ with
subdirectories:
 
  memory/         — SQLite memory DB, embeddings, consolidation state
  profile/        — USER.md profile/bio markdown  
  workspace/      — user workspace (code, projects)
  social/weekly/  — weekly social draft bundles (images, posts)
  logs/           — per-user log files

Key functions:
  - current_user_id()     — get the active user ID from session or env
  - user_state_dir()      — resolve <USER_SPACE_ROOT>/<user_id> for a user
  - user_state_path()     — resolve a file path under user state
  - user_workspace_root() — resolve workspace root for a user
  - user_profile_path()   — resolve profile path (defaults to profile/USER.md)
  - set_current_user_id() / reset_current_user_id() — per-request user context
  - current_display_name() — resolve display name: contextvar -> raw user_id
  - set_current_display_name() / reset_current_display_name() — per-request
                              display name context
  - normalize_user_id()    — build a filesystem-safe, provider-scoped id
                              for OAuth identities, with built-in owner
                              aliases (e.g. github_205369547 → OppaAI) so
                              a single canonical directory serves the box's
                              owner regardless of which provider logged in
 
The multi-user design allows running multiple Aiko instances (e.g., for
different team members) on the same machine, each with their own isolated
state, memories, and configurations.
"""

from __future__ import annotations                        # evaluates type annotations later

import contextvars
import os
from pathlib import Path
import re
from system.log import get_logger

log = get_logger(__name__)

_DEFAULT_USER_ID = "guest"
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_DOTDOT_RE = re.compile(r"\.{2,}")
# Display names users may not claim — reserved for the assistant itself.
_RESERVED_DISPLAY: frozenset[str] = frozenset({
    (os.getenv("AI_NAME") or "Aiko").casefold(),
    "assistant",
    "system",
    "aiko",
})
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
    n = (name or "").strip()
    if n.casefold() in _RESERVED_DISPLAY:
        raise ValueError("Display name 'Aiko' is reserved for the assistant")
    return _CURRENT_DISPLAY_NAME.set(name)


def reset_current_display_name(token: contextvars.Token[str | None]) -> None:
    """Reset the display name context var using a token from set_current_display_name()."""
    _CURRENT_DISPLAY_NAME.reset(token)


def current_display_name() -> str:
    """Return the user's display name (e.g. GitHub login).
    Order: request-local context var -> process-global env override
    -> raw user_id as last resort.

    The contextvar is preferred because web sessions and worker threads
    can overlap, so authenticated callers should set it per request/thread
    with set_current_display_name(). The env var is a single process-global
    override for headless/local runs where no session identity exists.
    """
    name = _CURRENT_DISPLAY_NAME.get()
    if name:
        return name
    name = os.getenv("CURRENT_DISPLAY_NAME")
    if name:
        return name
    return current_user_id()


# Built-in owner aliases: raw provider-scoped ids (e.g. github_205369547)
# resolve to the human owner name (OppaAI) so a single directory under
# USER_SPACE_ROOT serves as the canonical home for the box's owner,
# regardless of which OAuth provider issued the login. Mirrors the alias
# map in cognition/memory/entity.py so memory attribution and filesystem
# paths agree on the same canonical name.
_BUILTIN_USER_ALIASES: dict[str, str] = {
    "github_205369547": "OppaAI",
    "local_oppa.ai.bot": "OppaAI",
    "local_oppa.ai": "OppaAI",
    "local_@oppa.ai.bot": "OppaAI",
}


def normalize_user_id(provider: str | None, user_id: object) -> str:
    """Create a filesystem-safe, provider-scoped id for OAuth identities.

    Resolves built-in owner aliases (e.g. github_205369547 -> OppaAI) so
    the returned id is the canonical directory name for the box's owner
    rather than the raw provider id. Non-aliased ids pass through
    unchanged.
    """
    provider_part = _SAFE_RE.sub("_", str(provider or "local")).strip("._-") or "local"
    provider_part = _DOTDOT_RE.sub("_", provider_part)
    user_part = _SAFE_RE.sub("_", str(user_id or _DEFAULT_USER_ID)).strip("._-") or _DEFAULT_USER_ID
    user_part = _DOTDOT_RE.sub("_", user_part)
    raw = f"{provider_part}_{user_part}"
    return _BUILTIN_USER_ALIASES.get(raw, raw)


def _resolve_aliased_uid(raw: str) -> str:
    """Sanitise a raw user_id and collapse built-in owner aliases.

    Used by user_state_dir() so an already-canonical id (e.g. "OppaAI")
    or a raw provider-scoped one (e.g. "github_205369547") both resolve
    to the same on-disk directory name.
    """
    sanitised = _SAFE_RE.sub("_", str(raw or _DEFAULT_USER_ID)).strip("._-") or _DEFAULT_USER_ID
    sanitised = _DOTDOT_RE.sub("_", sanitised)
    return _BUILTIN_USER_ALIASES.get(sanitised, sanitised)


def _user_state_root_value() -> str:
    """Return the configured root for per-user mutable state.

    USER_SPACE_ROOT is the canonical name. USER_STATE_ROOT and
    AIKO_USER_STATE_ROOT are accepted as compatibility aliases so
    deployments and docs that used those names still point Aiko at
    the same per-user files.
    """
    return (
        os.getenv("USER_SPACE_ROOT")
        or os.getenv("USER_STATE_ROOT")
        or os.getenv("AIKO_USER_STATE_ROOT")
        or str(Path.home() / ".aiko")
    )


def user_state_dir(user_id: str | None = None) -> Path:
    """Root directory for user-private mutable state.

    Resolves to <USER_SPACE_ROOT>/<user_id>. For a real authenticated
    user_id, creates it (locked to owner-only) if missing. For the guest
    sentinel (no one authenticated yet), returns the path WITHOUT creating
    it — callers doing existence checks (e.g. profile lookup) correctly
    see nothing there, and no stray folder is left on disk before login.
    """
    root = Path(_user_state_root_value()).expanduser()
    raw = user_id if user_id is not None else current_user_id()
    # normalise through the same alias-aware pipeline as OAuth logins so a
    # raw github_205369547 collapses to the canonical OppaAI directory.
    uid = _resolve_aliased_uid(raw)

    if uid == _DEFAULT_USER_ID:
        return root / uid  # no mkdir — nothing on disk for an unauthenticated guest

    path = root / uid
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        log.warning("userspace: chmod 0o700 failed for %s", path)
    return path


def user_state_path(filename: str, user_id: str | None = None) -> Path:
    return user_state_dir(user_id) / filename


def all_known_user_ids() -> list[str]:
    """Enumerate every real (non-guest) user_id with a state directory
    under the resolved USER_SPACE_ROOT.

    Uses the same root resolution as user_state_dir() (USER_SPACE_ROOT /
    USER_STATE_ROOT / AIKO_USER_STATE_ROOT aliases, falling back to
    ~/.aiko) so callers that need to iterate all users — e.g. the
    scheduler daemon checking every user's due jobs each tick — never
    drift out of sync with wherever user_state_dir() actually resolves to.
    """
    root = Path(_user_state_root_value()).expanduser()
    if not root.exists():
        return []
    try:
        return sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and p.name and p.name != _DEFAULT_USER_ID
        )
    except OSError:
        log.warning("userspace: failed to list %s", root)
        return []


def resolve_owner_user_id() -> str | None:
    """Resolve the machine owner's user_id without requiring a login.

    Order: explicit AIKO_USER_ID env override, else the unique non-guest
    user directory under USER_SPACE_ROOT that has both a memory store and
    a profile. Returns None when identity is ambiguous (multi-user layout)
    so callers can skip instead of reading/writing the wrong person's data.
    """
    override = os.getenv("AIKO_USER_ID", "").strip()
    if override:
        return override
    try:
        root = Path(_user_state_root_value()).expanduser()
        candidates = []
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name == _DEFAULT_USER_ID:
                continue
            if (child / "memory" / "memory.db").is_file() and (
                child / "profile" / "USER.md"
            ).is_file():
                candidates.append(child.name)
    except OSError:
        return None
    except Exception:
        return None
    return candidates[0] if len(candidates) == 1 else None


def user_workspace_root(user_id: str | None = None) -> Path:
    """Workspace root isolated by user unless WORKSPACE_ROOT explicitly overrides."""
    if os.getenv("WORKSPACE_ROOT"):
        return Path(os.environ["WORKSPACE_ROOT"]).expanduser().resolve()
    return (user_state_dir(user_id) / "workspace").resolve()


def user_profile_path(user_id: str | None = None) -> Path:
    """Per-user editable profile/bio markdown path.

    Defaults to <USER_SPACE_ROOT>/<user_id>/profile/USER.md. The profile stores
    user-provided biographical information, preferences, and identity
    details that Aiko can use to personalize responses.
    """
    if os.getenv("USER_PROFILE_PATH"):
        return Path(os.environ["USER_PROFILE_PATH"]).expanduser().resolve()
    return user_state_path("profile/USER.md", user_id).resolve()
