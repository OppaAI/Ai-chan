"""
toolkit/common.py

Shared helpers for Aiko tool modules.

This module provides utilities used across multiple toolkit modules:

  - workspace_root()  — user-specific workspace directory
  - notes_dir()       — user notes subdirectory
  - now_stamp()       — UTC timestamp for generated files
  - slugify()         — stable file slug generation from text
  - safe_path()       — path resolution with traversal prevention
  - json_block()      — formatted JSON output for tool results

All functions respect the per-user isolation provided by core/userspace.py.
"""

from __future__ import annotations

import json
import os
import re
from system.bioclock import local_now, timezone_name
from pathlib import Path
from typing import Any

from system.userspace import user_workspace_root

MAX_WRITE_CHARS = int(os.getenv("MAX_WRITE_CHARS", 20_000))
MAX_READ_CHARS = int(os.getenv("MAX_READ_CHARS", 12_000))


def workspace_root() -> Path:
    """Resolve the active user workspace root lazily."""
    return user_workspace_root()


def notes_dir() -> Path:
    """Resolve the active user notes directory lazily."""
    return workspace_root() / "notes"


def now_stamp() -> str:
    """Return a compact local-timezone timestamp for generated notes and plans."""
    now = local_now()
    return now.strftime(f"%Y-%m-%d %H:%M {timezone_name()}")


def slugify(text: str, fallback: str = "task") -> str:
    """Create a stable lowercase file slug from arbitrary user text."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return (slug or fallback)[:80]


def safe_path(relative_path: str) -> Path:
    """Resolve a user path under the active WORKSPACE_ROOT, rejecting traversal."""
    root = workspace_root()
    cleaned = relative_path.strip().lstrip("/\\")
    path = (root / cleaned).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path escapes workspace: {relative_path}")
    return path


def json_block(title: str, payload: dict[str, Any]) -> str:
    """Render machine-readable tool output with a short human title."""
    return f"[{title}]\n" + json.dumps(payload, ensure_ascii=False, indent=2)
