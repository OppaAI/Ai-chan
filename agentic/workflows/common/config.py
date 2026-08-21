"""Shared config loading for workflow packages."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def load_workflow_config(workflow_dir: Path, filename: str = "config.json") -> dict[str, Any]:
    """Load workflow config.json.

    Preference order:
      1. ``<user_state>/agentic/workflows/<name>/config.json`` when present
      2. Package directory ``workflow_dir/config.json``
    """
    candidates: list[Path] = []
    try:
        from system.userspace import user_state_dir
        name = workflow_dir.name
        user_path = user_state_dir() / "agentic" / "workflows" / name / filename
        candidates.append(user_path)
    except Exception as e:
        # Failed to determine user config path; fall back to package config
        log.debug("Could not construct user config path: %s", e)
    candidates.append(workflow_dir / filename)

    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            # Invalid UTF-8 or malformed JSON; try next candidate
            log.debug("Could not read config from %s: %s", path, e)
            continue
        if isinstance(data, dict):
            return data
    return {}


def resolve_config_value(
    config: dict[str, Any],
    key: str,
    env_key: str = "",
    default: Any = None,
    *,
    as_type: str = "auto",
) -> Any:
    """Resolve env override → config[key] → default.

    as_type: "auto" | "str" | "int" | "bool" | "list"
    """
    raw: Any = None
    if env_key:
        env = os.getenv(env_key, "").strip()
        if env:
            raw = env
    if raw is None:
        raw = config.get(key, default)

    if as_type == "auto":
        return raw if raw is not None else default

    if as_type == "list":
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
        if isinstance(raw, str) and raw.strip():
            return [p.strip() for p in raw.split(",") if p.strip()]
        return list(default or [])

    if as_type == "int":
        try:
            return max(1, int(raw)) if raw not in (None, "") else int(default or 1)
        except (TypeError, ValueError):
            return int(default or 1)

    if as_type == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.lower() in {"1", "true", "yes", "on"}
        return bool(default)

    # str
    if raw is None:
        return default
    return str(raw)
