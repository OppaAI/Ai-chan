"""Shared config loading for workflow packages."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_workflow_config(workflow_dir: Path, filename: str = "config.json") -> dict[str, Any]:
    """Load workflow config.json from the workflow package directory."""
    path = workflow_dir / filename
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


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
