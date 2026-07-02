"""Configuration bootstrap for Aiko.

Loads secrets from .env and non-secret defaults from category YAML files.
Existing process environment variables win over both, and .env wins over YAML.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_LOADED = False


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        env_key = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, env_key.upper()))
        else:
            flat[env_key.upper()] = value
    return flat


def load_config(*, override: bool = False) -> None:
    """Load .env secrets and indexed config/*.yaml settings into os.environ."""
    global _LOADED
    if _LOADED and not override:
        return

    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env", override=False)

    config_dir = root / "config"
    if config_dir.exists():
        index_path = config_dir / "index.yaml"
        if index_path.exists():
            index_data = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
            config_names = index_data.get("configs", [])
            if not isinstance(config_names, list):
                raise ValueError(f"{index_path} configs must be a list")
            paths = [config_dir / str(name) for name in config_names]
        else:
            paths = sorted(
                path for path in config_dir.glob("*.y*ml")
                if path.name != "index.yaml"
            )

        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Configured YAML file not found: {path}")
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                raise ValueError(f"{path} must contain a YAML mapping")
            for key, value in _flatten(data).items():
                if override or key not in os.environ:
                    os.environ[key] = _stringify(value)

    _LOADED = True
