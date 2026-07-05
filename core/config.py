"""Configuration bootstrap for Aiko.

Loads non-secret defaults from category YAML files and secrets from .env.
Real process environment variables win over both, YAML wins over stale .env
constants, and only secret/API credential values are copied from .env.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - optional dependency fallback
    def dotenv_values(*_args, **_kwargs):
        return {}

_LOADED = False

_SECRET_KEY_PARTS = (
    "API_KEY",
    "ACCESS_TOKEN",
    "AUTH_TOKEN",
    "BEARER_TOKEN",
    "CLIENT_SECRET",
    "SECRET_KEY",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "PRIVATE_KEY",
)


def _is_secret_key(key: str) -> bool:
    """Return True for .env names that should remain secret-backed."""
    upper = key.upper()
    return upper in {"HF_TOKEN", "GITHUB_TOKEN"} or any(part in upper for part in _SECRET_KEY_PARTS)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return json.dumps([str(item) for item in value], ensure_ascii=False)
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
    """Load indexed config/*.yaml settings and .env secrets into os.environ.

    Precedence is:
    1. Real process environment variables, unless ``override=True``.
    2. Non-secret YAML constants from config/*.yaml.
    3. Secret/API credential values from .env.

    This keeps stale constants in .env from shadowing the YAML files while
    preserving .env as the local place for tokens and keys.
    """
    global _LOADED
    if _LOADED and not override:
        return

    root = Path(__file__).resolve().parent.parent
    original_env = set(os.environ)

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
                if override or key not in original_env:
                    os.environ[key] = _stringify(value)

    env_path = root / ".env"
    if env_path.exists():
        for key, value in dotenv_values(env_path).items():
            if not key or value is None or not _is_secret_key(key):
                continue
            if override or key not in os.environ:
                os.environ[key] = value

    _LOADED = True
