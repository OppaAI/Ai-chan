"""Configuration bootstrap for Aiko.
Loads non-secret defaults from category YAML files and local values from an
age-encrypted .env.age. Real process environment variables win over both, and
YAML wins over stale .env constants while .env still fills in secrets or
deployment-specific gaps.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - optional dependency fallback
    def dotenv_values(*_args, **_kwargs):
        return {}

_LOADED = False


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


def _decrypt_env(enc_path: Path, identity_path: Path) -> dict[str, str]:
    """Decrypt an age-encrypted dotenv file straight into memory.

    Plaintext is never written to disk: age's stdout is piped directly into
    dotenv_values via an in-memory buffer.
    """
    if not enc_path.exists():
        return {}
    if not identity_path.exists():
        raise FileNotFoundError(
            f"age identity file not found: {identity_path}. "
            "Set AGE_KEY to point at it, or place .env.age's key there."
        )
    try:
        result = subprocess.run(
            ["age", "-d", "-i", str(identity_path), str(enc_path)],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "the 'age' binary was not found on PATH; install it "
            "(e.g. `sudo apt install age`)"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"failed to decrypt {enc_path}: {exc.stderr.decode(errors='replace')}"
        ) from exc
    return dict(dotenv_values(stream=io.StringIO(result.stdout.decode())))


def load_config(*, override: bool = False) -> None:
    """Load indexed config/*.yaml settings and .env.age secrets into os.environ.

    Precedence is:
    1. Real process environment variables, unless ``override=True``.
    2. Non-secret YAML constants from config/*.yaml.
    3. Values from .env.age that YAML did not already define.

    This keeps stale constants in .env.age from shadowing the YAML files
    while preserving .env.age as the local place for tokens, keys, URLs,
    DSNs, and other deployment-specific values whose names may not follow a
    strict pattern.
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
                # Empty YAML values mean "unset": allow code defaults or
                # .env.age/deployment secrets to provide the value instead of
                # exporting an empty string.
                if value is None or (isinstance(value, str) and value == ""):
                    continue
                if override or key not in original_env:
                    os.environ[key] = _stringify(value)

    # --- Secrets: encrypted .env.age (preferred) with plaintext .env fallback ---
    identity_path = Path(
        os.environ.get("AGE_KEY", "/etc/aiko/age-key.txt")
    ).expanduser()
    enc_path = Path(os.environ.get("ENV_AGE_PATH", root / ".env.age"))

    if enc_path.exists():
        for key, value in _decrypt_env(enc_path, identity_path).items():
            if not key or value is None:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
    else:
        # Dev-machine fallback only — should not exist on the Jetson deployment.
        env_path = root / ".env"
        if env_path.exists():
            for key, value in dotenv_values(env_path).items():
                if not key or value is None:
                    continue
                if override or key not in os.environ:
                    os.environ[key] = value

    _LOADED = True
