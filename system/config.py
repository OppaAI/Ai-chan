"""
system/config.py

Configuration bootstrap for Aiko.

Usage:
    from system.config import load_config, load_yaml
    load_config()  # once at startup
    data = load_yaml("memory.yaml")

Loads non-secret defaults from category YAML files and local values from an
age-encrypted .env.age. Real process environment variables win over both, and
YAML wins over stale .env constants while .env still fills in secrets or
deployment-specific gaps.

Precedence is:
    1. Real process environment variables, unless override=True.
    2. Non-secret YAML constants from config/*.yaml.
    3. Values from .env.age that YAML did not already define.

This module only resolves YAML/.env.age at load_config() call (not at import
time), so main.py can call it once near startup before any subsystem init
— see main.py's module docstring for why import-time resolution would be
too late.

Flow:

                                      load_config()
                                           │
              ┌────────────────┼────────────────┼─────────────────┐
              ▼                ▼                ▼                 ▼
         config/*.yaml    .env.age (age)    .env (fallback)   os.environ
              │                │                │                 │
              ▼                ▼                ▼                 ▼
         _flatten+_stringify → os.environ (if not in original_env / override)
                                           │
                                           ▼
                                      _LOADED = True  (guarded by _load_lock)
"""
from __future__ import annotations            # evaluates type annotations later

# Public libraries
import io                                     # for in-memory age decrypt buffer
import json                                   # for list stringify + JSON YAML fallback
import os                                     # for os.environ precedence
import subprocess                             # for age decrypt subprocess
import threading                              # for thread-safe one-time load
from pathlib import Path                      # for config/state path resolution
from typing import Any                        # for YAML value types

try:
    import yaml                               # for YAML parsing (preferred)
except ImportError:  # pragma: no cover - lightweight fallback for minimal tooling envs
    yaml = None


try:
    from dotenv import dotenv_values          # for .env parsing
except ImportError:  # pragma: no cover - optional dependency fallback
    def dotenv_values(*_args: Any, **_kwargs: Any) -> dict[str, str | None]:  # type: ignore[no-redef]
        return {}

_LOADED = False
_load_lock = threading.Lock()

__all__ = ["load_config", "load_yaml"]        # external API — internal defs keep leading _

# Timeout for the `age` decrypt subprocess. If the binary hangs (bad
# identity file, unexpected passphrase prompt, etc.) this turns a silent
# boot-time hang into a fast, diagnosable failure.
_AGE_TIMEOUT_SECONDS = 10


def _strip_comment(line: str) -> str:
    """Strip a trailing `# comment`, but not a `#` inside quotes."""
    in_squote = in_dquote = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_dquote:
            in_squote = not in_squote
        elif ch == '"' and not in_squote:
            in_dquote = not in_dquote
        elif ch == "#" and not in_squote and not in_dquote:
            return line[:i]
    return line


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Tiny fallback parser for Aiko's simple config/*.yaml files."""
    data: dict[str, Any] = {}
    current_key: str | None = None
    pending_empty: set[str] = set()
    for raw in text.splitlines():
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("-") and current_key:
            existing = data.get(current_key)
            if current_key in pending_empty:
                data[current_key] = []
                pending_empty.discard(current_key)
            elif not isinstance(existing, list):
                # Malformed/mixed YAML (scalar followed by a "- item" line
                # under the same key) — don't crash the whole config load
                # over a hand-edited typo; start a fresh list instead.
                data[current_key] = []
            data[current_key].append(stripped[1:].strip().strip('"\''))
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        current_key = key
        if value == "":
            data[key] = ""
            pending_empty.add(key)
        else:
            data[key] = value.strip('"\'')
            pending_empty.discard(key)
    return data


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        lines = text.splitlines()
        start = 0
        for index, line in enumerate(lines):
            stripped_line = line.lstrip()
            if stripped_line and not stripped_line.startswith("#"):
                start = index
                break
        body = "\n".join(lines[start:])
        stripped = body.lstrip()
        data = json.loads(body) if stripped.startswith(("{", "[")) else _simple_yaml_load(text)  # merge startswith tuple (py312)
    return data or {}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file from the given path."""
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / "config" / path
    return _load_yaml_mapping(path)


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
        msg = f"age identity file not found: {identity_path}. Set AGE_KEY to point at it, or place .env.age's key there."  # for FileNotFoundError
        raise FileNotFoundError(msg)
    try:
        result = subprocess.run(
            ["age", "-d", "-i", str(identity_path), str(enc_path)],
            capture_output=True,
            check=True,
            timeout=_AGE_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "the 'age' binary was not found on PATH; install it "
            "(e.g. `sudo apt install age`)"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"decrypting {enc_path} timed out after {_AGE_TIMEOUT_SECONDS}s "
            "(age may be waiting on an unexpected prompt, or the identity "
            "file/binary is misbehaving)"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"failed to decrypt {enc_path}: {exc.stderr.decode(errors='replace')}"
        ) from exc
    vals = dotenv_values(stream=io.StringIO(result.stdout.decode(errors="replace")))
    return {k: v for k, v in vals.items() if v is not None}


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

    Thread-safe: guarded by _load_lock so concurrent callers can't both
    pass the _LOADED check and run the full load twice. Currently called
    once, synchronously, near the top of main() before any other thread
    exists — this guard is here so that stays true if that ever changes.
    """
    global _LOADED
    if _LOADED and not override:
        return

    with _load_lock:
        if _LOADED and not override:  # re-check inside the lock
            return

        root = Path(__file__).resolve().parent.parent
        original_env = set(os.environ)

        config_dir = root / "config"
        if config_dir.exists():
            index_path = config_dir / "index.yaml"
            if index_path.exists():
                index_data = _load_yaml_mapping(index_path)
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
                data = _load_yaml_mapping(path)
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
        # AGE_KEY / ENV_AGE_PATH may be bare filenames (e.g. from identity.yaml);
        # resolve them under USER_SPACE_ROOT unless they're already absolute,
        # so an explicit absolute override (env var or YAML) still wins outright.
        state_root = Path(os.environ.get("USER_SPACE_ROOT", str(Path.home() / ".aiko"))).expanduser()

        def _resolve_under(base: Path, value: str) -> Path:
            candidate = Path(value).expanduser()
            return candidate if candidate.is_absolute() else base / candidate

        identity_path = _resolve_under(state_root, os.environ.get("AGE_KEY", "age-key.txt"))
        enc_path = _resolve_under(state_root, os.environ.get("ENV_AGE_PATH", ".env.age"))

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
