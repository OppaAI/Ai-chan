from __future__ import annotations

import os


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


def bool_env(name: str, default: bool = False) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}