"""cognition/memory/env.py - shared environment-variable helpers.

Before this module, each of grasp.py / schema.py / entity.py (and later
session_anchor.py / cross_store.py) re-implemented the same
small parse helpers with subtly diverged semantics. Keep ONE canonical set
here and import it everywhere.

Semantics (chosen to match the majority / most conservative):
  * ``env_flag``  - any value NOT in {0, false, no, off, <empty>} is True.
                    This is the "grasp" convention used by the original
                    GRASP_* / session flags and by entity's arousal flag.
  * ``env_bool``  - whitelist: only 1/true/yes/on count as True. This is the
                    stricter schema/_env_bool + cross_store convention.
"""
from __future__ import annotations

import os


def env_flag(name: str, default: str = "1") -> bool:
    """Conservative flag: off only when the var is 0/false/no/off/empty."""
    return str(os.getenv(name, default)).strip().lower() not in (
        "0", "false", "no", "off", "",
    )


def env_bool(name: str, default: str = "1") -> bool:
    """Whitelist flag: True only for 1/true/yes/on."""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or str(v).strip() == "" else str(v).strip()