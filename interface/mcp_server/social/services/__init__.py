from __future__ import annotations

import os
import threading
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

import requests


_session: requests.Session | None = None
_session_lock = threading.Lock()


def get_session() -> requests.Session:
    """Get shared requests Session with connection pooling and retries."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = requests.Session()
                adapter = HTTPAdapter(
                    pool_connections=10,
                    pool_maxsize=20,
                    max_retries=Retry(
                        total=3,
                        backoff_factor=0.5,
                        status_forcelist=[429, 500, 502, 503, 504],
                        allowed_methods=["HEAD", "GET", "PUT", "POST", "OPTIONS", "DELETE"],
                    ),
                )
                _session.mount("http://", adapter)
                _session.mount("https://", adapter)
    return _session


def close_session() -> None:
    global _session
    if _session is not None:
        _session.close()
        _session = None


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


def bool_env(name: str, default: bool = False) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def err(provider: str, message: str) -> dict:
    return {"ok": False, "provider": provider, "error": message}