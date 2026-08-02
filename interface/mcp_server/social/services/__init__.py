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


def refresh_oauth_token(
    service: str,
    token_url: str,
    client_id: str,
    client_secret: str,
    refresh_token: str,
    *,
    grant_type: str = "refresh_token",
    extra_data: dict | None = None,
    timeout: int = 30,
) -> str | dict:
    """
    Generic OAuth2 token refresh with retry (uses shared session).

    Returns access_token (str) on success, or err dict on failure.
    """
    if not (client_id and client_secret and refresh_token):
        return err(service, "missing client_id, client_secret, or refresh_token")

    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": grant_type,
    }
    if extra_data:
        data.update(extra_data)

    session = get_session()
    try:
        resp = session.post(token_url, data=data, timeout=timeout)
        payload = resp.json()
        if not (200 <= resp.status_code < 300):
            return err(service, f"token refresh failed: {resp.status_code}, response: {payload}")
        access_token = payload.get("access_token")
        if not access_token:
            return err(service, f"no access_token in refresh response: {payload}")
        return access_token
    except Exception as e:
        return err(service, f"token refresh error: {e}")