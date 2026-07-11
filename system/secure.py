"""
core/secure.py

Optional at-rest encryption helpers for user-private Aiko state.

SQLite encryption using SQLCipher is optional and off by default. When
enabled via SQLITE_ENCRYPTION=1, user databases are encrypted at rest with
per-user keys derived from a server-secret. The key derivation uses HMAC-SHA256
with a per-user context salt, so the same user always gets the same key without
storing anything plaintext on disk.

Called by core/memorize.py (via connect_sqlite) and any other module that
needs an encrypted SQLite connection for a specific user. The encryption
setting is global (SQLITE_ENCRYPTION), but each user gets a unique key.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from pathlib import Path
from typing import Any


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def sqlite_encryption_enabled() -> bool:
    """Return True when SQLCipher-backed SQLite encryption is requested."""
    return _truthy(os.getenv("SQLITE_ENCRYPTION"))


def _data_secret() -> bytes:
    """Return the server-side secret used to derive per-user data keys."""
    secret = os.getenv("DATA_KEY_SECRET") or os.getenv("SECRET_KEY")
    if not secret:
        raise ValueError(
            "SQLITE_ENCRYPTION is enabled but neither DATA_KEY_SECRET "
            "nor SECRET_KEY is set. Set a high-entropy server secret first."
        )
    return secret.encode("utf-8")


def derive_user_sqlite_key(user_id: str) -> str:
    """Derive a stable per-user 256-bit SQLCipher raw key.

    OAuth user ids are public identifiers, so they are context/salt only. The
    secrecy comes from DATA_KEY_SECRET (preferred) or SECRET_KEY.
    """
    digest = hmac.new(_data_secret(), f"aiko-sqlite:{user_id}".encode("utf-8"), hashlib.sha256).digest()
    return digest.hex()


def _validate_sqlcipher_connection(conn: Any) -> None:
    # Force key validation immediately, so a wrong key fails at boot instead of
    # later after partial initialization.
    conn.execute("SELECT count(*) FROM sqlite_master")


def connect_sqlite(path: str | os.PathLike[str], *, user_id: str) -> Any:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not sqlite_encryption_enabled():
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    try:
        from pysqlcipher3 import dbapi2 as sqlcipher  # type: ignore
    except ImportError as exc:
        raise RuntimeError(...) from exc

    raw_key = derive_user_sqlite_key(user_id)
    conn = sqlcipher.connect(str(path), check_same_thread=False)
    conn.execute(f"PRAGMA key = \"x'{raw_key}'\"")
    conn.execute("PRAGMA cipher_page_size = 4096")
    _validate_sqlcipher_connection(conn)
    conn.row_factory = sqlcipher.Row
    return conn