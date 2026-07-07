"""Optional at-rest encryption helpers for user-private Aiko state."""

from __future__ import annotations

import base64
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
    return _truthy(os.getenv("AIKO_SQLITE_ENCRYPTION"))


def _data_secret() -> bytes:
    """Return the server-side secret used to derive per-user data keys."""
    secret = os.getenv("AIKO_DATA_KEY_SECRET") or os.getenv("SECRET_KEY")
    if not secret:
        raise RuntimeError(
            "AIKO_SQLITE_ENCRYPTION is enabled but neither AIKO_DATA_KEY_SECRET "
            "nor SECRET_KEY is set. Set a high-entropy server secret first."
        )
    return secret.encode("utf-8")


def derive_user_sqlite_key(user_id: str) -> str:
    """Derive a stable per-user SQLCipher passphrase from server secret + user id.

    OAuth user ids are public identifiers, so they are context/salt only. The
    secrecy comes from AIKO_DATA_KEY_SECRET (preferred) or SECRET_KEY.
    HMAC-SHA256 is intentionally fast because the input secret is server-side
    high entropy; Argon2id is mainly for low-entropy human passwords.
    """
    digest = hmac.new(_data_secret(), f"aiko-sqlite:{user_id}".encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii")


def _quote_sqlcipher_key(key: str) -> str:
    return "'" + key.replace("'", "''") + "'"


def connect_sqlite(path: str | os.PathLike[str], *, user_id: str) -> Any:
    """Connect to SQLite, using SQLCipher when AIKO_SQLITE_ENCRYPTION=1.

    The default path uses the stdlib sqlite3 module exactly as before. When
    encryption is enabled, pysqlcipher3 must be installed in the runtime image.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not sqlite_encryption_enabled():
        return sqlite3.connect(path, check_same_thread=False)

    try:
        from pysqlcipher3 import dbapi2 as sqlcipher  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "AIKO_SQLITE_ENCRYPTION=1 requires pysqlcipher3/SQLCipher in the runtime image. "
            "Install a SQLCipher-capable Python driver or disable AIKO_SQLITE_ENCRYPTION."
        ) from exc

    conn = sqlcipher.connect(str(path), check_same_thread=False)
    key = derive_user_sqlite_key(user_id)
    conn.execute(f"PRAGMA key = {_quote_sqlcipher_key(key)}")
    conn.execute("PRAGMA cipher_page_size = 4096")
    conn.execute("PRAGMA kdf_iter = 256000")
    # Force key validation immediately, so a wrong key fails at boot instead of
    # later after partial initialization.
    conn.execute("SELECT count(*) FROM sqlite_master")
    return conn
