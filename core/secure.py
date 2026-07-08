"""Optional at-rest encryption helpers for user-private Aiko state."""

from __future__ import annotations

import hashlib
import hmac
import os
import base64
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


def _derive_legacy_user_sqlite_key(user_id: str) -> str:
    """Return the legacy SQLCipher passphrase used before raw hex keys."""
    digest = hmac.new(_data_secret(), f"aiko-sqlite:{user_id}".encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii")


def _quote_sqlcipher_key(key: str) -> str:
    return "'" + key.replace("'", "''") + "'"


def _validate_sqlcipher_connection(conn: Any) -> None:
    # Force key validation immediately, so a wrong key fails at boot instead of
    # later after partial initialization.
    conn.execute("SELECT count(*) FROM sqlite_master")


def connect_sqlite(path: str | os.PathLike[str], *, user_id: str) -> Any:
    """Connect to SQLite, using SQLCipher when SQLITE_ENCRYPTION=1.

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
            "SQLITE_ENCRYPTION=1 requires pysqlcipher3/SQLCipher in the runtime image. "
            "Install a SQLCipher-capable Python driver or disable SQLITE_ENCRYPTION."
        ) from exc

    raw_key = derive_user_sqlite_key(user_id)
    conn = sqlcipher.connect(str(path), check_same_thread=False)
    try:
        conn.execute(f"PRAGMA key = \"x'{raw_key}'\"")
        conn.execute("PRAGMA cipher_page_size = 4096")
        _validate_sqlcipher_connection(conn)
        return conn
    except Exception as raw_exc:
        conn.close()

        # Databases created by the first SQLCipher integration used the derived
        # value as a SQLCipher passphrase, letting SQLCipher run its KDF. Keep
        # those databases readable and migrate them in place to the raw key.
        legacy_key = _derive_legacy_user_sqlite_key(user_id)
        legacy_conn = sqlcipher.connect(str(path), check_same_thread=False)
        try:
            legacy_conn.execute(f"PRAGMA key = {_quote_sqlcipher_key(legacy_key)}")
            legacy_conn.execute("PRAGMA cipher_page_size = 4096")
            _validate_sqlcipher_connection(legacy_conn)
            legacy_conn.execute(f"PRAGMA rekey = \"x'{raw_key}'\"")
            _validate_sqlcipher_connection(legacy_conn)
            return legacy_conn
        except Exception:
            legacy_conn.close()
            raise raw_exc
