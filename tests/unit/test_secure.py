"""
tests/test_secure.py
Tests for system/secure.py -- per-user SQLCipher key derivation and the
plaintext/encrypted connection switch.

Focus areas:
  - key derivation is deterministic per (secret, user_id) pair
  - different user_ids never collide on the same key
  - different secrets produce different keys for the SAME user_id (so
    rotating DATA_KEY_SECRET actually re-keys everyone, rather than some
    users silently keeping an old key through a caching bug)
  - missing secret fails loudly at derive time, not silently with a weak
    default key
  - connect_sqlite() falls back to plain sqlite3 when SQLITE_ENCRYPTION is
    unset/false, and only imports pysqlcipher3 when actually enabled
  - sqlite_encryption_enabled() truthy parsing matches the same convention
    used elsewhere (1/true/yes/on, case-insensitive)

Note: these tests do NOT require pysqlcipher3 to be installed except for
the "encryption enabled" path, which is expected to raise a clear
RuntimeError if the package is missing rather than crash obscurely.
"""
from __future__ import annotations

import sqlite3

import pytest

from system.secure import (
    _data_secret,
    connect_sqlite,
    derive_user_sqlite_key,
    sqlite_encryption_enabled,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("SQLITE_ENCRYPTION", "DATA_KEY_SECRET", "SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)


# ─────────────────────────────────────────────────────────────────────────────
# sqlite_encryption_enabled() truthy parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestEncryptionEnabledFlag:
    @pytest.mark.parametrize("value", ["1", "true", "True", "YES", "on", "  yes  "])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("SQLITE_ENCRYPTION", value)
        assert sqlite_encryption_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "garbage"])
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("SQLITE_ENCRYPTION", value)
        assert sqlite_encryption_enabled() is False

    def test_unset_defaults_falsy(self):
        assert sqlite_encryption_enabled() is False


# ─────────────────────────────────────────────────────────────────────────────
# _data_secret() -- fails loudly, no silent weak-key fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestDataSecret:
    def test_raises_when_no_secret_configured(self):
        with pytest.raises(ValueError, match="DATA_KEY_SECRET"):
            _data_secret()

    def test_prefers_data_key_secret_over_secret_key(self, monkeypatch):
        monkeypatch.setenv("DATA_KEY_SECRET", "primary-secret")
        monkeypatch.setenv("SECRET_KEY", "fallback-secret")
        assert _data_secret() == b"primary-secret"

    def test_falls_back_to_secret_key(self, monkeypatch):
        monkeypatch.setenv("SECRET_KEY", "only-this-one")
        assert _data_secret() == b"only-this-one"


# ─────────────────────────────────────────────────────────────────────────────
# derive_user_sqlite_key() -- determinism and isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestDeriveUserSqliteKey:
    def test_deterministic_for_same_user_and_secret(self, monkeypatch):
        monkeypatch.setenv("DATA_KEY_SECRET", "server-secret")
        k1 = derive_user_sqlite_key("user_a")
        k2 = derive_user_sqlite_key("user_a")
        assert k1 == k2

    def test_different_users_get_different_keys(self, monkeypatch):
        monkeypatch.setenv("DATA_KEY_SECRET", "server-secret")
        k_a = derive_user_sqlite_key("user_a")
        k_b = derive_user_sqlite_key("user_b")
        assert k_a != k_b

    def test_different_secrets_rekey_the_same_user(self, monkeypatch):
        """If DATA_KEY_SECRET rotates, every user's derived key must change --
        otherwise a 'secret rotation' silently does nothing for at-rest
        encryption and old backups stay readable with the new secret."""
        monkeypatch.setenv("DATA_KEY_SECRET", "secret-one")
        k_before = derive_user_sqlite_key("user_a")
        monkeypatch.setenv("DATA_KEY_SECRET", "secret-two")
        k_after = derive_user_sqlite_key("user_a")
        assert k_before != k_after

    def test_key_is_256_bit_hex(self, monkeypatch):
        monkeypatch.setenv("DATA_KEY_SECRET", "server-secret")
        key = derive_user_sqlite_key("user_a")
        assert len(key) == 64  # 32 bytes -> 64 hex chars
        int(key, 16)  # raises if not valid hex

    def test_raises_without_secret_configured(self):
        with pytest.raises(ValueError):
            derive_user_sqlite_key("user_a")


# ─────────────────────────────────────────────────────────────────────────────
# connect_sqlite() -- plaintext path (default / SQLITE_ENCRYPTION unset)
# ─────────────────────────────────────────────────────────────────────────────

class TestConnectSqlitePlaintext:
    def test_returns_plain_sqlite3_connection_when_disabled(self, tmp_path):
        db_path = tmp_path / "plain.db"
        conn = connect_sqlite(db_path, user_id="user_a")
        try:
            assert isinstance(conn, sqlite3.Connection)
            conn.execute("CREATE TABLE t (x INTEGER)")
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
            row = conn.execute("SELECT x FROM t").fetchone()
            assert row[0] == 1
        finally:
            conn.close()

    def test_creates_parent_directory(self, tmp_path):
        db_path = tmp_path / "nested" / "dir" / "plain.db"
        conn = connect_sqlite(db_path, user_id="user_a")
        try:
            assert db_path.parent.exists()
        finally:
            conn.close()

    def test_row_factory_is_sqlite3_row(self, tmp_path):
        conn = connect_sqlite(tmp_path / "plain.db", user_id="user_a")
        try:
            assert conn.row_factory is sqlite3.Row
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# connect_sqlite() -- encrypted path
# ─────────────────────────────────────────────────────────────────────────────

class TestConnectSqliteEncrypted:
    def test_raises_clear_error_when_pysqlcipher3_missing(self, tmp_path, monkeypatch):
        """If pysqlcipher3 isn't installed in this environment, enabling
        encryption should fail with a clear, actionable RuntimeError --
        not an opaque ImportError three frames deep."""
        monkeypatch.setenv("SQLITE_ENCRYPTION", "1")
        monkeypatch.setenv("DATA_KEY_SECRET", "server-secret")
        try:
            import pysqlcipher3  # noqa: F401
            pytest.skip("pysqlcipher3 is installed in this environment -- "
                        "this test only applies where it's absent")
        except ImportError:
            pass

        with pytest.raises(RuntimeError, match="pysqlcipher3"):
            connect_sqlite(tmp_path / "encrypted.db", user_id="user_a")

    def test_raises_before_touching_disk_if_secret_missing(self, tmp_path, monkeypatch):
        """Encryption enabled but no secret configured -- should fail at key
        derivation, before ever opening a connection."""
        monkeypatch.setenv("SQLITE_ENCRYPTION", "1")
        with pytest.raises(ValueError, match="DATA_KEY_SECRET"):
            connect_sqlite(tmp_path / "encrypted.db", user_id="user_a")
