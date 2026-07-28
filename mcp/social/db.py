from __future__ import annotations

import json
import os
import sqlite3
import time
import hashlib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

log: Any = None


def _get_log():
    global log
    if log is None:
        import logging
        log = logging.getLogger("social.db")
    return log


def _get_driver() -> str:
    encryption = os.getenv("SQLITE_ENCRYPTION", "0")
    if encryption.lower() in ("1", "true", "yes"):
        try:
            import pysqlcipher3.dbapi2 as sqlcipher
            _get_log().info("Using SQLCipher for encrypted DB")
            return "sqlcipher"
        except ImportError:
            _get_log().warning("SQLITE_ENCRYPTION=1 but pysqlcipher3 not installed — falling back to plain SQLite")
    return "sqlite"


_DB: "MCPDatabase | None" = None


def get_db() -> "MCPDatabase":
    global _DB
    if _DB is None:
        raise RuntimeError("MCP database not initialized — call init_db() first")
    return _DB


def init_db(db_path: str = "") -> "MCPDatabase":
    global _DB
    _DB = MCPDatabase(db_path=db_path)
    _DB.migrate()
    return _DB


def close_db() -> None:
    global _DB
    if _DB is not None:
        _DB.close()
        _DB = None


class MCPDatabase:
    def __init__(self, db_path: str = ""):
        path = db_path or os.getenv("SOCIAL_MCP_DB_PATH", "")
        if not path:
            base = Path(__file__).parent.resolve()
            path = str(base / "mcp.social.db")

        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._driver = _get_driver()
        self._connect()

    def _connect(self):
        if self._driver == "sqlcipher":
            import pysqlcipher3.dbapi2 as sqlcipher
            self._conn = sqlcipher.connect(str(self._path))
            key = os.getenv("DATA_KEY_SECRET", "")
            if key:
                safe_key = key.replace("'", "''")
            self._conn.execute(f"PRAGMA key = '{safe_key}'")
            self._conn.execute("PRAGMA cipher_use_hmac = OFF")
            self._conn.execute("PRAGMA cipher_page_size = 4096")
        else:
            self._conn = sqlite3.connect(str(self._path))

        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.row_factory = sqlite3.Row

    def migrate(self):
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS access_tokens (
                service TEXT PRIMARY KEY,
                access_token TEXT NOT NULL,
                expires_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS idempotency (
                request_hash TEXT PRIMARY KEY,
                tool TEXT NOT NULL,
                result TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rate_limits (
                service TEXT NOT NULL,
                period TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (service, period)
            );

            CREATE TABLE IF NOT EXISTS tool_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool TEXT NOT NULL,
                arguments TEXT,
                result TEXT,
                duration_ms REAL,
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_idempotency_expires ON idempotency(expires_at);
            CREATE INDEX IF NOT EXISTS idx_tool_log_created ON tool_log(created_at);
        """)
        self._conn.commit()

    # ── Access token cache ────────────────────────────────────────────────

    def get_cached_token(self, service: str) -> str | None:
        row = self._conn.execute(
            "SELECT access_token, expires_at FROM access_tokens WHERE service = ?",
            (service,),
        ).fetchone()
        if row and row["expires_at"] > time.time():
            return row["access_token"]
        return None

    def set_cached_token(self, service: str, access_token: str, expires_in: int):
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO access_tokens (service, access_token, expires_at, updated_at) VALUES (?, ?, ?, ?)",
            (service, access_token, now + expires_in, now),
        )
        self._conn.commit()

    # ── Idempotency ───────────────────────────────────────────────────────

    def _request_hash(self, tool: str, arguments: dict) -> str:
        raw = f"{tool}:{json.dumps(arguments, sort_keys=True, default=str)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get_idempotent_result(self, tool: str, arguments: dict) -> dict | None:
        h = self._request_hash(tool, arguments)
        row = self._conn.execute(
            "SELECT result FROM idempotency WHERE request_hash = ? AND expires_at > ?",
            (h, time.time()),
        ).fetchone()
        if row:
            return json.loads(row["result"])
        return None

    def set_idempotent_result(self, tool: str, arguments: dict, result: dict, ttl_hours: int = 24):
        h = self._request_hash(tool, arguments)
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO idempotency (request_hash, tool, result, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
            (h, tool, json.dumps(result), now, now + ttl_hours * 3600),
        )
        self._conn.commit()

    def _clean_expired_idempotency(self):
        self._conn.execute("DELETE FROM idempotency WHERE expires_at <= ?", (time.time(),))
        self._conn.commit()

    # ── Rate limiting ──────────────────────────────────────────────────────

    def _period_key(self, unit: str = "hour") -> str:
        now = time.gmtime()
        if unit == "hour":
            return time.strftime("%Y-%m-%d-%H", now)
        return time.strftime("%Y-%m-%d", now)

    def check_rate_limit(self, service: str, max_per_hour: int = 10, max_per_day: int = 50) -> tuple[bool, str]:
        hour_key = self._period_key("hour")
        day_key = self._period_key("day")

        hour_row = self._conn.execute(
            "SELECT count FROM rate_limits WHERE service = ? AND period = ?",
            (service, hour_key),
        ).fetchone()
        if hour_row and hour_row["count"] >= max_per_hour:
            return False, f"Rate limit exceeded for {service}: max {max_per_hour}/hour"

        day_row = self._conn.execute(
            "SELECT count FROM rate_limits WHERE service = ? AND period = ?",
            (service, day_key),
        ).fetchone()
        if day_row and day_row["count"] >= max_per_day:
            return False, f"Rate limit exceeded for {service}: max {max_per_day}/day"

        return True, ""

    def increment_rate_limit(self, service: str):
        hour_key = self._period_key("hour")
        day_key = self._period_key("day")

        for period in (hour_key, day_key):
            self._conn.execute(
                "INSERT INTO rate_limits (service, period, count) VALUES (?, ?, 1) "
                "ON CONFLICT(service, period) DO UPDATE SET count = count + 1",
                (service, period),
            )
        self._conn.commit()

    # ── Tool log ───────────────────────────────────────────────────────────

    def log_tool_call(self, tool: str, arguments: dict, result: dict, duration_ms: float):
        self._conn.execute(
            "INSERT INTO tool_log (tool, arguments, result, duration_ms, created_at) VALUES (?, ?, ?, ?, ?)",
            (tool, json.dumps(arguments), json.dumps(result), duration_ms, time.time()),
        )
        self._conn.commit()

    # ── Periodic cleanup ──────────────────────────────────────────────────

    def cleanup(self):
        self._clean_expired_idempotency()
        cutoff = time.time() - 86400 * 30  # 30 days
        self._conn.execute("DELETE FROM tool_log WHERE created_at < ?", (cutoff,))
        self._conn.execute("DELETE FROM access_tokens WHERE expires_at < ?", (time.time(),))
        self._conn.commit()

    def close(self):
        if self._conn:
            self.cleanup()
            self._conn.close()
            self._conn = None
