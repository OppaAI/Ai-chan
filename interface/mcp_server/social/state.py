from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from system.log import get_logger

load_dotenv()

log = get_logger("social.db")


_DB: "MCPDatabase | None" = None
_db_lock = threading.Lock()


def get_db() -> "MCPDatabase":
    global _DB
    if _DB is None:
        with _db_lock:
            if _DB is None:
                init_db()
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
        path = db_path or os.path.expanduser(os.getenv("SOCIAL_MCP_DB_PATH", ""))
        if not path:
            base = Path(__file__).parent.resolve()
            path = str(base / "mcp.social.db")

        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._connections_lock = threading.RLock()
        self._in_transaction = False
        self._connect()

    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            self._connect()
            conn = self._local.conn
        return conn

    @_conn.setter
    def _conn(self, value: sqlite3.Connection | None) -> None:
        self._local.conn = value

    def _connect(self):
        conn = sqlite3.connect(str(self._path))
        with self._connections_lock:
            self._connections.append(conn)
        self._local.conn = conn
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

            CREATE TABLE IF NOT EXISTS threads_posts (
                post_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                last_checked_at REAL
            );

            CREATE TABLE IF NOT EXISTS threads_processed_replies (
                reply_id TEXT PRIMARY KEY,
                post_id TEXT NOT NULL,
                processed_at REAL NOT NULL,
                response_id TEXT
            );

            CREATE TABLE IF NOT EXISTS threads_logged_replies (
                reply_id TEXT PRIMARY KEY,
                logged_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bluesky_processed_replies (
                reply_id TEXT PRIMARY KEY,
                post_id TEXT NOT NULL,
                processed_at REAL NOT NULL,
                response_id TEXT
            );

            CREATE TABLE IF NOT EXISTS bluesky_logged_replies (
                reply_id TEXT PRIMARY KEY,
                logged_at REAL NOT NULL
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
        self._commit()
        self._clear_failed_post_cache()

    def _clear_failed_post_cache(self) -> None:
        rows = self._conn.execute(
            "SELECT request_hash, result FROM idempotency WHERE tool LIKE 'post_%'"
        ).fetchall()
        stale = []
        for row in rows:
            try:
                if not json.loads(row["result"]).get("ok", True):
                    stale.append((row["request_hash"],))
            except (TypeError, json.JSONDecodeError):
                stale.append((row["request_hash"],))
        if stale:
            self._conn.executemany(
                "DELETE FROM idempotency WHERE request_hash = ?", stale
            )
            self._commit()

    def _commit(self) -> None:
        if not self._in_transaction:
            self._conn.commit()

    @contextmanager
    def transaction(self) -> Iterator["MCPDatabase"]:
        self._in_transaction = True
        ok = False
        try:
            yield self
            ok = True
        except BaseException:
            self._conn.rollback()
            raise
        finally:
            self._in_transaction = False
            if ok:
                self._conn.commit()

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
        self._commit()

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
        self._commit()

    def _clean_expired_idempotency(self):
        self._conn.execute("DELETE FROM idempotency WHERE expires_at <= ?", (time.time(),))
        self._commit()

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
        self._commit()

    def remember_threads_post(self, post_id: str) -> None:
        if post_id:
            self._conn.execute(
                "INSERT OR IGNORE INTO threads_posts (post_id, created_at) VALUES (?, ?)",
                (str(post_id), time.time()),
            )
            self._commit()

    def list_threads_posts(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT post_id FROM threads_posts ORDER BY created_at ASC"
        ).fetchall()
        return [str(row["post_id"]) for row in rows]

    def has_processed_threads_reply(self, reply_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM threads_processed_replies WHERE reply_id = ?",
            (str(reply_id),),
        ).fetchone()
        return row is not None

    def mark_processed_threads_reply(
        self, reply_id: str, post_id: str, response_id: str | None = None
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO threads_processed_replies "
            "(reply_id, post_id, processed_at, response_id) VALUES (?, ?, ?, ?)",
            (str(reply_id), str(post_id), time.time(), response_id),
        )
        self._commit()

    def has_logged_threads_reply(self, reply_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM threads_logged_replies WHERE reply_id = ?",
            (str(reply_id),),
        ).fetchone()
        return row is not None

    def mark_logged_threads_reply(self, reply_id: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO threads_logged_replies (reply_id, logged_at) VALUES (?, ?)",
            (str(reply_id), time.time()),
        )
        self._commit()

    def has_processed_bluesky_reply(self, reply_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM bluesky_processed_replies WHERE reply_id = ?",
            (str(reply_id),),
        ).fetchone()
        return row is not None

    def mark_processed_bluesky_reply(
        self, reply_id: str, post_id: str, response_id: str | None = None
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO bluesky_processed_replies "
            "(reply_id, post_id, processed_at, response_id) VALUES (?, ?, ?, ?)",
            (str(reply_id), str(post_id), time.time(), response_id),
        )
        self._commit()

    def has_logged_bluesky_reply(self, reply_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM bluesky_logged_replies WHERE reply_id = ?",
            (str(reply_id),),
        ).fetchone()
        return row is not None

    def mark_logged_bluesky_reply(self, reply_id: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO bluesky_logged_replies (reply_id, logged_at) VALUES (?, ?)",
            (str(reply_id), time.time()),
        )
        self._commit()

    def log_tool_call(self, tool: str, arguments: dict, result: dict, duration_ms: float):
        self._conn.execute(
            "INSERT INTO tool_log (tool, arguments, result, duration_ms, created_at) VALUES (?, ?, ?, ?, ?)",
            (tool, json.dumps(arguments), json.dumps(result), duration_ms, time.time()),
        )
        self._commit()

    def cleanup(self):
        self._clean_expired_idempotency()
        cutoff = time.time() - 86400 * 30
        self._conn.execute("DELETE FROM tool_log WHERE created_at < ?", (cutoff,))
        self._conn.execute("DELETE FROM access_tokens WHERE expires_at < ?", (time.time(),))
        self._commit()

    def close(self):
        if getattr(self._local, "conn", None):
            self.cleanup()
        with self._connections_lock:
            connections = list(self._connections)
            self._connections.clear()
        for conn in connections:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        self._local.conn = None
