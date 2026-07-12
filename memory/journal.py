"""
memory/journal.py

Encrypted daily journal store for faithful reflection records.

Daily journal rows are separate from memory facts: they keep the large,
verbatim day-level blob in ``journal.db`` beside ``memory.db`` while using the
same ``system.secure.connect_sqlite`` encryption path.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime

from memory.vecstore import delete_user_row, initialize_store_db, utc_now_iso
from system.userspace import current_user_id

JOURNAL_DB_PATH = os.getenv("JOURNAL_DB_PATH", "memory/journal.db")

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS journals (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    entry_date TEXT NOT NULL,
    tag        TEXT NOT NULL,
    body       TEXT NOT NULL,
    pinned     INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(user_id, entry_date)
);
CREATE INDEX IF NOT EXISTS idx_journals_user_date ON journals(user_id, entry_date);
"""


def _connect(user_id: str | None = None):
    return initialize_store_db(JOURNAL_DB_PATH, _DDL, user_id=user_id, vector=False)


def _now() -> str:
    return utc_now_iso()


def daily_journal_tag(date: datetime | str) -> str:
    date_str = date if isinstance(date, str) else date.strftime("%Y-%m-%d")
    return f"Daily journal of {date_str}:"


def pin_daily_journal(body: str, date: datetime, *, user_id: str | None = None) -> str | None:
    """Pin a journal entry for a given local calendar day.

    ``date`` must be the LOCAL date the entry belongs to (e.g. from
    system.bioclock.local_now()), since entry_date is used as a plain
    string key/index, not a timezone-aware comparison.
    """
    uid = user_id or current_user_id()
    entry_date = date.strftime("%Y-%m-%d")
    tag = daily_journal_tag(entry_date)
    now = _now()
    row_id = str(uuid.uuid4())
    conn = _connect(uid)
    try:
        conn.execute(
            """
            INSERT INTO journals(id,user_id,entry_date,tag,body,pinned,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(user_id, entry_date) DO UPDATE SET
                tag=excluded.tag,
                body=excluded.body,
                pinned=1,
                updated_at=excluded.updated_at
            """,
            (row_id, uid, entry_date, tag, body, 1, now, now),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM journals WHERE user_id=? AND entry_date=?",
            (uid, entry_date),
        ).fetchone()
        return str(row["id"]) if row else row_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_between(start: datetime, end: datetime, *, user_id: str | None = None) -> list[dict]:
    """Fetch pinned journals with entry_date in [start, end).

    ``start``/``end`` must be LOCAL dates (e.g. from system.bioclock.local_now()),
    matching the local basis entry_date is stored in. Passing UTC-shifted
    datetimes here will misalign the YYYY-MM-DD window near midnight and can
    put month-boundary entries in the wrong bucket or drop them entirely.
    """
    uid = user_id or current_user_id()
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")
    conn = _connect(uid)
    try:
        rows = conn.execute(
            """
            SELECT * FROM journals
            WHERE user_id=? AND entry_date >= ? AND entry_date < ? AND pinned=1
            ORDER BY entry_date ASC
            """,
            (uid, start_s, end_s),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def delete(entry_id: str, *, user_id: str | None = None) -> bool:
    uid = user_id or current_user_id()
    conn = _connect(uid)
    try:
        deleted = delete_user_row(conn, "journals", entry_id, uid)
        conn.commit()
        return deleted > 0
    finally:
        conn.close()