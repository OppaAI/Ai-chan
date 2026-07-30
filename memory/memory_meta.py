"""
memory/memory_meta.py

Phase A memory metadata: schema ensure, write-op classification, status constants.

Design constraints:
  - Zero extra LLM calls (latency-safe on Jetson / local models).
  - Additive SQLite columns only — no vector rebuild.
  - Idempotent migration (safe to run on every boot and via CLI).
"""
from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from system.log import get_logger

log = get_logger(__name__)

STATUS_ACTIVE = "active"
STATUS_SUPERSEDED = "superseded"

KIND_FACT = "fact"
KIND_PREFERENCE = "preference"
KIND_IDENTITY = "identity"
KIND_EVENT = "event"
KIND_PLAN = "plan"

SOURCE_CHAT = "chat"
SOURCE_REFLECT = "reflect"
SOURCE_PIN = "pin"
SOURCE_IMPORT = "import"
SOURCE_LEGACY = "legacy"

_WS_RE = re.compile(r"\s+")


def normalize_memory_text(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip()).lower()


def entities_to_json(entities: list[str] | None) -> str:
    if not entities:
        return "[]"
    cleaned: list[str] = []
    seen: set[str] = set()
    for e in entities:
        s = str(e).strip()
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s[:80])
        if len(cleaned) >= 16:
            break
    return json.dumps(cleaned, ensure_ascii=False)


def entities_from_json(raw: Any) -> list[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [str(x).strip() for x in data if str(x).strip()]


def classify_write_op(
    *,
    similarity: float | None,
    new_text: str,
    old_text: str | None,
    dedup_threshold: float,
) -> str:
    """Return 'noop' | 'supersede' | 'add' using only similarity + text.

    Policy (keeps current write cost: one KNN already paid):
      - No neighbor under dedup threshold → add
      - Same normalized text → noop (true duplicate)
      - High cosine but different text → supersede (value/wording update)
    """
    if similarity is None or similarity < dedup_threshold:
        return "add"
    if normalize_memory_text(new_text) == normalize_memory_text(old_text or ""):
        return "noop"
    return "supersede"


_PHASE_A_COLUMNS: tuple[tuple[str, str], ...] = (
    ("status", "TEXT NOT NULL DEFAULT 'active'"),
    ("supersedes_id", "TEXT"),
    ("kind", "TEXT NOT NULL DEFAULT 'fact'"),
    ("source", "TEXT NOT NULL DEFAULT 'legacy'"),
    ("entities", "TEXT NOT NULL DEFAULT '[]'"),
)


def existing_columns(conn: sqlite3.Connection, table: str = "memories") -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(r[1]) for r in rows}


def ensure_phase_a_schema(conn: sqlite3.Connection) -> list[str]:
    """Idempotent ALTER TABLE for Phase A columns + status index.

    Returns names of columns that were added this call (empty if already migrated).
    Does not rebuild vectors or touch FTS content.
    """
    cols = existing_columns(conn)
    added: list[str] = []
    for name, decl in _PHASE_A_COLUMNS:
        if name in cols:
            continue
        try:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {name} {decl}")
            added.append(name)
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).casefold():
                raise
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_user_status "
        "ON memories(user_id, status)"
    )
    conn.commit()
    if added:
        log.info("memory Phase A schema: added columns %s", added)
    return added


def phase_a_ddl_fragment() -> str:
    """Extra column lines for CREATE TABLE on fresh DBs."""
    return (
        "    status           TEXT NOT NULL DEFAULT 'active',\n"
        "    supersedes_id    TEXT,\n"
        "    kind             TEXT NOT NULL DEFAULT 'fact',\n"
        "    source           TEXT NOT NULL DEFAULT 'legacy',\n"
        "    entities         TEXT NOT NULL DEFAULT '[]',\n"
    )
