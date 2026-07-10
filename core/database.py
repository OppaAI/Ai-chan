"""Shared database helpers for Aiko's local RAG stores.

Memory, learned knowledge, and experience all use local SQLite/sqlite-vec
stores, optionally encrypted through core.secure. Keep common connection,
schema, FTS query, and ranking helpers here so store modules own their domain
schemas/queries but not repeated database bootstrap code.
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from core.secure import connect_sqlite
from core.userspace import current_user_id, user_state_path



def resolve_user_db_path(path_value: str | os.PathLike[str], *, user_id: str | None = None) -> Path:
    """Resolve an absolute or per-user relative database path.

    Relative paths live under <USER_STATE_ROOT>/<user_id>/, matching memory,
    knowledge, and experience storage conventions.
    """
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return user_state_path(str(path), user_id)


def connect_sqlite_vec(path: str | os.PathLike[str], *, user_id: str | None = None, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    """Open an optionally encrypted SQLite connection with sqlite-vec loaded."""
    uid = user_id or current_user_id()
    conn = connect_sqlite(path, user_id=uid)
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        conn.enable_load_extension(True)
    except Exception:
        pass
    import sqlite_vec
    sqlite_vec.load(conn)
    try:
        conn.enable_load_extension(False)
    except Exception:
        pass
    return conn


def initialize_sqlite_vec_db(path: str | os.PathLike[str], ddl: str, *, user_id: str | None = None) -> Any:
    """Open a sqlite-vec DB, apply schema DDL, commit, and return connection."""
    conn = connect_sqlite_vec(path, user_id=user_id)
    conn.executescript(ddl)
    conn.commit()
    return conn


_WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "how", "what", "who", "when", "where", "why",
    "we", "you", "i", "he", "she", "they", "this", "that", "these",
    "those", "some", "any", "all", "each", "can", "could", "will",
    "would", "should", "shall", "may", "might", "must", "to", "of", "in",
    "on", "at", "for", "with", "and", "or", "not", "no", "yes", "make",
    "made", "get", "got", "go", "going", "let", "lets", "want", "wants",
    "just", "so", "up", "down", "out", "about", "if", "then", "than",
})


def fts_or_query(text: str, *, max_terms: int = 16) -> str | None:
    """Build a conservative FTS5 OR query from literal terms.

    This intentionally stays lexical/deterministic; semantic ranking is handled
    separately by embeddings and fused later.
    """
    terms: list[str] = []
    for match in _WORD_RE.finditer(text or ""):
        term = match.group(0).strip().replace('"', "")
        if len(term) >= 2 and term.casefold() not in _STOPWORDS:
            terms.append(f'"{term}"')
    return " OR ".join(terms[:max_terms]) or None


def rrf_score(item_id: str, rank_knn: dict[str, int], rank_fts: dict[str, int], *, k: int) -> float:
    score = 0.0
    if item_id in rank_knn:
        score += 1.0 / (k + rank_knn[item_id])
    if item_id in rank_fts:
        score += 1.0 / (k + rank_fts[item_id])
    return score
