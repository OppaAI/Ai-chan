"""
memory/memorybank.py

Shared database helpers for Aiko's local RAG stores.

Memory, learned knowledge, and experience all use local SQLite/sqlite-vec
stores, optionally encrypted through system.secure. Keep common connection,
schema, FTS query, and ranking helpers here so store modules own their domain
schemas/queries but not repeated database bootstrap code.
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from system.secure import connect_sqlite
from system.userspace import current_user_id, user_state_path



def resolve_user_db_path(path_value: str | os.PathLike[str], *, user_id: str | None = None) -> Path:
    """Resolve an absolute or per-user relative database path.

    Relative paths live under <USER_STATE_ROOT>/<user_id>/, matching memory,
    knowledge, and experience storage conventions.
    """
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return user_state_path(str(path), user_id)



def connect_sqlite_db(path: str | os.PathLike[str], *, user_id: str | None = None, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    """Open an optionally encrypted SQLite connection without sqlite-vec."""
    uid = user_id or current_user_id()
    conn = connect_sqlite(path, user_id=uid)
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def initialize_sqlite_db(path: str | os.PathLike[str], ddl: str, *, user_id: str | None = None) -> sqlite3.Connection:
    """Open a standard SQLite DB, apply schema DDL, commit, and return connection."""
    conn = connect_sqlite_db(path, user_id=user_id)
    conn.executescript(ddl)
    conn.commit()
    return conn

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


def store_db_path(path_value: str | os.PathLike[str], *, user_id: str | None = None) -> Path:
    """Alias for resolving a store-owned DB path under the active user state."""
    return resolve_user_db_path(path_value, user_id=user_id)


def fetch_by_ids(conn: sqlite3.Connection, table: str, ids: set[str], *, id_column: str = "id") -> dict[str, sqlite3.Row]:
    """Fetch rows from a table by ids and return them keyed by id."""
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(f"SELECT * FROM {table} WHERE {id_column} IN ({placeholders})", list(ids)).fetchall()
    return {str(row[id_column]): row for row in rows}

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ident(name: str) -> str:
    """Validate a SQLite identifier used in shared generated SQL."""
    if not _IDENT_RE.fullmatch(name or ""):
        raise ValueError(f"unsafe SQLite identifier: {name!r}")
    return name


def utc_now_iso() -> str:
    """Current UTC timestamp as an ISO string, shared by durable stores."""
    from system.bioclock import utc_now
    return utc_now().isoformat()


def initialize_store_db(
    path_value: str | os.PathLike[str],
    ddl: str,
    *,
    user_id: str | None = None,
    vector: bool = True,
) -> sqlite3.Connection:
    """Resolve a store DB path, initialize its schema, and return a connection.

    Store modules pass their own env-configured path and schema. Relative paths
    live under the active user's state directory; absolute paths are respected.
    """
    uid = user_id or current_user_id()
    path = store_db_path(path_value, user_id=uid)
    init = initialize_sqlite_vec_db if vector else initialize_sqlite_db
    return init(path, ddl, user_id=uid)


def sqlite_vec_blob(vector: object) -> bytes:
    """Serialize a Python/numpy vector for sqlite-vec insertion/search."""
    import sqlite_vec
    return sqlite_vec.serialize_float32(vector)


def insert_vector(conn: sqlite3.Connection, table: str, item_id: str, vector: object) -> None:
    """Insert a serialized vector into a vec0 table with common id/embedding columns."""
    table = _ident(table)
    conn.execute(f"INSERT INTO {table}(id, embedding) VALUES(?, ?)", (item_id, sqlite_vec_blob(vector)))


def user_scoped_vec_knn(
    conn: sqlite3.Connection,
    *,
    vec_table: str,
    owner_table: str,
    owner_alias: str,
    vector: object,
    user_id: str,
    limit: int,
    owner_id_column: str = "id",
    vec_id_column: str = "id",
    user_column: str = "user_id",
) -> list[sqlite3.Row]:
    """Run the common sqlite-vec KNN pattern scoped through an owner table."""
    vec_table = _ident(vec_table)
    owner_table = _ident(owner_table)
    owner_alias = _ident(owner_alias)
    owner_id_column = _ident(owner_id_column)
    vec_id_column = _ident(vec_id_column)
    user_column = _ident(user_column)
    blob = sqlite_vec_blob(vector)
    return conn.execute(
        f"""
        SELECT v.{vec_id_column} AS id, vec_distance_cosine(v.embedding, ?) AS dist
        FROM {vec_table} v
        JOIN {owner_table} {owner_alias} ON {owner_alias}.{owner_id_column} = v.{vec_id_column}
        WHERE {owner_alias}.{user_column} = ?
        ORDER BY dist ASC
        LIMIT ?
        """,
        (blob, user_id, limit),
    ).fetchall()


def user_scoped_fts_search(
    conn: sqlite3.Connection,
    *,
    fts_table: str,
    owner_table: str,
    owner_alias: str,
    query: str,
    user_id: str,
    limit: int,
    owner_id_column: str = "id",
    fts_id_column: str = "id",
    user_column: str = "user_id",
) -> list[sqlite3.Row]:
    """Run the common FTS5 MATCH pattern scoped through an owner table."""
    fts = fts_or_query(query)
    if not fts:
        return []
    fts_table = _ident(fts_table)
    owner_table = _ident(owner_table)
    owner_alias = _ident(owner_alias)
    owner_id_column = _ident(owner_id_column)
    fts_id_column = _ident(fts_id_column)
    user_column = _ident(user_column)
    return conn.execute(
        f"""
        SELECT f.{fts_id_column} AS id
        FROM {fts_table} f
        JOIN {owner_table} {owner_alias} ON {owner_alias}.{owner_id_column} = f.{fts_id_column}
        WHERE {fts_table} MATCH ? AND {owner_alias}.{user_column} = ?
        ORDER BY rank
        LIMIT ?
        """,
        (fts, user_id, limit),
    ).fetchall()


def rank_by_id(rows: list[sqlite3.Row]) -> dict[str, int]:
    """Convert ordered id rows into an RRF rank mapping."""
    return {str(row["id"]): i + 1 for i, row in enumerate(rows)}


def delete_by_id(conn: sqlite3.Connection, table: str, item_id: str, *, id_column: str = "id") -> int:
    """Delete a single row by id from a table and return affected row count."""
    table = _ident(table)
    id_column = _ident(id_column)
    cur = conn.execute(f"DELETE FROM {table} WHERE {id_column}=?", (item_id,))
    return int(cur.rowcount or 0)


def delete_user_row(conn: sqlite3.Connection, table: str, item_id: str, user_id: str, *, id_column: str = "id", user_column: str = "user_id") -> int:
    """Delete one row by id + user_id and return affected row count."""
    table = _ident(table)
    id_column = _ident(id_column)
    user_column = _ident(user_column)
    cur = conn.execute(f"DELETE FROM {table} WHERE {user_column}=? AND {id_column}=?", (user_id, item_id))
    return int(cur.rowcount or 0)
