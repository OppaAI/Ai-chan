"""
memory/vecstore.py

Shared database helpers and text embedder for Aiko's local RAG stores.

Memory, learned knowledge, and experience all use local SQLite/sqlite-vec
stores, optionally encrypted through system.secure. Keep common connection,
schema, FTS query, and ranking helpers here so store modules own their domain
schemas/queries but not repeated database bootstrap code.

The HarrierEmbedder class provides HTTP-based text embeddings via llama-server.
"""
from __future__ import annotations

import os
import re
import sqlite3
import struct
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Generator, Iterable

import numpy as np
import requests

from system.secure import connect_sqlite
from system.userspace import current_user_id, user_state_path



# ═══════════════════════════════════════════════════════════════════════════════
#  Embedder
# ═══════════════════════════════════════════════════════════════════════════════

_EMBED_BASE_URL   = os.getenv("EMBED_BASE_URL", "http://127.0.0.1:8080")
_EMBED_MODEL      = os.getenv("EMBED_MODEL", "harrier")
_EMBED_DIMS       = int(os.getenv("EMBED_DIMS", "640"))
_BATCH_SIZE       = int(os.getenv("EMBED_BATCH_SIZE", "32"))
_EMBED_TIMEOUT    = float(os.getenv("EMBED_TIMEOUT_S", "30"))
_QUERY_INSTRUCT   = os.getenv(
    "EMBED_QUERY_INSTRUCT",
    "Retrieve relevant memories that answer the query",
)


class HarrierEmbedder:
    """
    HTTP-based text embedder for harrier-oss-v1-270m via llama-server.

    Talks to a running llama-server instance (started with ``embedding = true``,
    ``pooling-type = last``) over its /embedding endpoint. Connection is lazy —
    the first call just hits the HTTP endpoint, no local model loading happens
    in this process.

    Includes a small TTL-based LRU cache on _embed_texts so that repeated
    calls with the same texts (e.g. the same user query embedded by routing,
    memory search, and knowledge search in a single turn) skip the HTTP
    round-trip.
    """

    _CACHE_MAX: int = 256
    _CACHE_TTL: float = 30.0  # seconds

    def __init__(
        self,
        base_url: str   = _EMBED_BASE_URL,
        model: str      = _EMBED_MODEL,
        dims: int       = _EMBED_DIMS,
        batch_size: int = _BATCH_SIZE,
        timeout: float  = _EMBED_TIMEOUT,
    ) -> None:
        self.base_url   = base_url.rstrip("/")
        self.model      = model
        self.dims       = dims
        self.batch_size = batch_size
        self.timeout    = timeout
        self._session   = requests.Session()
        self._cache: OrderedDict[tuple[str, ...], tuple[float, np.ndarray]] = OrderedDict()
        self._cache_lock = threading.Lock()

        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=requests.adapters.Retry(
                total=2,
                connect=2,
                read=2,
                backoff_factor=0.2,
                status_forcelist=[502, 503, 504],
                allowed_methods=frozenset(["GET", "POST"]),
            ),
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of raw texts via llama-server's /embedding endpoint.
        Returns np.ndarray of shape (len(texts), dims), L2-normalised.

        Results are cached in a small TTL-based LRU cache keyed by the tuple
        of input texts, so duplicate calls within _CACHE_TTL seconds skip the
        HTTP round-trip.
        """
        key = tuple(texts)
        now = time.monotonic()
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None and now - cached[0] <= self._CACHE_TTL:
                self._cache.move_to_end(key)
                return cached[1]

        resp = self._session.post(
            f"{self.base_url}/embedding",
            json={"model": self.model, "content": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        vecs = []
        for item in data:
            emb = item["embedding"]
            if isinstance(emb[0], list):
                emb = emb[0]
            vecs.append(emb)

        arr = np.asarray(vecs, dtype=np.float32)

        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        result = arr / norms

        with self._cache_lock:
            self._cache[key] = (now, result)
            while len(self._cache) > self._CACHE_MAX:
                self._cache.popitem(last=False)

        return result

    def embed(self, texts: Iterable[str]) -> Generator[np.ndarray, None, None]:
        """Embed documents (no instruction prefix). Yields one np.ndarray(dim,) per text."""
        texts = list(texts)
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vecs  = self._embed_texts(batch)
            for v in vecs:
                yield v

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed documents and return all vectors as np.ndarray (N, dims)."""
        all_vecs = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            all_vecs.append(self._embed_texts(batch))
        return np.vstack(all_vecs)

    def embed_query(self, query: str, instruct: str = _QUERY_INSTRUCT) -> np.ndarray:
        """
        Embed a single search query with the instruction prefix.
        Returns np.ndarray(dims,).
        """
        prefixed = f"Instruct: {instruct}\nQuery: {query}"
        return self._embed_texts([prefixed])[0]

    def embed_queries(self, queries: list[str], instruct: str = _QUERY_INSTRUCT) -> np.ndarray:
        """Embed multiple search queries with the instruction prefix. Returns np.ndarray (N, dims)."""
        prefixed = [f"Instruct: {instruct}\nQuery: {q}" for q in queries]
        return self.embed_batch(prefixed)

    @staticmethod
    def serialize(vector: np.ndarray) -> bytes:
        """Serialise a float32 vector for sqlite-vec INSERT."""
        v = vector.astype(np.float32)
        return struct.pack(f"{len(v)}f", *v)


# ═══════════════════════════════════════════════════════════════════════════════
#  Database helpers
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_user_db_path(path_value: str | os.PathLike[str], *, user_id: str | None = None) -> Path:
    """Resolve an absolute or per-user relative database path.

    Relative paths live under <USER_STATE_ROOT>/<user_id>/, matching memory,
    knowledge, and experience storage conventions. ":memory:" is SQLite's
    special in-memory sentinel — it must pass through untouched, never
    joined onto a user directory, or it becomes a literal on-disk file
    named ":memory:" instead of an ephemeral RAM-only database.
    """
    raw = str(path_value)
    if raw == ":memory:":
        return Path(":memory:")

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
    "we", "you", "i", "he", "she", "it", "they", "this", "that", "these",
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
    threshold: float | None = None,
) -> list[sqlite3.Row]:
    """Run the common sqlite-vec KNN pattern scoped through an owner table.

    threshold, if given, is a minimum cosine similarity (0..1) — rows whose
    distance exceeds (1 - threshold) are excluded, so a lone unrelated
    candidate can't win purely by being the only thing in the table.
    """
    vec_table = _ident(vec_table)
    owner_table = _ident(owner_table)
    owner_alias = _ident(owner_alias)
    owner_id_column = _ident(owner_id_column)
    vec_id_column = _ident(vec_id_column)
    user_column = _ident(user_column)
    blob = sqlite_vec_blob(vector)

    if threshold is not None:
        dist_ceil = 1.0 - threshold
        return conn.execute(
            f"""
            SELECT v.{vec_id_column} AS id, vec_distance_cosine(v.embedding, ?) AS dist
            FROM {vec_table} v
            JOIN {owner_table} {owner_alias} ON {owner_alias}.{owner_id_column} = v.{vec_id_column}
            WHERE {owner_alias}.{user_column} = ?
              AND vec_distance_cosine(v.embedding, ?) <= ?
            ORDER BY dist ASC
            LIMIT ?
            """,
            (blob, user_id, blob, dist_ceil, limit),
        ).fetchall()

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
