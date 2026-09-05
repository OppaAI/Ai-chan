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
from system.config import env_float, env_int
from system.userspace import current_user_id, user_state_path
from system.log import get_logger

log = get_logger(__name__)



# ═══════════════════════════════════════════════════════════════════════════════
#  Embedder
# ═══════════════════════════════════════════════════════════════════════════════

_EMBED_BASE_URL   = os.getenv("EMBED_BASE_URL", "http://127.0.0.1:8080")
_EMBED_MODEL      = os.getenv("EMBED_MODEL", "harrier")
_EMBED_DIMS       = env_int("EMBED_DIMS", 640)
_BATCH_SIZE       = env_int("EMBED_BATCH_SIZE", 32)
_EMBED_TIMEOUT    = env_float("EMBED_TIMEOUT_S", 30)
_QUERY_INSTRUCT   = os.getenv(
    "EMBED_QUERY_INSTRUCT",
    "Retrieve relevant memories that answer the query",
)

# vec0 MATCH KNN oversampling — see user_scoped_vec_knn. Defaults mirror the
# memory-domain constants in cognition.memory.schema.
KNN_MATCH_OVERSCAN = env_int("KNN_MATCH_OVERSCAN", 16)
KNN_MATCH_K_MIN = env_int("KNN_MATCH_K_MIN", 32)


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

    Optional persistent disk cache (EMBED_CACHE_PATH) survives restarts.
    """

    _CACHE_MAX: int = 256
    # Nano default 300s: same greeting/query must not re-embed after 30s.
    # Override with EMBED_CACHE_TTL seconds when needed.
    _CACHE_TTL: float = env_float("EMBED_CACHE_TTL", 300.0)
    _DISK_CACHE_MAX: int = 2048  # cap on in-RAM disk-cache entries
    _DISK_CACHE_FILE_MAX_BYTES: int = 8 * 1024 * 1024  # compact JSONL past this

    def __init__(
        self,
        base_url: str   = _EMBED_BASE_URL,
        model: str      = _EMBED_MODEL,
        dims: int       = _EMBED_DIMS,
        batch_size: int = _BATCH_SIZE,
        timeout: float  = _EMBED_TIMEOUT,
        cache_path: str | None = None,
    ) -> None:
        self.base_url   = base_url.rstrip("/")
        self.model      = model
        self.dims       = dims
        self.batch_size = batch_size
        self.timeout    = timeout
        self._session   = requests.Session()
        self._cache: OrderedDict[tuple[str, ...], tuple[float, np.ndarray]] = OrderedDict()
        self._cache_lock = threading.Lock()

        # Persistent disk cache for embeddings across restarts
        self._disk_cache_path = Path(cache_path) if cache_path else None
        self._disk_cache: dict[tuple[str, ...], np.ndarray] = {}
        self._disk_cache_lock = threading.Lock()
        if self._disk_cache_path:
            self._load_disk_cache()

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

    def _load_disk_cache(self) -> None:
        """Load embeddings from disk cache (JSON Lines format).

        Keeps only the most recent _DISK_CACHE_MAX entries — the file is
        append-ordered, so the tail wins and a stale 100k-line cache from
        before the cap can't OOM the Nano on boot.
        """
        if not self._disk_cache_path or not self._disk_cache_path.exists():
            return
        try:
            import json
            self._disk_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._disk_cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        key = tuple(entry["texts"])
                        vec = np.asarray(entry["vector"], dtype=np.float32)
                        self._disk_cache[key] = vec
                    except Exception:
                        continue
            log.info(f"Loaded {len(self._disk_cache)} embeddings from disk cache: {self._disk_cache_path}")
            if len(self._disk_cache) > self._DISK_CACHE_MAX:
                tail = list(self._disk_cache.items())[-self._DISK_CACHE_MAX:]
                self._disk_cache = dict(tail)
                log.info(f"Trimmed disk embedding cache to {len(self._disk_cache)} most-recent entries")
        except Exception as e:
            log.warning(f"Failed to load disk embedding cache: {e}")

    def _save_to_disk_cache(self, key: tuple[str, ...], vec: np.ndarray) -> None:
        """Append embedding to disk cache (JSON Lines)."""
        if not self._disk_cache_path:
            return
        try:
            import json
            entry = {"texts": list(key), "vector": vec.tolist()}
            with open(self._disk_cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._maybe_compact_disk_cache()
        except Exception as e:
            log.debug(f"Failed to write disk embedding cache: {e}")

    def _maybe_compact_disk_cache(self) -> None:
        """Rewrite an overgrown JSONL cache from the capped in-RAM dict."""
        try:
            if not self._disk_cache_path or not self._disk_cache_path.exists():
                return
            if self._disk_cache_path.stat().st_size <= self._DISK_CACHE_FILE_MAX_BYTES:
                return
            import json
            with self._disk_cache_lock:
                entries = list(self._disk_cache.items())
            tmp = self._disk_cache_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for key, vec in entries:
                    f.write(json.dumps({"texts": list(key), "vector": np.asarray(vec).tolist()}, ensure_ascii=False) + "\n")
            tmp.replace(self._disk_cache_path)
            log.info(f"Compacted disk embedding cache to {len(entries)} entries")
        except Exception as e:
            log.debug(f"Disk embedding cache compaction skipped: {e}")

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of raw texts via llama-server's /embedding endpoint.
        Returns np.ndarray of shape (len(texts), dims), L2-normalised.

        Results are cached in a small TTL-based LRU cache keyed by the tuple
        of input texts, so duplicate calls within _CACHE_TTL seconds skip the
        HTTP round-trip. Also checks persistent disk cache.
        """
        key = tuple(texts)
        now = time.monotonic()

        # Check memory cache first
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None and now - cached[0] <= self._CACHE_TTL:
                self._cache.move_to_end(key)
                return cached[1]

        # Check disk cache
        with self._disk_cache_lock:
            disk_cached = self._disk_cache.get(key)
            if disk_cached is not None:
                # Promote to memory cache
                with self._cache_lock:
                    self._cache[key] = (now, disk_cached)
                    while len(self._cache) > self._CACHE_MAX:
                        self._cache.popitem(last=False)
                return disk_cached

        # HTTP call to embedding server
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

        # Store in both caches (both capped LRU-style). The disk-cache dict
        # mutation holds the lock; the file append below takes it again
        # briefly inside _maybe_compact_disk_cache, so it must run outside.
        with self._cache_lock:
            self._cache[key] = (now, result)
            while len(self._cache) > self._CACHE_MAX:
                self._cache.popitem(last=False)

        with self._disk_cache_lock:
            self._disk_cache[key] = result
            while len(self._disk_cache) > self._DISK_CACHE_MAX:
                self._disk_cache.popitem(next(iter(self._disk_cache)))
        self._save_to_disk_cache(key, result)

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

    Relative paths live under <USER_SPACE_ROOT>/<user_id>/, matching memory,
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
    # Resolve relative paths under <USER_SPACE_ROOT>/<user_id>/ so that
    # a bare "memory/memory.db" argument never lands in the process's
    # CWD (e.g. the repo root when launched via main.py).
    resolved = resolve_user_db_path(path, user_id=uid)
    conn = connect_sqlite(resolved, user_id=uid)
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
    resolved = resolve_user_db_path(path, user_id=uid)
    conn = connect_sqlite(resolved, user_id=uid)
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
    ensure_vec0_cosine_metric(conn)
    return conn


_VEC0_TABLE_RE = re.compile(
    r"CREATE\s+VIRTUAL\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(\w+)\s+USING\s+vec0\s*\((.*)\)\s*;?\s*$",
    re.S | re.I,
)


def ensure_vec0_cosine_metric(conn: sqlite3.Connection) -> list[str]:
    """Rebuild any vec0 table that was created without `distance_metric=cosine`.

    vec0's default metric is L2, but every KNN caller in this codebase
    interprets the returned `distance` as cosine (1 - similarity) and applies
    cosine-based thresholds (dedup, recall, dream merge). Running MATCH against
    an L2-metric table would silently rank by L2 and return L2 distances, so
    existing stores must be migrated once. sqlite-vec 0.1.9 cannot ALTER a
    vec0 table's metric, so we rebuild in place: copy rows out, drop, recreate
    with the metric, reinsert. Returns the list of migrated table names.
    """
    migrated: list[str] = []
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    except sqlite3.Error as e:
        log.debug("vec0 metric scan skipped: %s", e)
        return migrated
    for name, ddl_sql in rows:
        ddl_sql = ddl_sql or ""
        if "USING vec0" not in ddl_sql or "distance_metric=cosine" in ddl_sql:
            continue
        m = _VEC0_TABLE_RE.search(ddl_sql)
        if not m:
            continue
        cols_text = m.group(2).rstrip()
        new_ddl = f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} USING vec0({cols_text} distance_metric=cosine);"
        try:
            cols = conn.execute(f"PRAGMA table_info({name})").fetchall()
            id_col = next((c[1] for c in cols if c[1] == "id"), None)
            embed_col = next((c[1] for c in cols if c[1] == "embedding"), None)
            if not embed_col:
                continue
            if id_col:
                id_rows = conn.execute(f"SELECT {id_col} FROM {name}").fetchall()
            else:
                id_rows = conn.execute(f"SELECT rowid FROM {name}").fetchall()
            embed_rows = conn.execute(f"SELECT {embed_col} FROM {name}").fetchall()
            rows_out = [(r[0], e[0]) for r, e in zip(id_rows, embed_rows)]
            conn.execute(f"DROP TABLE {name}")
            conn.executescript(new_ddl)
            for rowid, embedding in rows_out:
                if id_col:
                    conn.execute(
                        f"INSERT INTO {name}({id_col}, embedding) VALUES (?, ?)",
                        (rowid, embedding),
                    )
                else:
                    conn.execute(
                        f"INSERT INTO {name}(rowid, embedding) VALUES (?, ?)",
                        (rowid, embedding),
                    )
            conn.commit()
            migrated.append(name)
            log.info("vec0 %s: migrated to distance_metric=cosine (%d rows)", name, len(rows_out))
        except sqlite3.Error as e:
            log.warning("vec0 %s metric migration skipped: %s", name, e)
    return migrated


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

    When the DDL defines the personal ``memories`` table, Phase A columns are
    ensured (idempotent ALTER) and write/recall hooks are installed once.
    """
    uid = user_id or current_user_id()
    path = store_db_path(path_value, user_id=uid)
    init = initialize_sqlite_vec_db if vector else initialize_sqlite_db
    conn = init(path, ddl, user_id=uid)
    # Phase A: lightweight schema migrate for personal memory only.
    # (Write/recall hooks are now native methods on memorize._MemoryBackend —
    #  no runtime monkey-patching needed.)
    if "memories_vec" in (ddl or "") or "CREATE TABLE IF NOT EXISTS memories" in (ddl or ""):
        try:
            from cognition.memory.memorize import ensure_phase_a_schema
            ensure_phase_a_schema(conn)
        except Exception:
            # Never block boot on optional Phase A wiring.
            pass
    return conn


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

    Uses the vec0 MATCH index (tables are created with distance_metric=cosine,
    so `distance` == 1 - cosine_similarity). k is oversampled because the
    owner-scope filter applies after the vec0 scan picks the k nearest rows.

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
    k = max(int(limit) * KNN_MATCH_OVERSCAN, KNN_MATCH_K_MIN)

    if threshold is not None:
        dist_ceil = 1.0 - threshold
        return conn.execute(
            f"""
            SELECT v.{vec_id_column} AS id, v.distance AS dist
            FROM {vec_table} v
            JOIN {owner_table} {owner_alias} ON {owner_alias}.{owner_id_column} = v.{vec_id_column}
            WHERE v.embedding MATCH ?
              AND v.k = ?
              AND {owner_alias}.{user_column} = ?
              AND v.distance <= ?
            ORDER BY v.distance ASC
            LIMIT ?
            """,
            (blob, k, user_id, dist_ceil, limit),
        ).fetchall()

    return conn.execute(
        f"""
        SELECT v.{vec_id_column} AS id, v.distance AS dist
        FROM {vec_table} v
        JOIN {owner_table} {owner_alias} ON {owner_alias}.{owner_id_column} = v.{vec_id_column}
        WHERE v.embedding MATCH ?
          AND v.k = ?
          AND {owner_alias}.{user_column} = ?
        ORDER BY v.distance ASC
        LIMIT ?
        """,
        (blob, k, user_id, limit),
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
