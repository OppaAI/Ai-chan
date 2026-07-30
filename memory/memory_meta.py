"""
memory/memory_meta.py

Phase A/B memory metadata: schema ensure, write-op classification, runtime hooks.

Design constraints:
  - Zero extra LLM calls (latency-safe on Jetson / local models).
  - Additive SQLite columns only — no vector rebuild.
  - Idempotent migration (boot + CLI).
  - Hooks install once into memory.memorize without requiring a full file rewrite.
  - Phase B: rule-based entity tags + kind on write.
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from system.log import get_logger

log = get_logger(__name__)

STATUS_ACTIVE = "active"
STATUS_SUPERSEDED = "superseded"

KIND_FACT = "fact"
SOURCE_CHAT = "chat"
SOURCE_PIN = "pin"
SOURCE_LEGACY = "legacy"

_WS_RE = re.compile(r"\s+")
_PHASE_A_INSTALLED = False

_PHASE_A_COLUMNS: tuple[tuple[str, str], ...] = (
    ("status", "TEXT NOT NULL DEFAULT 'active'"),
    ("supersedes_id", "TEXT"),
    ("kind", "TEXT NOT NULL DEFAULT 'fact'"),
    ("source", "TEXT NOT NULL DEFAULT 'legacy'"),
    ("entities", "TEXT NOT NULL DEFAULT '[]'"),
)


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
    """Return 'noop' | 'supersede' | 'add' — rule-only, no LLM."""
    if similarity is None or similarity < dedup_threshold:
        return "add"
    if normalize_memory_text(new_text) == normalize_memory_text(old_text or ""):
        return "noop"
    return "supersede"


def existing_columns(conn: sqlite3.Connection, table: str = "memories") -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(r[1]) for r in rows}


def ensure_phase_a_schema(conn: sqlite3.Connection) -> list[str]:
    """Idempotent ALTER TABLE for Phase A columns + status index."""
    try:
        cols = existing_columns(conn)
    except sqlite3.Error:
        return []
    if "id" not in cols and "memory" not in cols:
        return []

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
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user_status "
            "ON memories(user_id, status)"
        )
        conn.commit()
    except sqlite3.Error as e:
        log.debug("memory Phase A index: %s", e)
    if added:
        log.info("memory Phase A schema: added columns %s", added)
    return added


def backfill_entities(
    conn: sqlite3.Connection,
    *,
    user_id: str | None = None,
    limit: int = 0,
    only_empty: bool = True,
) -> int:
    """Fill entities/kind for existing rows using rule-based extractors.

    No re-embed. Returns number of rows updated.
    """
    from memory.entities import classify_kind, extract_entities

    ensure_phase_a_schema(conn)
    cols = existing_columns(conn)
    if "entities" not in cols:
        return 0

    sql = "SELECT id, memory, entities, kind FROM memories WHERE 1=1"
    params: list[Any] = []
    if user_id:
        sql += " AND user_id = ?"
        params.append(user_id)
    if only_empty:
        sql += " AND (entities IS NULL OR entities = '' OR entities = '[]')"
    sql += " ORDER BY created_at DESC"
    if limit and limit > 0:
        sql += " LIMIT ?"
        params.append(int(limit))

    rows = conn.execute(sql, params).fetchall()
    updated = 0
    for row in rows:
        text = row["memory"] or ""
        ents = extract_entities(text)
        kind = classify_kind(text, default=str(row["kind"] or KIND_FACT))
        conn.execute(
            "UPDATE memories SET entities = ?, kind = ? WHERE id = ?",
            (entities_to_json(ents), kind, row["id"]),
        )
        updated += 1
    if updated:
        conn.commit()
        log.info("memory Phase B backfill: updated %d rows", updated)
    return updated


def _active_sql(active_only: bool) -> str:
    if not active_only:
        return ""
    return " AND (m.status = 'active' OR m.status IS NULL)"


def install_phase_a_hooks() -> None:
    """Patch memory.memorize write/recall paths once (idempotent)."""
    global _PHASE_A_INSTALLED
    if _PHASE_A_INSTALLED:
        return
    try:
        from memory import memorize as m
    except Exception as e:
        log.debug("Phase A hooks deferred (memorize not importable): %s", e)
        return

    if getattr(m, "_PHASE_A_HOOKS", False):
        _PHASE_A_INSTALLED = True
        return

    import sqlite_vec
    from memory.entities import classify_kind, extract_entities

    def _sqlite_knn_search(
        conn: sqlite3.Connection,
        vector: list[float],
        user_id: str,
        limit: int,
        threshold: float | None = None,
        active_only: bool = True,
    ):
        vec_blob = sqlite_vec.serialize_float32(vector)
        status_sql = _active_sql(active_only)
        if threshold is not None:
            dist_ceil = 1.0 - threshold
            return conn.execute(
                f"""
                SELECT v.id, vec_distance_cosine(v.embedding, ?) AS dist
                FROM memories_vec v
                JOIN memories m ON m.id = v.id
                WHERE m.user_id = ?
                  AND vec_distance_cosine(v.embedding, ?) <= ?
                  {status_sql}
                ORDER BY dist ASC
                LIMIT ?
                """,
                (vec_blob, user_id, vec_blob, dist_ceil, limit),
            ).fetchall()
        return conn.execute(
            f"""
            SELECT v.id, vec_distance_cosine(v.embedding, ?) AS dist
            FROM memories_vec v
            JOIN memories m ON m.id = v.id
            WHERE m.user_id = ?
              {status_sql}
            ORDER BY dist ASC
            LIMIT ?
            """,
            (vec_blob, user_id, limit),
        ).fetchall()

    m._sqlite_knn_search = _sqlite_knn_search

    _Backend = m._MemoryBackend

    def _fts_pass(self, fts_query, user_id, fts_limit, active_only=True):
        if fts_query is None:
            return []
        status_sql = _active_sql(active_only)
        return self._conn.execute(
            f"""
            SELECT f.id
            FROM memories_fts f
            JOIN memories m ON m.id = f.id
            WHERE memories_fts MATCH ?
              AND m.user_id = ?
              {status_sql}
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, user_id, fts_limit),
        ).fetchall()

    def _insert_row(
        self,
        *,
        mem_id: str,
        user_id: str,
        text: str,
        now: str,
        vector: list[float],
        pinned: int = 0,
        source: str = SOURCE_CHAT,
        supersedes_id: str | None = None,
        kind: str | None = None,
        entities: list[str] | None = None,
    ) -> None:
        cols = existing_columns(self._conn)
        kind_val = kind or classify_kind(text, default=KIND_FACT)
        ents_json = entities_to_json(entities if entities is not None else extract_entities(text))
        if "status" in cols:
            self._conn.execute(
                """
                INSERT INTO memories
                    (id, user_id, memory, created_at, access_count, last_accessed_at, pinned,
                     status, supersedes_id, kind, source, entities)
                VALUES (?, ?, ?, ?, 0, 'never', ?, ?, ?, ?, ?, ?)
                """,
                (
                    mem_id, user_id, text, now, pinned,
                    STATUS_ACTIVE, supersedes_id, kind_val, source, ents_json,
                ),
            )
        else:
            self._conn.execute(
                """
                INSERT INTO memories
                    (id, user_id, memory, created_at, access_count, last_accessed_at, pinned)
                VALUES (?, ?, ?, ?, 0, 'never', ?)
                """,
                (mem_id, user_id, text, now, pinned),
            )
        self._conn.execute(
            "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
            (mem_id, sqlite_vec.serialize_float32(vector)),
        )

    def _maybe_supersede_neighbor(
        self, user_id: str, vector: list[float], text: str
    ) -> tuple[str, str | None]:
        existing = m._sqlite_knn_search(
            self._conn, vector, user_id,
            limit=1, threshold=m.WRITE_DEDUP_THRESHOLD, active_only=True,
        )
        if not existing:
            return "add", None
        sim = 1.0 - float(existing[0]["dist"])
        old_id = str(existing[0]["id"])
        row = self._conn.execute(
            "SELECT memory, pinned FROM memories WHERE id = ?", (old_id,)
        ).fetchone()
        old_text = (row["memory"] if row else "") or ""
        pinned = bool(row and row["pinned"])
        op = classify_write_op(
            similarity=sim,
            new_text=text,
            old_text=old_text,
            dedup_threshold=m.WRITE_DEDUP_THRESHOLD,
        )
        if op == "supersede" and pinned:
            return "add", None
        if op == "supersede":
            return "supersede", old_id
        return op, None

    def add(self, messages, user_id, display_name=None):
        facts = self._extract_facts(messages, display_name=display_name)
        if not facts:
            return []
        from system import bioclock

        now = bioclock.local_now()
        if not isinstance(now, str):
            now = now.isoformat() if hasattr(now, "isoformat") else str(now)
        ids: list[str] = []
        try:
            vectors = self._embed_batch(facts)
        except Exception as e:
            log.warning("Batch embedding failed, aborting write: %s", e)
            return []

        with self._db_lock:
            try:
                ensure_phase_a_schema(self._conn)
                for fact, vector in zip(facts, vectors):
                    op, supersedes_id = _maybe_supersede_neighbor(self, user_id, vector, fact)
                    if op == "noop":
                        log.debug("Skipping near-duplicate fact: %r", fact)
                        continue
                    if op == "supersede" and supersedes_id:
                        cols = existing_columns(self._conn)
                        if "status" in cols:
                            self._conn.execute(
                                "UPDATE memories SET status = ? WHERE id = ?",
                                (STATUS_SUPERSEDED, supersedes_id),
                            )
                            log.info("Superseded memory %s with new fact", supersedes_id)
                    mem_id = str(uuid.uuid4())
                    _insert_row(
                        self,
                        mem_id=mem_id,
                        user_id=user_id,
                        text=fact,
                        now=now,
                        vector=vector,
                        pinned=0,
                        source=SOURCE_CHAT,
                        supersedes_id=supersedes_id,
                    )
                    ids.append(mem_id)
                self._conn.commit()
            except Exception as e:
                log.warning("Failed to upsert fact batch: %s", e)
                self._conn.rollback()
                return []
        return ids

    def add_raw(self, memory, user_id, *, pinned=False):
        text = (memory or "").strip()
        if not text:
            return None
        with self._db_lock:
            try:
                ensure_phase_a_schema(self._conn)
                vector = self._embed(text)
                op, supersedes_id = _maybe_supersede_neighbor(self, user_id, vector, text)
                if op == "noop":
                    log.debug("Skipping near-duplicate raw memory: %r", text[:80])
                    return None
                if op == "supersede" and supersedes_id:
                    cols = existing_columns(self._conn)
                    if "status" in cols:
                        self._conn.execute(
                            "UPDATE memories SET status = ? WHERE id = ?",
                            (STATUS_SUPERSEDED, supersedes_id),
                        )
                mem_id = str(uuid.uuid4())
                now = datetime.now(timezone.utc).isoformat()
                _insert_row(
                    self,
                    mem_id=mem_id,
                    user_id=user_id,
                    text=text,
                    now=now,
                    vector=vector,
                    pinned=1 if pinned else 0,
                    source=SOURCE_PIN if pinned else SOURCE_CHAT,
                    supersedes_id=supersedes_id,
                )
                self._conn.commit()
                return mem_id
            except Exception as e:
                log.warning("Failed to insert raw memory: %s", e)
                self._conn.rollback()
                return None

    def search(self, query, user_id, limit=5, vector=None, include_history=False):
        active_only = not include_history
        if vector is None:
            vector = self._embed(query, query=True)
        fts_query = m._sanitize_fts_query(query)

        with self._db_lock:
            quick_knn_rows = m._sqlite_knn_search(
                self._conn, vector, user_id, m.QUICK_KNN_LIMIT, active_only=active_only
            )
        rank_knn_q = {row["id"]: i + 1 for i, row in enumerate(quick_knn_rows)}
        quick_fts_rows = _fts_pass(self, fts_query, user_id, m.QUICK_FTS_LIMIT, active_only=active_only)
        rank_fts_q = {row["id"]: i + 1 for i, row in enumerate(quick_fts_rows)}

        scored_ids, scores, row_by_id = self._rank_and_score(rank_knn_q, rank_fts_q)

        confident = (
            len(scored_ids) >= limit
            and scores.get(scored_ids[limit - 1], 0.0) >= m.MEMORY_RECALL_SCORE_THRESHOLD
        )
        if not confident:
            wide_knn_rows = m._sqlite_knn_search(
                self._conn, vector, user_id, m.KNN_LIMIT, active_only=active_only
            )
            rank_knn_w = {row["id"]: i + 1 for i, row in enumerate(wide_knn_rows)}
            wide_fts_rows = _fts_pass(self, fts_query, user_id, m.FTS_LIMIT, active_only=active_only)
            rank_fts_w = {row["id"]: i + 1 for i, row in enumerate(wide_fts_rows)}
            scored_ids, scores, row_by_id = self._rank_and_score(rank_knn_w, rank_fts_w)

        ordered_ids = self._apply_recency_rerank(scored_ids, scores, row_by_id)
        top_ids = ordered_ids[:limit]
        results = []
        for mid in top_ids:
            if mid not in row_by_id:
                continue
            d = dict(row_by_id[mid])
            d["_recall_score"] = scores.get(mid, 0.0)
            results.append(d)
        return results

    _Backend._fts_pass = _fts_pass
    _Backend.add = add
    _Backend.add_raw = add_raw
    _Backend.search = search

    def public_search(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 5,
        query_vector: list[float] | None = None,
        include_history: bool = False,
    ):
        user_id = self._resolve_user_id(user_id)
        if m._is_trivial_input(query or ""):
            log.debug("Skipping search for trivial input: %r", query)
            return []

        if m._BROAD_RECALL_RE.search(query or ""):
            results = self._recent_or_important_memories(user_id=user_id, limit=limit)
            if not include_history:
                results = [
                    r for r in results
                    if str(r.get("status") or STATUS_ACTIVE) == STATUS_ACTIVE
                ]
            self._touch_memories(results)
            return results[: int(limit)]

        cache_key = (
            user_id,
            " ".join((query or "").lower().split()),
            int(limit),
            bool(include_history),
        )
        import time

        now_s = time.monotonic()
        with self._search_cache_lock:
            cached = self._search_cache.get(cache_key)
            if cached and now_s - cached[0] <= m.MEMORY_SEARCH_CACHE_TTL:
                self._search_cache.move_to_end(cache_key)
                results = [dict(r) for r in cached[1]]
                self._touch_memories(results)
                return results
            if cached:
                self._search_cache.pop(cache_key, None)

        results = self._mem.search(
            query,
            user_id=user_id,
            limit=limit,
            vector=query_vector,
            include_history=include_history,
        )
        self._touch_memories(results)
        with self._search_cache_lock:
            self._search_cache[cache_key] = (now_s, [dict(r) for r in results])
            while len(self._search_cache) > m.MEMORY_SEARCH_CACHE_SIZE:
                self._search_cache.popitem(last=False)
        return results

    m.AikoMemorize.search = public_search
    m._PHASE_A_HOOKS = True
    _PHASE_A_INSTALLED = True
    log.info("memory Phase A/B hooks installed (active recall + supersede + entities)")
