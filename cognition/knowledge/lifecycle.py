"""Knowledge maintenance: prune, archive, dedupe."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import numpy as np

from cognition.memory.vecstore import utc_now_iso
from system.log import get_logger
from system.userspace import current_user_id

from .schema import connect, Embedder, KnowledgeSchema, vacuum_knowledge_db

log = get_logger(__name__)


class KnowledgeLifecycle:
    """Owns prune / archive / dedupe maintenance for the knowledge store."""

    def __init__(self, schema: KnowledgeSchema | None = None, embedder: Embedder | None = None):
        self.schema = schema or KnowledgeSchema()
        self.embedder = embedder

    def prune(
        self,
        *,
        min_access: int = 2,
        archive_days: int = 90,
        delete_days: int = 180,
        dedupe_threshold: float = 0.95,
        user_id: str | None = None,
        embedder=None,
    ) -> dict:
        return prune_knowledge(
            min_access=min_access,
            archive_days=archive_days,
            delete_days=delete_days,
            dedupe_threshold=dedupe_threshold,
            user_id=user_id,
            embedder=embedder if embedder is not None else self.embedder,
        )

    def vacuum(self, user_id: str | None = None) -> None:
        vacuum_knowledge_db(user_id)


def prune_knowledge(
    *,
    min_access: int = 2,
    archive_days: int = 90,
    delete_days: int = 180,
    dedupe_threshold: float = 0.95,
    user_id: str | None = None,
    embedder=None,
) -> dict:
    """
    Prune knowledge DB: archive cold chunks, delete never-accessed old chunks,
    deduplicate near-duplicates. Returns stats dict.
    """
    uid = user_id or current_user_id()
    conn = connect(uid)
    stats = {"archived": 0, "deleted": 0, "deduped": 0, "errors": 0}
    now = utc_now_iso()
    try:
        # 1. Archive: move cold chunks (old + low access) to archive table
        archive_cutoff = (datetime.fromisoformat(now.replace('Z', '+00:00')) - timedelta(days=archive_days)).isoformat()
        archive_cursor = conn.execute(
            """
            INSERT INTO learned_chunks_archive
            (id, doc_id, user_id, chunk_index, text, created_at, access_count, last_accessed, archived_at)
            SELECT id, doc_id, user_id, chunk_index, text, created_at, access_count, last_accessed, ?
            FROM learned_chunks
            WHERE user_id = ?
              AND created_at < ?
              AND access_count < ?
              AND id NOT IN (SELECT id FROM learned_chunks_archive WHERE user_id = ?)
            """,
            (now, uid, archive_cutoff, min_access, uid),
        )
        stats["archived"] = archive_cursor.rowcount

        # Delete archived from main table
        conn.execute(
            "DELETE FROM learned_chunks WHERE id IN (SELECT id FROM learned_chunks_archive WHERE user_id = ? AND archived_at = ?)",
            (uid, now),
        )

        # 2. Delete: remove never-accessed chunks older than delete_days
        delete_cutoff = (datetime.fromisoformat(now.replace('Z', '+00:00')) - timedelta(days=delete_days)).isoformat()
        delete_cursor = conn.execute(
            """
            DELETE FROM learned_chunks
            WHERE user_id = ?
              AND created_at < ?
              AND access_count = 0
              AND id NOT IN (SELECT id FROM learned_chunks_archive WHERE user_id = ?)
            """,
            (uid, delete_cutoff, uid),
        )
        stats["deleted"] = delete_cursor.rowcount

        # 3. Deduplicate: find near-duplicate chunks via embedding similarity
        if embedder is not None:
            stats["deduped"] = _deduplicate_chunks(conn, uid, embedder, dedupe_threshold)

        conn.commit()
    except Exception as exc:
        conn.rollback()
        log.warning("prune_knowledge failed: %s", exc)
        stats["errors"] = 1
    finally:
        conn.close()
    return stats


def _deduplicate_chunks(conn: sqlite3.Connection, user_id: str, embedder, threshold: float) -> int:
    """Find near-duplicate chunks via embedding similarity and archive duplicates."""
    MAX_CANDIDATES = 1000
    rows = conn.execute(
        """
        SELECT id, text, chunk_index, access_count, created_at FROM learned_chunks
        WHERE user_id = ?
          AND (status = 'active' OR status IS NULL)
          AND id NOT IN (SELECT id FROM learned_chunks_archive WHERE user_id = ?)
        ORDER BY chunk_index
        LIMIT ?
        """,
        (user_id, user_id, MAX_CANDIDATES),
    ).fetchall()

    if len(rows) < 2:
        return 0

    texts = [row["text"] for row in rows]
    ids = [row["id"] for row in rows]
    indices = [row["chunk_index"] for row in rows]
    access_counts = [row["access_count"] for row in rows]
    created_ats = [row["created_at"] for row in rows]

    try:
        if hasattr(embedder, "embed_queries"):
            batch = embedder.embed_queries(texts)
        else:
            batch = [embedder.embed_query(t) for t in texts]
        vectors = np.array(batch, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms
        sim = vectors @ vectors.T
    except Exception as exc:
        log.warning("Deduplication embedding/similarity failed: %s", exc)
        return 0

    to_archive = set()
    n = len(ids)
    for i in range(n):
        if ids[i] in to_archive:
            continue
        for j in range(i + 1, n):
            if ids[j] in to_archive:
                continue
            if sim[i, j] >= threshold:
                # Retain the chunk with higher access_count; on tie, retain earlier created_at
                if access_counts[i] > access_counts[j]:
                    to_archive.add(ids[j])
                elif access_counts[i] < access_counts[j]:
                    to_archive.add(ids[i])
                    break
                else:
                    if created_ats[i] <= created_ats[j]:
                        to_archive.add(ids[j])
                    else:
                        to_archive.add(ids[i])
                        break

    if to_archive:
        now = utc_now_iso()
        for cid in to_archive:
            row = conn.execute("SELECT * FROM learned_chunks WHERE id = ?", (cid,)).fetchone()
            if row:
                conn.execute(
                    """
                    INSERT INTO learned_chunks_archive
                    (id, doc_id, user_id, chunk_index, text, created_at, access_count, last_accessed, archived_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (row["id"], row["doc_id"], row["user_id"], row["chunk_index"],
                     row["text"], row["created_at"], row["access_count"],
                     row["last_accessed"], now),
                )
                conn.execute("DELETE FROM learned_chunks WHERE id = ?", (cid,))

    return len(to_archive)

