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
    Legacy free-function shim: sole implementation that
    :class:`KnowledgeLifecycle.prune` delegates to. Kept for the historical
    import path.

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


def _get_dedupe_cursor(conn: sqlite3.Connection, user_id: str) -> str:
    row = conn.execute(
        "SELECT last_id FROM knowledge_prune_meta WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    return str(row["last_id"]) if row and row["last_id"] is not None else ""


def _set_dedupe_cursor(conn: sqlite3.Connection, user_id: str, last_id: str) -> None:
    now = utc_now_iso()
    conn.execute(
        """
        INSERT INTO knowledge_prune_meta(user_id, last_id, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            last_id = excluded.last_id,
            updated_at = excluded.updated_at
        """,
        (user_id, last_id, now),
    )


def _deduplicate_chunks(
    conn: sqlite3.Connection,
    user_id: str,
    embedder,
    threshold: float,
) -> int:
    """Find near-duplicate active chunks via embedding similarity and archive dups.

    Scans in keyset pages ordered by id so each prune advances through the
    store; when the page is empty the cursor resets to the start.
    """
    MAX_CANDIDATES = 1000

    cursor = _get_dedupe_cursor(conn, user_id)

    rows = conn.execute(
        """
        SELECT id, text, chunk_index, access_count, created_at
        FROM learned_chunks
        WHERE user_id = ?
          AND (status = 'active' OR status IS NULL)
          AND id NOT IN (
              SELECT id FROM learned_chunks_archive WHERE user_id = ?
          )
          AND id > ?
        ORDER BY id
        LIMIT ?
        """,
        (user_id, user_id, cursor, MAX_CANDIDATES),
    ).fetchall()

    # End of store → reset for next prune
    if not rows:
        _set_dedupe_cursor(conn, user_id, "")
        return 0

    # Advance cursor past this page (even if no dups found)
    page_last_id = str(rows[-1]["id"])
    _set_dedupe_cursor(conn, user_id, page_last_id)

    if len(rows) < 2:
        return 0

    texts = [row["text"] for row in rows]
    ids = [row["id"] for row in rows]
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

    to_archive: set[str] = set()
    n = len(ids)
    for i in range(n):
        if ids[i] in to_archive:
            continue
        for j in range(i + 1, n):
            if ids[j] in to_archive:
                continue
            if sim[i, j] < threshold:
                continue
            # Keep higher access_count; on tie keep earlier created_at
            if access_counts[i] > access_counts[j]:
                to_archive.add(ids[j])
            elif access_counts[i] < access_counts[j]:
                to_archive.add(ids[i])
                break  # i is out; stop pairing against it
            else:
                if created_ats[i] <= created_ats[j]:
                    to_archive.add(ids[j])
                else:
                    to_archive.add(ids[i])
                    break

    if not to_archive:
        return 0

    now = utc_now_iso()
    for cid in to_archive:
        row = conn.execute(
            "SELECT * FROM learned_chunks WHERE id = ? AND user_id = ?",
            (cid, user_id),
        ).fetchone()
        if not row:
            continue
        conn.execute(
            """
            INSERT INTO learned_chunks_archive
            (id, doc_id, user_id, chunk_index, text, created_at,
             access_count, last_accessed, archived_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                row["doc_id"],
                row["user_id"],
                row["chunk_index"],
                row["text"],
                row["created_at"],
                row["access_count"],
                row["last_accessed"],
                now,
            ),
        )
        conn.execute(
            "DELETE FROM learned_chunks WHERE id = ? AND user_id = ?",
            (cid, user_id),
        )

    return len(to_archive)
