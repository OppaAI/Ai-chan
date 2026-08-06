"""Schema helpers — class façade over backend connection/DDL."""
from __future__ import annotations

from .backend import (
    EMBED_DIMS,
    KNOWLEDGE_CHUNK_CHARS,
    KNOWLEDGE_CONTEXT_CHARS,
    KNOWLEDGE_DB_PATH,
    KNOWLEDGE_ENTITY_BOOST,
    KNOWLEDGE_FTS_LIMIT,
    KNOWLEDGE_KNN_LIMIT,
    KNOWLEDGE_KNN_MIN_SIMILARITY,
    KNOWLEDGE_QUERY_INSTRUCT,
    KNOWLEDGE_RECALL_SCORE_THRESHOLD,
    KNOWLEDGE_RRF_K,
    KNOWLEDGE_SPREADING_ENABLED,
    KNOWLEDGE_SPREADING_MAX_EXTRA,
    KNOWLEDGE_SPREADING_SCORE_WEIGHT,
    KNOWLEDGE_SUPERSEDE_ON_DEDUP,
    KNOWLEDGE_WORKSPACE_DIR,
    KNOWLEDGE_WRITE_DEDUP_THRESHOLD,
    Embedder,
    _connect,
    _ensure_knowledge_schema_migrated,
    _now,
    vacuum_knowledge_db,
)


class KnowledgeSchema:
    """Owns DB connection, DDL, and schema migrations for learned knowledge."""

    def connect(self, user_id: str | None = None):
        return _connect(user_id)

    def ensure_migrated(self, conn, user_id: str | None = None) -> None:
        _ensure_knowledge_schema_migrated(conn, user_id)

    def vacuum(self, user_id: str | None = None) -> None:
        vacuum_knowledge_db(user_id)


__all__ = [
    "KnowledgeSchema",
    "Embedder",
    "_connect",
    "_ensure_knowledge_schema_migrated",
    "_now",
    "vacuum_knowledge_db",
    "EMBED_DIMS",
    "KNOWLEDGE_DB_PATH",
    "KNOWLEDGE_CHUNK_CHARS",
    "KNOWLEDGE_CONTEXT_CHARS",
    "KNOWLEDGE_RRF_K",
    "KNOWLEDGE_KNN_LIMIT",
    "KNOWLEDGE_FTS_LIMIT",
    "KNOWLEDGE_RECALL_SCORE_THRESHOLD",
    "KNOWLEDGE_KNN_MIN_SIMILARITY",
    "KNOWLEDGE_QUERY_INSTRUCT",
    "KNOWLEDGE_WORKSPACE_DIR",
    "KNOWLEDGE_ENTITY_BOOST",
    "KNOWLEDGE_WRITE_DEDUP_THRESHOLD",
    "KNOWLEDGE_SUPERSEDE_ON_DEDUP",
    "KNOWLEDGE_SPREADING_ENABLED",
    "KNOWLEDGE_SPREADING_MAX_EXTRA",
    "KNOWLEDGE_SPREADING_SCORE_WEIGHT",
]
