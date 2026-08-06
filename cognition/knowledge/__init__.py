"""Public knowledge-store façade.

Implementation is split by responsibility:

- :mod:`.schema` — connection, DDL, migrations, vacuum
- :mod:`.ingest` — file/text/workspace ingestion
- :mod:`.search` — hybrid retrieval, cache, context formatting
- :mod:`.lifecycle` — prune / archive / dedupe

``KnowledgeStore`` is the shared coordinator. Module-level functions preserve
the historical import path used by think, agentic, tests, and studio.
"""
from __future__ import annotations

from .schema import (
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
    KnowledgeSchema,
    _connect,
    _ensure_knowledge_schema_migrated,
    _now,
    vacuum_knowledge_db,
)
from .ingest import (
    KnowledgeIngest,
    extract_text_from_file,
    ingest_file,
    ingest_text,
    ingest_workspace_knowledge_folder,
)
from .search import (
    KnowledgeSearch,
    knowledge_context_for,
    search_knowledge,
    _maybe_clear_knowledge_cache,
)
from .lifecycle import (
    KnowledgeLifecycle,
    prune_knowledge,
)


class KnowledgeStore:
    """Shared backend coordinator: schema + ingest + search + lifecycle."""

    def __init__(self, user_id: str | None = None, embedder: Embedder | None = None):
        self.user_id = user_id
        self.embedder = embedder
        self.schema = KnowledgeSchema()
        self.ingest = KnowledgeIngest(self.schema, embedder)
        self.search = KnowledgeSearch(self.schema, embedder)
        self.lifecycle = KnowledgeLifecycle(self.schema, embedder)

    def extract_text_from_file(self, relative_path: str, **kwargs):
        kwargs.setdefault("user_id", self.user_id)
        return self.ingest.extract_text_from_file(relative_path, **kwargs)

    def ingest_file(self, relative_path: str, **kwargs):
        kwargs.setdefault("user_id", self.user_id)
        return self.ingest.ingest_file(relative_path, **kwargs)

    def ingest_text(self, title: str, text: str, **kwargs):
        kwargs.setdefault("user_id", self.user_id)
        return self.ingest.ingest_text(title, text, **kwargs)

    def ingest_workspace_knowledge_folder(self, **kwargs):
        kwargs.setdefault("user_id", self.user_id)
        return self.ingest.ingest_workspace_knowledge_folder(**kwargs)

    def search_knowledge(self, query: str, limit: int = 5, **kwargs):
        kwargs.setdefault("user_id", self.user_id)
        return self.search.search(query, limit=limit, **kwargs)

    def knowledge_context_for(self, query: str, limit: int = 5, **kwargs):
        kwargs.setdefault("user_id", self.user_id)
        return self.search.context_for(query, limit=limit, **kwargs)

    def prune_knowledge(self, **kwargs):
        kwargs.setdefault("user_id", self.user_id)
        return self.lifecycle.prune(**kwargs)

    def vacuum_knowledge_db(self, user_id: str | None = None):
        return self.lifecycle.vacuum(user_id or self.user_id)


__all__ = [
    "connect",
    "EMBED_DIMS",
    "Embedder",
    "ensure_knowledge_schema_migrated",
    "extract_text_from_file",
    "ingest_file",
    "ingest_text",
    "ingest_workspace_knowledge_folder",
    "KNOWLEDGE_CHUNK_CHARS",
    "KNOWLEDGE_CONTEXT_CHARS",
    "knowledge_context_for",
    "KNOWLEDGE_DB_PATH",
    "KNOWLEDGE_ENTITY_BOOST",
    "KNOWLEDGE_FTS_LIMIT",
    "KNOWLEDGE_KNN_LIMIT",
    "KNOWLEDGE_KNN_MIN_SIMILARITY",
    "KNOWLEDGE_QUERY_INSTRUCT",
    "KNOWLEDGE_RECALL_SCORE_THRESHOLD",
    "KNOWLEDGE_RRF_K",
    "KNOWLEDGE_SPREADING_ENABLED",
    "KNOWLEDGE_SPREADING_MAX_EXTRA",
    "KNOWLEDGE_SPREADING_SCORE_WEIGHT",
    "KNOWLEDGE_SUPERSEDE_ON_DEDUP",
    "KNOWLEDGE_WORKSPACE_DIR",
    "KNOWLEDGE_WRITE_DEDUP_THRESHOLD",
    "KnowledgeIngest",
    "KnowledgeLifecycle",
    "KnowledgeSchema",
    "KnowledgeSearch",
    "KnowledgeStore",
    "maybe_clear_knowledge_cache",
    "now",   
    "prune_knowledge",
    "search_knowledge",
    "vacuum_knowledge_db",
]
