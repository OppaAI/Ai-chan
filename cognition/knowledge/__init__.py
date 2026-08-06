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
        return self.ingest.extract_text_from_file(relative_path, **kwargs)

    def ingest_file(self, relative_path: str, **kwargs):
        return self.ingest.ingest_file(relative_path, **kwargs)

    def ingest_text(self, title: str, text: str, **kwargs):
        return self.ingest.ingest_text(title, text, **kwargs)

    def ingest_workspace_knowledge_folder(self, **kwargs):
        return self.ingest.ingest_workspace_knowledge_folder(**kwargs)

    def search_knowledge(self, query: str, limit: int = 5, **kwargs):
        return self.search.search(query, limit=limit, **kwargs)

    def knowledge_context_for(self, query: str, limit: int = 5, **kwargs):
        return self.search.context_for(query, limit=limit, **kwargs)

    def prune_knowledge(self, **kwargs):
        return self.lifecycle.prune(**kwargs)

    def vacuum_knowledge_db(self, user_id: str | None = None):
        return self.lifecycle.vacuum(user_id)


__all__ = [
    "KnowledgeStore",
    "KnowledgeSchema",
    "KnowledgeIngest",
    "KnowledgeSearch",
    "KnowledgeLifecycle",
    "Embedder",
    "extract_text_from_file",
    "ingest_file",
    "ingest_text",
    "ingest_workspace_knowledge_folder",
    "search_knowledge",
    "knowledge_context_for",
    "prune_knowledge",
    "vacuum_knowledge_db",
    "_connect",
    "_ensure_knowledge_schema_migrated",
    "_now",
    "_maybe_clear_knowledge_cache",
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
