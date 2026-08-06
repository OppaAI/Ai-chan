"""Knowledge search — class façade over backend retrieval."""
from __future__ import annotations

from .backend import (
    Embedder,
    knowledge_context_for,
    search_knowledge,
    _maybe_clear_knowledge_cache,
)
from .schema import KnowledgeSchema


class KnowledgeSearch:
    """Owns hybrid retrieval, spreading, context formatting, and access tracking."""

    def __init__(self, schema: KnowledgeSchema | None = None, embedder: Embedder | None = None):
        self.schema = schema or KnowledgeSchema()
        self.embedder = embedder

    def search(self, query: str, limit: int = 5, **kwargs):
        kwargs.setdefault("embedder", self.embedder)
        return search_knowledge(query, limit=limit, **kwargs)

    def context_for(self, query: str, limit: int = 5, **kwargs):
        kwargs.setdefault("embedder", self.embedder)
        return knowledge_context_for(query, limit=limit, **kwargs)


__all__ = [
    "KnowledgeSearch",
    "search_knowledge",
    "knowledge_context_for",
    "_maybe_clear_knowledge_cache",
]
