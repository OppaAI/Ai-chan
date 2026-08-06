"""Knowledge lifecycle — class façade over backend maintenance."""
from __future__ import annotations

from .backend import Embedder, prune_knowledge, vacuum_knowledge_db
from .schema import KnowledgeSchema


class KnowledgeLifecycle:
    """Owns prune / archive / dedupe maintenance for the knowledge store."""

    def __init__(self, schema: KnowledgeSchema | None = None, embedder: Embedder | None = None):
        self.schema = schema or KnowledgeSchema()
        self.embedder = embedder

    def prune(self, **kwargs):
        kwargs.setdefault("embedder", self.embedder)
        return prune_knowledge(**kwargs)

    def vacuum(self, user_id: str | None = None) -> None:
        vacuum_knowledge_db(user_id)


__all__ = [
    "KnowledgeLifecycle",
    "prune_knowledge",
    "vacuum_knowledge_db",
]
