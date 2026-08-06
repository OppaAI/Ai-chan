"""Knowledge ingestion — class façade over backend write path."""
from __future__ import annotations

from .backend import (
    Embedder,
    extract_text_from_file,
    ingest_file,
    ingest_text,
    ingest_workspace_knowledge_folder,
)
from .schema import KnowledgeSchema


class KnowledgeIngest:
    """Owns file extraction and learned-knowledge write path."""

    def __init__(self, schema: KnowledgeSchema | None = None, embedder: Embedder | None = None):
        self.schema = schema or KnowledgeSchema()
        self.embedder = embedder

    def extract_text_from_file(self, relative_path: str, **kwargs):
        return extract_text_from_file(relative_path, **kwargs)

    def ingest_file(self, relative_path: str, **kwargs):
        kwargs.setdefault("embedder", self.embedder)
        return ingest_file(relative_path, **kwargs)

    def ingest_text(self, title: str, text: str, **kwargs):
        kwargs.setdefault("embedder", self.embedder)
        return ingest_text(title, text, **kwargs)

    def ingest_workspace_knowledge_folder(self, **kwargs):
        kwargs.setdefault("embedder", self.embedder)
        return ingest_workspace_knowledge_folder(**kwargs)


__all__ = [
    "KnowledgeIngest",
    "extract_text_from_file",
    "ingest_file",
    "ingest_text",
    "ingest_workspace_knowledge_folder",
]
