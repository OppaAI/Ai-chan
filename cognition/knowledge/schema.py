"""Schema helpers for the knowledge store."""

from __future__ import annotations

from .backend import _connect, _ensure_knowledge_schema_migrated, vacuum_knowledge_db

__all__ = ["_connect", "_ensure_knowledge_schema_migrated", "vacuum_knowledge_db"]
