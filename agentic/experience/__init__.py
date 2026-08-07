"""Public experience-store façade.

Implementation is split by responsibility:

- :mod:`.schema` — connection, DDL, migrations, sanitization, constants
- :mod:`.write` — experience record write path (insert/embed/auto-relate/prune)
- :mod:`.search` — hybrid retrieval, spreading, context formatting
- :mod:`.lifecycle` — engram-relation graph

``ExperienceStore`` is the shared coordinator. Module-level functions preserve
the historical import path used by think, agentic, practice, tests, and studio.
The façade intentionally also re-exports legacy private helpers (``_connect``,
``_knn``, ``_fts``, ``_sanitize``, ``_now``, ``_prune``) because tests and studio
diagnostics import them directly.
"""
from __future__ import annotations

from .lifecycle import (
    ExperienceLifecycle,
    RELATION_TYPES,
    get_engram_relations,
    record_engram_relation,
)
from .schema import (
    EMBED_DIMS,
    EXPERIENCE_AUTO_RELATE_THRESHOLD,
    EXPERIENCE_CONTEXT_CHARS,
    EXPERIENCE_DB_PATH,
    EXPERIENCE_ENTITY_BOOST,
    EXPERIENCE_FTS_LIMIT,
    EXPERIENCE_KNN_LIMIT,
    EXPERIENCE_MAX_ROWS,
    EXPERIENCE_QUERY_INSTRUCT,
    EXPERIENCE_RECALL_SCORE_THRESHOLD,
    EXPERIENCE_RRF_K,
    EXPERIENCE_SPREADING_ENABLED,
    EXPERIENCE_SPREADING_MAX_EXTRA,
    EXPERIENCE_SPREADING_SCORE_WEIGHT,
    EXPERIENCE_SUPERSEDE_ON_NEAR_DUP,
    EXPERIENCE_SUPERSEDE_THRESHOLD,
    Embedder,
    ExperienceSchema,
    ExperienceStep,
    connect,
    ensure_experience_schema_migrated,
    now,
    sanitize,
    _connect,
    _DDL,
    _sanitize,
    _SECRET_RE,
    _now,
)
from .search import (
    ExperienceSearch,
    _experience_spread_extra,
    _attr,
    _fts,
    _knn,
    experience_context_for,
    search_experience,
)
from .write import (
    ExperienceWriter,
    _prune,
    record_experience,
    record_practice_experience,
)


class ExperienceStore:
    """Shared coordinator: schema + write + search + lifecycle."""

    def __init__(self, user_id: str | None = None):
        self.user_id = user_id
        self.schema = ExperienceSchema()
        self.writer = ExperienceWriter(self.schema)
        self.search = ExperienceSearch(self.schema)
        self.lifecycle = ExperienceLifecycle(self.schema)

    def record(self, goal: str, steps: list[dict], final_answer: str, verified_ok: bool, score: float, embedder=None, **kwargs) -> str | None:
        kwargs.setdefault("user_id", None)
        return self.writer.record(None, goal, steps, final_answer, verified_ok, score, embedder=embedder)

    def search(self, query: str, limit: int = 3, embedder=None, **kwargs) -> list[dict]:
        kwargs.setdefault("user_id", self.user_id)
        return self.search.search(query, limit=limit, embedder=embedder, **kwargs)

    def context_for(self, query: str, limit: int = 3, embedder=None) -> str:
        return self.search.context_for(query, limit=limit, embedder=embedder)

    def record_relation(self, from_engram: str, to_engram: str, relation_type: str, confidence: float = 1.0, **kwargs) -> bool:
        kwargs.setdefault("user_id", self.user_id)
        return self.lifecycle.record_relation(from_engram, to_engram, relation_type, confidence, **kwargs)

    def get_relations(self, engram_id: str, direction: str = "both", **kwargs) -> list[dict]:
        kwargs.setdefault("user_id", self.user_id)
        return self.lifecycle.get_relations(engram_id, direction=direction, **kwargs)


__all__ = [
    "EMBED_DIMS",
    "Embedder",
    "EXPERIENCE_AUTO_RELATE_THRESHOLD",
    "EXPERIENCE_CONTEXT_CHARS",
    "EXPERIENCE_DB_PATH",
    "EXPERIENCE_ENTITY_BOOST",
    "EXPERIENCE_FTS_LIMIT",
    "EXPERIENCE_KNN_LIMIT",
    "EXPERIENCE_MAX_ROWS",
    "EXPERIENCE_QUERY_INSTRUCT",
    "EXPERIENCE_RECALL_SCORE_THRESHOLD",
    "EXPERIENCE_RRF_K",
    "EXPERIENCE_SPREADING_ENABLED",
    "EXPERIENCE_SPREADING_MAX_EXTRA",
    "EXPERIENCE_SPREADING_SCORE_WEIGHT",
    "EXPERIENCE_SUPERSEDE_ON_NEAR_DUP",
    "EXPERIENCE_SUPERSEDE_THRESHOLD",
    "ExperienceLifecycle",
    "ExperienceSchema",
    "ExperienceSearch",
    "ExperienceStep",
    "ExperienceStore",
    "ExperienceWriter",
    "RELATION_TYPES",
    "_attr",
    "_connect",
    "_DDL",
    "_experience_spread_extra",
    "_fts",
    "_knn",
    "_now",
    "_prune",
    "_sanitize",
    "_SECRET_RE",
    "connect",
    "ensure_experience_schema_migrated",
    "experience_context_for",
    "get_engram_relations",
    "now",
    "record_engram_relation",
    "record_experience",
    "record_practice_experience",
    "sanitize",
    "search_experience",
]