"""Public memory API façade.

Most implementation details live in focused modules.  This module keeps the
stable import path used by the rest of the application:

    from cognition.memory.memorize import AikoMemorize, format_for_context, infer_valence_score
"""

from __future__ import annotations

from .backend import (
    AikoMemorize,
    BOOT_LABELS,
    EMBED_DIMS,
    _MemoryBackend,
    backfill_entities,
    classify_kind,
    classify_write_op,
    entities_from_json,
    entities_to_json,
    entity_overlap_score,
    ensure_entity_relations_schema,
    ensure_l2_scene_schema,
    ensure_phase_a_schema,
    existing_columns,
    extract_entities,
    infer_salience_hit,
    infer_valence_score,
    infer_valence_tag,
    normalize_memory_text,
    rebuild_entity_relations,
    tag_from_score,
    upsert_co_mentions,
    vacuum_memory_db,
)

format_for_context = AikoMemorize.format_for_context

__all__ = [
    "AikoMemorize",
    "BOOT_LABELS",
    "EMBED_DIMS",
    "_MemoryBackend",
    "backfill_entities",
    "classify_kind",
    "classify_write_op",
    "entities_from_json",
    "entities_to_json",
    "entity_overlap_score",
    "ensure_entity_relations_schema",
    "ensure_l2_scene_schema",
    "ensure_phase_a_schema",
    "existing_columns",
    "extract_entities",
    "format_for_context",
    "infer_salience_hit",
    "infer_valence_score",
    "infer_valence_tag",
    "normalize_memory_text",
    "rebuild_entity_relations",
    "tag_from_score",
    "upsert_co_mentions",
    "vacuum_memory_db",
]
