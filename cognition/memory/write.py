"""Write-path helpers for memory extraction and persistence."""

from __future__ import annotations

from .backend import (
    _MemoryBackend,
    classify_kind,
    classify_write_op,
    entities_from_json,
    entities_to_json,
    extract_entities,
    infer_salience_hit,
    infer_valence_score,
    infer_valence_tag,
    normalize_memory_text,
    tag_from_score,
)

extract_facts = _MemoryBackend._extract_facts
insert_row = _MemoryBackend._insert_row
add = _MemoryBackend.add
add_raw = _MemoryBackend.add_raw
maybe_supersede_neighbor = _MemoryBackend._maybe_supersede_neighbor

__all__ = [
    "_MemoryBackend",
    "add",
    "add_raw",
    "classify_kind",
    "classify_write_op",
    "entities_from_json",
    "entities_to_json",
    "extract_entities",
    "extract_facts",
    "infer_salience_hit",
    "infer_valence_score",
    "infer_valence_tag",
    "insert_row",
    "maybe_supersede_neighbor",
    "normalize_memory_text",
    "tag_from_score",
]
