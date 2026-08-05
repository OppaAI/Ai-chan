"""Search, ranking, spreading, and chain-expansion helpers."""

from __future__ import annotations

from .backend import _MemoryBackend, entity_overlap_score

search = _MemoryBackend.search
rank_and_score = _MemoryBackend._rank_and_score
spreading_extra_ids = _MemoryBackend._spreading_extra_ids
expand_supersession_chains = _MemoryBackend._expand_supersession_chains
fts_pass = _MemoryBackend._fts_pass
graph_pass = _MemoryBackend._graph_pass
apply_recency_rerank = _MemoryBackend._apply_recency_rerank

__all__ = [
    "_MemoryBackend",
    "apply_recency_rerank",
    "entity_overlap_score",
    "expand_supersession_chains",
    "fts_pass",
    "graph_pass",
    "rank_and_score",
    "search",
    "spreading_extra_ids",
]
