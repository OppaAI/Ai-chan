# Phase D — Thin entity relations

## Goal
Cheap multi-entity links without a graph database (Mem0/Zep-style linking, SQLite-scale).

## Schema
`entity_relations(user_id, entity_a, entity_b, relation, weight, memory_id, updated_at)`

Default relation: `co_mentions` (entities appearing on the same fact).

## Ops
```bash
uv run python -m util.migrate_memory_phase_d --dry-run
uv run python -m util.migrate_memory_phase_d
```

## Integration notes
- Depends on Phase A `entities` column; richer if Phase B tags exist.
- Graph Studio (Phase C) can merge `relations_as_graph_edges()` when both land.
- Write-path auto-upsert can be wired later into `memory_meta` hooks (optional).
