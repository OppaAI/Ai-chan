# Phase E — Core profile (Letta-lite)

## Goal
Always-on durable user facts without paying full RRF every turn.

## Sources (priority)
1. Optional `~/.aiko/<user>/memory/core_profile.json` → `{ "facts": ["..."] }`
2. Active personal memories ordered by: pinned → kind identity/preference → access_count

## API
```python
from memory.core_profile import core_profile_for_context, format_core_profile

block = core_profile_for_context(memorize)
# inject near system prompt / memory_context
```

## Config
- `CORE_PROFILE_MAX_FACTS` (default 12)
- `CORE_PROFILE_MAX_CHARS` (default 900)
- `CORE_PROFILE_PATH` optional override path

## Also in this PR (Phase D polish)
- **Live co-mentions:** `memory_meta._insert_row` calls `upsert_co_mentions` after each write (best-effort; no LLM).
- **Studio edges:** `graph_export.export_memory_graph` merges `relations_as_graph_edges` when `include_entities=True`.

Offline rebuild still available:
```bash
uv run python -m util.migrate_memory_phase_d
```

## Notes
- No extra LLM
- Wire into `think.py` / context assembly when ready (not required to merge module)
- Best after Phase A (status) + Phase B (kind tags) + Phase D (entity_relations table)
