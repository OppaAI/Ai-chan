# EMC-3 — episodic recall

## What
- `EpisodicStore.search` via `episode_recall.py` (KNN + FTS5 + RRF)
- `<episodic_context>` block appended in `AikoMemorize.format_for_context`
- Joint budget: when `EMC_JOINT_BUDGET=1`, EM+SM share `MEMORY_CONTEXT_TOTAL_CHARS` (EM reserved first)

## Flow
```
chat turn
  → memorize.search (semantic facts) — unchanged
  → format_for_context(memories, query=...)
       → SM block
       → store.search(query) → EM hits
       → append <episodic_context>
       → if joint: shrink SM so total fits
```

## Config (`config/memory.yaml`)
See EMC_RECALL_* / EMC_JOINT_BUDGET keys.

## Out of scope
- EMC-4 dream distillation EM→SM
- Grouping similar staging rows before flush
