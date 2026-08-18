# Aiko Workflow Spec (DAG + tool-result cache)

**Goal:** Every agentic DAG workflow is a declarative **spec** (JSON playbook).  
**Caching is a first-class step**, not an ad-hoc script side effect.

## Principles

1. **Tool outputs → cache by default** (full/near-full results on disk).
2. **Only a selected slice enters LLM context** for synthesis.
3. **Same record format** across email, RSS, web, fetch, etc.
4. **Same pipeline** for all workflows; separate *files* per run/source OK.
5. **Do not** dump bulk tool results into normal chat memory / persona cache.
6. Promote only **durable outcomes** (0–N facts) into L1 memory after synthesis.

## Pipeline (every DAG)

```text
fetch / tool
  → cache_write          # append JSONL records
  → cache_select         # rank/filter → small working set
  → synthesize / draft   # LLM sees only the selection
  → write_report / post
  → optional memory_add  # durable facts only
```

## Spec shape (playbook node)

Compatible with existing `PlanNode` fields in `graph_engine.py`:

```json
{
  "id": "cache_rss",
  "tool": "cache_write",
  "args": {
    "workflow": "lane_d_job_hunt",
    "source": "rss",
    "from_state": "rss_raw",
    "run_id": "${run_id}"
  },
  "depends_on": ["fetch_rss"]
}
```

### Standard cache tools

| Tool | Role |
|------|------|
| `cache_write` | Normalize + append tool results to JSONL under workflow cache dir |
| `cache_select` | Load cache → filter/rank → put compact list on graph state |
| `cache_read` | Read records by run_id / source (debug, drill-down) |
| `cache_gc` | Drop old runs (retention policy) |

## Directory layout

```text
~/.aiko/<user_id>/agentic/cache/
  <workflow_id>/
    2026-08-17T230000Z_rss.jsonl
    2026-08-17T230000Z_email.jsonl
    index.jsonl          # optional: one line per file written
```

## Record schema (one JSON object per line)

```json
{
  "id": "uuid",
  "workflow": "lane_d_job_hunt",
  "source": "rss|email|web|tool",
  "run_id": "2026-08-17T230000Z",
  "fetched_at": "ISO-8601",
  "title": "...",
  "body": "...",
  "url": "",
  "score": 0.0,
  "matched": true,
  "meta": {}
}
```

## What stays out of this cache

- Persona / `kind=identity` (L3 chat fast path)
- In-process `TTLCache` for search/fetch dedup within a single process
- Long-term `memories` table (only after explicit promote)

## Migration

| Today | Spec |
|-------|------|
| Lane D `fetch_*_email_*.jsonl` + `fetch_*_rss_*.jsonl` ad hoc | Same files optional; **same schema** + `cache_write` / `cache_select` nodes |
| Research DAG dumps big text into node results | `cache_write` then `cache_select` / condense before synthesize |
| Skill markdown only | Keep skills for human docs; **execution** is playbook JSON with cache steps |
