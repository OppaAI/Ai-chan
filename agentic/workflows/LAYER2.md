# Layer 2 — job_hunt on shared nodes

Builds on Layer 1 (`LAYER1.md`).

## What Layer 2 adds

1. **job_hunt adapter** in `common/nodes.py` — `adapter:job_hunt` (or `rss` / `email` markers) calls `job_hunt.toolset` fetch helpers and normalizes postings to shared items
2. **synthesis_data** job path — uses `post_fields` + optional LLM enrich (`format_job_post` / `enrich_posting_fields_with_llm`)
3. **verify_results** HITL — writes draft files via `save_single_job_draft`, keeps `verified: false`
4. **Primary graph** `gen_job_post` — five shared nodes only
5. **Legacy graph** `gen_job_post_legacy` — old fetch → loop → draft → save → report (rollback)

## Graph shape

```
ingest_data → store_data → synthesis_data → verify_results → output_user_results
```

Schedule can keep pointing at `gen_job_post`.

## Config mini-spec (`job_hunt/config.json`)

- `sources`: `[{ "type": "adapter", "name": "job_hunt" }]`
- `human_in_the_loop`: true
- `llm_enriched`: true
- `post_fields` / `post_signature` (existing)
- `email` / `social`: default off (approve drafts before post)

## Not in Layer 2

- Spec → codegen
- Studio UI
- Removing domain toolset (still the adapter implementation)
