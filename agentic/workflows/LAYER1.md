# Layer 1 — Runnable shared nodes

Builds on Layer 0 (`LAYER0.md`).

## What Layer 1 adds

1. **Catalog registration** — `ingest_data`, `store_data`, `synthesis_data`, `verify_results`, `output_user_results` in `config/tools.yaml`
2. **Graph handlers** — `agentic/workflows/common/nodes.py` (`@tool` + `graph=True`)
3. **Aurora adapter** — `ingest_data` source type `adapter` / name `aurora` calls the domain check and emits one normalized item
4. **Aurora graph** — wired only to the five shared nodes (domain `check_aurora` kept for ReAct / debug)
5. **Config mini-spec** — `aurora_forecast/config.json` carries sources, template, email, social, retain_days

## Aurora graph shape

```
ingest_data → store_data → synthesis_data → verify_results → output_user_results
```

Trigger remains schedule_graphs (`hourly_aurora_forecast`).

## Not in Layer 1

- Full job_hunt migration onto shared nodes
- Studio / Spec codegen
- LLM-backed `synthesis_data` enrichment (hook only)
