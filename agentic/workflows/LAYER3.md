# Layer 3 — Spec JSON → PlanGraph

Builds on Layer 2 (`LAYER2.md`).

## What Layer 3 adds

1. **`common/spec.py`** — versioned Spec schema (`spec_version: "1"`), validate, load, and **coerce** legacy `config.json` into a Spec
2. **`common/spec_graph.py`** — `build_plan_graph(spec)` compiles Spec → shared 5-node `PlanGraph`
3. **Aurora + job_hunt** — `graph.py` builders load Spec (or coerce config) then call `build_plan_graph` (no duplicated node lists)
4. Optional **`spec.json`** per workflow — if present, preferred over coerced config

## Spec shape (v1)

```json
{
  "spec_version": "1",
  "id": "gen_job_post",
  "name": "Job hunt (shared nodes)",
  "goal": "Fetch job listings, draft posts, save for human review",
  "pipeline": "shared_5",
  "workflow_id": "job_hunt",
  "sources": [{"type": "adapter", "id": "job_hunt", "name": "job_hunt"}],
  "filters": {},
  "max_items": 30,
  "retain_days": 3,
  "parallel": true,
  "template": "",
  "llm_enriched": true,
  "per_item": true,
  "human_in_the_loop": true,
  "auto_pass_if": null,
  "email": {"enabled": false},
  "social": [],
  "config": { }
}
```

`config` holds domain fields (RSS feeds, `post_fields`, lat/lon, …). Shared fields may also appear at the top level; coercion lifts them from legacy configs automatically.

## Pipeline

Only `shared_5` in Layer 3:

```text
ingest_data → store_data → synthesis_data → verify_results → output_user_results
```

## Not in Layer 3

- Studio UI (Layer 4)
- Arbitrary custom node DAGs in Spec (only `shared_5`)
- Auto-codegen of Python modules
- Removing domain toolsets / adapters
