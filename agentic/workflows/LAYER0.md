# Aiko Workflow Runtime — Layer 0

Foundation for shared workflow execution. Studio / spec codegen comes later.

```text
Spec (planning)          ← user / future studio
    ↓
Graph (orchestrate)      ← per-workflow graph.py + trigger
    ↓
Nodes (execution.py)     ← shared steps (this layer)
    ↓
Toolsets (registry/MCP)  ← concrete capabilities nodes call
```

## Layers

| Layer | Role | Code today |
|-------|------|------------|
| **0 Trigger** | time / interval / event / human | `schedule_graphs.json` + `system/schedule.py` |
| **1 Spec** | sources, steps, goals (JSON) | `*/config.json` (full Spec schema later) |
| **2 Graph** | order, deps, loops | `*/graph.py` → `PlanGraph` |
| **3 Nodes** | stable execution units | `common/execution.py` |
| **4 Tools** | RSS, HTTP, email, Threads… | `registry` + MCP |

## Shared nodes (`common/execution.py`)

1. **ingest_data** — sources + filters → normalized `items[]`
2. **store_data** — disk / state, dedup, retain_days
3. **synthesis_data** — template ± LLM → filled results
4. **verify_results** — HITL and/or rules / LLM check
5. **output_user_results** — email + social channels

Config is loaded once per run and sliced into node args (or `state.config`).

## Per-workflow packages

```text
workflows/
  common/           # Layer 0 nodes + store/notify/graphs
  job_hunt/         # graph.py + config.json (+ adapters until ingest is generic)
  aurora_forecast/  # graph.py + config.json
```

`graph.py` only **arranges** nodes. Domain URLs/thresholds live in **config.json**.

## Migration path

1. **Layer 0** (this PR): shared helpers + node stubs + registry
2. Aurora on shared nodes (replace domain check/store/notify gradually)
3. Job hunt on shared nodes
4. Spec JSON → graph (engine interprets)
5. Studio UI on Spec

## What is intentionally not Layer 0

- Visual DAG studio
- Full generic RSS/email adapters inside `ingest_data` (stubs + adapters OK)
- Auto-codegen of `.py` from Spec
