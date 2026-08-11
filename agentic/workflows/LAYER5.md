# Layer 5 — Unified Spec / Graph Studio

Builds on Layer 4 (`LAYER4.md`).

## What Layer 5 adds

Spec Studio at `/studio/spec` matches **DAG Studio** look-and-feel and lists **all playbooks**.

1. **Playbook browser** — `/api/playbooks` (same as DAG Studio via `graph_engine.load_playbooks`)
2. **Full DAG canvas** — level layout, depends_on / loop_to / fallback_to edges, zoom, node details
3. **Spec drawer** — for Spec-backed graphs (`gen_job_post` → job_hunt, `aurora_forecast` → aurora): edit / validate / preview / save

## Layout

```text
sidebar: all playbooks (spec-backed tagged)
canvas:  full PlanGraph (DAG Studio renderer)
drawer:  Spec editor (Spec-backed only)
```

## Not in Layer 5

- Drag-edit of arbitrary DAG topology
- Removing `/studio/dag` (still available by URL)
- Execution from the studio
