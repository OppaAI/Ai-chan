# Spec Studio (Layer 5)

Unified **Graph + Spec** studio at `/studio/spec`.

- Same look as DAG Studio: playbook list, full DAG canvas, zoom, node details
- All playbooks from `graph_engine`
- Spec drawer for Spec-backed workflows (job_hunt / aurora)

## Run (standalone)

```bash
python -m uvicorn interface.webui.studio.spec.backend.api:app --host 127.0.0.1 --port 8010
```

## Run (WebUI)

Mounted at `/studio/spec` (toolbar graph button).

## Docs

See `agentic/workflows/LAYER5.md`.
