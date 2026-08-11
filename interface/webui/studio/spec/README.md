# Spec Studio (Layer 4)

Edit / validate / preview Workflow Specs and see the compiled shared_5 PlanGraph.

## Run (standalone)

```bash
python -m uvicorn interface.webui.studio.spec.backend.api:app --host 0.0.0.0 --port 8010
```

Open http://localhost:8010

## Run (mounted under WebUI)

Mounted at `/studio/spec` from `interface/webui/auth.py` when the main WebUI is up.

## API

See `agentic/workflows/LAYER4.md`.
