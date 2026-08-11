# Spec Studio (Layer 4)

Edit / validate / preview Workflow Specs and see the compiled shared_5 PlanGraph.

## Run (standalone)

```bash
# Loopback-only (default, recommended for local dev)
python -m uvicorn interface.webui.studio.spec.backend.api:app --host 127.0.0.1 --port 8010

# For remote access (requires authentication boundary, e.g., reverse proxy with auth)
SPEC_STUDIO_HOST=0.0.0.0 python interface/webui/studio/spec/entrypoint.sh
```

Open http://localhost:8010

**Security note:** The standalone server has no built-in authentication. For remote access, deploy behind an authenticated reverse proxy or VPN.

## Run (mounted under WebUI)

Mounted at `/studio/spec` from `interface/webui/auth.py` when the main WebUI is up.

## API

See `agentic/workflows/LAYER4.md`.
