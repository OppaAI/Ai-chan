# Layer 4 — Spec Studio UI

Builds on Layer 3 (`LAYER3.md`).

## What Layer 4 adds

A **Spec Studio** at `/studio/spec` so you can:

1. **List** workflows that run on the shared 5-node spine
2. **View** the current Spec (`spec.json` if present, else coerced from `config.json`)
3. **Validate** Spec JSON against Spec v1
4. **Preview** the compiled PlanGraph (shared_5 nodes + edges)
5. **Save** a Spec as `spec.json` for that workflow (preferred over config coercion)

## Layout

```text
interface/webui/studio/spec/
  backend/api.py       # FastAPI: workflows, validate, preview, save
  frontend/            # list + JSON editor + DAG preview
  README.md
  entrypoint.sh
```

Mounted from `interface/webui/auth.py` at `/studio/spec`.

## API

| Method | Path | Role |
|--------|------|------|
| GET | `/api/workflows` | Known Spec-backed workflows + graph ids |
| GET | `/api/workflows/{id}/spec` | Current Spec (file or coerced) |
| POST | `/api/validate` | Validate Spec body → `WorkflowSpec` or errors |
| POST | `/api/preview` | Spec → PlanGraph JSON (nodes + edges) |
| PUT | `/api/workflows/{id}/spec` | Write `spec.json` under the workflow package |

## Not in Layer 4

- Drag-and-drop custom node DAGs beyond `shared_5`
- Live workflow *execution* from the studio (use schedule / chat)
- Full form builder for every domain `config` key (JSON editor is enough)
- Auth changes (uses the same WebUI session as other studios)
