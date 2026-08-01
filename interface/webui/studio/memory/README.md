# Aiko Memory Graph Studio (Phase C)

Visualize personal memory as a graph — same spirit as `agentic/studio`,
but for facts, supersession chains, and entity hubs.

## Run

```bash
# from repo root (deps: fastapi uvicorn)
uv run python -m memory.studio.backend.api
# → http://localhost:8001
```

Or:

```bash
uv run uvicorn memory.studio.backend.api:app --host 0.0.0.0 --port 8001
```

## API

| Endpoint | Description |
|----------|-------------|
| `GET /` | SPA frontend |
| `GET /api/graph` | `{nodes, edges, meta}` |
| `GET /api/health` | Liveness |

Query params for `/api/graph`:

- `user_id` — optional
- `limit` — max memory rows (default 200)
- `include_history` — include `status=superseded` (default true)
- `include_entities` — entity hub nodes + `mentions` edges (default true)

## Graph model

**Nodes**

- `type=memory` — fact rows (`label` truncated text, full `text` in details)
- `type=entity` — shared entity hubs from Phase B tags

**Edges**

- `supersedes` — newer fact → older fact it replaced (Phase A)
- `mentions` — memory → entity (Phase B)

Without Phase B tags, entity hubs stay empty; supersedes still show if Phase A ran.

## Notes

- Read-only — does not write memories
- No re-embed
- Draft / smoke-test after Phase A+B on a real DB before relying on it day-to-day
