# Aiko Memory Graph Studio

Visualize personal memory as a **galaxy graph** — facts, supersession chains, entity hubs, and retain scores.

## Run

```bash
# from repo root (deps: fastapi uvicorn)
uv run python -m interface.webui.studio.memory.backend.api
# → http://127.0.0.1:8001
```

Or:

```bash
uv run uvicorn interface.webui.studio.memory.backend.api:app --host 127.0.0.1 --port 8001
```

Local only (`127.0.0.1`). For remote access, put an authenticated TLS-terminating reverse proxy in front; do not expose plain HTTP on `0.0.0.0`.

## API

| Endpoint | Description |
|----------|-------------|
| `GET /` | SPA frontend (galaxy view) |
| `GET /api/graph` | `{nodes, edges, meta, legend}` |
| `GET /api/search` | Memory + knowledge search |
| `GET /api/health` | Liveness |

Query params for `/api/graph`:

- `user_id` — optional
- `limit` — max memory rows fetched (default 200; also capped by `MEMORY_STUDIO_MAX_MEMORIES`)
- `include_history` — include `status=superseded` (default true)
- `include_entities` — entity hub nodes + `mentions` edges (default true)

## Phase 12

**Server caps** (`config/memory.yaml`):

- `MEMORY_STUDIO_MAX_MEMORIES` (default 400) — prefer high retain
- `MEMORY_STUDIO_MAX_ENTITIES` (default 120)
- `MEMORY_STUDIO_MAX_EDGES` (default 200)

**Client filters** (sidebar; re-filter without reload):

- Status: all / active / superseded
- Valence: all / pos / neg / neutral
- Min retain (0–1)
- Entity contains (substring)

## Graph model

**Nodes**

- `type=memory` — fact rows; **`size`** = retain tendency; **`scores`** rim arcs
- `type=entity` — shared entity hubs; **`size`** ≈ $I_e$ when available

**Edges**

- `supersedes` — newer fact → older fact it replaced
- `mentions` — memory → entity
- `related_to` / co-mention — entity → entity

## Notes

- Read-only — does not write memories
- Scoring helpers live inside `graph_export.py`
