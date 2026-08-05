# Aiko Memory Graph Studio

Visualize personal memory as a **neural graph** — facts, supersession chains, entity hubs, knowledge, experience, and retain scores.

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
- `include_knowledge` — learned knowledge nodes (default true; Phase 13)
- `include_experience` — experience nodes (default true; Phase 13)

## Phase 12

**Server caps** (env vars read by `graph_export.py`; set via deploy / `.env` — `config/memory.yaml` is the documented source when your config loader exports them to the environment):

- `MEMORY_STUDIO_MAX_MEMORIES` (default 400)
- `MEMORY_STUDIO_MAX_ENTITIES` (default 120)
- `MEMORY_STUDIO_MAX_EDGES` (default 200)

Note: over-fetches ~3× `limit` (newest-first), then keeps the top `limit` by retain among that window (not a full-DB retain rank).

## Phase 13 — Cross-store layers

Related knowledge and experience appear as extra node types, linked by shared entities.

### New node types

| type | Color | Source |
|------|--------|--------|
| `knowledge` | green `#4ade80` | `learned_chunks` |
| `experience` | orange `#fb923c` | `experiences` |

### New edge types

| type | Meaning |
|------|---------|
| `about` | knowledge/experience → entity |
| `grounded_in` | memory → knowledge (shared entity) |
| `practiced_in` | memory → experience (shared entity) |

### API
```http
GET /api/graph?include_knowledge=true&include_experience=true
```

### UI layers

Sidebar:

- **Include knowledge / experience** — control what the export loads (reload)
- **Show memory / entities / knowledge / experience** — client filter without reload

### Env (optional)

- `MEMORY_STUDIO_INCLUDE_KNOWLEDGE` (default `1`)
- `MEMORY_STUDIO_INCLUDE_EXPERIENCE` (default `1`)
- `MEMORY_STUDIO_MAX_KNOWLEDGE` (default `80`)
- `MEMORY_STUDIO_MAX_EXPERIENCE` (default `40`)

**Client filters** (sidebar; re-filter without reload):

- Status: all / active / superseded
- Valence: all / pos / neg / neutral
- Min retain (0–1)
- Entity contains (substring)

## Graph model

**Nodes**

- `type=memory` — fact rows; **`size`** = retain tendency; **`scores`** rim arcs
- `type=entity` — shared entity hubs; **`size`** ≈ $I_e$ when available
- `type=knowledge` — learned chunks (green)
- `type=experience` — past agent runs (orange)

**Edges**

- `supersedes` — newer fact → older fact it replaced
- `mentions` — memory → entity
- `related_to` / co-mention — entity → entity

- `about` — knowledge/experience → entity
- `grounded_in` — memory → knowledge
- `practiced_in` — memory → experience

## Notes

- Read-only — does not write memories
- Scoring helpers live inside `graph_export.py`
