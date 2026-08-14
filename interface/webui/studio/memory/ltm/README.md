# Aiko LTM Graph Studio

Visualize personal memory as a **neural graph** — facts, supersession chains, entity hubs, knowledge, experience, episodes, and retain scores.

## Run

```bash
# from repo root (deps: fastapi uvicorn)
uv run python -m interface.webui.studio.memory.ltm.backend.api
# → http://127.0.0.1:8001
```

Or:

```bash
uv run uvicorn interface.webui.studio.memory.ltm.backend.api:app --host 127.0.0.1 --port 8001
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
- `include_episodes` — episodic memory (EMC) nodes (default true; EMC-5)

## Server caps

- `MEMORY_STUDIO_MAX_MEMORIES` (default 400)
- `MEMORY_STUDIO_MAX_ENTITIES` (default 120)
- `MEMORY_STUDIO_MAX_EDGES` (default 200)
- `MEMORY_STUDIO_MAX_EPISODES` (default 80)
- `MEMORY_STUDIO_INCLUDE_EPISODES` (default true)

### Node types

| type | Color | Source |
|------|--------|--------|
| `knowledge` | green `#4ade80` | `learned_chunks` |
| `experience` | orange `#fb923c` | `experiences` |
| `episode` | fuchsia `#e879f9` | `emc_storage` (EMC) |
