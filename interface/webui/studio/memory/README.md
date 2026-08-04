# Aiko Memory Graph Studio

Visualize personal memory as a **galaxy graph** — facts, supersession chains, entity hubs, and Phase 10 retain scores.

## Run

```bash
# from repo root (deps: fastapi uvicorn)
uv run python -m interface.webui.studio.memory.backend.api
# → http://localhost:8001
```

Or:

```bash
uv run uvicorn interface.webui.studio.memory.backend.api:app --host 0.0.0.0 --port 8001
Local only (`127.0.0.1`). For remote access, put an authenticated TLS-terminating reverse proxy in front; do not expose plain HTTP on `0.0.0.0`.
```

## API

| Endpoint | Description |
|----------|-------------|
| `GET /` | SPA frontend (galaxy view) |
| `GET /api/graph` | `{nodes, edges, meta, legend}` |
| `GET /api/search` | Memory + knowledge search |
| `GET /api/health` | Liveness |

Query params for `/api/graph`:

- `user_id` — optional
- `limit` — max memory rows (default 200)
- `include_history` — include `status=superseded` (default true)
- `include_entities` — entity hub nodes + `mentions` edges (default true)

## Graph model

**Nodes**

- `type=memory` — fact rows; **`size`** = retain tendency; **`scores`** rim arcs
- `type=entity` — shared entity hubs; **`size`** ≈ \(I_e\) when available

**Node fields (Phase 10)**

- `scores.retain` — keep-likelihood proxy (not full monthly R)
- `scores.salience | spacing | connectivity | valence | access` — rim arcs
- `valence_tag` — pos / neg / neutral
- `size` — derived from retain / \(I_e\)

**Edges**

- `supersedes` — newer fact → older fact it replaced
- `mentions` — memory → entity
- `related_to` / co-mention — entity → entity (`entity_relations`)

## Visual encoding

| Channel | Meaning |
|---------|---------|
| **Size** | Retain tendency (memories) / entity importance |
| **Rim arcs** | Factor breakdown |
| **Fill** | Valence / monthly / entity / pinned |
| **Dim** | Superseded |

## Notes

- Read-only — does not write memories
- No re-embed
- Backend: `graph_export.export_memory_graph` (scores computed at export time)
- Scoring helpers live **inside** `graph_export.py` (no separate `studio_scores.py`)
