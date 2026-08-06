# Knowledge Graph Studio

Neural graph of **learned knowledge only** (chunks + entities). No personal memory or experience nodes.

## Run

```bash
uv run uvicorn interface.webui.studio.kb.backend.api:app --host 127.0.0.1 --port 8002
```

Open `http://127.0.0.1:8002/` (or open `frontend/index.html` with `API_BASE` set).

## Visual encoding

| Node | Color | Size / brightness |
|------|--------|-------------------|
| Knowledge chunk | green `#4ade80` | **importance** = access + recency + entity count |
| Entity hub | purple `#a78bfa` | degree among visible chunks |

```text
importance ∈ [0,1]
size ≈ 0.20 + 1.10 × importance^1.25
```

## Edges

| Type | Meaning |
|------|---------|
| `about` | chunk → entity |
| `same_doc` | consecutive chunks of the same document |

## Env

- `KNOWLEDGE_STUDIO_MAX_CHUNKS` (default 200)
- `KNOWLEDGE_STUDIO_MAX_ENTITIES` (default 120)
- `KNOWLEDGE_STUDIO_MAX_EDGES` (default 300)
