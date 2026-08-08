# Knowledge Graph Studio

Neural graph of **learned knowledge only** (chunks + entities). No personal memory or experience nodes.

## Run

```bash
uv run uvicorn interface.webui.studio.memory.kb.backend.api:app --host 127.0.0.1 --port 8002
```

Open `http://127.0.0.1:8002/` (or open `frontend/index.html` with `API_BASE` set).

## Visual encoding

| Node | Color | Importance |
|------|-------|------------|
| Knowledge chunk | green `#4ade80` | weighted access, recency, and entity connectivity |
| Entity hub | purple `#a78bfa` | degree relative to the highest-degree visible entity |

```text
importance ∈ [0,1]
knowledge importance = 0.45 × access + 0.30 × recency + 0.25 × connectivity
connectivity = min(1.0, entity_count / 6.0)

entity importance = entity_degree / maximum_visible_entity_degree
```

Inactive knowledge records reduce access and recency before importance is calculated.

## Frontend rendering

```
knowledge radius = 5 + 24 × importance^1.35 +entity radius = 5 + 16 × importance^1.25 +opacity = 0.22 + 0.78 × importance +
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
