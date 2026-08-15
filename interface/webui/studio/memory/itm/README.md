# Aiko ITM Studio (Episodic Memory)

Visualize how Aiko stores memory in **episodes** — the episodic memory
pipeline (EMC): working-memory eviction → `emc_staging` → `emc_storage` →
distillation into semantic facts during dream() (EM→SM).

## Run

```bash
# from repo root (deps: fastapi uvicorn)
uv run python -m interface.webui.studio.memory.itm.backend.api
# → http://127.0.0.1:8004
```

Local only. For remote access, put an authenticated TLS-terminating reverse
proxy in front.

## API

| Endpoint | Description |
|----------|-------------|
| `GET /` | SPA frontend (episode timeline) |
| `GET /api/pipeline` | Stage counts: staging / storage / distilled / recalled |
| `GET /api/episodes` | Chronological `emc_storage` timeline (newest first) |
| `GET /api/staging` | Pending staged episodes (not yet flushed) |
| `GET /api/episode/{id}` | Full detail + distilled semantic facts (EM→SM) |
| `GET /api/health` | Liveness |

Query params for `/api/episodes`:

- `user_id` — optional
- `limit` — max rows (default 200)
- `stage` — `all` (default) | `storage` | `distilled`
- `date_from` / `date_to` — ISO timestamp filter
- `q` — substring filter on the episode trace

Read-only: mirrors the LTM/STM studio convention; no delete/flush/dream
actions.

## Data source

Episodes come from the same per-user SQLite file as the rest of memory
(`emc_storage` / `emc_staging` tables). Episodes marked `distilled_at`
carry a `distilled_into` JSON list of semantic-memory ids, which the detail
view resolves back into the durable facts Aiko consolidated them into.