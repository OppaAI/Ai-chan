# Aiko WMC Studio

Visualize the **Working Memory Cortex** — active slots, 8-factor scores, recall frequency, and eviction flow.

## Run

```bash
# from repo root
uv run python -m interface.webui.studio.wmc.backend.api
# → http://127.0.0.1:8003
```

Or:

```bash
uv run python interface/webui/studio/wmc/entrypoint.sh
```

## Features

- **Live buffer strip** — Miller slots (high score = left / focus)
- **Per-slot factor bars** — emotion, importance, recency, relevance, novelty, question, entity, recall_freq
- **Eviction log** — what left the buffer and at what score
- **Demo seed** — canned conversation that triggers scoring + overflow
- **Manual fill / touch / reset** — inject turns, simulate prompt injection (recall bump), clear

## API

| Endpoint | Description |
|----------|-------------|
| `GET /` | SPA frontend |
| `GET /api/health` | Liveness |
| `GET /api/state` | Current WMC `studio_state()` + eviction log |
| `POST /api/fill` | `{user, assistant}` → fill demo buffer |
| `POST /api/touch` | Simulate context injection (recall_count++) |
| `POST /api/reset` | Clear buffer + eviction log |
| `POST /api/demo/seed` | Load canned multi-turn demo |

## Note

This studio drives a **process-local demo WMC** instance. Once WMC is wired into `cognition/think.py`, a follow-up can expose the live session buffer via the same API shape.
