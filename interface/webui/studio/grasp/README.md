# Aiko Grasp Studio

Visualize **grasp** — temporary working-memory slots, scores, and eviction flow.

## Run

```bash
uv run python -m interface.webui.studio.grasp.backend.api
# → http://127.0.0.1:8003
```

## Features

- Live buffer strip (Miller slots)
- Per-slot factor bars (9 dimensions incl. primacy)
- Eviction log, demo seed, fill / touch / reset

Process-local demo instance until wired into `think.py`.
