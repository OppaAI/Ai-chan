# Grasp — temporary working memory

> Hold the current conversational focus for a few turns, then let go.

Status: **framework + daily journal** (not yet wired into `cognition/think.py`).  
Branch: `feat/working-memory-cortex`

Pairs with: `memorize` / `forget` / `imprint` / `learn`.

---

## Lifecycle

```
Induction  -> form turn-pair after assistant reply
Filling    -> append into buffer + async journal line (day = user_ts)
Sustaining -> re-score & reorder (high score = current focus)
Receding   -> low-score items drift to the end
Evicting   -> overflow → on_evict (+ optional event=evict journal line)
```

## Capacity (dual-guard)

| Guard | Default | Notes |
|-------|---------|-------|
| Slot limit (Miller) | min 5 / center 7 / max 9 | Soft preference for center |
| Token budget | 0 (unlimited) | `GRASP_TOKEN_BUDGET` |

## Scoring factors (9 — pure Python)

| Factor | Weight default | Source |
|--------|----------------|--------|
| Emotion | 0.14 | Emoji + lexicon |
| Importance | 0.17 | Keywords, length |
| Recency | 0.14 | Turn-distance decay |
| Relevance | 0.11 | Lexical overlap with precomputed static anchor |
| Novelty | 0.11 | 1 − max Jaccard vs peer slots |
| Question | 0.09 | Interrogatives / `?` |
| Entity | 0.08 | Name-like tokens + numbers |
| Recall frequency | 0.09 | Soft-capped context injections |
| Primacy | 0.07 | Early session turns stick a bit longer |

No embedder / LLM on the hot path.

## Daily journal (one JSONL)

```text
~/.local/share/aiko/journal/YYYY-MM-DD.jsonl
```

| Rule | Behavior |
|------|----------|
| When written | **On fill** — async daemon thread |
| Day boundary | **User turn** local datetime only |
| Straddle midnight | User `23:59:59`, assistant after midnight → previous day |
| Nightly job | ~00:05 processes **yesterday's** file → reflect / dream |
| Pinning / LTM | Unchanged (memory DB) |

```python
from cognition.memory.grasp import build_grasp, load_journal_day

rows = load_journal_day()  # yesterday
buf = build_grasp(static_anchor_tokens={"cats", "jetson"})
evicted = buf.fill(user, asst, user_ts=user_dt)
block = buf.get_context_block(max_tokens=1200)
```

## Config

| Env | Default |
|-----|----------|
| `GRASP_ENABLED` | `1` |
| `GRASP_MILLER_MIN/CENTER/MAX` | `5/7/9` |
| `GRASP_TOKEN_BUDGET` | `0` |
| `GRASP_W_*` | see `grasp.py` |
| `GRASP_JOURNAL_ENABLED` | `1` |
| `GRASP_JOURNAL_DIR` | `~/.local/share/aiko/journal` |

## Studio

`interface/webui/studio/grasp/` — buffer strip, factor bars, seed/fill/touch/reset.

```bash
uv run python -m interface.webui.studio.grasp.backend.api
# → http://127.0.0.1:8003
```

## Integration (follow-up)

1. One `GraspBuffer` per session in `AikoThink`
2. Pass real `user_ts` from inbound message
3. Nightly: `load_journal_day()` → reflect / dream
4. Pins + consolidate stay on existing memory DB
