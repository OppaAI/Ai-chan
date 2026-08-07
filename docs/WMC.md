# Working Memory Cortex (WMC)

> Cognitively-inspired active buffer for Aiko's *current conversational focus*.

Status: **framework + daily journal** (not yet wired into `cognition/think.py`).  
Branch: `feat/working-memory-cortex`

---

## Motivation

Most agent memory systems leave working memory as "whatever still fits in the context window".  
WMC adds an explicit, capacity-limited, multi-factor scored buffer plus a **single daily JSONL journal** for nightly dream/reflect.

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
| Token budget | 0 (unlimited) | `WMC_TOKEN_BUDGET` |

## Scoring factors (9 dimensions — pure Python)

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
| **Primacy** | 0.07 | Early session turns stick a bit longer |

No embedder / LLM on the hot path.

## Daily journal (one JSONL, not two)

**Single file per local calendar day:**

```text
~/.local/share/aiko/journal/YYYY-MM-DD.jsonl
```

| Rule | Behavior |
|------|----------|
| When written | **On fill** (faithful trail) — async daemon thread |
| Day boundary | **User turn** local datetime only |
| Straddle midnight | User `23:59:59`, assistant `00:00:10` → **still previous day** |
| New day | First user turn at `00:00:00` or later → new file |
| Eviction debug | Optional same-file line `event: "evict"` (not a second JSONL) |
| Nightly job | At ~00:05 process **yesterday's** file → reflect / dream |
| Pinning / LTM consolidate | Unchanged (memory DB) |

Record shape:

```json
{
  "ts": "2026-08-07T23:59:59+07:00",
  "ts_unix": 1786142399,
  "day": "2026-08-07",
  "event": "fill",
  "turn": 47,
  "user": "...",
  "assistant": "...",
  "tokens": 128,
  "score": 0.71,
  "factors": { "emotion": 0.6, "importance": 0.8 }
}
```

Helpers:

```python
from cognition.memory.wmc import load_journal_day, build_wmc

# Nightly reflect / dream input
rows = load_journal_day()  # yesterday by default
rows = load_journal_day("2026-08-07")

wmc = build_wmc()
wmc.fill(user, assistant, user_ts=user_msg_datetime)
wmc.flush_resident_to_journal(event="day_close")  # optional EOD
```

## Public API

```python
from cognition.memory.wmc import build_wmc, load_journal_day

wmc = build_wmc(
    static_anchor_tokens={"cats", "jetson"},
    on_evict=lambda t: ...,  # episodic hand-off
)
evicted = wmc.fill(user, asst, user_ts=user_dt)
block = wmc.get_context_block(max_tokens=1200)
state = wmc.studio_state()
```

## Config

| Env | Default | Meaning |
|-----|---------|---------|
| `WMC_ENABLED` | `1` | Master switch |
| `WMC_MILLER_MIN/CENTER/MAX` | `5/7/9` | Slot bounds |
| `WMC_TOKEN_BUDGET` | `0` | 0 = slot-only |
| `WMC_W_*` | see code | Factor weights incl. `WMC_W_PRIMACY` |
| `WMC_RECENCY_HALF_LIFE` | `4.0` | Turns |
| `WMC_RECALL_FREQ_CAP` | `6` | Soft cap |
| `WMC_PRIMACY_SPAN` | `6.0` | Turns over which primacy decays |
| `WMC_JOURNAL_ENABLED` | `1` | Daily JSONL on/off |
| `WMC_JOURNAL_DIR` | `~/.local/share/aiko/journal` | Journal root |

## Studio

`interface/webui/studio/wmc/` — buffer strip, factor bars, eviction log, demo seed.  
Run: `uv run python -m interface.webui.studio.wmc.backend.api` → `:8003`

## Integration plan (follow-up)

1. Wire one WMC per session in `AikoThink`
2. Pass real `user_ts` from the inbound message
3. Nightly job: `load_journal_day()` → reflect → `journal.db` / dream → blog
4. Pinning + consolidate pipelines stay on existing memory DB
5. Optional: warm-load recent journal lines into WMC on boot

## Non-goals (this PR)

- Wiring into `think.py`
- Replacing sqlite-vec LTM or pin paths
- Second debug-only JSONL file
