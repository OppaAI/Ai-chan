# Working Memory Cortex (WMC)

> Cognitively-inspired active buffer for Aiko's *current conversational focus*.

Status: **framework only** (not yet wired into `cognition/think.py`).  
Branch: `feat/working-memory-cortex`

## Motivation

Most agent memory systems are strong on long-term storage (facts, personas, scenarios) but leave working memory as "whatever still fits in the context window".  
WMC adds an explicit, capacity-limited, scored buffer that:

- Respects Miller's Law (7±2) as a soft slot limit
- Uses a token budget as the primary governor
- Ranks turn-pairs by fast, local signals (emotion, importance, recency, relevance to a static anchor)
- Evicts low-score items deterministically into an episodic hand-off path
- Adds essentially zero latency (no embeddings, no LLM, no disk on the hot path)

## Lifecycle

```
Induction  -> form turn-pair after assistant reply
Filling    -> append into the active buffer
Sustaining -> re-score & reorder (high score = current focus)
Receding   -> low-score items drift to the end
Evicting   -> overflow items leave the buffer -> episodic hand-off
```

When the active set is kept in the prompt, "recalling" is free.

## Capacity (dual-guard)

| Guard | Default | Notes |
|-------|---------|-------|
| Slot limit (Miller) | min 5 / center 7 / max 9 | Soft preference for center |
| Token budget | 0 (unlimited) | Set via `WMC_TOKEN_BUDGET` or per-call |

Both are checked on every `fill()`.

## Scoring factors (all O(1) / pure Python)

| Factor | Range | Source |
|--------|-------|--------|
| Emotion | -1 … +1 | Emoji + tiny pos/neg lexicon |
| Importance | 0 … 1 | Keywords ("remember", "prefer"…), questions, name-like tokens, length |
| Recency | 0 … 1 | Exponential decay by turn distance |
| Relevance | 0 … 1 | Lexical overlap with a static-anchor token set (precomputed at boot) |

Weights are configurable (`WMC_W_*`).

## Public API

```python
from cognition.memory.wmc import build_wmc, WMTurn

wmc = build_wmc(
    static_anchor_tokens={"cats", "jetson", "preference"},  # optional
    on_evict=lambda turn: episodic_buffer.append(turn),     # optional
)

# After every completed turn:
evicted = wmc.fill(user_text, assistant_text)

# When building the prompt:
block = wmc.get_context_block(max_tokens=1200)

# Session end / /reset:
wmc.clear()
```

## Integration plan (follow-up)

1. Instantiate one `WorkingMemoryCortex` per user/session inside `AikoThink` (or a thin session object).
2. Replace (or sit in front of) the raw `CONTEXT_WINDOW_TURNS` list.
3. Call `fill()` after the assistant response is committed.
4. Inject `get_context_block()` into the prompt assembly path.
5. Wire `on_evict` into the existing async memory write / episodic path.
6. Optionally build the static anchor once at boot from high-retain / pinned memories (can reuse ideas from `session_anchor.py`).

## Config knobs

| Env key | Default | Meaning |
|---------|---------|---------|
| `WMC_ENABLED` | `1` | Master switch |
| `WMC_MILLER_MIN` | `5` | Soft lower slot bound |
| `WMC_MILLER_CENTER` | `7` | Preferred size |
| `WMC_MILLER_MAX` | `9` | Hard slot ceiling |
| `WMC_TOKEN_BUDGET` | `0` | 0 = slot-only; else max tokens for whole block |
| `WMC_W_EMOTION` | `0.25` | Weight |
| `WMC_W_IMPORTANCE` | `0.30` | Weight |
| `WMC_W_RECENCY` | `0.25` | Weight |
| `WMC_W_RELEVANCE` | `0.20` | Weight |
| `WMC_RECENCY_HALF_LIFE` | `4.0` | Turns |

## Relationship to other memory layers

| Layer | Role | WMC interaction |
|-------|------|-----------------|
| WMC (this) | Active focus, <= ~9 turn-pairs | Always (or mostly) in context |
| Episodic buffer / LTM | Facts, experiences, consolidated knowledge | Receives evicted turn-pairs |
| Session anchor (`session_anchor.py`) | Query-embedding mean for LTM rank boost | Complementary; different purpose |
| Persona / skills / wiki | Stable identity & procedural knowledge | Independent |

## Non-goals (for this PR)

- Full wiring into `think.py` / agentic paths
- Vector-based relevance (kept lexical for zero latency)
- Persistence of the WMC buffer across process restarts
- Replacement of the existing long-term sqlite-vec store
