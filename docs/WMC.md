# Working Memory Cortex (WMC)

> Cognitively-inspired active buffer for Aiko's *current conversational focus*.

Status: **framework only** (not yet wired into `cognition/think.py`).  
Branch: `feat/working-memory-cortex`

---

## Motivation

Most agent memory systems are strong on long-term storage but leave working memory as "whatever still fits in the context window".  
WMC adds an explicit, capacity-limited, multi-factor scored buffer that:

- Respects Miller's Law (7±2) as a soft slot limit
- Uses a token budget as the primary governor
- Ranks turn-pairs by **8 lightweight dimensions** (no embedder / LLM on the hot path)
- Tracks **recall frequency** while resident (frequently injected slots gain score)
- Evicts low-score items deterministically into an episodic hand-off path
- Adds essentially zero latency

## Lifecycle

```
Induction  -> form turn-pair after assistant reply
Filling    -> append into the active buffer
Sustaining -> re-score & reorder (high score = current focus)
Receding   -> low-score items drift to the end
Evicting   -> overflow items leave the buffer -> episodic hand-off
```

When the active set is kept in the prompt, "recalling" is free.  
Each `get_context_block(touch=True)` increments `recall_count` on included slots.

## Capacity (dual-guard)

| Guard | Default | Notes |
|-------|---------|-------|
| Slot limit (Miller) | min 5 / center 7 / max 9 | Soft preference for center |
| Token budget | 0 (unlimited) | Set via `WMC_TOKEN_BUDGET` or per-call |

## Scoring factors (8 dimensions — all pure Python)

| Factor | Range | Weight default | Source |
|--------|-------|----------------|--------|
| Emotion | −1…+1 → 0…1 | 0.15 | Emoji + tiny pos/neg lexicon |
| Importance | 0…1 | 0.18 | Keywords ("remember", "prefer", "from now on"…), length |
| Recency | 0…1 | 0.15 | Exponential decay by turn distance |
| Relevance | 0…1 | 0.12 | Lexical overlap with **precomputed** static-anchor token set |
| Novelty | 0…1 | 0.12 | 1 − max Jaccard vs other resident slots |
| Question | 0…1 | 0.10 | Interrogatives / `?` density on user side |
| Entity | 0…1 | 0.08 | Proper-name-like tokens + numbers |
| **Recall frequency** | 0…1 | 0.10 | Soft-capped count of times included in context while resident |

Static anchor is built **once at boot** (or after consolidation) from high-retain / pinned memory text — lexical token set only. No embedder on the hot path.

**Embedder involvement:** none inside WMC. Optional once-at-boot if you later want a vector centroid; evicted turns still go through the existing LTM extract+embed path asynchronously.

## Public API

```python
from cognition.memory.wmc import build_wmc, WMTurn

wmc = build_wmc(
    static_anchor_tokens={"cats", "jetson", "preference"},
    on_evict=lambda turn: episodic_buffer.append(turn),
)

evicted = wmc.fill(user_text, assistant_text)
block = wmc.get_context_block(max_tokens=1200)  # touches recall_count
state = wmc.studio_state()  # for WMC Studio / debug
wmc.clear()
```

## Config knobs

| Env key | Default | Meaning |
|---------|---------|---------|
| `WMC_ENABLED` | `1` | Master switch |
| `WMC_MILLER_MIN` | `5` | Soft lower slot bound |
| `WMC_MILLER_CENTER` | `7` | Preferred size |
| `WMC_MILLER_MAX` | `9` | Hard slot ceiling |
| `WMC_TOKEN_BUDGET` | `0` | 0 = slot-only |
| `WMC_W_EMOTION` | `0.15` | Weight |
| `WMC_W_IMPORTANCE` | `0.18` | Weight |
| `WMC_W_RECENCY` | `0.15` | Weight |
| `WMC_W_RELEVANCE` | `0.12` | Weight |
| `WMC_W_NOVELTY` | `0.12` | Weight |
| `WMC_W_QUESTION` | `0.10` | Weight |
| `WMC_W_ENTITY` | `0.08` | Weight |
| `WMC_W_RECALL_FREQ` | `0.10` | Weight |
| `WMC_RECENCY_HALF_LIFE` | `4.0` | Turns |
| `WMC_RECALL_FREQ_CAP` | `6` | Soft cap for recall_freq normalization |

## Context size vs before

Current Aiko uses a flat rolling window (`CONTEXT_WINDOW_TURNS` ≈ 8).  
WMC targets the same ballpark (5–9) with **better ordering** and earlier eviction of weak turns, so typical prompt tokens should be similar or slightly lower, with higher signal density.

## Studio (planned)

`studio_state()` exposes slots + full factor breakdown for a WMC Studio sibling to the Memory Graph Studio:

- Live buffer strip (slots, scores)
- Per-slot radar/bars for all 8 factors
- Eviction log
- Static-anchor panel
- Live config knobs

## Integration plan (follow-up)

1. One `WorkingMemoryCortex` per user/session in `AikoThink`
2. Sit in front of / replace raw `CONTEXT_WINDOW_TURNS`
3. `fill()` after assistant commit; inject `get_context_block()` into prompt assembly
4. `on_evict` → existing async memory / episodic path
5. Build static anchor at boot from pinned / high-retain memories

## Non-goals (this PR)

- Full wiring into `think.py`
- Vector relevance on the hot path
- Cross-process persistence of the buffer
- Replacing sqlite-vec LTM
