# EMC-6 — coherent episode formation (staging → one episode)

## What

On `EpisodicStore.flush_staging()`, related **staging** rows are optionally
merged into a single `emc_storage` episode before embed/FTS.

Human analogy: hippocampal episode binding — successive moments of the same
experience become one episodic trace, not a bag of turn fragments.

This was explicitly **out of scope** for EMC-4 (dream = EM→SM). EMC-6 is the
missing **WM buffer → EM** shaping step.

## Grouping rules

All must hold to append a staging row to the current group:

| Rule | Default |
|------|---------|
| Same `session_id` (both NULL counts as same) | — |
| Time gap from previous row ≤ `EMC_GROUP_MAX_GAP_SEC` | 900 s |
| Group size < `EMC_GROUP_MAX_TURNS` | 6 |
| Merged trace ≤ `EMC_GROUP_MAX_CHARS` | 2000 |

No embedding similarity pass and **no LLM** on the flush path (Jetson-safe).

## Merge policy

- `timestamp` / `date` — first row
- `trace` — non-empty traces joined with blank lines
- `valence_tag` — first non-null
- `arousal_score` / `salience_score` — max of non-null
- `entities` — JSON union (casefold-deduped)
- `source` — first non-null, else `"emc_group"`
- `session_id` — first non-null
- **Never invent** who / where / why / other 5W fields

## Config

```yaml
EMC_GROUP_ENABLED: "1"
EMC_GROUP_MAX_GAP_SEC: "900"
EMC_GROUP_MAX_TURNS: "6"
EMC_GROUP_MAX_CHARS: "2000"
```

Set `EMC_GROUP_ENABLED: "0"` to restore EMC-1/2 1:1 flush behaviour.

## Out of scope

- Embedding / semantic clustering across sessions
- LLM 5W extraction into episode columns
- Emotional reprocessing of episodes
- Changing EMC-3 recall or EMC-4 dream contracts

## Test plan

- [ ] `EMC_GROUP_ENABLED=0` → same 1:1 flush as before
- [ ] Same session, turns within 15 min → one storage row, multi-turn trace
- [ ] Gap > max → separate episodes
- [ ] Different `session_id` → never merged
- [ ] Empty / short traces still respect existing ingest filters
- [ ] Dream + recall still see valid `emc_storage` rows
