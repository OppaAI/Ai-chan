# Self-model via attention (not a second SOUL.md)

Aiko's lived self is learned in `cognition/attention.py`, not hardcoded.

## What was added

- Detect agency cues in Aiko's own replies: refuse / bargain / initiate / stance
- Evidence-gate into durable `self_preferences` (needs ≥2 observations, same as user prefs)
- Persist self_preferences, self_notes, self_decisions with the existing cognitive_state row
- `self_model_context()` surfaces only patterns that have actually locked in
- `record_self_decision(kind, summary, promote=False)` for agentic/reflection hooks

## What was not added

- No new persona markdown
- No hand-authored self_state.json
- No one-shot identity rewrite

SOUL.md remains the constitution. Edge state remains the lived, updating self.
