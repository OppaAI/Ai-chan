# Phase E — Core profile (Letta-lite)

## Goal
Always-on durable user facts without paying full RRF every turn.

## Sources (priority)
1. Optional `~/.aiko/<user>/memory/core_profile.json` → `{ "facts": ["..."] }`
2. Active personal memories ordered by: pinned → kind identity/preference → access_count

## API
```python
from memory.core_profile import core_profile_for_context, format_core_profile

block = core_profile_for_context(memorize)
# inject near system prompt / memory_context
```

## Config
- `CORE_PROFILE_MAX_FACTS` (default 12)
- `CORE_PROFILE_MAX_CHARS` (default 900)
- `CORE_PROFILE_PATH` optional override path

## Notes
- No extra LLM
- Wire into `think.py` / context assembly when ready (not required to merge module)
- Best after Phase A (status) + Phase B (kind tags)
