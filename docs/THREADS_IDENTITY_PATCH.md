# Threads identity follow-up (same PR as Bluesky two-way)

Apply these edits in `interface/mcp_server/social/services/threads.py` so Threads matches Bluesky.

Shared helpers live in `interface/mcp_server/social/services/identity.py`.

## 1. Replace hardcoded constants (near top)

**Delete:**
```python
THREADS_REPLY_TRIGGER = "Hi Aiko"
THREADS_REPLY_MENTION = "@oppa.ai.bot"
```

**Add after the imports / before `_LLM_CLIENT`:**
```python
try:
    from social.services.identity import (
        ai_name,
        reply_trigger_phrase,
        platform_username,
        mention_trigger,
        owner_display_name,
        is_trigger as _identity_is_trigger,
    )
except ModuleNotFoundError:
    from .identity import (
        ai_name,
        reply_trigger_phrase,
        platform_username,
        mention_trigger,
        owner_display_name,
        is_trigger as _identity_is_trigger,
    )
```

## 2. Replace `_is_trigger`

```python
def _is_trigger(text: str) -> bool:
    return _identity_is_trigger(
        text,
        phrase=reply_trigger_phrase("threads"),
        mention=mention_trigger("THREADS_USERNAME"),
    )
```

Optional override: set `THREADS_REPLY_TRIGGER` in env to force a custom phrase.

## 3. Owner username checks (3 places)

Replace every:
```python
env("THREADS_USERNAME", "oppa.ai.bot")
# and
env("THREADS_USERNAME", THREADS_REPLY_MENTION.lstrip("@"))
```
with:
```python
platform_username("THREADS_USERNAME")
```

Locations:
- `_save_requested_memory` → `owner = ...`
- `_save_interaction_memory` → `owner = ...`
- `_infer_reply` → `owner = ...`

## 4. Owner display name in `_infer_reply`

Replace:
```python
env('THREADS_OWNER_NAME', 'OppaAI')
```
with:
```python
owner_display_name()
```

## 5. AI name in reply prompt (`_infer_reply`)

Replace hardcoded `Aiko` in the prompt with `ai_name()`:

- `in Aiko's voice` → `in {ai_name()}'s voice`
- `*Aiko considers the question.*` → `*{ai_name()} considers the question.*`
- In `_save_interaction_memory`: `Aiko replied:` → `f"{ai_name()} replied:"`

## 6. Env

In `.env.age` (already documented in `.env.example` on this branch):

```
THREADS_USERNAME=oppa.ai.bot
```

Mention trigger becomes `@oppa.ai.bot` automatically. Phrase trigger is `Hi {AI_NAME}` from `config/system.yaml`.
