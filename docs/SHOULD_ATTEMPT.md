# should_attempt — execution gate

State used to only shape prompts. This gate lets Aiko **branch** on her own energy / uncertainty / tool health.

## Placement (current)

1. **Early in `route()`** — *before* quaternary semantic routing (`mode="route"`).
   Soft outcomes apply to **every** turn, including ones that would have been localchat.
2. **Again in `agentic_chat()`** — for direct entry (scheduled jobs, approval resume paths) (`mode="agentic"`).

## Module

`cognition/memory/attempt_gate.py`

- `should_attempt(...)` → `(ok, reason, action)`
  - `proceed` | `degrade_chat` | `defer` | `clarify`
- `capability_from_outcomes` — coarse reliability from recent tool outcomes
- `is_critical_task` — urgent/safety/approval paths skip soft gates

## Soft actions

| action | Effect |
|--------|--------|
| `proceed` | Continue to normal routing / agentic |
| `defer` | Short in-character “later” via `chat()` |
| `clarify` | One clarifying question via `chat()` |
| `degrade_chat` | Force localchat-style `chat()` (no tool loop) |

Disable: `EDGE_ATTEMPT_GATE=0`

Critical requests always proceed. Existing tool guardrails unchanged.
