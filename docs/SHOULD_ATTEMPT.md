# should_attempt — execution gate

State used to only shape prompts. This gate lets Aiko **branch** before the agentic tool loop.

## Module

`cognition/memory/attempt_gate.py`

- `should_attempt(...)` → `(ok, reason, action)`
  - `proceed` | `degrade_chat` | `defer` | `clarify`
- `capability_from_outcomes` — coarse reliability from recent tool outcomes
- `is_critical_task` — urgent/safety/approval paths skip soft gates

## Wire-up (required)

1. **edge_state.py** — thin wrappers (see PR discussion / local patches)
2. **think.py `agentic_chat`** — call gate before `run_agentic_chat`

Disable: `EDGE_ATTEMPT_GATE=0`

Critical requests always proceed. Existing tool guardrails unchanged.
