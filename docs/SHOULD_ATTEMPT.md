# should_attempt — execution gate

After PR self-preferences, Aiko’s state could shape prompts but not whether agentic work ran.

## What this adds

### `EdgeCognitiveState.should_attempt(user_input, mode="agentic")`
Returns `(ok, reason, action)` where action is:
- `proceed` — run agentic as usual
- `degrade_chat` — use normal chat instead of the tool loop
- `defer` — short in-character “later” (non-critical, low energy)
- `clarify` — one clarifying question (high uncertainty, short/ambiguous ask)

Also:
- `capability_for(domain)` — coarse success rate from recent tool outcomes
- `is_critical_task(text)` — urgent/safety/approval paths skip soft gates

### `AikoThink.agentic_chat`
Calls `should_attempt` **before** `run_agentic_chat`. Soft actions are logged, recorded via `record_self_decision`, and branch to `chat()` instead of the agent loop.

## Safety / conservatism
- Critical requests always `proceed`
- Gate can be disabled with `EDGE_ATTEMPT_GATE=0`
- Existing tool guardrails and human approval are unchanged

## Apply patches (if source files not yet updated on this branch)

```bash
patch -p1 < patches/should_attempt_edge_state.patch
patch -p1 < patches/should_attempt_think.patch
```
