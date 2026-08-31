# Pre-route should_attempt gate

Moves the self-assessment gate **in front of** quaternary semantic routing so localchat/webchat get executable freedom of choice (not only agentic).

## Already on this branch

- `cognition/attention.py` — soft rules for `mode="route"` and `mode="agentic"`
- `docs/SHOULD_ATTEMPT.md` — updated placement notes

## Restore + finish think.py (required)

`cognition/think.py` on this branch was briefly overwritten during upload. Restore and wire it in one step:

```bash
git fetch origin
git checkout feat/pre-route-attempt-gate
python3 scripts/apply_pre_route_gate.py
# requires origin/dev (or local dev) with a good cognition/think.py

git add cognition/think.py
git commit -m "think: should_attempt before quaternary routing"
git push
```

The script:

1. Loads `origin/dev:cognition/think.py`
2. Inserts early `should_attempt(..., mode="route")` in `route()`
3. Adds `_soft_gate_reply` and reuses it from `agentic_chat`

## Behaviour

```text
input → should_attempt(mode=route)
          soft → chat (defer / clarify / degrade)
          proceed → existing quaternary route
```

Disable: `EDGE_ATTEMPT_GATE=0`
