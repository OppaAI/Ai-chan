"""TEMPORARY PLACEHOLDER — do not merge until restored.

This file was replaced during a large-file push failure.

Restore + apply the pre-route gate in one step:

    python3 scripts/apply_pre_route_gate.py

That loads origin/dev:cognition/think.py and inserts:
  - should_attempt(mode=\"route\") before quaternary routing
  - _soft_gate_reply helper shared with agentic_chat

See docs/PRE_ROUTE_GATE.md
"""

raise ImportError(
    "cognition/think.py is a placeholder on this branch. "
    "Run: python3 scripts/apply_pre_route_gate.py"
)
