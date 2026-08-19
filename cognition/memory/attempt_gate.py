"""Bounded self-assessment gate for execution paths.

Used by EdgeCognitiveState.should_attempt and AikoThink.route / agentic_chat.
Keeps thresholds conservative: critical work always proceeds.

mode:
  - agentic — gate before the tool loop (legacy path; also used for direct agentic entry)
  - route   — gate *before* quaternary semantic routing so localchat/webchat
              also get executable soft outcomes (defer / clarify / degrade_chat)
"""
from __future__ import annotations

import re

_CRITICAL_TASK_RE = re.compile(
    r"\b("
    r"urgent|emergency|asap|right now|immediately|"
    r"safety|danger|hurt|injured|crisis|"
    r"deadline today|due today|production (?:is )?down|outage|"
    r"approve run-|cancel (?:the )?(?:job|schedule|reminder)"
    r")\b",
    re.I,
)
_WORD_RE = re.compile(r"[a-z0-9_]{3,}")
# Modes where soft opt-out (defer/clarify/degrade) is allowed.
_SOFT_MODES = frozenset({"agentic", "route"})


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall((text or "").lower()))


def is_critical_task(user_input: str) -> bool:
    return bool(_CRITICAL_TASK_RE.search(user_input or ""))


def capability_from_outcomes(outcomes: list[dict], domain: str = "") -> dict:
    domain = (domain or "").strip().lower()
    rows = list(outcomes or [])
    if domain:
        rows = [
            o for o in rows
            if domain in str(o.get("tool") or "").lower()
            or domain in str(o.get("detail") or "").lower()
        ]
    n = len(rows)
    if n == 0:
        return {
            "domain": domain or "any",
            "samples": 0,
            "success_rate": None,
            "confidence": "unknown",
            "avoid": False,
        }
    successes = sum(1 for o in rows if o.get("ok"))
    rate = successes / n
    avoid = n >= 3 and rate <= 0.34
    confidence = "high" if n >= 4 else "moderate" if n >= 2 else "low"
    return {
        "domain": domain or "any",
        "samples": n,
        "success_rate": round(rate, 3),
        "confidence": confidence,
        "avoid": avoid,
    }


def should_attempt(
    *,
    user_input: str,
    mode: str = "agentic",
    energy: float = 0.5,
    uncertainty: float = 0.0,
    tool_outcomes: list[dict] | None = None,
    enabled: bool = True,
) -> tuple[bool, str, str]:
    """Return (ok, reason, action) with action in proceed|degrade_chat|defer|clarify."""
    if not enabled:
        return True, "attempt gate disabled", "proceed"
    text = (user_input or "").strip()
    if not text:
        return True, "empty input", "proceed"
    if is_critical_task(text):
        return True, "critical or time-sensitive request", "proceed"

    mode_norm = (mode or "").strip().lower()
    soft = mode_norm in _SOFT_MODES

    if mode_norm == "agentic":
        text_tokens = _tokens(text)
        scoped_outcomes = [
            o for o in (tool_outcomes or [])
            if text_tokens & _tokens(f"{o.get('tool') or ''} {o.get('detail') or ''}")
        ]
        cap = capability_from_outcomes(scoped_outcomes, "")
    else:
        # Pre-route (mode="route"): task/tool isn't resolved yet, so there is
        # no reliable way to scope outcomes to "the current task/tool" — skip
        # the avoid-rule rather than risk unrelated failures triggering it.
        cap = {"domain": "any", "samples": 0, "success_rate": None, "confidence": "unknown", "avoid": False}

    if soft and energy < 0.28:
        return False, "energy low; discretionary work can wait", "defer"

    if soft and uncertainty > 0.62:
        if len(text) < 80 and ("?" in text or len(_tokens(text)) <= 6):
            return False, "uncertainty elevated; need a clearer ask", "clarify"
        return False, "uncertainty elevated; prefer lighter chat path", "degrade_chat"

    if soft and cap.get("avoid") and (cap.get("samples") or 0) >= 3:
        return False, "recent tool outcomes are weak; prefer lighter path", "degrade_chat"

    return True, "self-assessment clear", "proceed"
