"""Bounded self-assessment gate for agentic execution.

Used by EdgeCognitiveState.should_attempt and AikoThink.agentic_chat.
Keeps thresholds conservative: critical work always proceeds.
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

    cap = capability_from_outcomes(tool_outcomes or [], "")

    if mode == "agentic" and energy < 0.28:
        return False, "energy low; discretionary agentic work can wait", "defer"

    if mode == "agentic" and uncertainty > 0.62:
        if len(text) < 80 and ("?" in text or len(_tokens(text)) <= 6):
            return False, "uncertainty elevated; need a clearer ask", "clarify"
        return False, "uncertainty elevated; prefer chat over agentic loop", "degrade_chat"

    if mode == "agentic" and cap.get("avoid") and (cap.get("samples") or 0) >= 3:
        return False, "recent tool outcomes are weak; prefer lighter path", "degrade_chat"

    return True, "self-assessment clear", "proceed"
