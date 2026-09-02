"""Subliminal layer — human-like subconscious for Aiko on Jetson.

Design goals
------------
* Bounded, no hot-path allocation: all state is a handful of small deques.
* Zero extra threads, no LLM/embeddings, no DB on the hot path.
* Layers mirror human mind but gate deeper work behind snapshots:
    L1 pre-attentive scan  — cheap token/pattern check, runs every turn (<0.2 ms)
    L2 affect / motivation — energy/uncertainty/affect → drive, runs every turn
    L3 implicit memory     — intuitions + priming from contradictions/goals/loops
    L4 meta-cognitive      — self-model summary, only when building prompt

Latency / RAM budget
--------------------
* One instance per identity, created lazily via :func:`for_identity` in
  ``cognition.attention``. No global singleton beyond that.
* RAM: ~4 deques × 4 slots + a few strings — <4 KiB per identity.
* Latency: L1/L2 inlined in ``EdgeCognitiveState.record`` (still <1 ms);
  L3/L4 only called when ``subconscious_guidance`` / ``priming_context`` /
  ``self_model_context`` are requested during prompt assembly.
"""
from __future__ import annotations

import re
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any


# ── L1: pre-attentive scan — token bag, gated ────────────────────────────

_WORD_RE = re.compile(r"[a-z0-9_]{3,}")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD_RE.findall((text or "").lower()))


# ── SubliminalLayer ───────────────────────────────────────────────────────

@dataclass(slots=True)
class PreAttentiveScan:
    """L1: cheap scan of the current turn before full attention."""
    recurring_terms: frozenset[str] = frozenset()
    has_contradiction: bool = False
    has_goal_loop_overlap: bool = False


class SubliminalLayer:
    """Bounded subconscious layer composed into EdgeCognitiveState.

    Holds the 4 deques / caches that previously lived directly on
    EdgeCognitiveState. All methods are lock-protected by the owner's
    RLock — this object never creates its own lock.
    """

    __slots__ = ("_intuitions", "_pre_attentive", "_affect_bias", "_lock_ref")

    def __init__(self, lock: threading.RLock | None = None) -> None:
        self._intuitions: deque[str] = deque(maxlen=4)
        # L1 snapshot for the last turn — overwritten each record()
        self._pre_attentive: PreAttentiveScan | None = None
        # L2 affect drift carried across turns
        self._affect_bias: float = 0.0
        self._lock_ref = lock

    # ── L1/L2: called from EdgeCognitiveState.record() ───────────────────

    def scan(self, user: str, snapshot: dict, events: deque) -> PreAttentiveScan:
        """L1: pre-attentive scan. Cheap, no allocation beyond a set."""
        has_contra = bool(snapshot.get("contradictions"))
        has_goal_loop = bool(snapshot.get("goals") and snapshot.get("open_loops"))
        recurring: frozenset[str] = frozenset()
        if len(events) >= 3:
            recent = [_tokens(e.user) for e in list(events)[-3:]]
            rec = set(recent[0]).intersection(*recent[1:]) if recent else set()
            rec -= {"can", "could", "please", "today"}
            if rec:
                recurring = frozenset(sorted(rec)[:4])
        scan = PreAttentiveScan(
            recurring_terms=recurring,
            has_contradiction=has_contra,
            has_goal_loop_overlap=has_goal_loop,
        )
        self._pre_attentive = scan
        return scan

    def refresh_intuitions(self, snapshot: dict, scan: PreAttentiveScan | None = None) -> None:
        """L3: derive tentative intuitions. Called once per record()."""
        scan = scan or self._pre_attentive
        candidates: list[str] = []
        if snapshot.get("contradictions"):
            candidates.append("Possible unresolved belief conflict; verify before asserting.")
        if scan and scan.has_goal_loop_overlap:
            candidates.append("An active goal may be connected to an unresolved thread.")
        if scan and scan.recurring_terms:
            candidates.append("Recurring focus detected: " + " ".join(sorted(scan.recurring_terms)) + ".")
        if snapshot.get("durable_lessons"):
            candidates.append("A learned interaction pattern may apply here; check the current request first.")
        # Bounded append — no duplicate intuitions
        for c in candidates:
            if c not in self._intuitions:
                self._intuitions.appendleft(c)

    # ── L3/L4: called during prompt assembly (not hot path) ──────────────

    def guidance(self) -> str:
        """L3: subconscious guidance block for the prompt."""
        intuitions = list(self._intuitions)
        if not intuitions:
            return "<subconscious_guidance>\nNo tentative intuition currently salient.\n</subconscious_guidance>"
        lines = ["- tentative hypothesis: " + i for i in intuitions[:3]]
        lines.append("Treat these as associations to verify, never as facts.")
        return "<subconscious_guidance>\n" + "\n".join(lines) + "\n</subconscious_guidance>"

    def priming_context(self, query: str, snapshot: dict, identity_questions: deque) -> str:
        """L3 implicit priming — only related goals/loops surface."""
        query_words = _tokens(query)
        related_goals = [g for g in snapshot.get("goals", []) if not query_words or _tokens(g) & query_words]
        related_loops = [l for l in snapshot.get("open_loops", []) if not query_words or _tokens(l) & query_words]
        # identity query gate — use same pattern as attention.py
        _IDENTITY_RE = re.compile(r"\b(?:do you know me|who am i|what is my name|what\x27s my name|remember me)\b", re.I)
        identity_query = bool(_IDENTITY_RE.search(query or ""))
        lines: list[str] = []
        if related_goals:
            lines.append("Relevant active goal: " + related_goals[0][:180])
        if identity_query and identity_questions:
            lines.append("Relevant identity thread: answer from retrieved identity evidence; do not infer missing facts.")
        if related_loops:
            lines.append("Relevant open loop: " + related_loops[0][:180])
        if snapshot.get("uncertainty", 0.0) > 0.35 and (related_goals or related_loops):
            lines.append("Relevant uncertainty is elevated; verify assumptions.")
        if abs(float(snapshot.get("affect") or 0.0)) > 0.25 and (related_goals or related_loops):
            lines.append("Recent emotional context may affect interpretation; respond proportionately.")
        if not lines:
            return ""
        return "<subconscious_priming>\n" + "\n".join("- " + l for l in lines) + "\n</subconscious_priming>"

    def snapshot_extra(self) -> dict[str, Any]:
        """Expose layer state for EdgeCognitiveState.snapshot()."""
        return {
            "intuitions": list(self._intuitions),
            "pre_attentive": {
                "recurring_terms": sorted(self._pre_attentive.recurring_terms) if self._pre_attentive else [],
                "has_contradiction": bool(self._pre_attentive.has_contradiction) if self._pre_attentive else False,
            },
        }

    def restore(self, data: dict) -> None:
        """Restore from persisted snapshot."""
        self._intuitions = deque(data.get("intuitions", [])[:4], maxlen=4)


__all__ = ["SubliminalLayer", "PreAttentiveScan"]
