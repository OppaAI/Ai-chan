"""
system/notice.py
Per-session mailbox for runtime notices — subsystem failures worth
surfacing to the LLM (and through it, the user) mid-conversation.

This is deliberately NOT a logger. aiko.log/aiko.error.log are the
permanent record; NoticeBus is a small, bounded, ephemeral queue that
orchestrate.py drains once per turn so Aiko can react gracefully to a
failure instead of the user shipping into silence (e.g. TTS dies ->
Aiko says "my voice cut out, I'll keep chatting in text" instead of
typing while the user waits for audio that never comes).

Scoped per session_id: a TTS failure in one user's session must never
leak a notice into another user's next turn. Call drop_session() on
logout/disconnect so the dict doesn't grow unbounded across the
multi-user/multi-session lifetime of the process.

Rules of thumb for callers:
- Only push from user-visible subsystems: TTS, ASR, memory read/write,
  messenger adapters. Not every exception belongs here — that's what
  aiko.log is for.
- One line, human-readable, no tracebacks. Models fixate on raw
  tracebacks; "TTS: synthesis failed (timeout)" is more useful to Aiko
  than a stack trace.
- drain() is destructive by design — notices don't haunt future turns.

Usage:
    from system.notice import get_notice_bus

    bus = get_notice_bus(session_id)
    bus.push("TTS", "synthesis failed (timeout)")

    # in orchestrate, before calling the LLM for this turn:
    notes = bus.drain()
    if notes:
        system_note = "\n".join(f"[system notice] {n}" for n in notes)
        # prepend/inject system_note into this turn's context
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

MAX_NOTICES_PER_DRAIN = 3   # keep prompts from being flooded by a failure burst
MAX_PENDING           = 20  # hard cap per session so a runaway loop can't leak memory


@dataclass
class Notice:
    area: str
    brief: str
    ts: float = field(default_factory=time.time)

    def format(self) -> str:
        return f"{self.area}: {self.brief}"


class NoticeBus:
    """Bounded, thread-safe mailbox for one session's runtime notices."""

    def __init__(self, max_pending: int = MAX_PENDING):
        self._lock = threading.Lock()
        self._notices: deque[Notice] = deque(maxlen=max_pending)

    def push(self, area: str, brief: str) -> None:
        with self._lock:
            self._notices.append(Notice(area=area, brief=brief))

    def drain(self, limit: int = MAX_NOTICES_PER_DRAIN) -> list[str]:
        """Pop and clear pending notices, most recent `limit` first.

        If more than `limit` notices piled up between turns, the oldest
        ones in the burst are dropped silently rather than delivered late
        — a stale notice about last turn's hiccup is just noise by the
        time this turn's response is being generated.
        """
        with self._lock:
            if not self._notices:
                return []
            pending = list(self._notices)[-limit:]
            self._notices.clear()
            return [n.format() for n in pending]

    def peek_count(self) -> int:
        with self._lock:
            return len(self._notices)


_buses: dict[str, NoticeBus] = {}
_buses_lock = threading.Lock()


def get_notice_bus(session_id: str) -> NoticeBus:
    """Return the NoticeBus for a session, creating it on first use."""
    with _buses_lock:
        bus = _buses.get(session_id)
        if bus is None:
            bus = NoticeBus()
            _buses[session_id] = bus
        return bus


def drop_session(session_id: str) -> None:
    """Remove a session's bus. Call on logout/disconnect to avoid leaking
    entries across the process's multi-user/multi-session lifetime."""
    with _buses_lock:
        _buses.pop(session_id, None)
