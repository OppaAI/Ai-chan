"""Live Grasp hub — shared working-memory buffer + studio snapshot.

Aiko's think process owns the in-memory GraspBuffer. After each turn it
publishes an atomic JSON snapshot so Grasp Studio (separate process) can
poll live state without sharing an address space.

Path (override with GRASP_LIVE_STATE_PATH):
  ~/.local/share/aiko/grasp/live_state.json
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from cognition.memory.grasp import GraspBuffer, GraspTurn, build_grasp


def _env_flag(name: str, default: str = "1") -> bool:
    return str(os.getenv(name, default)).strip().lower() not in ("0", "false", "no", "off", "")


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or str(v).strip() == "" else str(v).strip()


GRASP_LIVE_ENABLED = _env_flag("GRASP_LIVE_ENABLED", "1")
GRASP_LIVE_STATE_PATH = _env_str(
    "GRASP_LIVE_STATE_PATH",
    str(Path.home() / ".local" / "share" / "aiko" / "grasp" / "live_state.json"),
)

_lock = threading.RLock()
_buffer: GraspBuffer | None = None
_evictions: list[dict[str, Any]] = []
_MAX_EVICT = 40
_last_publish = 0.0


def _on_evict(turn: GraspTurn) -> None:
    with _lock:
        _evictions.insert(
            0,
            {
                "user": (turn.user or "")[:160],
                "assistant": (turn.assistant or "")[:160],
                "score": round(float(turn.score), 4),
                "recall_count": int(turn.recall_count),
                "tokens": int(turn.tokens),
                "created_turn": int(turn.created_turn),
            },
        )
        del _evictions[_MAX_EVICT:]


def get_live_buffer() -> GraspBuffer:
    """Lazy singleton used by the Aiko process."""
    global _buffer
    with _lock:
        if _buffer is None:
            _buffer = build_grasp(on_evict=_on_evict)
        return _buffer


def set_static_anchor_tokens(tokens: set[str] | list[str] | None) -> None:
    if not tokens:
        return
    buf = get_live_buffer()
    buf.set_static_anchor(tokens)


def record_turn(
    user: str,
    assistant: str,
    *,
    user_ts: float | None = None,
    assistant_ts: float | None = None,
) -> list[GraspTurn]:
    """Fill live buffer after a completed conversation turn. No-op if disabled."""
    if not GRASP_LIVE_ENABLED:
        return []
    user = (user or "").strip()
    assistant = (assistant or "").strip()
    if not user and not assistant:
        return []
    buf = get_live_buffer()
    with _lock:
        evicted = buf.fill(
            user,
            assistant,
            user_ts=user_ts if user_ts is not None else time.time(),
            assistant_ts=assistant_ts if assistant_ts is not None else time.time(),
        )
        # Context injection rehearsal for items that remain in focus
        try:
            buf.get_context_block(touch=True)
        except Exception:
            pass
        _publish_unlocked()
        return list(evicted)


def clear_live() -> None:
    with _lock:
        if _buffer is not None:
            _buffer.clear()
        _evictions.clear()
        _publish_unlocked()


def live_studio_state() -> dict[str, Any]:
    """In-process snapshot (same process as Aiko)."""
    buf = get_live_buffer()
    with _lock:
        state = buf.studio_state()
        state["mode"] = "live"
        state["evictions"] = list(_evictions)
        state["updated_at"] = time.time()
        return state


def _publish_unlocked() -> None:
    """Atomic write of live state JSON for the studio process."""
    global _last_publish
    if not GRASP_LIVE_ENABLED:
        return
    try:
        buf = get_live_buffer()
        state = buf.studio_state()
        state["mode"] = "live"
        state["evictions"] = list(_evictions)
        state["updated_at"] = time.time()
        path = Path(GRASP_LIVE_STATE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = json.dumps(state, ensure_ascii=False)
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
        _last_publish = time.time()
    except Exception:
        pass


def read_live_snapshot() -> dict[str, Any] | None:
    """Read published snapshot (studio process). None if missing/stale empty."""
    path = Path(GRASP_LIVE_STATE_PATH)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        data.setdefault("mode", "live")
        return data
    except Exception:
        return None


def snapshot_age_seconds() -> float | None:
    path = Path(GRASP_LIVE_STATE_PATH)
    if not path.is_file():
        return None
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None
