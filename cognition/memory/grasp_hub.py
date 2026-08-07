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
        emo = float(turn.emotion)
        if hasattr(turn, "valence_label"):
            vtag = turn.valence_label()
        elif emo >= 0.25:
            vtag = "positive"
        elif emo <= -0.25:
            vtag = "negative"
        else:
            vtag = "neutral"
        _evictions.insert(
            0,
            {
                "user": (turn.user or "")[:160],
                "assistant": (turn.assistant or "")[:160],
                "score": round(float(turn.score), 4),
                "emotion": round(emo, 4),
                "valence_tag": vtag,
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
    buf = get_live_buffer()
    with _lock:
        state = _enrich_valence(buf.studio_state())
        state["mode"] = "live"
        state["evictions"] = list(_evictions)
        state["updated_at"] = time.time()
        return state


def _publish_unlocked() -> None:
    global _last_publish
    if not GRASP_LIVE_ENABLED:
        return
    try:
        buf = get_live_buffer()
        state = _enrich_valence(buf.studio_state())
        state["mode"] = "live"
        state["evictions"] = list(_evictions)
        state["updated_at"] = time.time()
        path = Path(GRASP_LIVE_STATE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        _last_publish = time.time()
    except Exception:
        pass


def read_live_snapshot() -> dict[str, Any] | None:
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


def get_context_block(*, max_tokens: int | None = 1200, touch: bool = True) -> str:
    """Return the scored WM block for injection into the LLM system prompt.

    Empty string when disabled or buffer is empty. touch=True bumps recall_count
    on included slots so frequently-used focus items stick longer.
    """
    if not GRASP_LIVE_ENABLED:
        return ""
    try:
        buf = get_live_buffer()
        with _lock:
            block = buf.get_context_block(max_tokens=max_tokens, touch=touch)
            if block:
                _publish_unlocked()
            return block or ""
    except Exception:
        return ""


def _enrich_valence(state: dict[str, Any]) -> dict[str, Any]:
    """Attach valence_tag to studio slots from emotion factor or raw emotion."""
    try:
        from cognition.memory.grasp import valence_tag
    except Exception:
        def valence_tag(emotion: float) -> str:  # type: ignore[misc]
            if emotion >= 0.25:
                return "positive"
            if emotion <= -0.25:
                return "negative"
            return "neutral"
    slots = state.get("slots") or []
    for s in slots:
        if s.get("valence_tag"):
            continue
        emo = s.get("emotion")
        if emo is None:
            fe = (s.get("factors") or {}).get("emotion")
            if fe is not None:
                emo = float(fe) * 2.0 - 1.0
            else:
                emo = 0.0
        s["emotion"] = round(float(emo), 4)
        s["valence_score"] = s["emotion"]
        s["valence_tag"] = valence_tag(float(emo))
    return state


def install_into_think(think: Any) -> bool:
    """Wrap AikoThink for live WM recording + prompt injection.

    Called once from system.wakeup after think is constructed.
    - _store_async: fill Grasp buffer after every completed turn
    - reset_context: clear buffer on /reset
    - _stream_response: inject scored <grasp> block into system prompt
      (covers localchat + webchat; agentic builds its own system string)
    """
    if think is None or not GRASP_LIVE_ENABLED:
        return False
    if getattr(think, "_grasp_live_installed", False):
        return True

    orig_store = think._store_async
    orig_reset = think.reset_context
    orig_stream = think._stream_response

    def _store_async(user_input: str, response_text: str) -> None:
        orig_store(user_input, response_text)
        try:
            record_turn(user_input, response_text)
        except Exception:
            pass

    def _reset_context() -> None:
        orig_reset()
        try:
            clear_live()
        except Exception:
            pass

    def _stream_response(messages: list, system: str = "", token_callback=None) -> str:
        try:
            block = get_context_block(max_tokens=1200, touch=True)
            if block:
                system = f"{system}\n\n{block}" if system else block
        except Exception:
            pass
        return orig_stream(messages, system=system, token_callback=token_callback)

    think._store_async = _store_async  # type: ignore[method-assign]
    think.reset_context = _reset_context  # type: ignore[method-assign]
    think._stream_response = _stream_response  # type: ignore[method-assign]
    think._grasp_live_installed = True
    try:
        install_into_agentic()
    except Exception:
        pass
    return True


def install_into_agentic() -> bool:
    """Inject Grasp into agentic turns by wrapping _stream_agent_message.

    That helper receives the full messages list (system first); we append the
    scored <grasp> block onto messages[0]["content"] once per call.
    """
    if not GRASP_LIVE_ENABLED:
        return False
    try:
        from agentic import agentic as ag
    except Exception:
        return False
    if getattr(ag, "_grasp_live_installed", False):
        return True
    if not hasattr(ag, "_stream_agent_message"):
        return False
    orig = ag._stream_agent_message

    def _stream_agent_message(owner, messages, tools, token_callback=None):
        try:
            if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
                block = get_context_block(max_tokens=1200, touch=True)
                if block:
                    content = messages[0].get("content") or ""
                    if "<grasp>" not in content:
                        messages[0] = dict(messages[0])
                        messages[0]["content"] = f"{content}\n\n{block}" if content else block
        except Exception:
            pass
        return orig(owner, messages, tools, token_callback)

    ag._stream_agent_message = _stream_agent_message  # type: ignore[method-assign]
    ag._grasp_live_installed = True
    return True
