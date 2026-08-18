"""Live Grasp hub — shared working-memory buffer + studio snapshot.

Aiko's think process owns the in-memory GraspBuffer. After each turn it
publishes an atomic JSON snapshot so Grasp Studio (separate process) can
poll live state without sharing an address space.

Path (override with GRASP_LIVE_STATE_PATH):
  ~/.local/share/aiko/grasp/live_state.json
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from cognition.memory.grasp import GraspBuffer, GraspTurn, build_grasp
from cognition.memory.env import env_flag, env_str

log = logging.getLogger("aiko.grasp_hub")


GRASP_LIVE_ENABLED = env_flag("GRASP_LIVE_ENABLED", "1")
GRASP_LIVE_STATE_PATH = env_str(
    "GRASP_LIVE_STATE_PATH",
    str(Path.home() / ".local" / "share" / "aiko" / "grasp" / "live_state.json"),
)

_lock = threading.RLock()
_buffers: dict[str, GraspBuffer] = {}
_evictions: dict[str, list[dict[str, Any]]] = {}
_MAX_EVICT = 40
_last_publish: dict[str, float] = {}
_last_publish_error: dict[str, str | None] = {}

_DEFAULT_IDENTITY = "default"


def _resolve_identity(identity: str | None) -> str:
    """Resolve identity from argument or current_user_id() context or fallback."""
    if identity:
        return identity
    try:
        from system.userspace import current_user_id
        return current_user_id() or _DEFAULT_IDENTITY
    except Exception:
        return _DEFAULT_IDENTITY


def _live_state_path(identity: str) -> Path:
    """Return per-identity snapshot path."""
    base = Path(GRASP_LIVE_STATE_PATH)
    return base.with_name(f"{base.stem}.{identity}{base.suffix}")


def _on_evict(identity: str, turn: GraspTurn) -> None:
    """Eviction callback bound to a specific identity."""
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
        bucket = _evictions.setdefault(identity, [])
        bucket.insert(
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
        del bucket[_MAX_EVICT:]


def get_live_buffer(identity: str | None = None) -> GraspBuffer:
    """Lazy per-identity buffer used by the Aiko process."""
    ident = _resolve_identity(identity)
    with _lock:
        if ident not in _buffers:
            _buffers[ident] = build_grasp(on_evict=lambda turn, _id=ident: _on_evict(_id, turn))
        return _buffers[ident]


def set_static_anchor_tokens(tokens: set[str] | list[str] | None, identity: str | None = None) -> None:
    if not tokens:
        return
    buf = get_live_buffer(identity=identity)
    buf.set_static_anchor(tokens)


def record_turn(
    user: str,
    assistant: str,
    *,
    user_ts: float | None = None,
    assistant_ts: float | None = None,
    identity: str | None = None,
) -> list[GraspTurn]:
    """Fill live buffer after a completed conversation turn. No-op if disabled."""
    if not GRASP_LIVE_ENABLED:
        return []
    user = (user or "").strip()
    assistant = (assistant or "").strip()
    if not user and not assistant:
        return []
    ident = _resolve_identity(identity)
    buf = get_live_buffer(identity=ident)
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
        _publish_unlocked(identity=ident)
        return list(evicted)


def clear_live(identity: str | None = None) -> None:
    ident = _resolve_identity(identity)
    with _lock:
        if ident in _buffers:
            _buffers[ident].clear()
        _evictions.pop(ident, None)
        _publish_unlocked(identity=ident)


def live_studio_state(identity: str | None = None) -> dict[str, Any]:
    ident = _resolve_identity(identity)
    buf = get_live_buffer(identity=ident)
    with _lock:
        state = _enrich_valence(buf.studio_state())
        state["mode"] = "live"
        state["evictions"] = list(_evictions.get(ident, []))
        state["updated_at"] = time.time()
        return state


def _publish_unlocked(identity: str) -> None:
    """Publish per-identity snapshot. Caller must resolve identity and hold _lock."""
    if not GRASP_LIVE_ENABLED:
        return
    try:
        path = _live_state_path(identity)
        buf = get_live_buffer(identity=identity)
        state = _enrich_valence(buf.studio_state())
        state["mode"] = "live"
        state["evictions"] = list(_evictions.get(identity, []))
        state["updated_at"] = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        # Drop stale temp from a prior crashed publish, then create 0600
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, json.dumps(state, ensure_ascii=False).encode("utf-8"))
        finally:
            os.close(fd)
        # Ensure temp file has 0600 before replace
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        # Ensure final file has 0600 after replace
        os.chmod(path, 0o600)
        _last_publish[identity] = time.time()
        _last_publish_error[identity] = None
    except Exception as e:
        _last_publish_error[identity] = f"{type(e).__name__}: {e}"
        log.warning(
            "grasp live snapshot publish failed identity=%s path=%s err=%s",
            identity,
            path,
            e,
            exc_info=True,
        )


def read_live_snapshot(identity: str | None = None) -> dict[str, Any] | None:
    ident = _resolve_identity(identity)
    path = _live_state_path(ident)
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


def snapshot_age_seconds(identity: str | None = None) -> float | None:
    ident = _resolve_identity(identity)
    path = _live_state_path(ident)
    if not path.is_file():
        return None
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def publish_health(identity: str | None = None) -> dict[str, Any]:
    """Snapshot publication status for Studio /api/health."""
    ident = _resolve_identity(identity)
    return {
        "last_publish_at": _last_publish.get(ident) or None,
        "last_publish_error": _last_publish_error.get(ident),
        "path": str(_live_state_path(ident)),
    }


def get_context_block(*, max_tokens: int | None = 1200, touch: bool = True, identity: str | None = None) -> str:
    """Return the scored WM block for injection into the LLM system prompt.

    Empty string when disabled or buffer is empty. touch=True bumps recall_count
    on included slots so frequently-used focus items stick longer.
    """
    if not GRASP_LIVE_ENABLED:
        return ""
    try:
        ident = _resolve_identity(identity)
        buf = get_live_buffer(identity=ident)
        with _lock:
            block = buf.get_context_block(max_tokens=max_tokens, touch=touch)
            if block:
                _publish_unlocked(identity=ident)
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

    with _lock:
        if getattr(think, "_grasp_live_installed", False):
            return True

        orig_store = think._store_async
        orig_reset = think.reset_context
        orig_stream = think._stream_response

        def _store_async(user_input: str, response_text: str) -> None:
            orig_store(user_input, response_text)
            try:
                ident = _resolve_identity(None)
                record_turn(user_input, response_text, identity=ident)
            except Exception:
                pass

        def _reset_context() -> None:
            orig_reset()
            try:
                ident = _resolve_identity(None)
                clear_live(identity=ident)
            except Exception:
                pass

        def _stream_response(messages: list, system: str = "", token_callback=None, emit: bool = True) -> str:
            try:
                ident = _resolve_identity(None)
                block = get_context_block(max_tokens=1200, touch=True, identity=ident)
                if block:
                    system = f"{system}\n\n{block}" if system else block
            except Exception:
                pass
            return orig_stream(messages, system=system, token_callback=token_callback, emit=emit)

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
                ident = _resolve_identity(None)
                block = get_context_block(max_tokens=1200, touch=True, identity=ident)
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
