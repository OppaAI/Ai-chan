"""system/brain_trace.py

Per-step tracer for Aiko's cognitive pipeline. When AIKO_TRACE_BRAIN=1
(set automatically by `python main.py --debug`), every instrumented
function emits a structured step to:

  1. Live UI  — colored line via ui.add_message('sys', ...) so the trace
                scrolls past in the WebUI / CLI terminal in real time.
                Each step is one "screen" separated by a header rule.
  2. File     — appended to /tmp/aiko_trace_<YYYYMMDD-HHMMSS>.txt with
                explicit `--- screen N: <step> ---` separators so you can
                page through a session after the fact.

Off by default. Cost when off is a single `os.getenv` per instrumented
function call (negligible).

Design:
  - brain_trace.record_step(name, inputs, outputs, **meta)
      Lightweight call-site helper. Wraps the value rendering so call
      sites stay one-liners.
  - brain_trace.step(name, **inputs) context manager
      Used when a step has a duration to measure or when you want the
      start frame rendered before the body runs and the result frame
      rendered after.

Every step records:
  index        monotonic sequence number
  name         function/method name (e.g. "think.route")
  layer        "transport" | "gate" | "route" | "recall" | "rerank" |
               "context" | "stream" | "review" | "write" | "consolidate"
  inputs       dict of what flowed in (truncated strings, no secrets)
  outputs      dict of what came out (truncated strings)
  duration_ms  wall-clock time the step took (when step() is used)
  factors      list of human-readable reasons that drove the outcome
  extras       arbitrary structured fields
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from system.config import env_int

# ── configuration ────────────────────────────────────────────────────────────

TRACE_ENABLED = os.getenv("AIKO_TRACE_BRAIN", "0").lower() in {"1", "true", "yes", "on"}
TRACE_FILE_PATH = os.getenv("AIKO_TRACE_FILE", "")  # default: /tmp/aiko_trace_<ts>.txt
TRACE_UI_ENABLED = os.getenv("AIKO_TRACE_UI", "1").lower() in {"1", "true", "yes", "on"}
TRACE_MAX_VALUE_CHARS = env_int("AIKO_TRACE_MAX_VALUE_CHARS", 400)

# ── ANSI colours ──────────────────────────────────────────────────────────────
# Distinct color per layer so the eye can group steps in the live stream.
# Auto-disabled when stdout isn't a TTY (matches system/orchestrate.py).
_COLOR_ENABLED = sys.stdout.isatty()

_RESET = "\033[0m"
_BOLD = "\033[1m"

LAYER_COLORS = {
    "transport":    "\033[38;5;245m",   # grey
    "preflight":    "\033[38;5;245m",   # grey
    "gate":         "\033[38;5;226m",   # yellow
    "route":        "\033[38;5;141m",   # purple
    "recall":       "\033[38;5;183m",   # lavender
    "rerank":       "\033[38;5;218m",   # pink
    "context":      "\033[38;5;215m",   # orange
    "stream":       "\033[38;5;87m",    # cyan
    "review":       "\033[38;5;117m",   # light blue
    "write":        "\033[38;5;154m",   # lime
    "consolidate":  "\033[38;5;80m",    # med cyan
    "schedule":     "\033[38;5;159m",   # light cyan
}
_NAME_COLOR = "\033[38;5;222m"     # tan
_INPUT_COLOR = "\033[38;5;47m"     # green
_OUTPUT_COLOR = "\033[38;5;51m"    # bright cyan
_FACTOR_COLOR = "\033[38;5;223m"   # peach
_DIM = "\033[2m"
_HEADER = "\033[38;5;231;1m"       # bold white

_SENTINEL = "🧠"  # visible marker for the trace header line

# ── global state ──────────────────────────────────────────────────────────────

_lock = threading.Lock()
_steps: deque[dict] = deque(maxlen=2000)   # in-memory ring buffer for the session
_pending_file_steps: list[dict] = []        # buffered steps for batched file writes
_ui_sink = None                # injected by main.py → ui.add_message("sys", ...)
_file_handle = None            # opened lazily on first record
_session_id: str = ""           # YYYYMMDD-HHMMSS for the report file name
_turn_counter: int = 0
_turn_started_at: float | None = None


# ── small render helpers ─────────────────────────────────────────────────────

def _c(code: str, text: str) -> str:
    if not _COLOR_ENABLED:
        return text
    return f"{code}{text}{_RESET}"


def _trunc(value: Any, limit: int = TRACE_MAX_VALUE_CHARS) -> str:
    """Render an arbitrary value as a short, single-line string."""
    if value is None:
        return "None"
    if isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, str):
        one = value.replace("\n", "↵").replace("\r", "")
        if len(one) > limit:
            return one[: limit - 1] + "…"
        return one
    if isinstance(value, dict):
        try:
            rendered = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            rendered = repr(value)
        return _trunc(rendered, limit)
    if isinstance(value, (list, tuple)):
        try:
            rendered = json.dumps(list(value), ensure_ascii=False, default=str)
        except Exception:
            rendered = repr(value)
        return _trunc(rendered, limit)
    return _trunc(repr(value), limit)


def _short_type(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return f"str({len(value)})"
    if isinstance(value, dict):
        return f"dict({len(value)})"
    if isinstance(value, list):
        return f"list({len(value)})"
    if isinstance(value, tuple):
        return f"tuple({len(value)})"
    return type(value).__name__


# ── public surface ───────────────────────────────────────────────────────────

def set_ui_sink(sink) -> None:
    """Inject the UI object whose add_message('sys', ...) we'll call.

    Called once at startup after the UI is built. Passing None reverts to
    file-only output (useful when running --cli where the sink might not
    support add_message of arbitrary sys text without breaking the layout).
    """
    global _ui_sink
    _ui_sink = sink


def _ensure_file():
    global _file_handle, _session_id
    if _file_handle is not None:
        return _file_handle
    if not TRACE_ENABLED:
        return None
    if not TRACE_FILE_PATH:
        _session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path(f"/tmp/aiko_trace_{_session_id}.txt")
    else:
        path = Path(TRACE_FILE_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _file_handle = open(path, "a", encoding="utf-8", buffering=1)
        _file_handle.write(
            f"=== Aiko brain trace — session {_session_id} ===\n"
            f"=== started {datetime.now().isoformat()} ===\n\n"
        )
        return _file_handle
    except Exception:
        _file_handle = None
        return None


def begin_turn(label: str = "") -> None:
    """Mark the start of a new turn (one user prompt → response)."""
    global _turn_counter, _turn_started_at
    if not TRACE_ENABLED:
        return
    _turn_counter += 1
    _turn_started_at = time.monotonic()
    banner = (
        f"{_SENTINEL}  ── turn #{_turn_counter}"
        f"{f'  [{label}]' if label else ''}"
        f"  @ {datetime.now().strftime('%H:%M:%S')} ──"
    )
    _emit(banner, _HEADER)


def end_turn() -> None:
    """Mark the end of the current turn — flush a footer with total ms."""
    global _turn_started_at
    if not TRACE_ENABLED or _turn_started_at is None:
        return
    elapsed_ms = int((time.monotonic() - _turn_started_at) * 1000)
    banner = (
        f"{_SENTINEL}  ── turn #{_turn_counter} done in {elapsed_ms} ms ──"
    )
    _emit(banner, _HEADER)
    _emit("")  # blank line separator
    _turn_started_at = None
    flush()


def record_step(
    name: str,
    *,
    layer: str = "context",
    inputs: dict | None = None,
    outputs: dict | None = None,
    factors: list[str] | None = None,
    extras: dict | None = None,
    duration_ms: float | None = None,
) -> None:
    """Record one named step to both the live UI sink and the trace file.

    name:    "think.route", "memorize.search", "attention.situation_context", …
    layer:   one of LAYER_COLORS keys — drives the colour of the live line
    inputs:  short labels of what flowed into the step
    outputs: short labels of what came out
    factors: human-readable strings explaining WHY (route chosen, memory
             selected, rerank boosted, etc.)
    """
    if not TRACE_ENABLED:
        return

    step = {
        "index": len(_steps) + 1,
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "name": name,
        "layer": layer,
        "inputs": inputs or {},
        "outputs": outputs or {},
        "factors": factors or [],
        "extras": extras or {},
        "duration_ms": duration_ms,
    }
    with _lock:
        _steps.append(step)

    _emit_step(step)


@contextmanager
def step(name: str, *, layer: str = "context", inputs: dict | None = None, factors: list[str] | None = None):
    """Context manager: render the start frame immediately, then the end
    frame with `outputs` populated when the body finishes.

    Usage:
        with brain_trace.step("memorize.search", layer="recall",
                              inputs={"query": q}) as ctx:
            results = backend.search(...)
            ctx.set(outputs={"hits": results})
    """
    if not TRACE_ENABLED:
        yield _NullCtx()
        return

    start = time.monotonic()
    inputs = inputs or {}
    factors = factors or []
    step_obj = {
        "index": len(_steps) + 1,
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "name": name,
        "layer": layer,
        "inputs": inputs,
        "outputs": {},
        "factors": factors,
        "extras": {},
        "duration_ms": None,
    }
    with _lock:
        _steps.append(step_obj)
    _emit_step(step_obj, phase="start")

    ctx = _StepCtx(step_obj, start)
    try:
        yield ctx
    finally:
        step_obj["duration_ms"] = int((time.monotonic() - start) * 1000)
        _emit_step(step_obj, phase="end")


class _NullCtx:
    def set(self, **_):
        pass
    def add_factor(self, _):
        pass
    def add_extra(self, **_):
        pass


class _StepCtx:
    def __init__(self, step_obj: dict, start: float):
        self._step = step_obj
        self._start = start

    def set(self, *, outputs: dict | None = None, factors: list[str] | None = None) -> None:
        if outputs:
            self._step["outputs"].update(outputs)
        if factors:
            self._step["factors"].extend(factors)

    def add_factor(self, factor: str) -> None:
        self._step["factors"].append(factor)

    def add_extra(self, **kwargs) -> None:
        self._step["extras"].update(kwargs)


# ── internal rendering ───────────────────────────────────────────────────────

def _emit(text: str, color: str = "") -> None:
    """Push a line to the UI sink (if configured) and buffer for file."""
    if TRACE_UI_ENABLED and _ui_sink is not None:
        try:
            _ui_sink.add_message("sys", _c(color, text) if color else text)
        except Exception:
            pass
    # Buffer for batched file write
    with _lock:
        _pending_file_steps.append(text)


def _format_kv(label: str, value: Any, color: str) -> str:
    rendered = _trunc(value)
    return _c(_DIM, f"    {label}: ") + _c(color, f"{rendered}  ") + _c(_DIM, f"({_short_type(value)})")


def _emit_step(step: dict, *, phase: str = "end") -> None:
    layer_color = LAYER_COLORS.get(step["layer"], "")
    name_colored = _c(_NAME_COLOR, step["name"])
    layer_colored = _c(layer_color, f"[{step['layer']}]") if layer_color else f"[{step['layer']}]"
    duration = step.get("duration_ms")
    dur_str = _c(_DIM, f"  ({duration} ms)") if duration is not None else ""

    header_line = (
        f"  {_c(layer_color, '┌─')} "
        f"{layer_colored} {name_colored}{dur_str}"
    )
    _emit(header_line)

    if step["inputs"]:
        for k, v in step["inputs"].items():
            _emit(_format_kv(k, v, _INPUT_COLOR))
    if step["outputs"]:
        for k, v in step["outputs"].items():
            _emit(_format_kv(k, v, _OUTPUT_COLOR))
    if step["factors"]:
        _emit(_c(_DIM, "    factors:"))
        for f in step["factors"]:
            _emit(_c(_FACTOR_COLOR, f"      • {f}"))
    if step["extras"]:
        for k, v in step["extras"].items():
            _emit(_format_kv(k, v, _DIM))

    if phase == "start":
        _emit(_c(layer_color, "  │ running…"))
    else:
        _emit(_c(layer_color, "  └─ done"))


def get_recent_steps(limit: int = 50) -> list[dict]:
    """Return the last `limit` recorded steps — useful for /trace command."""
    with _lock:
        return list(_steps)[-limit:]


def reset() -> None:
    """Clear the in-memory ring buffer (does not touch the file)."""
    with _lock:
        _steps.clear()


def flush() -> None:
    """Flush buffered trace lines to the trace file."""
    if not TRACE_ENABLED:
        return
    fh = _ensure_file()
    if not fh:
        return
    with _lock:
        for line in _pending_file_steps:
            fh.write(line + "\n")
        _pending_file_steps.clear()
    fh.flush()


def shutdown() -> None:
    global _file_handle
    flush()
    if _file_handle is not None:
        try:
            _file_handle.write(f"\n=== trace closed {datetime.now().isoformat()} ===\n")
            _file_handle.close()
        except Exception:
            pass
        _file_handle = None


__all__ = [
    "TRACE_ENABLED",
    "set_ui_sink",
    "begin_turn",
    "end_turn",
    "record_step",
    "step",
    "get_recent_steps",
    "reset",
    "shutdown",
]