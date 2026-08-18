"""STM Studio backend — demo buffer + live snapshot from Aiko process."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Aiko STM Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8003",
        "http://localhost:8003",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
SHARED_DIR = Path(__file__).resolve().parents[3] / "shared"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="stm-frontend")

app.mount("/shared", StaticFiles(directory=str(SHARED_DIR), html=True), name="studio-shared")

_demo = None
_eviction_log: list[dict[str, Any]] = []
_MAX_EVICT_LOG = 40


def _get_demo():
    global _demo
    if _demo is None:
        from cognition.memory.grasp import build_grasp

        def _on_evict(turn):
            _eviction_log.insert(
                0,
                {
                    "user": turn.user[:160],
                    "assistant": turn.assistant[:160],
                    "score": round(turn.score, 4),
                    "recall_count": turn.recall_count,
                    "tokens": turn.tokens,
                    "created_turn": turn.created_turn,
                },
            )
            del _eviction_log[_MAX_EVICT_LOG:]

        _demo = build_grasp(
            static_anchor_tokens={
                "aiko", "jetson", "memory", "preference", "cats",
                "schedule", "voice", "persona", "working", "focus",
            },
            on_evict=_on_evict,
            journal_enabled=False,
        )
    return _demo


def _live_state() -> dict[str, Any] | None:
    """Read the snapshot, falling back to the live buffer in this process."""
    try:
        from cognition.memory.grasp_hub import read_live_snapshot, snapshot_age_seconds
        snap = read_live_snapshot()
        age = snapshot_age_seconds() if snap else None
        if not snap:
            from cognition.memory.grasp_hub import live_studio_state
            snap = live_studio_state()
            age = 0.0
        if not snap:
            return None
        snap["mode"] = "live"
        snap["live_age_s"] = age
        snap["live_fresh"] = age is not None and age < 120.0
        return snap
    except Exception:
        return None


@app.get("/api/health")
def health():
    live = _live_state()
    pub: dict = {}
    try:
        from cognition.memory.grasp_hub import publish_health
        pub = publish_health()
    except Exception:
        pub = {}
    return {
        "ok": True,
        "service": "stm-studio",
        "live_available": live is not None,
        "live_fresh": bool(live and live.get("live_fresh")),
        "publish_error": pub.get("last_publish_error"),
        "last_publish_at": pub.get("last_publish_at"),
    }


@app.get("/api/cognition")
def cognition():
    try:
        from cognition.memory.edge_state import for_identity
        from system.userspace import current_user_id
        state = for_identity(current_user_id())
        return {"ok": True, "evaluation": state.evaluation_snapshot(), "health": state.cognitive_health()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.get("/api/state")
def get_state(mode: str = Query("auto")):
    """mode=auto|live|demo — auto prefers live whenever it is available."""
    mode = (mode or "auto").strip().lower()
    if mode in ("auto", "live"):
        live = _live_state()
        if live:
            return live
        if mode == "live":
            return {
                "mode": "live",
                "slots": [],
                "size": 0,
                "total_tokens": 0,
                "turn_counter": 0,
                "miller": {"min": 5, "center": 7, "max": 9},
                "evictions": [],
                "live_available": False,
                "hint": "No live snapshot yet — talk to Aiko with GRASP_LIVE_ENABLED=1",
            }
    buf = _get_demo()
    state = buf.studio_state()
    state["evictions"] = list(_eviction_log)
    state["mode"] = "demo"
    return state


@app.post("/api/fill")
def fill_turn(payload: dict = Body(...)):
    user = str(payload.get("user") or "").strip()
    assistant = str(payload.get("assistant") or "").strip()
    if not user and not assistant:
        return {"ok": False, "error": "empty turn"}
    buf = _get_demo()
    evicted = buf.fill(user, assistant)
    return {
        "ok": True,
        "evicted": len(evicted),
        "state": {**buf.studio_state(), "mode": "demo"},
        "evictions": list(_eviction_log),
    }


@app.post("/api/touch")
def touch_context():
    buf = _get_demo()
    _ = buf.get_context_block(touch=True)
    return {"ok": True, "state": {**buf.studio_state(), "mode": "demo"}, "evictions": list(_eviction_log)}


@app.post("/api/reset")
def reset():
    global _eviction_log
    buf = _get_demo()
    buf.clear()
    _eviction_log = []
    return {"ok": True, "state": {**buf.studio_state(), "mode": "demo"}, "evictions": []}


@app.post("/api/demo/seed")
def seed_demo():
    global _eviction_log
    buf = _get_demo()
    buf.clear()
    _eviction_log = []
    script = [
        ("hey aiko", "Hey! How can I help?"),
        ("remember I prefer dark mode on the jetson", "Got it — dark mode preference noted for the Jetson."),
        ("what time is it?", "It's about mid-morning locally."),
        ("ok thanks", "You're welcome!"),
        ("can you schedule a voice test later?", "Sure — I can set a reminder for a voice test."),
        ("my cat Mochi is allergic to fish", "Noted: Mochi is allergic to fish."),
        ("lol", "😄"),
        ("why is working memory limited to 7±2?", "Miller's Law — limited capacity keeps focus sharp."),
        ("pin that explanation", "Pinned the working-memory explanation for you."),
        ("random filler", "Okay."),
        ("another ack", "Yep."),
        ("show me the persona summary", "I can pull from SOUL.md / persona when needed."),
    ]
    for u, a in script:
        buf.fill(u, a)
    buf.get_context_block(touch=True)
    buf.get_context_block(touch=True)
    return {"ok": True, "state": {**buf.studio_state(), "mode": "demo"}, "evictions": list(_eviction_log)}


@app.get("/")
async def serve_studio():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)
