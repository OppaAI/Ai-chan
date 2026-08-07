"""Grasp Studio backend — visualize temporary working-memory slots."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Aiko Grasp Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="grasp-frontend")

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


@app.get("/api/health")
async def health():
    return {"ok": True, "service": "grasp-studio"}


@app.get("/api/state")
async def get_state():
    buf = _get_demo()
    state = buf.studio_state()
    state["evictions"] = list(_eviction_log)
    state["mode"] = "demo"
    return state


@app.post("/api/fill")
async def fill_turn(payload: dict = Body(...)):
    user = str(payload.get("user") or "").strip()
    assistant = str(payload.get("assistant") or "").strip()
    if not user and not assistant:
        return {"ok": False, "error": "empty turn"}
    buf = _get_demo()
    evicted = buf.fill(user, assistant)
    return {
        "ok": True,
        "evicted": len(evicted),
        "state": buf.studio_state(),
        "evictions": list(_eviction_log),
    }


@app.post("/api/touch")
async def touch_context():
    buf = _get_demo()
    _ = buf.get_context_block(touch=True)
    return {"ok": True, "state": buf.studio_state(), "evictions": list(_eviction_log)}


@app.post("/api/reset")
async def reset():
    global _eviction_log
    buf = _get_demo()
    buf.clear()
    _eviction_log = []
    return {"ok": True, "state": buf.studio_state(), "evictions": []}


@app.post("/api/demo/seed")
async def seed_demo():
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
    return {"ok": True, "state": buf.studio_state(), "evictions": list(_eviction_log)}


@app.get("/")
async def serve_studio():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)
