"""WMC Studio backend — visualize Working Memory Cortex slots, scores, eviction."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Aiko WMC Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="wmc-frontend")

_demo_wmc = None
_eviction_log: list[dict[str, Any]] = []
_MAX_EVICT_LOG = 40


def _get_demo():
    global _demo_wmc
    if _demo_wmc is None:
        from cognition.memory.wmc import build_wmc

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

        _demo_wmc = build_wmc(
            static_anchor_tokens={
                "aiko", "jetson", "memory", "preference", "cats",
                "schedule", "voice", "persona", "working", "focus",
            },
            on_evict=_on_evict,
        )
    return _demo_wmc


@app.get("/api/health")
async def health():
    return {"ok": True, "service": "wmc-studio"}


@app.get("/api/state")
async def get_state():
    wmc = _get_demo()
    state = wmc.studio_state()
    state["evictions"] = list(_eviction_log)
    state["mode"] = "demo"
    return state


@app.post("/api/fill")
async def fill_turn(payload: dict = Body(...)):
    user = str(payload.get("user") or "").strip()
    assistant = str(payload.get("assistant") or "").strip()
    if not user and not assistant:
        return {"ok": False, "error": "empty turn"}
    wmc = _get_demo()
    evicted = wmc.fill(user, assistant)
    return {
        "ok": True,
        "evicted": len(evicted),
        "state": wmc.studio_state(),
        "evictions": list(_eviction_log),
    }


@app.post("/api/touch")
async def touch_context():
    wmc = _get_demo()
    _ = wmc.get_context_block(touch=True)
    return {"ok": True, "state": wmc.studio_state(), "evictions": list(_eviction_log)}


@app.post("/api/reset")
async def reset():
    global _eviction_log
    wmc = _get_demo()
    wmc.clear()
    _eviction_log = []
    return {"ok": True, "state": wmc.studio_state(), "evictions": []}


@app.post("/api/demo/seed")
async def seed_demo():
    global _eviction_log
    wmc = _get_demo()
    wmc.clear()
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
        wmc.fill(u, a)
    wmc.get_context_block(touch=True)
    wmc.get_context_block(touch=True)
    return {"ok": True, "state": wmc.studio_state(), "evictions": list(_eviction_log)}


@app.get("/")
async def serve_studio():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)
