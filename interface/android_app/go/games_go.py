"""
Go (囲碁) game API for Aiko.

Endpoints (mounted at /api/games/go):
  POST /start          — start a new game vs Aiko (or practice)
  POST /move           — GTP move (e.g. D4, pass); Aiko replies if vs_ai
  GET  /state          — board stones + status
  GET  /legal-moves    — legal GTP moves (+ pass)
  GET  /engine         — whether KataGo is available
  POST /warmup         — pre-spawn KataGo GTP
  POST /resign         — end the game

Board: pure Python (interface/webui/go/board.py).
AI: optional KataGo GTP (KATAGO_PATH + KATAGO_MODEL); else random legal.
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import random
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .board import BLACK, WHITE, GoBoard, color_name

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/games/go", tags=["games"])

# Process-local game store. Requires a **single worker process** (or sticky
# sessions). Multi-worker / restart will drop or 404 in-flight games — same
# constraint as interface/webui/shogi. Shared store is a follow-up if needed.
_games: dict[str, dict] = {}

_TAILSCALE_IPV4_NETWORK = ipaddress.ip_network("100.64.0.0/10")
_PROXY_CLIENT_IP_HEADERS = (
    "forwarded",
    "x-forwarded-for",
    "x-real-ip",
    "cf-connecting-ip",
    "true-client-ip",
)


class StartRequest(BaseModel):
    mode: str = Field(default="vs_ai", description="vs_ai | practice")
    size: int = Field(default=9, description="9 | 13 | 19")
    difficulty: Optional[str] = Field(
        default=None, description="easy | medium | hard"
    )
    side: Optional[str] = Field(
        default=None, description="black (first) | white (second)"
    )


class MoveRequest(BaseModel):
    move: str = Field(..., description="GTP move, e.g. D4 or pass")


class GameState(BaseModel):
    size: int
    turn: str
    last_move: Optional[str] = None
    status: str  # playing | finished | resigned
    mode: str = "vs_ai"
    side: str = "black"
    stones: List[dict] = Field(default_factory=list)
    captured_black: int = 0
    captured_white: int = 0
    ai_comment: Optional[str] = None
    engine: Optional[str] = None
    moves: List[str] = Field(default_factory=list)


def _allow_owner_fallback(request: Request) -> bool:
    """Gate the AIKO_USER_ID fallback used by the Android companion apps.

    Allows fallback only for:
      * direct loopback clients, or
      * direct IPv4 clients in Tailscale CGNAT (100.64.0.0/10), or
      * requests carrying X-Aiko-App-Secret matching GAMES_APP_SECRET
        (when that env is set).
    """
    host = (request.client.host if request.client else "") or ""
    has_proxy_client_ip = any(header in request.headers for header in _PROXY_CLIENT_IP_HEADERS)
    if not has_proxy_client_ip:
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and (
            address.is_loopback
            or (
                isinstance(address, ipaddress.IPv4Address)
                and address in _TAILSCALE_IPV4_NETWORK
            )
        ):
            return True
    secret = (os.getenv("GAMES_APP_SECRET") or "").strip()
    if secret and request.headers.get("X-Aiko-App-Secret") == secret:
        return True
    return False


async def _require_user(request: Request) -> dict:
    """Session auth; limited owner fallback for the Android app."""
    from interface.webui import auth

    try:
        session = await auth.require_session(request)
        return await auth.require_accepted_session(session)
    except HTTPException:
        owner = (os.getenv("AIKO_USER_ID") or "").strip()
        if owner and _allow_owner_fallback(request):
            log.warning("Go session auth failed — falling back to app owner")
            return {"user_id": owner, "username": owner}
        raise
    except Exception as e:
        # Do not convert unexpected errors into an owner session.
        log.warning("Go auth failed: %s", e)
        raise HTTPException(status_code=401, detail="Authentication required") from e


_DIFFICULTY_PRESETS = {
    "easy": {"blunder": 0.40},
    "medium": {"blunder": 0.12},
    "hard": {"blunder": 0.0},
}
_DEFAULT_DIFFICULTY = "medium"


def difficulty_name(value: Optional[str] = None) -> str:
    raw = (value if value is not None else os.getenv("GO_DIFFICULTY", _DEFAULT_DIFFICULTY))
    raw = (raw or "").strip().lower()
    return raw if raw in _DIFFICULTY_PRESETS else _DEFAULT_DIFFICULTY


def _state_response(uid: str, ai_comment: Optional[str] = None) -> GameState:
    game = _games[uid]
    board: GoBoard = game["board"]
    return GameState(
        size=board.size,
        turn=color_name(board.turn),
        last_move=game.get("last_move"),
        status=board.status if board.status != "playing" else game.get("status", board.status),
        mode=game.get("mode", "vs_ai"),
        side=game.get("side", "black"),
        stones=board.stones_list(),
        captured_black=board.captured.get(BLACK, 0),
        captured_white=board.captured.get(WHITE, 0),
        ai_comment=ai_comment,
        engine=game.get("engine"),
        moves=list(board.history),
    )


def _ai_move(board: GoBoard, difficulty: Optional[str] = None):
    """Return (gtp_move | None, engine_name)."""
    legal = board.legal_moves_gtp()
    if not legal:
        return None, None

    preset = _DIFFICULTY_PRESETS[difficulty_name(difficulty)]
    if preset["blunder"] > 0 and random.random() < preset["blunder"]:
        # Prefer non-pass when possible for casual play
        non_pass = [m for m in legal if m != "pass"]
        return random.choice(non_pass or legal), "random"

    try:
        from . import katago

        if katago.available():
            color = "B" if board.turn == BLACK else "W"
            mv = katago.best_move_gtp(board.size, board.history, color=color)
            if mv:
                # Validate against our rules engine
                try:
                    test = board.copy()
                    test.play_gtp(mv)
                    return mv, "katago"
                except Exception:
                    log.warning("KataGo move illegal under our rules: %s", mv)
    except Exception:
        log.warning("KataGo bridge error", exc_info=True)

    non_pass = [m for m in legal if m != "pass"]
    return random.choice(non_pass or legal), "random"


@router.get("/engine")
async def engine_status(session: dict = Depends(_require_user)):
    try:
        from . import katago

        return {
            "katago": katago.available(),
            "path": katago.engine_path(),
            "model": katago.model_path(),
            "fallback": "random",
            "difficulty": difficulty_name(),
            "difficulties": sorted(_DIFFICULTY_PRESETS),
            "sizes": [9, 13, 19],
        }
    except Exception as e:
        return {"katago": False, "error": str(e), "fallback": "random"}


@router.post("/warmup")
async def warmup_engine(session: dict = Depends(_require_user)):
    try:
        from . import katago

        if not katago.available():
            return {"warmed": False, "reason": "KataGo not configured"}
        ok = await asyncio.to_thread(katago.ensure_ready)
        return {"warmed": ok}
    except Exception as e:
        return {"warmed": False, "error": str(e)}


@router.post("/start", response_model=GameState)
async def start_game(body: StartRequest, session: dict = Depends(_require_user)):
    uid = session["user_id"]
    size = body.size if body.size in (9, 13, 19) else 9
    mode = body.mode if body.mode in ("vs_ai", "practice") else "vs_ai"
    diff = difficulty_name(body.difficulty)
    side = (body.side or "").strip().lower()
    side = side if side in ("black", "white") else "black"

    board = GoBoard(size=size)
    eng = "random"
    try:
        from . import katago

        eng = "katago" if katago.available() else "random"
    except Exception:
        eng = "random"

    _games[uid] = {
        "board": board,
        "mode": mode,
        "difficulty": diff,
        "side": side,
        "last_move": None,
        "status": "playing",
        "engine": eng,
    }

    if side == "black":
        comment = f"Let's play Go ({size}×{size})! You are Black — move first ⚫"
    else:
        comment = f"You are White on {size}×{size} — Aiko moves first ⚪"
    if eng == "katago":
        comment += f" (Aiko asks KataGo · {diff})"
    else:
        comment += " (KataGo offline — casual moves)"

    if side == "white" and mode == "vs_ai":
        ai, eng2 = await asyncio.to_thread(_ai_move, board, diff)
        _games[uid]["engine"] = eng2
        if ai is not None:
            board.play_gtp(ai)
            _games[uid]["last_move"] = ai
            comment = f"Aiko opens with {ai} — your move!"

    log.info(
        "Go game started for %s size=%s mode=%s engine=%s difficulty=%s side=%s",
        uid,
        size,
        mode,
        _games[uid]["engine"],
        diff,
        side,
    )
    return _state_response(uid, ai_comment=comment)


@router.post("/move", response_model=GameState)
async def make_move(body: MoveRequest, session: dict = Depends(_require_user)):
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=400, detail="No active game — call POST /start first")

    game = _games[uid]
    board: GoBoard = game["board"]
    if board.status != "playing" or game.get("status") != "playing":
        raise HTTPException(
            status_code=400,
            detail=f"Game already over: {board.status}",
        )

    user_side = game.get("side", "black")
    user_color = BLACK if user_side == "black" else WHITE
    move_str = (body.move or "").strip()
    if not move_str:
        raise HTTPException(status_code=400, detail="move is required (GTP, e.g. D4 or pass)")

    if game.get("mode") == "vs_ai" and board.turn != user_color:
        raise HTTPException(status_code=400, detail="Not your turn")

    try:
        board.play_gtp(move_str)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    game["last_move"] = move_str.strip().upper() if move_str.upper() not in ("PASS", "PA") else "pass"
    if board.status != "playing":
        game["status"] = board.status
        return _state_response(uid, ai_comment="Game over.")

    ai_comment = None
    if game["mode"] == "vs_ai" and board.status == "playing":
        ai, engine = await asyncio.to_thread(_ai_move, board, game.get("difficulty"))
        game["engine"] = engine
        if ai is not None:
            try:
                board.play_gtp(ai)
                game["last_move"] = ai
                if board.status != "playing":
                    game["status"] = board.status
                if engine == "katago":
                    ai_comment = f"Aiko (via KataGo) plays {ai}"
                else:
                    ai_comment = f"Aiko plays {ai}"
            except Exception as e:
                log.warning("AI move failed: %s", e)
                ai_comment = "Aiko hesitated… your move again?"

    return _state_response(uid, ai_comment=ai_comment)


@router.get("/state", response_model=GameState)
async def get_state(session: dict = Depends(_require_user)):
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=404, detail="No active game")
    return _state_response(uid)


@router.get("/legal-moves")
async def legal_moves(session: dict = Depends(_require_user)):
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=404, detail="No active game")
    board: GoBoard = _games[uid]["board"]
    return {
        "moves": board.legal_moves_gtp(),
        "turn": color_name(board.turn),
        "status": board.status,
        "size": board.size,
    }


@router.post("/resign")
async def resign(session: dict = Depends(_require_user)):
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=404, detail="No active game")
    board: GoBoard = _games[uid]["board"]
    board.status = "resigned"
    _games[uid]["status"] = "resigned"
    return {"status": "resigned", "message": "You resigned. Aiko wins this one."}
