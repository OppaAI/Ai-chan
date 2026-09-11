"""
Shogi game API for Aiko.

Endpoints (mounted at /api/games/shogi):
  POST /start          — start a new game vs Aiko (or practice)
  POST /move           — play a USI move; Aiko replies if vs_ai
  GET  /state          — current board + status
  GET  /legal-moves    — legal USI moves for the side to move
  GET  /engine         — whether YaneuraOu is available
  POST /resign         — end the game

Board state is SFEN (Shogi FEN). Moves use USI, e.g. "7g7f", "B*5e".

AI: Aiko asks YaneuraOu (USI) for the best move when YANEURAOU_PATH is set;
otherwise falls back to a random legal move.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/games/shogi", tags=["games"])

# In-memory games keyed by user_id. Fine for single-user / MVP.
_games: dict[str, dict] = {}


class StartRequest(BaseModel):
    mode: str = Field(default="vs_ai", description="vs_ai | practice")


class MoveRequest(BaseModel):
    move: str = Field(..., description="USI move, e.g. 7g7f or B*5e")


class GameState(BaseModel):
    sfen: str
    turn: str  # "black" (user / 先手) or "white" (Aiko / 後手)
    last_move: Optional[str] = None
    status: str  # playing | checkmate | stalemate | draw | resigned
    mode: str = "vs_ai"
    ai_comment: Optional[str] = None
    engine: Optional[str] = None  # "yaneuraou" | "random" | None


async def _require_user(request: Request) -> dict:
    """Real auth only — no AIKO_USER_ID fallback."""
    from interface.webui import auth

    try:
        session = await auth.require_session(request)
        return await auth.require_accepted_session(session)
    except HTTPException:
        raise
    except Exception as e:
        log.warning("Shogi auth failed: %s", e)
        raise HTTPException(status_code=401, detail="Authentication required")


def _import_shogi():
    try:
        import shogi

        return shogi
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail="python-shogi is not installed. Run: pip install python-shogi",
        ) from e


def _status_for(board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_game_over():
        return "draw"
    return "playing"


def _turn_label(board) -> str:
    shogi = _import_shogi()
    return "black" if board.turn == shogi.BLACK else "white"


def _ai_move(board):
    """
    Aiko asks YaneuraOu for the right move when available;
    otherwise picks a random legal move.
    Returns (move | None, engine_name).
    """
    shogi = _import_shogi()
    legal = list(board.legal_moves)
    if not legal:
        return None, None

    # 1) Ask YaneuraOu
    try:
        from . import yaneuraou

        usi = yaneuraou.best_move_usi(board.sfen())
        if usi:
            try:
                mv = shogi.Move.from_usi(usi)
                if mv in board.legal_moves:
                    return mv, "yaneuraou"
                log.warning("YaneuraOu returned illegal move %s — ignoring", usi)
            except Exception:
                log.warning("Could not parse YaneuraOu move %s", usi, exc_info=True)
    except Exception:
        log.warning("YaneuraOu bridge error", exc_info=True)

    # 2) Fallback
    return random.choice(legal), "random"


def _state_response(
    uid: str,
    ai_comment: Optional[str] = None,
) -> GameState:
    game = _games[uid]
    board = game["board"]
    return GameState(
        sfen=board.sfen(),
        turn=_turn_label(board),
        last_move=game.get("last_move"),
        status=game.get("status", _status_for(board)),
        mode=game.get("mode", "vs_ai"),
        ai_comment=ai_comment,
        engine=game.get("engine"),
    )


@router.get("/engine")
async def engine_status(session: dict = Depends(_require_user)):
    """Report whether YaneuraOu is configured and runnable."""
    try:
        from . import yaneuraou

        path = yaneuraou.engine_path()
        ok = yaneuraou.available()
        return {
            "yaneuraou": ok,
            "path": path if ok else path,
            "movetime_ms": yaneuraou.normalized_movetime_ms(),
            "fallback": "random",
        }
    except Exception as e:
        return {"yaneuraou": False, "error": str(e), "fallback": "random"}


@router.post("/start", response_model=GameState)
async def start_game(body: StartRequest, session: dict = Depends(_require_user)):
    """Start a new Shogi game. User plays 先手 (black, first move)."""
    shogi = _import_shogi()
    uid = session["user_id"]
    mode = body.mode if body.mode in ("vs_ai", "practice") else "vs_ai"
    _games[uid] = {
        "board": shogi.Board(),
        "mode": mode,
        "last_move": None,
        "status": "playing",
    }
    eng = None
    try:
        from . import yaneuraou

        eng = "yaneuraou" if yaneuraou.available() else "random"
    except Exception:
        eng = "random"
    _games[uid]["engine"] = eng
    comment = "Let's play Shogi! You move first ♟️"
    if eng == "yaneuraou":
        comment += " (Aiko will ask YaneuraOu for strong moves)"
    else:
        comment += " (engine offline — Aiko plays casual moves)"
    log.info("Shogi game started for %s mode=%s engine=%s", uid, mode, eng)
    return _state_response(uid, ai_comment=comment)


@router.post("/move", response_model=GameState)
async def make_move(body: MoveRequest, session: dict = Depends(_require_user)):
    """Apply user USI move; if vs_ai and still playing, Aiko replies."""
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=400, detail="No active game — call POST /start first")

    game = _games[uid]
    board = game["board"]
    if game.get("status") != "playing":
        raise HTTPException(status_code=400, detail=f"Game already over: {game['status']}")

    move_str = (body.move or "").strip()
    if not move_str:
        raise HTTPException(status_code=400, detail="move is required (USI, e.g. 7g7f)")

    shogi = _import_shogi()
    try:
        move = shogi.Move.from_usi(move_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid USI move: {e}") from e

    if move not in board.legal_moves:
        raise HTTPException(status_code=400, detail=f"Illegal move: {move_str}")

    board.push(move)
    game["last_move"] = move_str
    status = _status_for(board)
    game["status"] = status

    ai_comment = None
    engine = None
    if game["mode"] == "vs_ai" and status == "playing":
        ai, engine = await asyncio.to_thread(_ai_move, board)
        game["engine"] = engine
        if ai is not None:
            board.push(ai)
            usi = ai.usi()
            game["last_move"] = usi
            game["status"] = _status_for(board)
            if engine == "yaneuraou":
                ai_comment = f"Aiko (via YaneuraOu) plays {usi}"
            else:
                ai_comment = f"Aiko plays {usi}"
            if game["status"] == "checkmate":
                ai_comment += " — checkmate! 🐱"

    return _state_response(uid, ai_comment=ai_comment)


@router.get("/state", response_model=GameState)
async def game_state(session: dict = Depends(_require_user)):
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=404, detail="No active game")
    return _state_response(uid)


@router.get("/legal-moves")
async def legal_moves(session: dict = Depends(_require_user)):
    """List legal USI moves for the current side to move."""
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=404, detail="No active game")
    board = _games[uid]["board"]
    return {
        "moves": [m.usi() for m in board.legal_moves],
        "turn": _turn_label(board),
        "status": _games[uid].get("status", _status_for(board)),
    }


@router.post("/resign", response_model=GameState)
async def resign(session: dict = Depends(_require_user)):
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=404, detail="No active game")
    _games[uid]["status"] = "resigned"
    return _state_response(uid)
