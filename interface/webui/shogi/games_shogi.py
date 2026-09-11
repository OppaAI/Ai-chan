"""
Shogi game API for Aiko.

Endpoints (mounted at /api/games/shogi):
  POST /start          — start a new game vs Aiko (or practice)
  POST /move           — play a USI move; Aiko replies if vs_ai
  GET  /state          — current board + status
  GET  /legal-moves    — legal USI moves for the side to move
  GET  /engine         — whether YaneuraOu is available
  POST /warmup         — pre-spawn + handshake YaneuraOu without playing a move
  POST /resign         — end the game

Board state is SFEN (Shogi FEN). Moves use USI, e.g. "7g7f", "B*5e".

AI: Aiko asks YaneuraOu (USI) for the best move when YANEURAOU_PATH
(config/android_app.yaml, or env) is set;
otherwise falls back to a random legal move.

Strength: SHOGI_DIFFICULTY (config/android_app.yaml, or per-game
`difficulty` in POST /start) — easy | medium | hard. Lower levels cap
the search depth and mix in random moves.

Clock: real byoyomi (SHOGI_MAIN_TIME_S + SHOGI_BYOYOMI_S in
config/android_app.yaml). Each side's main time ticks on their moves;
once it hits zero, every move must come within the byoyomi allowance
or that side flags (status "timeout"). 0/0 disables the clock.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/games/shogi", tags=["games"])

# In-memory games keyed by user_id. Fine for single-user / MVP.
_games: dict[str, dict] = {}


def _clock_settings() -> tuple[Optional[float], float]:
    """(main_ms per side or None, byoyomi_ms). (None, 0) = clock off."""
    try:
        main_s = float(os.getenv("SHOGI_MAIN_TIME_S", "600"))
    except (TypeError, ValueError):
        main_s = 600.0
    try:
        byoyomi_s = float(os.getenv("SHOGI_BYOYOMI_S", "30"))
    except (TypeError, ValueError):
        byoyomi_s = 30.0
    main_ms = max(0.0, main_s) * 1000.0
    byoyomi_ms = max(0.0, byoyomi_s) * 1000.0
    if main_ms <= 0 and byoyomi_ms <= 0:
        return None, 0.0
    return main_ms, byoyomi_ms


def _new_clock() -> Optional[dict]:
    """Fresh clock state, or None when the clock is disabled."""
    main_ms, byoyomi_ms = _clock_settings()
    if main_ms is None:
        return None
    return {
        "remaining": {"black": main_ms, "white": main_ms},
        "byoyomi_ms": byoyomi_ms,
        "stamp": time.monotonic(),
    }


def _charge_clock(game: dict, side: str, elapsed_ms: float) -> bool:
    """Deduct elapsed_ms from side's clock. False = flagged.

    Main time first; once it is gone, the move must still fall inside
    one byoyomi period (classic single-period byoyomi).
    """
    clock = game.get("clock")
    if not clock:
        return True
    remaining = clock["remaining"][side] - elapsed_ms
    if remaining >= 0:
        clock["remaining"][side] = remaining
        return True
    if elapsed_ms <= clock["byoyomi_ms"]:
        clock["remaining"][side] = 0.0
        return True
    clock["remaining"][side] = 0.0
    return False


def _clock_view(game: dict) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """(black_ms, white_ms, byoyomi_ms) snapshot for API responses."""
    clock = game.get("clock")
    if not clock:
        return None, None, None
    return (
        max(0, int(clock["remaining"]["black"])),
        max(0, int(clock["remaining"]["white"])),
        int(clock["byoyomi_ms"]),
    )


class StartRequest(BaseModel):
    mode: str = Field(default="vs_ai", description="vs_ai | practice")
    difficulty: Optional[str] = Field(
        default=None, description="easy | medium | hard (None = server default)"
    )


class MoveRequest(BaseModel):
    move: str = Field(..., description="USI move, e.g. 7g7f or B*5e")


class GameState(BaseModel):
    sfen: str
    turn: str  # "black" (user / 先手) or "white" (Aiko / 後手)
    last_move: Optional[str] = None
    status: str  # playing | checkmate | stalemate | draw | resigned | timeout
    mode: str = "vs_ai"
    ai_comment: Optional[str] = None
    engine: Optional[str] = None  # "yaneuraou" | "random" | None
    clock_black_ms: Optional[int] = None  # remaining main time (None = no clock)
    clock_white_ms: Optional[int] = None
    byoyomi_ms: Optional[int] = None


async def _require_user(request: Request) -> dict:
    """Session auth with owner fallback for the Android app.

    Mirrors lingo's get_lingo_session: the Aiko-Shogi client has no
    CookieJar/login flow, so unauthenticated calls fall back to the app
    owner (AIKO_USER_ID) instead of hard-401ing every move.
    """
    from interface.webui import auth

    try:
        session = await auth.require_session(request)
        return await auth.require_accepted_session(session)
    except HTTPException:
        # Auth attempted but rejected (no/invalid cookie, terms unaccepted)
        # — fall back to owner before giving up.
        owner = (os.getenv("AIKO_USER_ID") or "").strip()
        if owner:
            log.warning("Shogi session auth failed — falling back to app owner")
            return {"user_id": owner, "username": owner}
        raise
    except Exception as e:
        log.warning("Shogi auth failed: %s", e)
        owner = (os.getenv("AIKO_USER_ID") or "").strip()
        if owner:
            return {"user_id": owner, "username": owner}
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
    try:
        if board.is_fourfold_repetition():
            return "draw"
    except AttributeError:
        pass
    if board.is_game_over():
        return "draw"
    return "playing"


def _turn_label(board) -> str:
    shogi = _import_shogi()
    return "black" if board.turn == shogi.BLACK else "white"


# Difficulty presets: depth caps the engine search (USI `go depth`),
# movetime_ms bounds the read loop (None = YANEURAOU_MOVETIME_MS),
# blunder = probability of a uniform-random legal move instead of asking
# the engine at all (skips the search, saving Nano CPU too).
_DIFFICULTY_PRESETS: dict[str, dict] = {
    "easy": {"depth": 3, "movetime_ms": 150, "blunder": 0.35},
    "medium": {"depth": 6, "movetime_ms": 400, "blunder": 0.10},
    "hard": {"depth": None, "movetime_ms": None, "blunder": 0.0},
}

_DEFAULT_DIFFICULTY = "medium"


def difficulty_name(value: Optional[str] = None) -> str:
    """Normalize a difficulty level; unknown/blank falls back to default."""
    raw = (value if value is not None else os.getenv("SHOGI_DIFFICULTY", _DEFAULT_DIFFICULTY))
    raw = (raw or "").strip().lower()
    return raw if raw in _DIFFICULTY_PRESETS else _DEFAULT_DIFFICULTY


def _engine_bridge():
    """Resolve the YaneuraOu bridge module (separate hook so tests can patch it)."""
    try:
        from . import yaneuraou

        return yaneuraou
    except ImportError:
        return None


def _banter_enabled() -> bool:
    """LLM move chatter on/off (SHOGI_BANTER, default on — shogi is slow anyway)."""
    return (os.getenv("SHOGI_BANTER", "1") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _banter_for(usi: str, difficulty: Optional[str], status: str) -> Optional[str]:
    """One short Aiko line about her just-played move, or None.

    Never raises and never blocks the game: any failure (no LLM
    instance, timeout, empty reply) falls back to the template comment.
    """
    try:
        from interface.webui import auth

        if not auth.aiko_web_instance or not auth.aiko_web_instance._think:
            return None
        think = auth.aiko_web_instance._think
        ending = " and checkmated the user" if status == "checkmate" else ""
        response = think._client.chat.completions.create(
            model=think._llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Aiko, a playful cat-girl AI playing shogi "
                        "(Japanese chess) as White against the user. "
                        f"You just played {usi} in a {difficulty_name(difficulty)} game{ending}. "
                        "Reply with ONE short playful line (under 20 words, "
                        "English with a touch of Japanese flavor). No board "
                        "analysis, no notation lecture, no quotes."
                    ),
                },
                {"role": "user", "content": "React to your move."},
            ],
            max_tokens=60,
            timeout=30.0,
        )
        text = (response.choices[0].message.content or "").strip().splitlines()
        line = (text[0] if text else "").strip().strip("\"'")[:140]
        return line or None
    except Exception:
        log.debug("Shogi banter failed, using template comment", exc_info=True)
        return None


def _ai_move(
    board,
    difficulty: Optional[str] = None,
    movetime_cap_ms: Optional[float] = None,
):
    """
    Aiko asks YaneuraOu for the right move when available;
    otherwise picks a random legal move.
    Returns (move | None, engine_name).

    movetime_cap_ms bounds the search so the engine can never flag
    itself when the clock is on (hard mode only; depth-capped levels
    finish in milliseconds anyway).
    """
    shogi = _import_shogi()
    legal = list(board.legal_moves)
    if not legal:
        return None, None

    preset = _DIFFICULTY_PRESETS[difficulty_name(difficulty)]
    movetime_ms = preset["movetime_ms"]
    if preset["depth"] is None and movetime_cap_ms is not None:
        # best_move_usi() clamps this again via normalized_movetime_ms.
        if movetime_ms is None or movetime_cap_ms < movetime_ms:
            movetime_ms = max(50, int(movetime_cap_ms))

    # 0) Human-like blunder: occasional random move, no engine search.
    if preset["blunder"] > 0 and random.random() < preset["blunder"]:
        return random.choice(legal), "random"

    # 1) Ask YaneuraOu (depth-capped unless hard)
    try:
        bridge = _engine_bridge()
        usi = (
            bridge.best_move_usi(
                board.sfen(),
                movetime_ms=movetime_ms,
                depth=preset["depth"],
            )
            if bridge is not None
            else None
        )
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
    black_ms, white_ms, byoyomi_ms = _clock_view(game)
    return GameState(
        sfen=board.sfen(),
        turn=_turn_label(board),
        last_move=game.get("last_move"),
        status=game.get("status", _status_for(board)),
        mode=game.get("mode", "vs_ai"),
        ai_comment=ai_comment,
        engine=game.get("engine"),
        clock_black_ms=black_ms,
        clock_white_ms=white_ms,
        byoyomi_ms=byoyomi_ms,
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
            "path": path,
            "movetime_ms": yaneuraou.normalized_movetime_ms(),
            "fallback": "random",
            "difficulty": difficulty_name(),
            "difficulties": sorted(_DIFFICULTY_PRESETS),
        }
    except Exception as e:
        return {"yaneuraou": False, "error": str(e), "fallback": "random"}


@router.post("/warmup")
async def warmup_engine(session: dict = Depends(_require_user)):
    """
    Pre-spawn YaneuraOu and complete its USI handshake now, without playing
    a move. Meant to be called when the app opens (or the lobby loads) so
    the process is already up and ready by the time the user starts a game
    — the first real move then has no extra spawn/handshake latency.

    Safe to call anytime, including with no active game and repeatedly;
    it's a fast no-op if the engine is already warm.
    """
    try:
        from . import yaneuraou

        if not yaneuraou.available():
            return {"warmed": False, "reason": "engine binary not found/configured"}
        ok = await asyncio.to_thread(yaneuraou.ensure_ready)
        return {"warmed": ok}
    except Exception as e:
        log.warning("Engine warmup failed: %s", e)
        return {"warmed": False, "error": str(e)}


@router.post("/start", response_model=GameState)
async def start_game(body: StartRequest, session: dict = Depends(_require_user)):
    """Start a new Shogi game. User plays 先手 (black, first move)."""
    shogi = _import_shogi()
    uid = session["user_id"]
    mode = body.mode if body.mode in ("vs_ai", "practice") else "vs_ai"
    diff = difficulty_name(body.difficulty)
    _games[uid] = {
        "board": shogi.Board(),
        "mode": mode,
        "difficulty": diff,
        "clock": _new_clock(),
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
        comment += f" (Aiko will ask YaneuraOu for {diff} moves)"
    else:
        comment += " (engine offline — Aiko plays casual moves)"
    log.info("Shogi game started for %s mode=%s engine=%s difficulty=%s", uid, mode, eng, diff)
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

    # Charge the user's (black) clock for time since the last stamp.
    now = time.monotonic()
    if game.get("clock"):
        elapsed_ms = (now - game["clock"]["stamp"]) * 1000.0
        if not _charge_clock(game, "black", elapsed_ms):
            game["status"] = "timeout"
            log.info("Shogi flag: %s ran out of time", uid)
            return _state_response(uid, ai_comment="Flag! You ran out of time — Aiko wins. ⏰")
        game["clock"]["stamp"] = now

    board.push(move)
    game["last_move"] = move_str
    status = _status_for(board)
    game["status"] = status

    ai_comment = None
    engine = None
    if game["mode"] == "vs_ai" and status == "playing":
        cap_ms = None
        if game.get("clock"):
            cap_ms = max(50.0, game["clock"]["remaining"]["white"] - 100.0)
        ai_start = time.monotonic()
        ai, engine = await asyncio.to_thread(
            _ai_move, board, game.get("difficulty"), cap_ms
        )
        if game.get("clock"):
            ai_elapsed_ms = (time.monotonic() - ai_start) * 1000.0
            if not _charge_clock(game, "white", ai_elapsed_ms):
                game["status"] = "timeout"
                game["engine"] = engine
                return _state_response(uid, ai_comment="Flag! Aiko ran out of time — you win! 🐱⏰")
            game["clock"]["stamp"] = time.monotonic()
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
            elif _banter_enabled():
                # LLM chatter runs after the move is committed, in a worker
                # so the event loop stays free; template above survives any
                # failure.
                line = await asyncio.to_thread(
                    _banter_for, usi, game.get("difficulty"), game["status"]
                )
                if line:
                    ai_comment = f"{ai_comment} — {line}"

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
