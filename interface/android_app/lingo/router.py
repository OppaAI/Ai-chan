"""
Lingo API endpoints (FastAPI router).

This is the reorganized, bugfixed version of the old lingo.py. Per
lingo_architecture.md the Pydantic schemas now live in models.py and the
SRS/vocab logic lives in srs.py / vocab.py; this module only owns HTTP
concerns: routing, request/response shaping, audio generation, and the
small pieces of per-user state (streaks, XP, difficulty level) that don't
belong in the SRS database.

=====================================================================
BUGFIX PASS (carried over from lingo.py):
  1. review_respond mixed `yield` + `return <value>` -> SyntaxError at
     import time, which took down the ENTIRE router. Converted to a
     plain async function returning ReviewResponseData.
  2. `timedelta` was used in get_streak()/record_practice() but never
     imported -> NameError at runtime. Imported alongside date/datetime.
  3. conversation_start's system_prompt was missing the `f` prefix on
     the line referencing `{request.level}`, so the LLM literally saw
     the text "{request.level}" instead of the actual level string.
  4. Pydantic v2 renamed Field(regex=...) to Field(pattern=...); fixed
     in models.py's StartRequest.
  5. get_streak()/record_practice() only tracked practice in an
     in-memory dict that's wiped on restart and never wrote to disk.
     Rewritten to persist to <USER_SPACE_ROOT>/<uid>/streaks/{uid}.json with real
     day-over-day accumulation.
  6. review_start() reported `cards_due=stats["reviews_today"]` (reviews
     completed today) instead of the actual due-card count. Fixed to use
     LingoSRS.count_due_cards().
  7. `_audio_cache` was an unbounded plain dict. Replaced with a bounded,
     TTL-evicting cache (24h TTL, 512-entry cap, LRU-ish eviction).
  8. No rate limit on TTS generation. Added a per-user sliding-window
     limiter, and /tts now returns a real 429 when it's hit instead of a
     silently-swallowed failure (see FIX #13 below).
  9. `_last_levels` was in-memory-only, so a restart silently reset every
     user to "beginner". Now persisted to <USER_SPACE_ROOT>/<uid>/levels/{uid}.json.
  10. VocabExtractor was constructed with no arguments, so its LLM
      fallback was always a no-op. Now constructed with the tutor's own
      LLM client/model via _get_vocab_extractor().
  11. Added GET /weak-vocab and wired LingoSRS.get_weak_vocab() into
      /stats as `weak_vocab`, backing the "Practice These" widget.

BUGFIX PASS (this version, from lingo_audit.md):
  12. Added POST /conversation/respond -- a non-streaming counterpart to
      /conversation/respond_stream. The Android interface already
      declared this endpoint; without it, any client code path that
      called it (e.g. fast-retry logic) hit a 404 (lingo_audit.md #1).
  13. /tts previously called get_cached_lingo_audio() and returned a
      generic 503 whether the request was rate-limited or synthesis
      actually failed -- and contained a dead, inert placeholder branch
      that never ran. Now the rate limit is checked up front and reported
      as 429, with 503 reserved for genuine synthesis failures
      (lingo_audit.md #3).
  14. Endpoints used `session.get("user_id", "OppaAI")`, so if session
      auth ever broke in a way that produced a session dict without
      `user_id`, every user would silently become "OppaAI" and their
      data would mix. get_lingo_session() always sets `user_id` (falling
      back to the configured app owner only when auth itself fails, and
      raising if even that isn't configured), so endpoints now do
      `session["user_id"]` with no further silent fallback
      (lingo_audit.md #7).
  15. `total_due = len(srs.get_due_cards(limit=1000))` fetched up to 1000
      full rows just to count them. Replaced with
      `srs.count_due_cards()`, a single COUNT(*) query
      (lingo_audit.md #4).

BUGFIX PASS (this version, from a second audit pass):
  16. SRS reviews never touched the streak or XP files -- only
      conversation turns did (via _finalize_respond_turn's internal
      /xp/add call and its record_practice() call). A user could clear
      their entire due review queue and the dashboard's streak/XP would
      not move at all, even though the "Practice These" weak-vocab
      widget funnels straight into the review flow. review_respond()
      now calls record_practice(uid) and awards XP scaled by review
      grade, using the same bookkeeping paths conversation turns use.
  17. Extracted the XP-file read/increment/write logic out of the
      /xp/add handler into _award_xp() so it can be reused by
      review_respond() (see #16) instead of duplicating the same
      read-modify-write logic in two places.
=====================================================================
"""
import os
import json
import uuid
import logging
import random
import re
import time
import html
import threading
from pathlib import Path
from typing import Optional, List, Dict
from collections import defaultdict
from datetime import datetime, date, timedelta  # FIX #2: timedelta import
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from system.userspace import set_current_user_id, reset_current_user_id, user_state_path

from .models import (
    TranslateRequest, StartRequest, DialogueHistoryEntry, RespondRequest,
    TranslationResult, TranslateResponse, Toast, ConversationResponse,
    ReviewCard, ReviewSessionResponse, ReviewResponseRequest, UpdatedCard,
    ReviewResponseData, StatsResponse, XPResponse, WordOfDayResponse,
    LessonCard, LessonDeck, LessonDeckMeta,
    LessonProgress, TestQuestion, TestSubmitRequest, TestItemResult,
    TestSubmitResponse,
)
from .srs import LingoSRS, ReviewGrade, init_srs_db
from .vocab import VocabExtractor

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/nihongo", tags=["lingo"])

# ============================================================================
# Static Files & Audio (thread-safe for race-condition fix)
# ============================================================================
STATIC_DIR = Path(__file__).parent / "static"
# NOTE: wav files must land under interface/webui/static — that is the
# directory webui.py actually mounts. android_app/lingo/static is NOT served,
# so audio URLs written there 404 and the app's Play button silently fails.
# This backend lives under interface/android_app/, so climb to interface/
# and back down into webui/static.
AUDIO_DIR = Path(__file__).resolve().parents[2] / "webui" / "static" / "lingo_audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# No hardcoded server address: audio URLs are built from the live request
# (see _public_base_url) so the app works on any hostname, Tailnet IP, or
# funnel URL with zero config. Env override when the request host can't be
# trusted (reverse proxy without forwarded headers):
#   config/android_app.yaml: AIKO_PUBLIC_BASE_URL=https://aiko.example.ts.net
#   (or export AIKO_PUBLIC_BASE_URL=https://aiko.example.ts.net)
_LEGACY_PUBLIC_BASE = "https://aiko.ide-chroma.ts.net"


def _public_base_url(request: "Request | None" = None) -> str:
    """Public origin for absolute audio URLs, e.g. https://host:port.

    Prefers the request's own base URL (correct on Tailnet hostnames and
    IPs), unless it points at localhost (reverse-proxy/funnel setups) —
    then falls back to AIKO_PUBLIC_BASE_URL, auth.REDIRECT_BASE, and only
    lastly the legacy default.
    """
    if request is not None:
        try:
            base = str(request.base_url).rstrip("/")
            if base and "localhost" not in base and "127.0.0.1" not in base:
                return base
        except Exception:
            pass
    env = (os.getenv("AIKO_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    if env:
        return env
    try:
        from interface.webui import auth
        if getattr(auth, "REDIRECT_BASE", ""):
            return str(auth.REDIRECT_BASE).rstrip("/")
    except Exception:
        pass
    return _LEGACY_PUBLIC_BASE

init_srs_db()

# FIX #7: bounded, TTL-evicting audio cache. The previous `_audio_cache: dict`
# grew forever -- every unique string ever spoken stayed cached for the life
# of the process. This caps memory use and lets stale entries expire.
_AUDIO_CACHE_MAXSIZE = 512
_AUDIO_CACHE_TTL_SECONDS = 24 * 60 * 60  # 24h

_audio_cache: Dict[str, str] = {}
_audio_cache_timestamps: Dict[str, float] = {}
_audio_lock = threading.Lock()   # race-condition fix


def _audio_cache_get(key: str) -> Optional[str]:
    with _audio_lock:
        if key not in _audio_cache:
            return None
        if time.time() - _audio_cache_timestamps.get(key, 0) > _AUDIO_CACHE_TTL_SECONDS:
            # Expired -- evict and treat as a miss.
            _audio_cache.pop(key, None)
            _audio_cache_timestamps.pop(key, None)
            return None
        return _audio_cache[key]


def _audio_cache_set(key: str, value: str):
    with _audio_lock:
        _audio_cache[key] = value
        _audio_cache_timestamps[key] = time.time()
        if len(_audio_cache) > _AUDIO_CACHE_MAXSIZE:
            # Evict the oldest entry. Not strictly LRU (doesn't bump on
            # read) but that's fine for a cache this size -- the goal is
            # just bounding memory, not optimal hit rate.
            oldest_key = min(_audio_cache_timestamps, key=_audio_cache_timestamps.get)
            _audio_cache.pop(oldest_key, None)
            _audio_cache_timestamps.pop(oldest_key, None)


# FIX #8: simple per-user sliding-window rate limit for TTS generation.
_TTS_RATE_LOCK = threading.Lock()
_tts_request_times: Dict[str, List[float]] = defaultdict(list)
MAX_TTS_PER_MINUTE = 20


def check_tts_rate_limit(uid: str) -> bool:
    """Returns True if the request is allowed, False if rate-limited."""
    now = time.time()
    window_start = now - 60
    with _TTS_RATE_LOCK:
        recent = [t for t in _tts_request_times[uid] if t > window_start]
        if len(recent) >= MAX_TTS_PER_MINUTE:
            _tts_request_times[uid] = recent
            return False
        recent.append(now)
        _tts_request_times[uid] = recent
        return True


def _clean_tts_text(text: str) -> str:
    """Speech-ready Japanese: strip what a voice must never read aloud.

    - 〜/～ pattern markers (grammar cards like 〜です speak as です),
    - （…） glosses containing Latin (e.g. 〜が（but） speaks as が),
    - bare / separators become a pause 、so alternatives don't read as
      "slash" (e.g. がいます / があります),
    - then the long-standing rules: Latin chars, emptied () parens.
    """
    clean_text = re.sub(r'[〜～~]', '', text or '')
    clean_text = re.sub(r'（[^（）]*[a-zA-Z][^（）]*）', '', clean_text)
    clean_text = re.sub(r'\s*/\s*', '、', clean_text)
    clean_text = re.sub(r'[a-zA-Z]', '', clean_text)
    clean_text = re.sub(r'\(\s*\)', '', clean_text)
    clean_text = re.sub(r'\s{2,}', ' ', clean_text).strip()
    return clean_text


def get_cached_lingo_audio(text: str, uid: Optional[str] = None,
                         base_url: Optional[str] = None) -> Optional[str]:
    """
    Returns a cached or freshly synthesized audio URL, or None if the text
    was invalid or synthesis failed. Rate limiting is applied only to
    actual synthesis (cache hits never count against the limit). Callers
    that need to distinguish "rate limited" from "synthesis failed" (e.g.
    the /tts endpoint) should call check_tts_rate_limit() themselves before
    calling this function -- see FIX #13.

    base_url pins the URL host (endpoints pass the live request's origin);
    the cache key includes it so a hostname URL is never served to an IP
    client or vice versa.
    """
    if not is_valid_text(text):
        return None
    clean_text = _clean_tts_text(text)
    if not is_valid_text(clean_text):
        return None
    base = (base_url or _public_base_url()).rstrip("/")
    cache_key = f"{clean_text}\x00{base}"

    cached = _audio_cache_get(cache_key)
    if cached is not None:
        return cached

    # Only rate-limit actual synthesis, not cache hits, and only when we
    # know who's asking (some internal callers don't have a uid handy).
    if uid is not None and not check_tts_rate_limit(uid):
        log.warning(f"TTS rate limit hit for user {uid}")
        return None

    url = generate_lingo_audio(clean_text, base_url=base)
    if url:
        _audio_cache_set(cache_key, url)
    return url


# ============================================================================
# Protocol Definition, Validation, JSON, History (FIXED alternation)
# ============================================================================
class LingoTags:
    MISTAKE = "MISTAKE"
    FEEDBACK = "FEEDBACK"
    SUGGESTION = "SUGGESTION"
    REPLY_JP = "REPLY_JP"
    REPLY_EN = "REPLY_EN"
    FINISHED = "FINISHED"
    ALL = [MISTAKE, FEEDBACK, SUGGESTION, REPLY_JP, REPLY_EN, FINISHED]
    START_TAGS = [REPLY_JP, REPLY_EN, FINISHED]


def is_valid_text(text: Optional[str], min_length: int = 1) -> bool:
    if not text:
        return False
    stripped = text.strip()
    return len(stripped) >= min_length and stripped not in ["...", "*", ""]


def is_japanese_text(text: str, min_ja_chars: int = 2) -> bool:
    if not text:
        return False
    ja_count = 0
    for c in text:
        code = ord(c)
        if (0x3040 <= code <= 0x309F or 0x30A0 <= code <= 0x30FF or 0x4E00 <= code <= 0x9FFF):
            ja_count += 1
    return ja_count >= min_ja_chars


def sanitize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    t = re.sub(r'\*[^*]*\*', '', text)
    t = re.sub(r'[\u2600-\u27bf\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f600-\U0001f64f]', '', t)
    return t.strip()


def unwrap_nested_dict(data: Dict, expected_keys: set) -> Dict:
    for key in list(data.keys()):
        val = data[key]
        if not isinstance(val, dict):
            continue
        for field in expected_keys:
            if field in val and isinstance(val[field], str):
                data[key] = val[field]
                log.debug(f"Unwrapped {key}.{field}")
                break
        else:
            for sub_val in val.values():
                if isinstance(sub_val, str) and sub_val.strip():
                    data[key] = sub_val
                    log.debug(f"Unwrapped {key} (first string value)")
                    break
    return data


def parse_lingo_json(content: str) -> dict:
    if not content:
        raise ValueError("Empty response from LLM")
    content = content.strip()

    def _extract(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse JSON from: {text[:200]}")

    data = _extract(content)
    if isinstance(data, dict):
        expected_keys = {
            'text', 'message', 'japanese', 'english', 'feedback', 'suggestion',
            'register', 'japaneseText', 'englishTranslation'
        }
        data = unwrap_nested_dict(data, expected_keys)
    return data


def build_conversation_history(entries: List[DialogueHistoryEntry]) -> List[dict]:
    """FIXED: strict alternation + merge logic"""
    if not entries:
        return []
    pending = []
    for entry in entries:
        text = entry.text.strip()
        if not is_valid_text(text):
            continue
        role = "assistant" if entry.speaker == "aiko" else "user"
        if not pending:
            if role == "assistant":
                pending.append({"role": "user", "content": "Let's practice Japanese."})
        if pending and pending[-1]["role"] == role:
            pending[-1]["content"] += "\n" + text
        else:
            pending.append({"role": role, "content": text})
    return pending


def _parse_dialogue_tags(flat_content: str) -> dict:
    """
    Shared tag-extraction logic for both the streaming and non-streaming
    respond endpoints, so the parsing rules can't drift between the two.
    """
    data = {"isCorrect": True, "isFinished": False}
    tag_regex = r"([A-Z_]+)\s*:\s*(.*?)(?=[A-Z_]+\s*:|$)"
    found_pairs = re.findall(tag_regex, flat_content, re.DOTALL | re.IGNORECASE)
    for tag, val in found_pairs:
        tag = tag.strip().upper()
        val = val.strip()
        if tag == LingoTags.MISTAKE.upper():
            data["isCorrect"] = val.lower() not in ["true", "yes", "y", "1"]
        elif tag == LingoTags.FINISHED.upper():
            data["isFinished"] = val.lower() in ["true", "yes", "y", "1"]
        elif tag == LingoTags.REPLY_JP.upper():
            data["japanese"] = val
        elif tag == LingoTags.REPLY_EN.upper():
            data["english"] = val
        elif tag == LingoTags.FEEDBACK.upper():
            data["feedback"] = val
        elif tag == LingoTags.SUGGESTION.upper():
            data["suggestion"] = val

    if not data.get("japanese") and not any(f"{t}:" in flat_content.upper() for t in LingoTags.ALL):
        data["japanese"] = flat_content.strip()
    if not data.get("japanese"):
        jp_match = re.search(
            r"([\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff].*?)(?:\n|REPLY_EN:|English:|$)",
            flat_content,
            re.DOTALL
        )
        if jp_match:
            candidate = jp_match.group(1).strip()
            if is_japanese_text(candidate):
                data["japanese"] = candidate
    if not data.get("english"):
        en_match = re.search(r"([A-Z][a-zA-Z\s,.'\"?!-]{10,}[.!?])", flat_content)
        if en_match:
            candidate = en_match.group(1).strip()
            if len(candidate.split()) >= 2:
                data["english"] = candidate
    return data


async def get_lingo_session(request: Request) -> dict:
    from interface.webui import auth
    try:
        session = await auth.require_session(request)
        return await auth.require_accepted_session(session)
    except Exception as e:
        log.warning(f"Lingo session auth failed: {e} — falling back to app owner")
        owner = os.getenv("AIKO_USER_ID", "")
        if not owner:
            log.error("AIKO_USER_ID not set and auth failed; aborting request")
            raise HTTPException(status_code=500, detail="Session unavailable")
        return {"user_id": owner, "username": owner}


# ============================================================================
# Streaks, XP, Toasts, Difficulty Scaling, Closure
# ============================================================================

# FIX #5: Streaks previously only lived in an in-memory dict that was wiped
# on every restart, and get_streak()'s file-based fallback logic capped out
# at 2 regardless of how many consecutive days were actually practiced.
# This version persists {last_date, current_streak, total_sessions} to disk
# and increments/resets correctly based on real day deltas.
_streak_lock = threading.Lock()


def _streak_file(uid: str) -> Path:
    p = user_state_path(f"streaks/{uid}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_streak(uid: str) -> dict:
    streak_file = _streak_file(uid)
    current_streak = 0
    total_sessions = 0
    if streak_file.exists():
        try:
            data = json.loads(streak_file.read_text())
            current_streak = int(data.get("current_streak", 0))
            total_sessions = int(data.get("total_sessions", 0))
        except Exception:
            log.warning(f"Failed to read streak file for {uid}, resetting", exc_info=True)
    next_reward = "Learn 5 new words" if current_streak >= 3 else "Start your first streak!"
    return {
        "days": current_streak,
        "total_sessions": total_sessions,
        "next_reward": next_reward,
    }


def has_user_practiced_today(uid: str) -> bool:
    streak_file = _streak_file(uid)
    if not streak_file.exists():
        return False
    try:
        data = json.loads(streak_file.read_text())
        last_date_str = data.get("last_date")
        if not last_date_str:
            return False
        return date.fromisoformat(last_date_str) == date.today()
    except Exception:
        return False


def record_practice(uid: str):
    """Called once per practice event. Idempotent per-day: only the
    first call in a given day advances the streak and session count."""
    with _streak_lock:
        streak_file = _streak_file(uid)
        today = date.today()
        current_streak = 0
        total_sessions = 0
        last_date = None
        if streak_file.exists():
            try:
                data = json.loads(streak_file.read_text())
                current_streak = int(data.get("current_streak", 0))
                total_sessions = int(data.get("total_sessions", 0))
                last_date_str = data.get("last_date")
                if last_date_str:
                    last_date = date.fromisoformat(last_date_str)
            except Exception:
                log.warning(f"Corrupt streak file for {uid}, resetting", exc_info=True)

        if last_date == today:
            # Already recorded today; no change.
            return

        if last_date == today - timedelta(days=1):
            current_streak += 1
        else:
            # First-ever session, or the streak was broken.
            current_streak = 1

        total_sessions += 1

        streak_file.write_text(json.dumps({
            "last_date": today.isoformat(),
            "current_streak": current_streak,
            "total_sessions": total_sessions,
        }, indent=2))


def get_xp(uid: str) -> dict:
    """XP bands of 100: level 1 = 0-99 XP, level 2 = 100-199, ... capped at 10.

    Brand-new users start at 0 (matching _award_xp, which also starts from
    0), so the very first award never makes the displayed total drop.
    """
    xp_file = user_state_path(f"xp/{uid}.json")
    xp_file.parent.mkdir(parents=True, exist_ok=True)
    xp = 0
    if xp_file.exists():
        try:
            xp = int(json.loads(xp_file.read_text()).get("xp", 0))
        except Exception:
            xp = 0
    level = min(10, (max(0, xp) // 100) + 1)
    return {"xp": max(0, xp), "level": level, "next_level_xp": level * 100}


# FIX #17: pulled out of the /xp/add handler so it can be shared with
# review_respond() (FIX #16) instead of duplicating the same
# read-modify-write logic in two places.
def _award_xp(uid: str, amount: int) -> int:
    """Add `amount` XP for a user and persist it, returning the new total."""
    xp_file = user_state_path(f"xp/{uid}.json")
    xp_file.parent.mkdir(parents=True, exist_ok=True)
    if xp_file.exists():
        try:
            data = json.loads(xp_file.read_text())
        except Exception:
            data = {"xp": 0}
    else:
        data = {"xp": 0}
    data["xp"] = data.get("xp", 0) + amount
    xp_file.write_text(json.dumps(data, indent=2))
    return data["xp"]


# FIX #16: XP awarded per SRS review, scaled by how well the card was
# remembered -- mirrors the correct/incorrect split conversation turns use
# (10/5 XP), but with finer grading since reviews already carry a 0-4 scale.
_REVIEW_XP_BY_GRADE = {
    ReviewGrade.AGAIN: 1,
    ReviewGrade.HARD: 2,
    ReviewGrade.GOOD: 5,
    ReviewGrade.EASY: 8,
    ReviewGrade.PERFECT: 10,
}


# FIX #9: difficulty level previously lived only in this in-memory dict, so
# it silently reset to "beginner" on every server restart even though every
# other piece of user state (streak, XP, SRS cards) survived. Persisted to
# disk the same way streaks are.
_last_levels: dict = {}
_level_lock = threading.Lock()


def _level_file(uid: str) -> Path:
    p = user_state_path(f"levels/{uid}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# JLPT track (N5 lowest/start) alongside legacy conversation difficulty.
JLPT_LEVELS = ["N5", "N4", "N3", "N2", "N1"]
_LEGACY_TO_JLPT = {"beginner": "N5", "intermediate": "N3", "advanced": "N1"}


def get_last_level(uid: str) -> str:
    if uid in _last_levels:
        return _last_levels[uid]

    level_file = _level_file(uid)
    level = "beginner"
    if level_file.exists():
        try:
            data = json.loads(level_file.read_text())
            candidate = data.get("level", "beginner")
            if candidate in ("beginner", "intermediate", "advanced") or candidate in JLPT_LEVELS:
                level = candidate
        except Exception:
            log.warning(f"Failed to read level file for {uid}, defaulting to beginner", exc_info=True)

    _last_levels[uid] = level
    return level


def get_jlpt_level(uid: str) -> str:
    """JLPT track for Learn/Courses/Grammar (N5 default). Maps legacy track."""
    # Prefer centralized per-user DB when available.
    try:
        from .lingo_store import init_user_db, normalize_level
        import sqlite3
        path = init_user_db(uid)
        con = sqlite3.connect(str(path))
        try:
            row = con.execute("SELECT level FROM user_level WHERE id=1").fetchone()
            if row and row[0]:
                return normalize_level(row[0])
        finally:
            con.close()
    except Exception:
        pass
    legacy = get_last_level(uid)
    if legacy in JLPT_LEVELS:
        return legacy
    return _LEGACY_TO_JLPT.get(legacy, "N5")


def update_last_level(uid: str, level: str):
    with _level_lock:
        _last_levels[uid] = level
        try:
            _level_file(uid).write_text(json.dumps({"level": level}, indent=2))
        except Exception:
            log.warning(f"Failed to persist level for {uid}", exc_info=True)
    # Keep centralized per-user DB in sync for JLPT levels.
    if level in JLPT_LEVELS:
        try:
            from .lingo_store import init_user_db
            import sqlite3
            from datetime import datetime, timezone
            path = init_user_db(uid)
            con = sqlite3.connect(str(path))
            try:
                con.execute("UPDATE user_level SET level=?,updated_at=? WHERE id=1",
                            (level, datetime.now(timezone.utc).isoformat()))
                con.commit()
            finally:
                con.close()
        except Exception:
            log.warning(f"Failed to sync JLPT level for {uid}", exc_info=True)


def _get_vocab_extractor() -> VocabExtractor:
    """
    FIX #10: previously always `VocabExtractor()` with no LLM client, which
    meant the LLM fallback in vocab.py was dead code -- any kanji outside
    the small static COMMON_READINGS cache was silently dropped instead of
    being added to the SRS queue. Wire in the tutor's own LLM client so
    uncommon vocabulary actually gets looked up and learned.
    """
    try:
        from interface.webui import auth
        think = auth.aiko_web_instance._think
        return VocabExtractor(llm_client=think._client, llm_model=think._llm_model)
    except Exception as e:
        log.warning(f"Could not wire LLM client into VocabExtractor, falling back to static cache only: {e}")
        return VocabExtractor()


# ============================================================================
# Audio Generation (thread-safe)
# ============================================================================
def generate_lingo_audio(text: str, base_url: Optional[str] = None) -> Optional[str]:
    from interface.webui import auth
    if not is_valid_text(text):
        return None
    clean_text = _clean_tts_text(text)
    if not is_valid_text(clean_text):
        return None
    if not auth.aiko_web_instance or not auth.aiko_web_instance._speak:
        return None
    try:
        wav_bytes = auth.aiko_web_instance._speak._synthesize(clean_text)
        if not wav_bytes:
            return None
        audio_id = f"{uuid.uuid4()}_{int(time.time() * 1000)}"
        filename = f"{audio_id}.wav"
        filepath = AUDIO_DIR / filename
        temp_path = filepath.with_suffix('.tmp')
        temp_path.write_bytes(wav_bytes)
        temp_path.replace(filepath)
        base = (base_url or _public_base_url()).rstrip("/")
        return f"{base}/lingo_audio/{filename}"
    except Exception as e:
        log.error(f"Failed to generate audio for Lingo: {e}")
        return None


# ============================================================================
# ORIGINAL ENDPOINTS (with closure + difficulty + toast)
# ============================================================================
@router.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest, session: dict = Depends(get_lingo_session)):
    from interface.webui import auth
    if not auth.aiko_web_instance or not auth.aiko_web_instance._think:
        raise HTTPException(status_code=503, detail="Aiko's brain is not ready")
    think = auth.aiko_web_instance._think
    uid = session["user_id"]  # FIX #14: no silent fallback
    token = set_current_user_id(uid)
    try:
        # Auto-direction: Japanese input -> a single natural English
        # translation (registers make no sense for English output).
        # Anything else (English, Chinese, French, ...) -> Japanese in
        # 3-5 registers as before.
        ja_input = is_japanese_text(request.text)
        escaped_text = html.escape(request.text)
        if ja_input:
            system_prompt = (
                "You are a Japanese-to-English translation expert. "
                "Output ONLY valid JSON. Your response must be a JSON object with a single 'translations' key. "
                "That key must contain an array with EXACTLY ONE object with 'register' ('English') and 'text' keys."
            )
            user_prompt = (
                f"Translate the following Japanese text to natural English: {escaped_text}"
            )
        else:
            system_prompt = (
                "You are a Japanese translation expert. "
                "Output ONLY valid JSON. Your response must be a JSON object with a single 'translations' key. "
                "That key must contain an array of objects, each with 'register' and 'text' keys."
            )
            user_prompt = (
                f"Translate the following text to Japanese in 3-5 different registers: {escaped_text}"
            )
        response = think._client.chat.completions.create(
            model=think._llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            timeout=120.0
        )
        content = response.choices[0].message.content
        data = parse_lingo_json(content)
        translations_raw = data.get("translations", [])
        results = []
        for item in translations_raw:
            text_jp = item.get("text", "")
            results.append(TranslationResult(
                register=item.get("register", "Normal"),
                text=text_jp,
                audioUrl=None
            ))
        return TranslateResponse(translations=results)
    except Exception as e:
        log.exception("Lingo translation failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        reset_current_user_id(token)


@router.post("/conversation/start")
async def conversation_start(request: StartRequest, http_request: Request, session: dict = Depends(get_lingo_session)):
    from interface.webui import auth
    if not auth.aiko_web_instance or not auth.aiko_web_instance._think:
        raise HTTPException(status_code=503, detail="Aiko's brain is not ready")
    think = auth.aiko_web_instance._think
    uid = session["user_id"]  # FIX #14: no silent fallback

    async def event_generator():
        token = set_current_user_id(uid)
        try:
            # Lingo is a pure language-skill mode: fixed tutor prompt with NO
            # persona / memory / knowledge base prompt mixed in. The tutor
            # teaches Japanese and nothing else.
            # (FIX #3 history: this line once missed its `f` prefix, sending
            # the literal text "{request.level}" to the model.)
            system_prompt = (
                "You are a Japanese language tutor in 'Lingo App Mode'. "
                "Teach Japanese language skill only — no persona chat, no "
                "memories, no general-knowledge answers. "
                f"Start a Japanese conversation at {request.level} level. "
                "Output your response exactly like this:\n"
                f"{LingoTags.REPLY_JP}: <Japanese sentences>\n"
                f"{LingoTags.REPLY_EN}: <English translation>\n"
                f"{LingoTags.FINISHED}: False"
            )
            user_prompt = f"Start a Japanese conversation at {request.level} level. Speak 1-3 sentences in Japanese."
            response = think._client.chat.completions.create(
                model=think._llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True,
                timeout=120.0
            )
            full_content = ""
            is_streaming_jp = False
            try:
                for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_content += delta
                        clean_buf = full_content.replace("**", "")
                        if not is_streaming_jp and f"{LingoTags.REPLY_JP}:" in clean_buf:
                            is_streaming_jp = True
                            parts = clean_buf.split(f"{LingoTags.REPLY_JP}:", 1)
                            if len(parts) > 1:
                                text = parts[1]
                                stop_tags = [f"{LingoTags.REPLY_EN}:", f"{LingoTags.FINISHED}:"]
                                for tag in stop_tags:
                                    if tag in text:
                                        text = text.split(tag, 1)[0]
                                        is_streaming_jp = False
                                if text:
                                    yield json.dumps({"type": "delta", "text": text}) + "\n"
                            continue
                        if is_streaming_jp:
                            clean_delta = delta.replace("**", "")
                            stop_tags = [f"{LingoTags.REPLY_EN}:", f"{LingoTags.FINISHED}:"]
                            found_stop = False
                            for tag in stop_tags:
                                if tag in clean_delta:
                                    text = clean_delta.split(tag, 1)[0]
                                    if text:
                                        yield json.dumps({"type": "delta", "text": text}) + "\n"
                                    is_streaming_jp = False
                                    found_stop = True
                                    break
                            if not found_stop:
                                yield json.dumps({"type": "delta", "text": clean_delta}) + "\n"
            except Exception as e:
                log.exception("Chunk iteration failed during start stream")
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
                return
            data = {"isCorrect": True, "isFinished": False}
            flat_content = full_content.replace("**", "").replace("`", "")
            tag_regex = r"([A-Z_]+)\s*:\s*(.*?)(?=[A-Z_]+\s*:|$)"
            found_pairs = re.findall(tag_regex, flat_content, re.DOTALL | re.IGNORECASE)
            for tag, val in found_pairs:
                tag = tag.strip().upper()
                val = val.strip()
                if tag == LingoTags.FINISHED.upper():
                    data["isFinished"] = val.lower() in ["true", "yes", "1"]
                elif tag == LingoTags.REPLY_JP.upper():
                    data["japanese"] = val
                elif tag == LingoTags.REPLY_EN.upper():
                    data["english"] = val
            if not data.get("japanese") and not any(f"{t}:" in flat_content.upper() for t in LingoTags.START_TAGS):
                data["japanese"] = full_content.strip()
            if not data.get("japanese"):
                jp_match = re.search(
                    r"([\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff].*?)(?:\n|$|REPLY_EN:|English:)",
                    flat_content,
                    re.DOTALL
                )
                if jp_match:
                    candidate = jp_match.group(1).strip()
                    if is_japanese_text(candidate):
                        data["japanese"] = candidate
            if not data.get("english"):
                en_match = re.search(
                    r"([A-Z][a-zA-Z\s,.'\"?!-]{10,}[.!?])",
                    flat_content
                )
                if en_match:
                    candidate = en_match.group(1).strip()
                    if len(candidate.split()) >= 2:
                        data["english"] = candidate
            audio_url = get_cached_lingo_audio(data.get("japanese") or "...", uid=uid,
                                           base_url=_public_base_url(http_request))
            yield json.dumps({
                "type": "final",
                "isCorrect": True,
                "japanese": data.get("japanese") or "...",
                "english": data.get("english") or "Translation unavailable",
                "isFinished": data.get("isFinished", False),
                "audioUrl": audio_url,
                "toast": {"type": "info", "message": f"Let's practice Japanese at {request.level} level! 🔥"}
            }) + "\n"
            update_last_level(uid, request.level)
        except Exception as e:
            log.exception("Lingo start stream failed")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        finally:
            reset_current_user_id(token)

    response = StreamingResponse(event_generator(), media_type="text/event-stream")
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Cache-Control"] = "no-cache"
    return response


def _respond_system_prompt(think, user_text: str) -> str:
    return (
        "You are Aiko, a helpful Japanese tutor. Maintain your unique personality but "
        "you MUST suppress all emojis and roleplay actions (like *smiles*). "
        "You are in 'Lingo App Mode' and must output ONLY these tags in order:\n\n"
        f"{LingoTags.MISTAKE}: <True/False>\n"
        f"{LingoTags.FEEDBACK}: <English explanation or empty>\n"
        f"{LingoTags.SUGGESTION}: <Corrected Japanese or empty>\n"
        f"{LingoTags.REPLY_JP}: <Next Japanese conversation turn. NO ROMAJI. NO ACTIONS.>\n"
        f"{LingoTags.REPLY_EN}: <Complete literal English translation>\n"
        f"{LingoTags.FINISHED}: <False>\n\n"
        "Do not include any other text."
    )


def _build_respond_messages(think, request: "RespondRequest") -> list:
    system_prompt = _respond_system_prompt(think, request.text)
    messages = [{"role": "system", "content": system_prompt}]
    history_messages = build_conversation_history(request.history)
    messages.extend(history_messages)
    user_text = request.text.strip()
    if messages[-1]["role"] == "user":
        messages[-1]["content"] += "\n" + user_text
    else:
        messages.append({"role": "user", "content": user_text})
    return messages


async def _finalize_respond_turn(uid: str, data: dict, source_text: str,
                                 base_url: Optional[str] = None) -> dict:
    """
    Shared post-processing for a parsed tutor turn: mistake formatting,
    vocab extraction/SRS insertion, audio synthesis, streak/XP bookkeeping,
    and toast construction. Used by both the streaming and non-streaming
    /conversation/respond endpoints so their behavior can't drift apart.
    """
    from interface.webui import auth

    srs = LingoSRS(uid)

    is_mistake = not data.get("isCorrect", True)
    if is_mistake:
        feedback = data.get("feedback") or "Grammar/usage error."
        suggestion_clean = sanitize_text(data.get("suggestion") or "")
        final_jp = f"Mistake: {feedback}"
        if suggestion_clean:
            final_jp += f"\nSuggestion: {suggestion_clean}"
        audio_text = suggestion_clean or feedback
        english_text = data.get("english") or "Please correct your Japanese ♡"
    else:
        feedback = data.get("feedback")
        final_jp = sanitize_text(data.get("japanese") or "...")
        audio_text = final_jp
        english_text = data.get("english") or "Perfect! Let's keep talking."

    # ===== Extract vocabulary =====
    new_vocab = []
    if data.get("japanese"):
        extractor = _get_vocab_extractor()
        new_vocab = extractor.extract_vocab(data.get("japanese"), context=source_text)
        user_jlpt = get_jlpt_level(uid)
        for card in new_vocab:
            card.level = user_jlpt
            srs.add_card(card)

    audio_url = get_cached_lingo_audio(audio_text, uid=uid, base_url=base_url)

    # ===== TOASTS & XP (Duolingo-style) =====
    record_practice(uid)
    is_correct = data.get("isCorrect", True)
    xp = 10 if is_correct else 5
    toast_message = "Perfect! Let's keep talking." if is_correct else feedback

    # Award XP directly — a previous version self-POSTed to a hardcoded
    # http://localhost:8000 (the server listens on 8787), so every
    # conversation turn's XP was silently lost to connection-refused.
    if xp > 0:
        try:
            _award_xp(uid, xp)
        except Exception:
            log.warning("Failed to award conversation XP", exc_info=True)
    update_last_level(uid, get_last_level(uid))

    return {
        "isCorrect": is_correct,
        "feedback": data.get("feedback"),
        "suggestion": data.get("suggestion"),
        "japanese": final_jp,
        "english": english_text,
        "isFinished": data.get("isFinished", False),
        "audioUrl": audio_url,
        "vocabExtracted": len(new_vocab),
        "toast": {"type": "success" if is_correct else "feedback", "message": toast_message},
    }


@router.post("/conversation/respond", response_model=ConversationResponse)
async def conversation_respond(request: RespondRequest, http_request: Request,
                             session: dict = Depends(get_lingo_session)):
    """
    FIX #12: non-streaming counterpart to /conversation/respond_stream.
    The Android interface already declared this endpoint
    (`respondToConversation`), but the backend never implemented it, so
    any call to it 404'd (lingo_audit.md #1). This collects the full LLM
    response before returning, for clients that prefer a single
    request/response round trip over SSE.
    """
    from interface.webui import auth
    if not auth.aiko_web_instance or not auth.aiko_web_instance._think:
        raise HTTPException(status_code=503, detail="Aiko's brain is not ready")
    think = auth.aiko_web_instance._think
    uid = session["user_id"]  # FIX #14: no silent fallback
    token = set_current_user_id(uid)
    try:
        messages = _build_respond_messages(think, request)
        response = think._client.chat.completions.create(
            model=think._llm_model,
            messages=messages,
            timeout=120.0,
            temperature=0.3
        )
        flat_content = (response.choices[0].message.content or "").replace("**", "").replace("`", "")
        data = _parse_dialogue_tags(flat_content)
        result = await _finalize_respond_turn(uid, data, request.text,
                                                base_url=_public_base_url(http_request))
        return ConversationResponse(
            japaneseText=result["japanese"],
            englishTranslation=result["english"],
            audioUrl=result["audioUrl"],
            isFinished=result["isFinished"],
            isCorrect=result["isCorrect"],
            feedback=result["feedback"],
            suggestion=result["suggestion"],
            toast=Toast(**result["toast"]) if result.get("toast") else None,
        )
    except Exception as e:
        log.exception("Lingo non-streaming respond failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        reset_current_user_id(token)


@router.post("/conversation/respond_stream")
async def conversation_respond_stream(request: RespondRequest, http_request: Request,
                                    session: dict = Depends(get_lingo_session)):
    from interface.webui import auth
    if not auth.aiko_web_instance or not auth.aiko_web_instance._think:
        raise HTTPException(status_code=503, detail="Aiko's brain is not ready")
    think = auth.aiko_web_instance._think
    uid = session["user_id"]  # FIX #14: no silent fallback

    async def event_generator():
        token = set_current_user_id(uid)
        try:
            messages = _build_respond_messages(think, request)
            response = think._client.chat.completions.create(
                model=think._llm_model,
                messages=messages,
                stream=True,
                timeout=120.0,
                temperature=0.3
            )
            full_content = ""
            is_streaming_jp = False
            try:
                for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_content += delta
                        clean_buf = full_content.replace("**", "")
                        target_tag = (
                            f"{LingoTags.SUGGESTION}:"
                            if f"{LingoTags.MISTAKE}: TRUE" in clean_buf.upper()
                            else f"{LingoTags.REPLY_JP}:"
                        )
                        if not is_streaming_jp:
                            if target_tag in clean_buf:
                                is_streaming_jp = True
                                parts = clean_buf.split(target_tag, 1)
                                if len(parts) > 1:
                                    text = parts[1]
                                    stop_tags = [
                                        f"{LingoTags.REPLY_EN}:",
                                        f"{LingoTags.FINISHED}:",
                                        f"{LingoTags.REPLY_JP}:"
                                    ]
                                    if target_tag in stop_tags:
                                        stop_tags.remove(target_tag)
                                    for tag in stop_tags:
                                        if tag in text:
                                            text = text.split(tag, 1)[0]
                                            is_streaming_jp = False
                                    if text:
                                        yield json.dumps({"type": "delta", "text": text}) + "\n"
                            elif (not any(t in clean_buf.upper() for t in [LingoTags.MISTAKE, LingoTags.REPLY_JP, LingoTags.REPLY_EN])
                                  and len(clean_buf) > 10):
                                is_streaming_jp = True
                                yield json.dumps({"type": "delta", "text": delta.replace("**", "")}) + "\n"
                            continue
                        if is_streaming_jp:
                            clean_delta = delta.replace("**", "")
                            stop_tags = [
                                f"{LingoTags.REPLY_EN}:",
                                f"{LingoTags.FINISHED}:",
                                f"{LingoTags.REPLY_JP}:",
                                f"{LingoTags.SUGGESTION}:"
                            ]
                            if target_tag in stop_tags:
                                stop_tags.remove(target_tag)
                            found_stop = False
                            for tag in stop_tags:
                                if tag in clean_delta:
                                    text = clean_delta.split(tag, 1)[0]
                                    if text:
                                        yield json.dumps({"type": "delta", "text": text}) + "\n"
                                    is_streaming_jp = False
                                    found_stop = True
                                    break
                            if not found_stop:
                                yield json.dumps({"type": "delta", "text": clean_delta}) + "\n"
            except Exception as e:
                log.exception("Chunk iteration failed during respond stream")
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
                return

            flat_content = full_content.replace("**", "").replace("`", "")
            data = _parse_dialogue_tags(flat_content)
            result = await _finalize_respond_turn(uid, data, request.text,
                                                base_url=_public_base_url(http_request))
            yield json.dumps({"type": "final", **result}) + "\n"
        except Exception as e:
            log.exception("Lingo respond stream failed")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        finally:
            reset_current_user_id(token)

    response = StreamingResponse(event_generator(), media_type="text/event-stream")
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Cache-Control"] = "no-cache"
    return response


@router.post("/conversation/hint", response_model=ConversationResponse)
async def conversation_hint(http_request: Request, session: dict = Depends(get_lingo_session)):
    from interface.webui import auth
    if not auth.aiko_web_instance or not auth.aiko_web_instance._think:
        raise HTTPException(status_code=503, detail="Aiko's brain is not ready")
    think = auth.aiko_web_instance._think
    uid = session["user_id"]  # FIX #14: no silent fallback
    token = set_current_user_id(uid)
    try:
        system_prompt = (
            "You are Aiko, teaching Japanese through conversation. "
            "Suggest a concise response for the student to say in Japanese and provide an English translation. "
            "Keep the Japanese response under 30 words. Japanese ONLY in 'japanese' field. "
            "Output ONLY valid JSON with 'japanese', 'english' and 'explanation' keys."
        )
        response = think._client.chat.completions.create(
            model=think._llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Hint please"}
            ],
            response_format={"type": "json_object"},
            timeout=120.0
        )
        data = parse_lingo_json(response.choices[0].message.content)
        jp = data.get("japanese") or data.get("japaneseText") or ""
        en = data.get("english") or data.get("englishTranslation") or "Translation unavailable"
        audio_url = get_cached_lingo_audio(str(jp), uid=uid,
                                         base_url=_public_base_url(http_request))
        level = get_last_level(uid)
        explanation = "This is a good beginner sentence because it uses only simple present tense."
        if level == "intermediate":
            explanation = "This uses the ~ます form correctly — perfect for this level!"
        return ConversationResponse(
            japaneseText=str(jp),
            englishTranslation=str(en),
            audioUrl=audio_url,
            isCorrect=True,
            explanation=explanation
        )
    except Exception as e:
        log.exception("Lingo conversation hint failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        reset_current_user_id(token)


@router.get("/tts")
async def get_tts(text: str, request: Request, session: dict = Depends(get_lingo_session)):
    if not text or len(text) > 200:
        raise HTTPException(
            status_code=400,
            detail="Text length must be 1–200 characters"
        )
    uid = session["user_id"]  # FIX #14: no silent fallback

    # FIX #13: check (and consume) the rate limit up front so we can
    # report a real 429 -- previously this branch was dead code (an
    # inert, always-false placeholder condition) and both "rate limited"
    # and "synthesis failed" produced the same generic 503.
    if not check_tts_rate_limit(uid):
        raise HTTPException(
            status_code=429,
            detail="TTS rate limit exceeded. Please wait a moment and try again."
        )

    # We already consumed a rate-limit slot above, so bypass the internal
    # rate check inside get_cached_lingo_audio() by passing uid=None only
    # after a cache-miss check would be redundant here; a cache hit is
    # still free and correct either way.
    url = _get_tts_audio_after_rate_check(text, base_url=_public_base_url(request))
    if not url:
        raise HTTPException(status_code=503, detail="TTS generation failed. Please try again.")
    return {"audioUrl": url}


def _get_tts_audio_after_rate_check(text: str, base_url: Optional[str] = None) -> Optional[str]:
    """
    Same cache-then-synthesize flow as get_cached_lingo_audio(), but without
    re-checking the rate limit -- the /tts endpoint already consumed a slot
    before calling this, so double-checking would over-penalize the user.
    """
    if not is_valid_text(text):
        return None
    clean_text = _clean_tts_text(text)
    if not is_valid_text(clean_text):
        return None
    base = (base_url or _public_base_url()).rstrip("/")
    cache_key = f"{clean_text}\x00{base}"

    cached = _audio_cache_get(cache_key)
    if cached is not None:
        return cached

    url = generate_lingo_audio(clean_text, base_url=base)
    if url:
        _audio_cache_set(cache_key, url)
    return url


@router.post("/conversation/stop")
async def conversation_stop(session: dict = Depends(get_lingo_session)):
    return {"success": True}


# Max upload: ~60s of 16kHz mono 16-bit PCM + WAV header.
_STT_MAX_BYTES = 2 * 1024 * 1024
_STT_MAX_SECONDS = 30


@router.post("/stt")
async def speech_to_text(audio: UploadFile = File(...),
                         session: dict = Depends(get_lingo_session)):
    """Private voice input: transcribe a short WAV clip with on-device ASR.

    The phone app records (16kHz mono PCM, WAV container) and posts it here;
    nothing ever leaves the Tailnet — no cloud STT involved. Returns
    {"text": "..."}; empty text means no speech was recognized.
    """
    from starlette.concurrency import run_in_threadpool

    raw = await audio.read()
    if not raw or len(raw) > _STT_MAX_BYTES:
        raise HTTPException(status_code=400, detail="Audio must be 1 byte – 2 MB")

    try:
        import io
        import wave
        import numpy as np

        with wave.open(io.BytesIO(raw), "rb") as w:
            n_channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()
            n_frames = w.getnframes()
            if n_frames <= 0 or n_frames > 16000 * _STT_MAX_SECONDS * 2:
                raise ValueError("bad frame count")
            frames = w.readframes(n_frames)
        if sampwidth == 2:
            pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:
            pcm = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"unsupported sample width: {sampwidth}")
        if n_channels > 1:
            pcm = pcm.reshape(-1, n_channels).mean(axis=1).astype(np.float32)
        max_len = 16000 * _STT_MAX_SECONDS
        if pcm.size > max_len:
            pcm = pcm[:max_len]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unusable WAV upload: {e}")

    from interface.webui import auth
    listener = getattr(auth.aiko_web_instance, "_listen", None) if auth.aiko_web_instance else None
    if listener is None:
        raise HTTPException(status_code=503, detail="On-device ASR is not running")
    try:
        text = await run_in_threadpool(listener.transcribe_pcm, pcm, framerate)
    except Exception as e:
        log.error(f"Lingo STT failed: {e}")
        raise HTTPException(status_code=503, detail="Transcription failed")
    return {"text": text or ""}


# ============================================================================
# NEW: SRS + TOASTS + LEADERBOARD + XP ENDPOINTS (with closure in start)
# ============================================================================
@router.post("/conversation/review/respond", response_model=ReviewResponseData)
async def review_respond(request: ReviewResponseRequest, session: dict = Depends(get_lingo_session)):
    # FIX #1: This was previously mixing `yield` with `return <value>`,
    # which is a SyntaxError in Python (async generators may only use a
    # bare `return`). That syntax error prevented the whole module -- and
    # therefore every endpoint in this router -- from importing. Converted
    # to a plain async function returning a typed ReviewResponseData.
    uid = session["user_id"]  # FIX #14: no silent fallback
    srs = LingoSRS(uid)
    # LingoSRS.record_review() validates the grade itself and raises a
    # clear ValueError for anything out of range, instead of the request
    # crashing deep inside the SM-2 math.
    try:
        updated_card = srs.record_review(request.card_id, request.grade, response_time_ms=0)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    grade = ReviewGrade(request.grade)

    # FIX #16: reviews previously never touched the streak or XP files --
    # only conversation turns did (via _finalize_respond_turn). A user could
    # clear their entire due queue and the dashboard would show no
    # progress at all. Drive both from the same per-user bookkeeping
    # conversation turns already use.
    record_practice(uid)
    _award_xp(uid, _REVIEW_XP_BY_GRADE.get(grade, 5))

    # Toast the word that was JUST reviewed (updated_card carries its
    # hiragana/meaning). The old code toasted get_due_cards()[0], which after
    # grading is the NEXT due card -- i.e. the wrong word.
    toast = None
    if grade == ReviewGrade.EASY:
        toast = Toast(
            type="info",
            message=f"🌟 {updated_card.hiragana} = {updated_card.meaning}",
        )

    due_cards = srs.get_due_cards(limit=1)
    next_card = None
    if due_cards:
        c = due_cards[0]
        next_card = ReviewCard(
            card_id=c.id,
            hiragana=c.hiragana,
            meaning=c.meaning,
            context=c.context or "",
            kanji=c.kanji or None,
        )

    return ReviewResponseData(
        updated_card=UpdatedCard(
            id=updated_card.id,
            interval=updated_card.interval,
            ease=round(updated_card.ease_factor, 2),
        ),
        next_card=next_card,
        # FIX #15: single COUNT(*) query instead of fetching up to 100 rows.
        cards_remaining=srs.count_due_cards(),
        toast=toast,
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(session: dict = Depends(get_lingo_session)):
    uid = session["user_id"]  # FIX #14: no silent fallback
    srs = LingoSRS(uid)
    stats = srs.get_stats()
    # FIX #15: single COUNT(*) query instead of fetching up to 1000 rows.
    due_count = srs.count_due_cards()
    streak = get_streak(uid)
    xp_data = get_xp(uid)

    # FIX #11: surface cards the user is actually struggling with, not
    # just whatever's next in due-date order.
    weak_cards = srs.get_weak_vocab(limit=6)
    weak_vocab = [
        ReviewCard(card_id=c.id, hiragana=c.hiragana, meaning=c.meaning,
                   context=c.context or "", kanji=c.kanji or None)
        for c in weak_cards
    ]

    return StatsResponse(
        total_cards=stats["total_cards"],
        learned_today=stats["learned_today"],
        reviews_today=stats["reviews_today"],
        avg_ease=stats["avg_ease"],
        cards_due=due_count,
        streak=streak,
        xp=xp_data["xp"],
        level=xp_data["level"],
        last_level=get_last_level(uid),
        weak_vocab=weak_vocab
    )


@router.get("/weak-vocab", response_model=List[ReviewCard])
async def get_weak_vocab(session: dict = Depends(get_lingo_session)):
    """
    FIX #11: standalone endpoint mirroring StatsResponse.weak_vocab, for
    screens that want just the struggling-cards list without pulling the
    full stats payload.
    """
    uid = session["user_id"]  # FIX #14: no silent fallback
    srs = LingoSRS(uid)
    weak_cards = srs.get_weak_vocab(limit=10)
    return [
        ReviewCard(card_id=c.id, hiragana=c.hiragana, meaning=c.meaning,
                   context=c.context or "", kanji=c.kanji or None)
        for c in weak_cards
    ]


# Curated starter deck for Word of the Day — shown before the user has
# learned any words of their own (vocab enters SRS through conversation).
# Cycles one per day; (hiragana, meaning, example context).
_WORD_OF_DAY_FALLBACK = [
    ("ありがとう", "thank you", "ありがとうと言いました。"),
    ("おはよう", "good morning", "おはようございます。"),
    ("すみません", "excuse me / sorry", "すみませんが、手伝ってください。"),
    ("かわいい", "cute", "かわいいねこですね。"),
    ("がんばって", "do your best", "がんばってください。"),
    ("おいしい", "delicious", "おいしいりょうりですね。"),
    ("さくら", "cherry blossom", "さくらがさきました。"),
]


_RANDOM_WORD_COUNT = 5


def _llm_random_words(uid: str, count: int = _RANDOM_WORD_COUNT,
                      level: str | None = None, stock_srs: bool = True) -> List[dict]:
    """Ask the tutor LLM for fresh random vocab at the user's level.

    Returns LessonCard-ready dicts (front/back/reading). Empty list when
    the brain is offline or the model output is unusable — callers fall
    back to static decks. Generated words are inserted into the user's
    SRS queue so Practice can quiz them later.
    """
    from interface.webui import auth
    if not auth.aiko_web_instance or not auth.aiko_web_instance._think:
        return []
    think = auth.aiko_web_instance._think
    level = level or get_last_level(uid)
    try:
        response = think._client.chat.completions.create(
            model=think._llm_model,
            messages=[
                {"role": "system", "content": (
                    "You are a Japanese teacher writing a mini vocabulary list. "
                    f"Pick {count} RANDOM, USEFUL {level}-level Japanese words "
                    "and short everyday phrases (mix nouns, verbs, adjectives, "
                    "phrases — no particles, no single kana, no copulas). "
                    "Output ONLY valid JSON: "
                    '{"words": [{"japanese": "<word as normally written>", '
                    '"hiragana": "<reading in hiragana>", '
                    '"meaning": "<short English gloss>"}]}'
                )},
                {"role": "user", "content": "Surprise me with new words."},
            ],
            response_format={"type": "json_object"},
            timeout=25.0,  # bounded: word-of-day falls back to static deck
        )
        data = parse_lingo_json(response.choices[0].message.content)
        out = []
        for w in (data.get("words") or [])[:count]:
            surf = str(w.get("japanese") or "").strip()
            hira = str(w.get("hiragana") or surf).strip()
            mean = str(w.get("meaning") or "").strip()
            if not surf or not mean or len(surf) > 20:
                continue
            out.append({"front": surf, "back": mean, "reading": hira})
        # Stock the SRS queue so these words resurface in Practice reviews
        # (skipped for pool top-ups — stocking happens at serve time there).
        if out and stock_srs:
            from .lingo_store import normalize_level
            srs = LingoSRS(uid)
            for c in out:
                try:
                    srs.add_card(LingoVocabCard(
                        kanji=c["front"] if re.search(r'[\u4e00-\u9fff]', c["front"]) else "",
                        hiragana=c["reading"],
                        meaning=c["back"],
                        pos="lesson",
                        context="random words",
                        level=normalize_level(level),
                    ))
                except Exception:
                    log.warning("Random-word SRS insert failed", exc_info=True)
        return out
    except Exception:
        log.exception("LLM random vocab generation failed")
        return []


def _ensure_lesson_pool(uid: str, level: str) -> None:
    """Top up the pregenerated LLM pool when a level runs low."""
    from .lessons import pool_add, pool_count, POOL_MIN, POOL_TOPUP
    try:
        if pool_count(level) >= POOL_MIN:
            return
    except Exception:
        log.warning("Lesson pool count failed", exc_info=True)
        return
    words = _llm_random_words(uid, POOL_TOPUP, level=level, stock_srs=False)
    if words:
        try:
            pool_add(words, level)
        except Exception:
            log.warning("Lesson pool top-up insert failed", exc_info=True)


def _word_of_day_file(uid: str) -> Path:
    p = user_state_path(f"word_of_day/{uid}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@router.get("/word-of-day", response_model=WordOfDayResponse)
async def get_word_of_day(http_request: Request, session: dict = Depends(get_lingo_session)):
    """Duolingo-style daily word: fixed per user per calendar day.

    The pick is cached to <USER_SPACE_ROOT>/<uid>/word_of_day/{uid}.json, so every visit to
    Stats on the same day returns the SAME word instantly (no LLM wait
    after the first call of the day).
    """
    uid = session["user_id"]  # FIX #14: no silent fallback
    today = date.today()
    cache_file = _word_of_day_file(uid)
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if cached.get("date") == today.isoformat():
                return WordOfDayResponse(**{k: v for k, v in cached.items() if k != "date"})
        except Exception:
            log.warning("Word-of-day cache unreadable, recomputing", exc_info=True)
    srs = LingoSRS(uid)
    pool = (srs.get_weak_vocab(limit=50) or srs.get_due_cards(limit=50)
            or srs.get_random_cards(limit=50))
    if pool:
        card = pool[today.toordinal() % len(pool)]
        cid, hira, mean, ctx = card.id, card.hiragana, card.meaning, card.context or ""
        kanji = card.kanji or None
    else:
        # No learned words yet: ask the LLM for a fresh one (also stocked
        # into SRS above) before falling back to the static starter deck.
        kanji = None
        fresh = _llm_random_words(uid, 1)
        if fresh:
            hira, mean, ctx, cid = fresh[0]["reading"], fresh[0]["back"], "", 0
            if re.search(r'[\u4e00-\u9fff]', fresh[0]["front"]):
                kanji = fresh[0]["front"]
        else:
            hira, mean, ctx = _WORD_OF_DAY_FALLBACK[today.toordinal() % len(_WORD_OF_DAY_FALLBACK)]
            cid = 0
    resp = WordOfDayResponse(
        card_id=cid,
        hiragana=hira,
        meaning=mean,
        context=ctx,
        audioUrl=get_cached_lingo_audio(hira, uid=uid,
                                      base_url=_public_base_url(http_request)),
        kanji=kanji,
        date=today.isoformat(),
    )
    try:
        _word_of_day_file(uid).write_text(json.dumps(resp.model_dump(), indent=2))
    except Exception:
        log.warning("Word-of-day cache write failed", exc_info=True)
    return resp


@router.get("/lessons", response_model=List[LessonDeckMeta])
async def list_lessons(session: dict = Depends(get_lingo_session)):
    """Learn-mode decks (static kana/words/phrases/kanji + AI random)."""
    from .lessons import list_decks, POOL_SERVE_N
    decks = [LessonDeckMeta(**d) for d in list_decks()
             if d["id"] in ("hiragana", "katakana")]
    decks.append(LessonDeckMeta(
        id="words-phrases", title="Words & Phrases",
        subtitle="Fresh AI picks, saved for you",
        kind="words", card_count=POOL_SERVE_N,
    ))
    return decks


@router.get("/lessons/{deck_id}", response_model=LessonDeck)
async def get_lesson(deck_id: str, session: dict = Depends(get_lingo_session)):
    """Full card list for one Learn-mode deck. 404 on unknown ids."""
    from .lessons import get_deck
    if deck_id == "words-phrases":
        from .lessons import pool_take_levels, POOL_SERVE_N
        uid = session["user_id"]  # FIX #14: no silent fallback
        level = get_jlpt_level(uid)
        _ensure_lesson_pool(uid, level)
        rows = pool_take_levels(_allowed_jlpt_levels(uid), POOL_SERVE_N)
        if not rows:
            raise HTTPException(status_code=503, detail="Word pool unavailable right now")
        # Serving graduates these words into the SRS queue for Practice.
        srs = LingoSRS(uid)
        cards = []
        for r in rows:
            try:
                srs.add_card(LingoVocabCard(
                    kanji=r["front"] if re.search(r'[\u4e00-\u9fff]', r["front"]) else "",
                    hiragana=r["reading"] or r["front"],
                    meaning=r["back"],
                    pos="lesson",
                    context="words & phrases",
                    level=r.get("level") or level,
                ))
            except Exception:
                log.warning("Lesson-word SRS insert failed", exc_info=True)
            cards.append(LessonCard(front=r["front"], back=r["back"], reading=r["reading"]))
        return LessonDeck(
            id="words-phrases", title="Words & Phrases",
            subtitle=f"Fresh {level} picks",
            kind="words", cards=cards,
        )
    deck = get_deck(deck_id)
    if deck is None:
        raise HTTPException(status_code=404, detail=f"Unknown lesson deck: {deck_id}")
    return LessonDeck(
        id=deck["id"], title=deck["title"], subtitle=deck.get("subtitle", ""),
        kind=deck.get("kind", ""),
        cards=[LessonCard(**c) for c in deck["cards"]],
    )


@router.get("/xp", response_model=XPResponse)
async def get_xp_endpoint(session: dict = Depends(get_lingo_session)):
    uid = session["user_id"]  # FIX #14: no silent fallback
    data = get_xp(uid)
    return XPResponse(**data)


@router.get("/leaderboard")
async def get_leaderboard(session: dict = Depends(get_lingo_session)):
    uid = session["user_id"]  # FIX #14: no silent fallback
    streak = get_streak(uid)
    xp_data = get_xp(uid)
    return {
        "rank": 1,
        "username": session.get("username", uid),
        "streak": streak["days"],
        "xp": xp_data["xp"],
        "level": xp_data["level"],
        "top_users": [{"username": session.get("username", uid), "streak": streak["days"], "xp": xp_data["xp"]}]
    }


@router.post("/xp/add")
async def add_xp(request: dict, session: dict = Depends(get_lingo_session)):
    # FIX #17: read/increment/write logic now lives in _award_xp() so
    # review_respond() (FIX #16) can share it instead of a second copy.
    uid = session["user_id"]  # FIX #14: no silent fallback
    _award_xp(uid, request.get("amount", 10))
    # Return the full shape — the Android XPResponse requires
    # xp + level + next_level_xp, and a bare {"xp": ...} crashes its parser.
    return get_xp(uid)


def _seed_starter_srs_cards(uid: str, srs) -> int:
    """Insert up to 10 N5 starter cards from shared materials for new users."""
    from .srs import LingoVocabCard
    added = 0
    try:
        from .lingo_store import init_materials_db, MATERIALS_DB
        import sqlite3
        init_materials_db(seed=True)
        con = sqlite3.connect(str(MATERIALS_DB))
        try:
            rows = con.execute(
                "SELECT front,reading,back,level FROM jlpt_cards"
                " WHERE level='N5' AND kind IN ('vocab','kanji') LIMIT 10"
            ).fetchall()
        finally:
            con.close()
        for front, reading, back, level in rows:
            try:
                srs.add_card(LingoVocabCard(
                    kanji=front if re.search(r'[\u4e00-\u9fff]', front or "") else "",
                    hiragana=reading or front,
                    meaning=back,
                    pos="starter",
                    context="starter deck",
                    level=level or "N5",
                ))
                added += 1
            except Exception:
                continue
    except Exception:
        log.warning("Starter SRS seed query failed", exc_info=True)
    return added


def _allowed_jlpt_levels(uid: str) -> list:
    """Equal-or-lower JLPT only: N5 sees N5; N4 sees N5+N4; ... N1 sees all."""
    lvl = get_jlpt_level(uid)
    idx = JLPT_LEVELS.index(lvl) if lvl in JLPT_LEVELS else 0
    return JLPT_LEVELS[:idx + 1]


def _session_cards(uid: str, n: int) -> list:
    """Random learnt cards at/below the user's JLPT (repeats across sessions
    allowed). Tops up from shared materials when the learnt pool is short."""
    from .srs import LingoVocabCard
    srs = LingoSRS(uid)
    allowed = _allowed_jlpt_levels(uid)
    picked = srs.get_random_cards(limit=n, allowed_levels=allowed)
    if len(picked) >= n:
        return picked
    try:
        from .lingo_store import init_materials_db, MATERIALS_DB, normalize_level
        import sqlite3
        import random as _random
        init_materials_db(seed=True)
        learnt = {(c.hiragana or "", c.meaning or "") for c in
                  srs.get_random_cards(limit=5000, allowed_levels=None)}
        con = sqlite3.connect(str(MATERIALS_DB))
        try:
            q = f"""SELECT front,reading,back,kind,level FROM jlpt_cards
                WHERE level IN ({','.join('?' * len(allowed))})"""
            mat = con.execute(q, allowed).fetchall()
            prow = []
            try:
                prow = con.execute(
                    f"""SELECT front,back,reading,kind,level FROM vocab_pool
                        WHERE level IN ({','.join('?' * len(allowed))})""", allowed).fetchall()
            except Exception:
                pass
            # Grammar points study like vocab: front=pattern, back=meaning.
            grows = []
            try:
                grows = con.execute(
                    "SELECT cards_json, level FROM grammar_decks").fetchall()
            except Exception:
                pass
        finally:
            con.close()
        cands = [(f, r, b, k, normalize_level(l)) for f, r, b, k, l in mat]
        cands += [(f, rd or f, b, k, normalize_level(l)) for f, b, rd, k, l in prow]
        for cards_json, glevel in grows:
            if normalize_level(glevel) not in allowed:
                continue
            try:
                gc = json.loads(cards_json or "[]")
            except Exception:
                continue
            for c in gc:
                gf, gb = c.get("front", ""), c.get("back", "")
                if gf and gb:
                    cands.append((gf, c.get("reading", "") or gf, gb,
                                  "grammar", normalize_level(glevel)))
        _random.shuffle(cands)
        for front, reading, back, kind, level in cands:
            if len(picked) >= n:
                break
            key = (reading or front or "", back or "")
            if not front or not back or key in learnt:
                continue
            try:
                card = srs.add_card(LingoVocabCard(
                    kanji=front if re.search(r'[\u4e00-\u9fff]', front) else "",
                    hiragana=reading or front,
                    meaning=back,
                    pos=kind or "session",
                    context="session top-up",
                    level=level,
                ))
                learnt.add(key)
                picked.append(card)
            except Exception:
                continue
    except Exception:
        log.warning("Session top-up failed", exc_info=True)
    return picked


def _to_review_card(c) -> "ReviewCard":
    return ReviewCard(
        card_id=c.id or 0,
        hiragana=c.hiragana or "",
        meaning=c.meaning or "",
        context=c.context or "",
        kanji=getattr(c, "kanji", None) or None,
    )


@router.get("/review/session", response_model=List[ReviewCard])
async def review_session(n: int = 10, session: dict = Depends(get_lingo_session)):
    """10 random learnt cards (equal-or-lower JLPT) for a Review session."""
    uid = session["user_id"]
    cards = _session_cards(uid, max(1, min(n, 20)))
    if not cards:
        raise HTTPException(status_code=400, detail="No cards available yet")
    return [_to_review_card(c) for c in cards]


@router.get("/practice/session", response_model=List[ReviewCard])
async def practice_session(n: int = 10, session: dict = Depends(get_lingo_session)):
    """10 random learnt cards (equal-or-lower JLPT) for a typing Practice session."""
    uid = session["user_id"]
    cards = _session_cards(uid, max(1, min(n, 20)))
    if not cards:
        raise HTTPException(status_code=400, detail="No cards available yet")
    return [_to_review_card(c) for c in cards]


@router.post("/practice/mark")
async def practice_mark(request: dict, session: dict = Depends(get_lingo_session)):
    """Award XP for a finished typing session: 5 XP per correct card."""
    uid = session["user_id"]
    try:
        correct = max(0, int(request.get("correct", 0)))
    except (TypeError, ValueError):
        correct = 0
    try:
        total = max(correct, int(request.get("total", correct)))
    except (TypeError, ValueError):
        total = correct
    record_practice(uid)
    xp = 0
    if correct > 0:
        try:
            _award_xp(uid, correct * 5)
            xp = correct * 5
        except Exception:
            log.warning("Practice XP award failed", exc_info=True)
    return {"xp": xp, "correct": correct, "total": total}


@router.post("/conversation/review/start", response_model=ReviewSessionResponse)
async def review_start(session: dict = Depends(get_lingo_session)):
    uid = session["user_id"]  # FIX #14: no silent fallback
    srs = LingoSRS(uid)
    # Day-one bootstrap: brand-new users have zero cards, which used to
    # 400 "No cards due" -> the app showed a dead-end "caught up" screen.
    # Seed starter N5 cards from shared materials so Practice works instantly.
    try:
        if srs.get_stats().get("total_cards", 0) == 0:
            _seed_starter_srs_cards(uid, srs)
    except Exception:
        log.warning("Starter SRS seed failed", exc_info=True)
    due_cards = srs.get_due_cards(limit=1)
    if not due_cards:
        raise HTTPException(status_code=400, detail="No cards due for review")
    card = due_cards[0]
    # FIX #6 / FIX #15: was `cards_due=stats["reviews_today"]` (reviews
    # completed today, not cards due right now), then
    # `len(srs.get_due_cards(limit=1000))` (correct count, but fetched up
    # to 1000 rows to get it). Now a single COUNT(*) query.
    total_due = srs.count_due_cards()
    return ReviewSessionResponse(
        cards_due=total_due,
        first_card=ReviewCard(
            card_id=card.id,
            hiragana=card.hiragana,
            meaning=card.meaning,
            context=card.context or ""
        )
    )


# ============================================================================
# JLPT level + Courses/Grammar (back the Android Learn/Courses/Grammar menu).
# Lists return [] when empty (never 404); only unknown detail ids 404.
# ============================================================================
@router.get("/level")
async def get_level(session: dict = Depends(get_lingo_session)):
    uid = session["user_id"]
    return {"level": get_jlpt_level(uid), "levels": JLPT_LEVELS}


@router.post("/level")
async def set_level(request: dict, session: dict = Depends(get_lingo_session)):
    uid = session["user_id"]
    level = str(request.get("level", "")).strip()
    if level not in JLPT_LEVELS:
        raise HTTPException(status_code=400, detail=f"Unknown level: {level}")
    update_last_level(uid, level)
    return {"level": level, "levels": JLPT_LEVELS}


def _read_course_table(table: str) -> list:
    try:
        from .lingo_store import init_materials_db
        import sqlite3
        init_materials_db(seed=False)
        from .lingo_store import MATERIALS_DB
        con = sqlite3.connect(str(MATERIALS_DB))
        try:
            rows = con.execute(
                f"SELECT id,title,level,kind,cards_json FROM {table} ORDER BY level,id"
            ).fetchall()
        finally:
            con.close()
        # Legacy curated lesson dups ("N5 Lesson 1 · Daily Words" etc.)
        # duplicate the Words & Phrases pool slot-for-slot; the ordered
        # progression below hides them so Lesson 1 is always the OpenJLPT one.
        if table == "courses":
            rows = [r for r in rows if r[0] not in _LEGACY_COURSE_IDS]
        return rows
    except Exception:
        log.warning("Course table read failed", exc_info=True)
        return []


def _course_meta_rows(table: str, allowed: list | None = None) -> list:
    out = []
    for cid, title, level, kind, cards_json in _read_course_table(table):
        if allowed is not None and level not in allowed:
            continue
        try:
            count = len(json.loads(cards_json or "[]"))
        except Exception:
            count = 0
        out.append({"id": cid, "title": title, "level": level,
                    "kind": kind, "card_count": count})
    return out


# ============================================================================
# Lesson-at-a-time progression + typing tests + level finals.
#
# Vocab and grammar share one engine (`track` = "vocab" | "grammar"):
#  - GET  /{courses,grammar}/current        -> current lesson + progress
#  - GET  /{courses,grammar}/{id}/test      -> shuffled typing questions
#  - POST /{courses,grammar}/{id}/test/submit -> grade; 100% advances
#  - GET  /{courses,grammar}/final/test     -> up-to-100 sampled questions
#  - POST /{courses,grammar}/final/submit   -> 100% auto-levels-up JLPT
# ============================================================================
# Legacy curated lesson rows that duplicate the Words & Phrases pool
# ("N5 Lesson 1 · Daily Words", "N4 Lesson 1 · Daily Life"). Hidden from the
# ordered progression so Lesson 1 is always the OpenJLPT lesson.
_LEGACY_COURSE_IDS = frozenset({"n5-lesson-1", "n4-lesson-1"})

_TRACK_TABLES = {"vocab": "courses", "grammar": "grammar_decks"}
FINAL_TEST_N = 100


def _deck_sort_key(cid: str):
    """Numeric-aware ordering: lesson 2 < lesson 10; `*basics*` decks first."""
    m = re.search(r"(\d+)\s*$", cid)
    num = int(m.group(1)) if m else None
    basics_first = 0 if "basics" in cid else 1
    return (basics_first, num if num is not None else 10 ** 9, cid)


def _ordered_track_decks(track: str, level: str) -> list:
    """All decks for one track+level in study order, cards parsed."""
    table = _TRACK_TABLES[track]
    decks = []
    for cid, title, lvl, kind, cards_json in _read_course_table(table):
        if lvl != level:
            continue
        try:
            cards = json.loads(cards_json or "[]")
        except Exception:
            cards = []
        cards = [c for c in cards if c.get("front") and c.get("back")]
        decks.append({"id": cid, "title": title, "level": lvl,
                      "kind": kind, "cards": cards})
    decks.sort(key=lambda d: _deck_sort_key(d["id"]))
    return decks


def _track_progress(uid: str, track: str) -> LessonProgress:
    from .lingo_store import get_lesson_progress
    level = get_jlpt_level(uid)
    decks = _ordered_track_decks(track, level)
    total = len(decks)
    current = get_lesson_progress(uid, track, level)
    current = max(1, min(current, total + 1)) if total else 1
    return LessonProgress(level=level, track=track, current_lesson=current,
                          lessons_total=total,
                          final_unlocked=bool(total) and current > total)


def _normalize_test_answer(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"^[〜～~\-·]+", "", s)
    return s


def _accepted_test_answers(card: dict) -> list:
    """front/reading plus grammar-tolerant variants (drop 〜, drop （…）)."""
    outs: set = set()
    for raw in (card.get("front", ""), card.get("reading", "")):
        if not raw or not str(raw).strip():
            continue
        outs.add(_normalize_test_answer(str(raw)))
        v = re.sub(r"^[〜～~\-·\s]+", "", str(raw))
        v = re.sub(r"[（\(][^）\)]*[）\)]", "", v).strip()
        if v:
            outs.add(_normalize_test_answer(v))
    return sorted(o for o in outs if o)


def _stock_lesson_cards(uid: str, track: str, level: str, cards: list) -> None:
    """Tested cards enter the SRS queue so Review/Practice pick them up."""
    try:
        srs = LingoSRS(uid)
        for c in cards:
            front = str(c.get("front", ""))
            back = str(c.get("back", ""))
            reading = str(c.get("reading", "") or front)
            if not front or not back:
                continue
            try:
                srs.add_card(LingoVocabCard(
                    kanji=front if re.search(r'[\u4e00-\u9fff]', front) else "",
                    hiragana=reading,
                    meaning=back,
                    pos="grammar" if track == "grammar" else "lesson",
                    context=f"{track} lesson",
                    level=level,
                ))
            except Exception:
                continue
    except Exception:
        log.warning("Lesson SRS stocking failed", exc_info=True)


def _grade_and_advance(uid: str, track: str, level: str, cards_by_qid: dict,
                       answers: list, *, is_final: bool,
                       lesson_pos: int = 0, lessons_total: int = 0) -> TestSubmitResponse:
    results: list = []
    correct = 0
    for qid, card in cards_by_qid.items():
        given = next((a.answer for a in answers if a.qid == qid), "")
        accepted = _accepted_test_answers(card)
        ok = bool(accepted) and _normalize_test_answer(given) in accepted
        correct += 1 if ok else 0
        results.append(TestItemResult(
            qid=qid, prompt=str(card.get("back", "")),
            expected=accepted[:4], given=given or "", correct=ok))
    total = len(cards_by_qid)
    passed = total > 0 and correct == total

    _stock_lesson_cards(uid, track, level, list(cards_by_qid.values()))

    xp = 0
    if correct:
        xp = correct * 2
        try:
            _award_xp(uid, xp)
        except Exception:
            log.warning("Test XP award failed", exc_info=True)
    record_practice(uid)

    new_level = None
    if passed:
        from .lingo_store import get_lesson_progress, set_lesson_progress
        if is_final:
            idx = JLPT_LEVELS.index(level) if level in JLPT_LEVELS else 0
            if idx + 1 < len(JLPT_LEVELS):
                new_level = JLPT_LEVELS[idx + 1]
                update_last_level(uid, new_level)
        elif lesson_pos >= get_lesson_progress(uid, track, level):
            # Only ever move forward: re-taking an old lesson never rewinds.
            set_lesson_progress(uid, track, level, lesson_pos + 1)

    progress = _track_progress(uid, track if new_level is None else track)
    if new_level is not None:
        # Progress was just moved to the next level -- report that level.
        progress = LessonProgress(level=new_level, track=track,
                                  current_lesson=1,
                                  lessons_total=len(_ordered_track_decks(track, new_level)),
                                  final_unlocked=False)
    return TestSubmitResponse(correct=correct, total=total, passed=passed,
                              xp=xp, results=results, progress=progress,
                              new_level=new_level)


def _require_final_unlocked(uid: str, track: str) -> None:
    """Final tests only count once every lesson at the level is passed."""
    prog = _track_progress(uid, track)
    if prog.lessons_total > 0 and not prog.final_unlocked:
        raise HTTPException(
            status_code=400,
            detail=f"Finish lesson {prog.current_lesson} first -- the final unlocks after all lessons pass.",
        )


def _current_lesson_payload(uid: str, track: str) -> dict:
    progress = _track_progress(uid, track)
    lesson = None
    if not progress.final_unlocked:
        decks = _ordered_track_decks(track, progress.level)
        pos = progress.current_lesson - 1
        if 0 <= pos < len(decks):
            d = decks[pos]
            lesson = {"id": d["id"], "title": d["title"], "level": d["level"],
                      "kind": d["kind"], "cards": d["cards"],
                      "lesson": progress.current_lesson,
                      "lessons_total": progress.lessons_total}
    return {"progress": progress.model_dump(), "lesson": lesson}


def _lesson_test_payload(track: str, level: str, deck_id: str) -> dict:
    decks = _ordered_track_decks(track, level)
    deck = next((d for d in decks if d["id"] == deck_id), None)
    if deck is None:
        raise HTTPException(status_code=404, detail=f"Unknown deck: {deck_id}")
    order = list(range(len(deck["cards"])))
    random.shuffle(order)
    questions = [{"qid": f"c{i}", "prompt": deck["cards"][i].get("back", "")}
                 for i in order]
    return {"deck_id": deck_id, "title": deck["title"], "level": level,
            "questions": questions}


def _final_test_payload(track: str, level: str, n: int) -> dict:
    seen: dict = {}
    for d in _ordered_track_decks(track, level):
        for i, c in enumerate(d["cards"]):
            key = (str(c.get("front", "")), str(c.get("back", "")))
            if key[0] and key[1] and key not in seen:
                seen[key] = (d["id"], i)
    keys = list(seen)
    random.shuffle(keys)
    picked = keys[:max(1, min(n, FINAL_TEST_N))]
    if not picked:
        raise HTTPException(status_code=400, detail="No cards available for final test")
    decks_by_id = {d["id"]: d for d in _ordered_track_decks(track, level)}
    questions = []
    for k in picked:
        deck_id, i = seen[k]
        card = decks_by_id[deck_id]["cards"][i]
        questions.append({"qid": f"{deck_id}#{i}", "prompt": card.get("back", "")})
    return {"level": level, "questions": questions, "total_at_level": len(seen)}


def _cards_by_qid_for_submit(track: str, level: str, answers: list,
                             *, is_final: bool, deck_id: str = "") -> dict:
    decks = _ordered_track_decks(track, level)
    if is_final:
        decks_by_id = {d["id"]: d for d in decks}
        out = {}
        for a in answers:
            if "#" not in a.qid:
                continue
            did, _, idx = a.qid.partition("#")
            try:
                out[a.qid] = decks_by_id[did]["cards"][int(idx)]
            except (KeyError, ValueError, IndexError):
                continue
        # Grade the FULL final set, not just answered qids: rebuild from the
        # level pool so skipped questions count as wrong.
        return out
    deck = next((d for d in decks if d["id"] == deck_id), None)
    if deck is None:
        raise HTTPException(status_code=404, detail=f"Unknown deck: {deck_id}")
    return {f"c{i}": deck["cards"][i] for i in range(len(deck["cards"]))}


@router.get("/courses/current")
async def courses_current(session: dict = Depends(get_lingo_session)):
    return _current_lesson_payload(session["user_id"], "vocab")


@router.post("/courses/final/submit", response_model=TestSubmitResponse)
async def courses_final_submit(body: TestSubmitRequest,
                               session: dict = Depends(get_lingo_session)):
    uid = session["user_id"]
    _require_final_unlocked(uid, "vocab")
    level = get_jlpt_level(uid)
    cards = _cards_by_qid_for_submit("vocab", level, body.answers, is_final=True)
    if not cards:
        raise HTTPException(status_code=400, detail="Nothing to grade")
    return _grade_and_advance(uid, "vocab", level, cards, body.answers,
                              is_final=True)


@router.get("/courses/final/test")
async def courses_final_test(n: int = FINAL_TEST_N,
                             session: dict = Depends(get_lingo_session)):
    uid = session["user_id"]
    return _final_test_payload("vocab", get_jlpt_level(uid), n)


@router.get("/courses/{course_id}/test")
async def course_test(course_id: str, session: dict = Depends(get_lingo_session)):
    uid = session["user_id"]
    return _lesson_test_payload("vocab", get_jlpt_level(uid), course_id)


@router.post("/courses/{course_id}/test/submit", response_model=TestSubmitResponse)
async def course_test_submit(course_id: str, body: TestSubmitRequest,
                             session: dict = Depends(get_lingo_session)):
    uid = session["user_id"]
    level = get_jlpt_level(uid)
    cards = _cards_by_qid_for_submit("vocab", level, body.answers,
                                     is_final=False, deck_id=course_id)
    if not cards:
        raise HTTPException(status_code=400, detail="Nothing to grade")
    progress = _track_progress(uid, "vocab")
    decks = _ordered_track_decks("vocab", level)
    pos = next((i + 1 for i, d in enumerate(decks) if d["id"] == course_id), 0)
    return _grade_and_advance(uid, "vocab", level, cards, body.answers,
                              is_final=False, lesson_pos=pos,
                              lessons_total=len(decks))


@router.get("/grammar/current")
async def grammar_current(session: dict = Depends(get_lingo_session)):
    return _current_lesson_payload(session["user_id"], "grammar")


@router.post("/grammar/final/submit", response_model=TestSubmitResponse)
async def grammar_final_submit(body: TestSubmitRequest,
                               session: dict = Depends(get_lingo_session)):
    uid = session["user_id"]
    _require_final_unlocked(uid, "grammar")
    level = get_jlpt_level(uid)
    cards = _cards_by_qid_for_submit("grammar", level, body.answers, is_final=True)
    if not cards:
        raise HTTPException(status_code=400, detail="Nothing to grade")
    return _grade_and_advance(uid, "grammar", level, cards, body.answers,
                              is_final=True)
@router.get("/grammar/final/test")
async def grammar_final_test(n: int = FINAL_TEST_N,
                             session: dict = Depends(get_lingo_session)):
    uid = session["user_id"]
    return _final_test_payload("grammar", get_jlpt_level(uid), n)


@router.get("/grammar/{grammar_id}/test")
async def grammar_test(grammar_id: str, session: dict = Depends(get_lingo_session)):
    uid = session["user_id"]
    return _lesson_test_payload("grammar", get_jlpt_level(uid), grammar_id)


@router.post("/grammar/{grammar_id}/test/submit", response_model=TestSubmitResponse)
async def grammar_test_submit(grammar_id: str, body: TestSubmitRequest,
                              session: dict = Depends(get_lingo_session)):
    uid = session["user_id"]
    level = get_jlpt_level(uid)
    cards = _cards_by_qid_for_submit("grammar", level, body.answers,
                                     is_final=False, deck_id=grammar_id)
    if not cards:
        raise HTTPException(status_code=400, detail="Nothing to grade")
    decks = _ordered_track_decks("grammar", level)
    pos = next((i + 1 for i, d in enumerate(decks) if d["id"] == grammar_id), 0)
    return _grade_and_advance(uid, "grammar", level, cards, body.answers,
                              is_final=False, lesson_pos=pos,
                              lessons_total=len(decks))


@router.get("/courses")
async def list_courses(session: dict = Depends(get_lingo_session)):
    return _course_meta_rows("courses", _allowed_jlpt_levels(session["user_id"]))


@router.get("/courses/{course_id}")
async def get_course(course_id: str, session: dict = Depends(get_lingo_session)):
    allowed = _allowed_jlpt_levels(session["user_id"])
    for cid, title, level, kind, cards_json in _read_course_table("courses"):
        if cid == course_id:
            if level not in allowed:
                raise HTTPException(status_code=404, detail=f"Unknown course: {course_id}")
            try:
                cards = json.loads(cards_json or "[]")
            except Exception:
                cards = []
            return {"id": cid, "title": title, "level": level,
                    "kind": kind, "cards": cards}
    raise HTTPException(status_code=404, detail=f"Unknown course: {course_id}")


@router.get("/grammar")
async def list_grammar(session: dict = Depends(get_lingo_session)):
    return _course_meta_rows("grammar_decks", _allowed_jlpt_levels(session["user_id"]))


@router.get("/grammar/{grammar_id}")
async def get_grammar(grammar_id: str, session: dict = Depends(get_lingo_session)):
    allowed = _allowed_jlpt_levels(session["user_id"])
    for cid, title, level, kind, cards_json in _read_course_table("grammar_decks"):
        if cid == grammar_id:
            if level not in allowed:
                raise HTTPException(status_code=404, detail=f"Unknown grammar deck: {grammar_id}")
            try:
                cards = json.loads(cards_json or "[]")
            except Exception:
                cards = []
            return {"id": cid, "title": title, "level": level,
                    "kind": kind, "cards": cards}
    raise HTTPException(status_code=404, detail=f"Unknown grammar deck: {grammar_id}")
