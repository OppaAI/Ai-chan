"""
X/Twitter reply monitor — polls for mentions using twitterapi.io /twitter/user/mentions endpoint.

Architecture mirrors Threads monitor with toggleable polling via X_POLLING_ENABLED.

Requires:
  - TWITTERAPI_KEY: API key from https://twitterapi.io
  - TWITTER_USERNAME: Your X handle (no @ symbol)
  - X_POLLING_ENABLED: Set to 1 to enable monitoring (default: 0)
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from social.services import env, int_env, get_session, err
    from social.state import get_db
    from social.services.identity import (
        ai_name,
        reply_trigger_phrase,
    )
    from social.services.x import post_x  # The post_x function we already created
except ModuleNotFoundError:
    from ..services import env, int_env, get_session, err
    from ..state import get_db
    from .identity import (
        ai_name,
        reply_trigger_phrase,
    )
    from .x import post_x


_MONITOR_RUN_LOCK = threading.Lock()


def _is_x_trigger(text: str) -> bool:
    """Check if text contains trigger phrase or mention."""
    text_lower = (text or "").lower()
    
    # Check for explicit trigger phrase (e.g., "Hi Aiko")
    trigger_phrase = reply_trigger_phrase("x")
    if trigger_phrase and trigger_phrase.lower() in text_lower:
        return True
    
    # Check for @mention
    username = env("TWITTER_USERNAME", "").lstrip("@")
    if username and f"@{username}".lower() in text_lower:
        return True
    
    return False


def _fetch_mentions_from_x(limit: int = 50) -> list[dict]:
    """
    Fetch recent mentions from X/Twitter via twitterapi.io.
    
    Returns list of mention tweets, most recent first.
    Handles pagination internally to fetch up to `limit` mentions.
    """
    api_key = env("TWITTERAPI_KEY", "")
    base_url = env("TWITTERAPI_BASE_URL", "https://api.twitterapi.io").rstrip("/")
    username = env("TWITTER_USERNAME", "").lstrip("@")
    
    if not api_key:
        return []
    
    if not username:
        return []
    
    session = get_session()
    mentions = []
    cursor = ""
    page_count = 0
    max_pages = (limit + 19) // 20  # ~20 mentions per page
    
    try:
        while len(mentions) < limit and page_count < max_pages:
            params = {
                "userName": username,
            }
            if cursor:
                params["cursor"] = cursor
            
            resp = session.get(
                f"{base_url}/twitter/user/mentions",
                params=params,
                headers={"X-API-Key": api_key},
                timeout=30,
            )
            
            if resp.status_code != 200:
                break
            
            data = resp.json() or {}
            if data.get("status") != "success":
                break
            
            tweets = data.get("tweets", [])
            if not tweets:
                break
            
            mentions.extend(tweets)
            
            # Check for next page
            if not data.get("has_next_page"):
                break
            
            cursor = data.get("next_cursor", "")
            if not cursor:
                break
            
            page_count += 1
        
        return mentions[:limit]
    
    except Exception as e:
        return []


def _append_x_reply_log(event: dict) -> None:
    """Log X reply activity (JSONL format, one event per line)."""
    try:
        log_dir = Path(env("X_REPLY_LOG_DIR", "logs/x")).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        day = datetime.now().strftime("%Y-%m-%d")
        path = log_dir / f"{day}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _infer_x_reply(mention: dict) -> Optional[str]:
    """
    Generate a reply to an X mention using the same LLM pipeline as Threads.
    
    For now, returns a simple placeholder. In production, integrate:
    - _infer_reply() from threads.py (reuse entire logic)
    - Memory recall via memorize.search()
    - Web research if mention contains "internet/web/search/verify"
    - Image generation if mentioned
    """
    text = mention.get("text", "")
    author = mention.get("author", {}).get("userName", "user")
    
    # TODO: Replace with full _infer_reply() pipeline from threads.py
    # For now, return a simple acknowledgment
    return f"Thanks for mentioning me, {author}! 🙂"


def monitor_x_replies(memorize=None) -> dict:
    """
    Poll X for mentions and generate replies.
    
    Guarded by run lock (like Threads) to prevent overlapping polls.
    Respects X_POLLING_ENABLED toggle.
    
    Args:
        memorize: Optional memory manager (same interface as Threads)
    
    Returns:
        Status dict: {"ok": bool, "provider": "x", "matched": int, "answered": int, ...}
    """
    if env("X_POLLING_ENABLED", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return {
            "ok": True,
            "provider": "x",
            "polling_disabled": True,
            "matched": 0,
            "answered": 0,
        }
    
    if not _MONITOR_RUN_LOCK.acquire(blocking=False):
        return {
            "ok": True,
            "provider": "x",
            "skipped": "monitor_run_in_progress",
            "matched": 0,
            "answered": 0,
        }
    
    try:
        return _monitor_x_replies_locked(memorize)
    finally:
        _MONITOR_RUN_LOCK.release()


def _monitor_x_replies_locked(memorize=None) -> dict:
    """Core polling loop (with run lock held)."""
    api_key = env("TWITTERAPI_KEY")
    username = env("TWITTER_USERNAME", "").lstrip("@")
    
    if not api_key:
        return err("x", "TWITTERAPI_KEY not set")
    
    if not username:
        return err("x", "TWITTER_USERNAME not set")
    
    db = get_db()
    batch_size = int_env("X_POLL_BATCH_SIZE", 20)
    
    # Fetch recent mentions
    mentions = _fetch_mentions_from_x(limit=batch_size)
    
    matched = answered = 0
    errors = []
    
    for mention in mentions:
        mention_id = str(mention.get("id") or "")
        
        if not mention_id:
            continue
        
        # Skip if already processed
        if db.has_processed_x_reply(mention_id):
            continue
        
        # Check if this mention triggers a response
        text = str(mention.get("text") or "")
        if not _is_x_trigger(text):
            # Still log that we saw it, but don't reply
            if not db.has_logged_x_reply(mention_id):
                _append_x_reply_log({
                    "kind": "mention_seen",
                    "mention_id": mention_id,
                    "author": mention.get("author", {}).get("userName", ""),
                    "timestamp": mention.get("createdAt", ""),
                    "text": text[:500],
                })
                db.mark_logged_x_reply(mention_id)
            continue
        
        matched += 1
        
        # Log that we're about to reply
        if not db.has_logged_x_reply(mention_id):
            _append_x_reply_log({
                "kind": "mention_matched",
                "mention_id": mention_id,
                "author": mention.get("author", {}).get("userName", ""),
                "timestamp": mention.get("createdAt", ""),
                "text": text[:500],
            })
            db.mark_logged_x_reply(mention_id)
        
        # Generate reply
        try:
            reply_text = _infer_x_reply(mention)
            if not reply_text:
                raise RuntimeError("LLM returned empty reply")
        
        except Exception as e:
            errors.append({
                "mention_id": mention_id,
                "stage": "inference",
                "error": str(e),
            })
            continue
        
        # Post reply
        try:
            result = post_x(reply_text)
            
            if not result.get("ok"):
                errors.append({
                    "mention_id": mention_id,
                    "stage": "post",
                    "error": result,
                })
                continue
            
            # Mark as processed and log success
            db.mark_processed_x_reply(mention_id)
            _append_x_reply_log({
                "kind": "aiko_reply",
                "mention_id": mention_id,
                "in_reply_to": mention_id,
                "author": mention.get("author", {}).get("userName", ""),
                "timestamp": mention.get("createdAt", ""),
                "text": reply_text[:500],
            })
            answered += 1
        
        except Exception as e:
            errors.append({
                "mention_id": mention_id,
                "stage": "publish",
                "error": str(e),
            })
    
    return {
        "ok": not errors,
        "provider": "x",
        "mentions_checked": len(mentions),
        "matched": matched,
        "answered": answered,
        "errors": errors,
    }


def load_tools(mcp):
    """Register X monitoring as an MCP tool."""
    @mcp.tool(
        name="monitor_x",
        description="Poll X/Twitter for mentions and reply to triggers",
    )
    def monitor_x_tool() -> dict:
        """Manually trigger X reply monitoring."""
        return monitor_x_replies()
