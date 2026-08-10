from typing import Optional, List, Dict
import asyncio
import os
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from social.services import env, err
from system.userspace import user_state_path

# Suppress CryptographyDeprecationWarning for TripleDES (used by protonmail-api-client)
# Note: CryptographyDeprecationWarning subclasses UserWarning, not DeprecationWarning.
warnings.filterwarnings(
    "ignore",
    category=Warning,
    message="TripleDES has been moved to cryptography.hazmat.decrepit.ciphers.algorithms.TripleDES",
)

# Global client cache (kept alive across tool calls)
_client_cache = None
_cache_username = None

# protonmail-api-client stores sessions as a binary pickle, not JSON.
_SESSION_FILE = str(user_state_path("profile/protonmail_session.pickle"))

# stderr-bound print for ProtonMail's internal logger.
# ProtonMail.__init__ defaults logging_func=print (stdout). Since the MCP
# server runs on stdio transport any write to stdout corrupts the JSON-RPC
# wire, causing the client to hang waiting for a valid response.
def _stderr_print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def _run_client_call(method, *args, **kwargs):
    """Keep third-party client output (including tqdm progress bars) off MCP stdout."""
    with redirect_stdout(sys.stderr):
        return method(*args, **kwargs)


def _get_client():
    """Return an authenticated client using the documented session flow."""
    global _client_cache, _cache_username
    try:
        from protonmail import ProtonMail
    except ImportError:
        return None, {"ok": False, "error": "protonmail-api-client not installed", "provider": "protonmail"}
    username = env("PROTONMAIL_USERNAME")
    password = env("PROTONMAIL_PASSWORD")
    if not username:
        return None, {"ok": False, "error": "PROTONMAIL_USERNAME not set", "provider": "protonmail"}
    if _client_cache is not None and _cache_username == username:
        print("[PROTONMAIL] Using cached client", file=sys.stderr, flush=True)
        return _client_cache, None
    if not os.path.exists(_SESSION_FILE) and not password:
        return None, {"ok": False, "error": "PROTONMAIL_PASSWORD not set for first login", "provider": "protonmail"}
    print(f"[PROTONMAIL] Authenticating as {username[:3]}{chr(42) * max(0, len(username) - 3)}...", file=sys.stderr, flush=True)
    print(f"[PROTONMAIL] Session file exists: {os.path.exists(_SESSION_FILE)} ({_SESSION_FILE})", file=sys.stderr, flush=True)
    try:
        client = ProtonMail(logging_func=_stderr_print)
        if os.path.exists(_SESSION_FILE):
            print(f"[PROTONMAIL] Loading session: {_SESSION_FILE}", file=sys.stderr, flush=True)
            _run_client_call(client.load_session, _SESSION_FILE, auto_save=True)
            print("[PROTONMAIL] Session loaded successfully", file=sys.stderr, flush=True)
        else:
            print("[PROTONMAIL] No saved session; performing login...", file=sys.stderr, flush=True)
            _run_client_call(client.login, username, password)
            Path(_SESSION_FILE).parent.mkdir(parents=True, exist_ok=True)
            _run_client_call(client.save_session, _SESSION_FILE)
            print(f"[PROTONMAIL] Session saved: {_SESSION_FILE}", file=sys.stderr, flush=True)
        _client_cache = client
        _cache_username = username
        return client, None
    except Exception as e:
        print(f"[PROTONMAIL] Authentication failed: {e}", file=sys.stderr, flush=True)
        return None, {"ok": False, "error": f"authentication failed: {e}", "provider": "protonmail"}


def get_client():
    """Get ProtonMail client. Returns (client, error_dict) or (client, None)."""
    return _get_client()


async def read_messages(client, folder: str, unread: bool, max_results: int, query: str, list_only: bool, message_id: str = "") -> Dict:
    """Read messages using ProtonMail client."""
    try:
        # Translate generic folder names to ProtonMail label IDs
        folder_map = {
            "inbox": "0",
            "spam": "4",
            "trash": "3",
        }
        folder_lower = folder.lower()
        protonmail_label = folder_map.get(folder_lower, folder_lower)
        
        # Get all messages
        all_messages = await asyncio.to_thread(_run_client_call, client.get_messages)
        
        # Filter by folder (0=inbox, 3=trash, 4=spam)
        folder = protonmail_label
        messages = []
        folder_counts = {}
        
        for msg in all_messages:
            msg_folder = ""
            if hasattr(msg, "labels") and msg.labels:
                label_list = msg.labels if isinstance(msg.labels, list) else [msg.labels]
                first_label = str(label_list[0]) if label_list else ""
                if first_label == "0":
                    msg_folder = "inbox"
                elif first_label == "3":
                    msg_folder = "trash"
                elif first_label == "4":
                    msg_folder = "spam"
                else:
                    msg_folder = first_label
            folder_counts[msg_folder] = folder_counts.get(msg_folder, 0) + 1
            
            # Only include inbox (label 0) and unread messages if requested
            is_inbox = (first_label == "0")
            is_unread = getattr(msg, "unread", False)
            if folder == "inbox" and is_inbox and (not unread or is_unread):
                messages.append(msg)
            elif folder != "inbox" and msg_folder == folder:
                messages.append(msg)
        
        print(f"[EMAIL] Provider=protonmail Folder counts: {folder_counts}, filtered={len(messages)}", file=sys.stderr, flush=True)
        
        # If message_id provided, return full message
        if message_id:
            target_msg = None
            for msg in messages:
                if getattr(msg, "id", "") == message_id:
                    target_msg = msg
                    break
            if target_msg is None:
                return {"ok": False, "error": f"message not found: {message_id}", "provider": "protonmail"}
            
            full = await asyncio.to_thread(_run_client_call, client.read_message, target_msg)
            return {
                "ok": True,
                "provider": "protonmail",
                "id": message_id,
                "from": full.sender.address if full.sender else "",
                "subject": full.subject or "",
                "date": str(full.time) if full.time else "",
                "body": full.body or "",
            }
        
        # Filter by query
        if query:
            q = query.lower()
            messages = [m for m in messages if q in (m.subject or "").lower() or q in (m.sender.address if m.sender else "").lower()]
        
        results = []
        for i, msg in enumerate(messages[:max_results]):
            if list_only:
                results.append({
                    "id": getattr(msg, "id", ""),
                    "from": msg.sender.address if getattr(msg, "sender", None) else "",
                    "subject": getattr(msg, "subject", "") or "",
                    "date": str(getattr(msg, "time", "") or ""),
                    "snippet": "",
                })
                continue
            try:
                print(f"[EMAIL] Reading message {i+1}/{max_results} ({getattr(msg, 'id', 'unknown')[:10]}...)...", file=sys.stderr, flush=True)
                full = await asyncio.to_thread(_run_client_call, client.read_message, msg)
                print(f"[EMAIL] Read message {i+1}/{max_results} OK", file=sys.stderr, flush=True)
                results.append({
                    "id": getattr(msg, "id", ""),
                    "from": full.sender.address if full.sender else "",
                    "subject": full.subject or "",
                    "date": str(full.time) if full.time else "",
                    "snippet": (full.body or ""),
                })
            except Exception:
                continue
        
        return {"ok": True, "provider": "protonmail", "count": len(results), "messages": results}
    except Exception as e:
        return {"ok": False, "error": f"read failed: {e}", "provider": "protonmail"}


async def send_message(client, recipients: List[str], subject: str, body: str, cc: List[str], bcc: List[str], attachments: List[Dict]) -> Dict:
    """Send email using ProtonMail client."""
    if not recipients:
        return {"ok": False, "error": "recipients required", "provider": "protonmail"}
    
    try:
        # protonmail-api-client requires two-step: create_message then send_message
        new_message = await asyncio.to_thread(
            _run_client_call,
            client.create_message,
            recipients=recipients,
            subject=subject,
            body=body,
            cc=cc if cc else [],
            bcc=bcc if bcc else [],
        )
        
        # Send the created message
        sent_message = await asyncio.to_thread(_run_client_call, client.send_message, new_message)
        
        return {
            "ok": True,
            "provider": "protonmail",
            "message_id": getattr(sent_message, "id", "unknown"),
            "status": "sent"
        }
    except Exception as e:
        return {"ok": False, "error": f"send failed: {e}", "provider": "protonmail"}


async def delete_message(client, message_id: str) -> Dict:
    """Delete email using ProtonMail client."""
    if not message_id:
        return {"ok": False, "error": "message_id required", "provider": "protonmail"}
    
    try:
        messages = await asyncio.to_thread(_run_client_call, client.get_messages)
        target = next((msg for msg in messages if getattr(msg, "id", "") == message_id), None)
        if target is None:
            return {"ok": False, "error": f"message not found: {message_id}", "provider": "protonmail"}
        await asyncio.to_thread(_run_client_call, client.delete_messages, [target])
        return {"ok": True, "provider": "protonmail", "message_id": message_id, "status": "deleted"}
    except Exception as e:
        return {"ok": False, "error": f"delete failed: {e}", "provider": "protonmail"}