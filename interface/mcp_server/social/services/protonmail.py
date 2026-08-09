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
        return None, err("protonmail", "protonmail-api-client not installed")
    username = env("PROTONMAIL_USERNAME")
    password = env("PROTONMAIL_PASSWORD")
    if not username:
        return None, err("protonmail", "PROTONMAIL_USERNAME not set")
    if _client_cache is not None and _cache_username == username:
        return _client_cache, None
    if not os.path.exists(_SESSION_FILE) and not password:
        return None, err("protonmail", "PROTONMAIL_PASSWORD not set for first login")
    print(f"[PROTONMAIL] Authenticating as {username[:3]}{chr(42) * max(0, len(username) - 3)}...", file=sys.stderr, flush=True)
    try:
        # logging_func=_stderr_print routes ALL internal ProtonMail log output
        # (including tqdm_asyncio progress bars in get_messages) to stderr so
        # it never touches the MCP stdio wire.
        client = ProtonMail(logging_func=_stderr_print)
        if os.path.exists(_SESSION_FILE):
            print(f"[PROTONMAIL] Loading session: {_SESSION_FILE}", file=sys.stderr, flush=True)
            _run_client_call(client.load_session, _SESSION_FILE, auto_save=True)
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
        return None, err("protonmail", f"authentication failed: {e}")

def load_tools(mcp):
    @mcp.tool(
        name="read_protonmail",
        description="Read ProtonMail messages. If message_id is provided, returns full message body. Otherwise lists messages from inbox (or specified folder) with optional query filter. If list_only is true, returns sender/subject/date only (no body read). Returns sender, subject, date, and snippet (or full body if message_id given).",
    )
    async def read_protonmail(
        message_id: str = "",
        folder: str = "inbox",
        query: str = "",
        max_results: int = 10,
        list_only: bool = False,
    ) -> Dict:
        client, err_resp = await asyncio.to_thread(_get_client)
        if err_resp:
            return err_resp

        try:
            # Get all messages (protonmail-api-client doesn't support folder filtering directly)
            print("[PROTONMAIL] Fetching messages list...", file=sys.stderr, flush=True)
            all_messages = await asyncio.to_thread(_run_client_call, client.get_messages)
            print(f"[PROTONMAIL] Got {len(all_messages)} messages", file=sys.stderr, flush=True)

            # Filter by folder (protonmail-api-client returns all folders; filter client-side)
            # Message objects have .label or .folder attribute indicating the folder
            folder = folder.lower()
            messages = []
            for msg in all_messages:
                msg_folder = (getattr(msg, "label", "") or getattr(msg, "folder", "") or "").lower()
                if folder == "inbox" and msg_folder in ("inbox", ""):
                    messages.append(msg)
                elif folder != "inbox" and msg_folder == folder:
                    messages.append(msg)
                # Default: if folder attr is empty/missing, assume inbox
            print(f"[PROTONMAIL] After folder filter ({folder}): {len(messages)} messages", file=sys.stderr, flush=True)

            # If message_id provided, return full message
            if message_id:
                target_msg = None
                for msg in messages:
                    if getattr(msg, "id", "") == message_id:
                        target_msg = msg
                        break
                if target_msg is None:
                    return err("protonmail", f"message not found: {message_id}")
                
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

            # Otherwise list messages with optional query filter
            if query:
                q = query.lower()
                messages = [m for m in messages if q in (m.subject or "").lower() or q in (m.sender.address if m.sender else "").lower()]

            results = []
            for i, msg in enumerate(messages[:max_results]):
                # list_only: screen messages by subject/sender WITHOUT reading
                # the body — the list metadata already carries both. This stops
                # the periodic check_email job from downloading 20 message
                # bodies every cycle just to see if a subject mentions a job.
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
                    print(f"[PROTONMAIL] Reading message {i+1}/{max_results} ({getattr(msg, 'id', 'unknown')[:10]}...)...", file=sys.stderr, flush=True)
                    full = await asyncio.to_thread(_run_client_call, client.read_message, msg)
                    print(f"[PROTONMAIL] Read message {i+1}/{max_results} OK", file=sys.stderr, flush=True)
                    results.append({
                        "id": getattr(msg, "id", ""),
                        "from": full.sender.address if full.sender else "",
                        "subject": full.subject or "",
                        "date": str(full.time) if full.time else "",
                        "snippet": (full.body or "")[:300],
                    })
                except Exception:
                    continue

            return {"ok": True, "provider": "protonmail", "folder": folder, "count": len(results), "messages": results}
        except Exception as e:
            return err("protonmail", f"read failed: {e}")

    @mcp.tool(
        name="send_protonmail",
        description="Send an email via ProtonMail. Supports HTML body, CC/BCC.",
    )
    async def send_protonmail(
        recipients: List[str],
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        attachments: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Send email via ProtonMail using protonmail-api-client.

        Args:
            recipients: List of email addresses (to)
            subject: Email subject
            body: HTML or plain text body
            cc: Optional CC recipients
            bcc: Optional BCC recipients
            attachments: Not yet implemented
        """
        client, err_resp = await asyncio.to_thread(_get_client)
        if err_resp:
            return err_resp

        if not recipients:
            return err("protonmail", "recipients required")

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
            return err("protonmail", f"send failed: {e}")

    @mcp.tool(
        name="delete_protonmail",
        description="Delete a ProtonMail message by ID.",
    )
    async def delete_protonmail(message_id: str) -> Dict:
        client, err_resp = await asyncio.to_thread(_get_client)
        if err_resp:
            return err_resp
        if not message_id:
            return err("protonmail", "message_id required")
        try:
            messages = await asyncio.to_thread(_run_client_call, client.get_messages)
            target = next((msg for msg in messages if getattr(msg, "id", "") == message_id), None)
            if target is None:
                return err("protonmail", f"message not found: {message_id}")
            await asyncio.to_thread(_run_client_call, client.delete_messages, [target])
            return {"ok": True, "provider": "protonmail", "message_id": message_id, "status": "deleted"}
        except Exception as e:
            return err("protonmail", f"delete failed: {e}")