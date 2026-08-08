from typing import Optional, List, Dict
import asyncio
import os
import sys
from social.services import env, err

# Global client cache (kept alive across tool calls)
_client_cache = None
_cache_username = None

# protonmail-api-client stores sessions as a binary pickle, not JSON.
_SESSION_FILE = "/home/oppa-ai/Aiko-chan/.protonmail_session.pickle"


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
        client = ProtonMail()
        if os.path.exists(_SESSION_FILE):
            print(f"[PROTONMAIL] Loading session: {_SESSION_FILE}", file=sys.stderr, flush=True)
            client.load_session(_SESSION_FILE, auto_save=True)
        else:
            print("[PROTONMAIL] No saved session; performing login...", file=sys.stderr, flush=True)
            client.login(username, password)
            client.save_session(_SESSION_FILE)
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
        description="Read messages from ProtonMail inbox (or specified folder). Returns list of messages with sender, subject, date, snippet.",
    )
    async def read_protonmail(folder: str = "inbox", query: str = "", max_results: int = 10) -> Dict:
        client, err_resp = await asyncio.to_thread(_get_client)
        if err_resp:
            return err_resp

        try:
            # Get all messages (protonmail-api-client doesn't support folder filtering directly)
            print("[PROTONMAIL] Fetching messages list...", file=sys.stderr, flush=True)
            messages = await asyncio.to_thread(client.get_messages)
            print(f"[PROTONMAIL] Got {len(messages)} messages", file=sys.stderr, flush=True)

            if query:
                q = query.lower()
                messages = [m for m in messages if q in (m.subject or "").lower() or q in (m.sender.address if m.sender else "").lower()]

            results = []
            for i, msg in enumerate(messages[:max_results]):
                try:
                    print(f"[PROTONMAIL] Reading message {i+1}/{max_results} ({getattr(msg, 'id', 'unknown')[:10]}...)...", file=sys.stderr, flush=True)
                    full = await asyncio.to_thread(client.read_message, msg)
                    print(f"[PROTONMAIL] Read message {i+1}/{max_results} OK", file=sys.stderr, flush=True)
                    results.append({
                        "id": getattr(msg, "id", ""),
                        "from": full.sender.address if full.sender else "",
                        "subject": full.subject or "",
                        "date": str(full.date) if full.date else "",
                        "snippet": (full.body or "")[:300],
                    })
                except Exception:
                    continue

            return {"ok": True, "provider": "protonmail", "folder": folder, "count": len(results), "messages": results}
        except Exception as e:
            return err("protonmail", f"read failed: {e}")

    @mcp.tool(
        name="search_protonmail",
        description="Search ProtonMail messages by keyword across all folders. Returns matching messages.",
    )
    async def search_protonmail(query: str, max_results: int = 20) -> Dict:
        client, err_resp = await asyncio.to_thread(_get_client)
        if err_resp:
            return err_resp

        if not query:
            return err("protonmail", "query required")

        try:
            messages = await asyncio.to_thread(client.get_messages)
            q = query.lower()
            filtered = [m for m in messages if q in (m.subject or "").lower() or q in (m.sender.address if m.sender else "").lower()]

            results = []
            for i, msg in enumerate(filtered[:max_results]):
                try:
                    print(f"[PROTONMAIL] Reading message {i+1}/{max_results} ({getattr(msg, 'id', 'unknown')[:10]}...)...", file=sys.stderr, flush=True)
                    full = await asyncio.to_thread(client.read_message, msg)
                    print(f"[PROTONMAIL] Read message {i+1}/{max_results} OK", file=sys.stderr, flush=True)
                    results.append({
                        "id": getattr(msg, "id", ""),
                        "from": full.sender.address if full.sender else "",
                        "subject": full.subject or "",
                        "date": str(full.date) if full.date else "",
                        "snippet": (full.body or "")[:300],
                    })
                except Exception:
                    continue

            return {"ok": True, "provider": "protonmail", "query": query, "count": len(results), "messages": results}
        except Exception as e:
            return err("protonmail", f"search failed: {e}")

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
                client.create_message,
                recipients=recipients,
                subject=subject,
                body=body,
                cc=cc if cc else [],
                bcc=bcc if bcc else [],
            )

            # Send the created message
            sent_message = await asyncio.to_thread(client.send_message, new_message)

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
            messages = await asyncio.to_thread(client.get_messages)
            target = next((msg for msg in messages if getattr(msg, "id", "") == message_id), None)
            if target is None:
                return err("protonmail", f"message not found: {message_id}")
            await asyncio.to_thread(client.delete_messages, [target])
            return {"ok": True, "provider": "protonmail", "message_id": message_id, "status": "deleted"}
        except Exception as e:
            return err("protonmail", f"delete failed: {e}")

    @mcp.tool(
        name="read_protonmail_full",
        description="Fetch the complete body of a specific ProtonMail message by ID. Use after read_protonmail/search_protonmail to get full content (links, full description).",
    )
    async def read_protonmail_full(message_id: str) -> Dict:
        client, err_resp = await asyncio.to_thread(_get_client)
        if err_resp:
            return err_resp

        try:
            # Find the message by ID in the full message list
            print("[PROTONMAIL] Fetching messages list (read_protonmail_full)...", file=sys.stderr, flush=True)
            messages = await asyncio.to_thread(client.get_messages)
            print(f"[PROTONMAIL] Got {len(messages)} messages", file=sys.stderr, flush=True)
            target_msg = None
            for msg in messages:
                if getattr(msg, "id", "") == message_id:
                    target_msg = msg
                    print(f"[PROTONMAIL] Found target message {message_id}", file=sys.stderr, flush=True)
                    break
            
            if target_msg is None:
                return err("protonmail", f"message not found: {message_id}")

            full = await asyncio.to_thread(client.read_message, target_msg)
            return {
                "ok": True,
                "provider": "protonmail",
                "id": message_id,
                "from": full.sender.address if full.sender else "",
                "subject": full.subject or "",
                "date": str(full.date) if full.date else "",
                "body": full.body or "",
            }
        except Exception as e:
            return err("protonmail", f"read full failed: {e}")