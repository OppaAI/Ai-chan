from typing import Optional, List, Dict
import asyncio
from social.services import env, err
from social.state import get_db

# Global client cache (kept alive across tool calls)
_client_cache = None
_cache_username = None


def _get_client():
    """Get or create cached ProtonMail client."""
    global _client_cache, _cache_username
    
    try:
        from protonmail import ProtonMail
    except ImportError:
        return None, err("protonmail", "protonmail-api-client not installed — pip install protonmail-api-client")

    username = env("PROTONMAIL_USERNAME")
    password = env("PROTONMAIL_PASSWORD")
    if not username or not password:
        return None, err("protonmail", "PROTONMAIL_USERNAME and PROTONMAIL_PASSWORD not set")

    # Always create a fresh client to avoid ServerProof errors
    # protonmail-api-client caches sessions internally via save_session
    try:
        client = ProtonMail()
        client.login(username, password)
        return client, None
    except Exception as e:
        return None, err("protonmail", f"login failed: {e}")


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
            messages = await asyncio.to_thread(client.get_messages)

            if query:
                q = query.lower()
                messages = [m for m in messages if q in (m.subject or "").lower() or q in (m.sender.address if m.sender else "").lower()]

            results = []
            for msg in messages[:max_results]:
                try:
                    full = await asyncio.to_thread(client.read_message, msg)
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
            for msg in filtered[:max_results]:
                try:
                    full = await asyncio.to_thread(client.read_message, msg)
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
        finally:
            # protonmail-api-client doesn't have close() method
            pass