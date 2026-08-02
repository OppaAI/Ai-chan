from __future__ import annotations

from typing import Optional

from social.services import env, err
from social.state import get_db


def _get_client():
    try:
        from protonmail import ProtonMail
    except ImportError:
        return None, err("protonmail", "protonmail-api-client not installed — pip install protonmail-api-client")

    username = env("PROTONMAIL_USERNAME")
    password = env("PROTONMAIL_PASSWORD")
    if not username or not password:
        return None, err("protonmail", "PROTONMAIL_USERNAME and PROTONMAIL_PASSWORD not set")

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
    def read_protonmail(folder: str = "inbox", query: str = "", max_results: int = 10) -> dict:
        client, err_resp = _get_client()
        if err_resp:
            return err_resp

        try:
            if folder.lower() != "inbox":
                messages = client.get_messages(folder=folder)
            else:
                messages = client.get_messages()

            if query:
                q = query.lower()
                messages = [m for m in messages if q in (m.subject or "").lower() or q in (m.sender.address if m.sender else "").lower()]

            results = []
            for msg in messages[:max_results]:
                try:
                    full = client.read_message(msg)
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
        finally:
            try:
                client.close()
            except Exception:
                pass

    @mcp.tool(
        name="search_protonmail",
        description="Search ProtonMail messages by keyword across all folders. Returns matching messages.",
    )
    def search_protonmail(query: str, max_results: int = 20) -> dict:
        client, err_resp = _get_client()
        if err_resp:
            return err_resp

        if not query:
            return err("protonmail", "query required")

        try:
            messages = client.get_messages()
            q = query.lower()
            filtered = [m for m in messages if q in (m.subject or "").lower() or q in (m.sender.address if m.sender else "").lower()]

            results = []
            for msg in filtered[:max_results]:
                try:
                    full = client.read_message(msg)
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
        finally:
            try:
                client.close()
            except Exception:
                pass

    @mcp.tool(
        name="send_protonmail",
        description="Send an email via ProtonMail. Supports HTML body, attachments, CC/BCC, reply-to.",
    )
    def send_protonmail(
        recipients: list[str],
        subject: str,
        body: str,
        cc: Optional[list[str]] = None,
        bcc: Optional[list[str]] = None,
        attachments: Optional[list[dict]] = None,
    ) -> dict:
        """
        Send email via ProtonMail.

        Args:
            recipients: List of email addresses (to)
            subject: Email subject
            body: HTML or plain text body
            cc: Optional CC recipients
            bcc: Optional BCC recipients
            attachments: Optional list of dicts with keys: name, content (bytes), mime_type
        """
        client, err_resp = _get_client()
        if err_resp:
            return err_resp

        try:
            # Create attachments if provided
            attach_objs = []
            if attachments:
                for att in attachments:
                    attach_objs.append(client.create_attachment(
                        content=att.get("content", b""),
                        name=att.get("name", "attachment"),
                    ))

            msg = client.create_message(
                recipients=recipients,
                subject=subject,
                body=body,
                cc=cc or [],
                bcc=bcc or [],
                attachments=attach_objs if attach_objs else None,
            )

            result = client.send_message(msg)

            return {"ok": True, "provider": "protonmail", "message_id": getattr(result, "id", ""), "status": "sent"}
        except Exception as e:
            return err("protonmail", f"send failed: {e}")
        finally:
            try:
                client.close()
            except Exception:
                pass