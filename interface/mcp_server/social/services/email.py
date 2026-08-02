
import base64
import os
from email.message import EmailMessage

from social.services import env, get_session, err, refresh_oauth_token
from social.state import get_db


# ── Gmail API (OAuth 2.0) ────────────────────────────────────────────────

def _gmail_access_token() -> tuple[str | None, dict | None]:
    db = get_db()
    cached = db.get_cached_token("gmail")
    if cached:
        return cached, None

    client_id = env("GMAIL_CLIENT_ID")
    client_secret = env("GMAIL_CLIENT_SECRET")
    refresh_token = env("GMAIL_REFRESH_TOKEN")

    token = refresh_oauth_token(
        "gmail",
        "https://oauth2.googleapis.com/token",
        client_id,
        client_secret,
        refresh_token,
    )
    if isinstance(token, dict) and not token.get("ok", True):
        return None, token

    db.set_cached_token("gmail", token, 3600)
    return token, None


def _gmail_send(token: str, to: str, subject: str, body: str) -> dict:
    msg = EmailMessage()
    msg.set_content(body)
    msg["To"] = to
    msg["Subject"] = subject
    encoded = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    try:
        session = get_session()
        resp = session.post(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"raw": encoded},
            timeout=30,
        )
        ok = 200 <= resp.status_code < 300
        return {"ok": ok, "provider": "gmail", "action": "send", "status_code": resp.status_code, "response": resp.text[:500]}
    except Exception as e:
        return err("gmail", str(e))


def _gmail_read(token: str, query: str = "", max_results: int = 10) -> dict:
    session = get_session()
    try:
        params = {"maxResults": min(max_results, 50)}
        if query:
            params["q"] = query
        resp = session.get(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=30,
        )
        if not (200 <= resp.status_code < 300):
            return {"ok": False, "provider": "gmail", "action": "list", "status_code": resp.status_code, "response": resp.text[:500]}
        messages = resp.json().get("messages", [])

        results = []
        for msg_ref in messages[:max_results]:
            detail = session.get(
                f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg_ref['id']}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
            if detail.status_code == 200:
                payload = detail.json()
                headers = {h["name"]: h["value"] for h in payload.get("payload", {}).get("headers", [])}
                snippet = payload.get("snippet", "")[:200]
                results.append({
                    "id": msg_ref["id"],
                    "from": headers.get("From", ""),
                    "subject": headers.get("Subject", ""),
                    "date": headers.get("Date", ""),
                    "snippet": snippet,
                })
        return {"ok": True, "provider": "gmail", "action": "read", "count": len(results), "messages": results}
    except Exception as e:
        return err("gmail", str(e))


# ── Microsoft Graph API (OAuth 2.0) ──────────────────────────────────────

def _outlook_access_token() -> tuple[str | None, dict | None]:
    db = get_db()
    cached = db.get_cached_token("outlook")
    if cached:
        return cached, None

    client_id = env("OUTLOOK_CLIENT_ID")
    client_secret = env("OUTLOOK_CLIENT_SECRET")
    refresh_token = env("OUTLOOK_REFRESH_TOKEN")
    tenant = env("OUTLOOK_TENANT", "common")

    token = refresh_oauth_token(
        "outlook",
        f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
        client_id,
        client_secret,
        refresh_token,
        extra_data={"scope": "https://graph.microsoft.com/Mail.Send https://graph.microsoft.com/Mail.Read offline_access"},
    )
    if isinstance(token, dict) and not token.get("ok", True):
        return None, token

    db.set_cached_token("outlook", token, 3600)
    return token, None


def _outlook_send(token: str, to: str, subject: str, body: str) -> dict:
    try:
        session = get_session()
        resp = session.post(
            "https://graph.microsoft.com/v1.0/me/sendMail",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "message": {
                    "subject": subject,
                    "body": {"contentType": "Text", "content": body},
                    "toRecipients": [{"emailAddress": {"address": to}}],
                },
                "saveToSentItems": True,
            },
            timeout=30,
        )
        ok = 200 <= resp.status_code < 300
        return {"ok": ok, "provider": "outlook", "action": "send", "status_code": resp.status_code, "response": resp.text[:500]}
    except Exception as e:
        return err("outlook", str(e))


def _outlook_read(token: str, query: str = "", max_results: int = 10) -> dict:
    session = get_session()
    try:
        params = {"$top": min(max_results, 50), "$select": "id,from,subject,receivedDateTime,bodyPreview"}
        if query:
            params["$search"] = f'"{query}"'
        resp = session.get(
            "https://graph.microsoft.com/v1.0/me/messages",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=30,
        )
        if not (200 <= resp.status_code < 300):
            return {"ok": False, "provider": "outlook", "action": "list", "status_code": resp.status_code, "response": resp.text[:500]}
        data = resp.json()
        messages = []
        for msg in data.get("value", [])[:max_results]:
            from_addr = msg.get("from", {}).get("emailAddress", {}).get("address", "") if isinstance(msg.get("from"), dict) else ""
            messages.append({
                "id": msg.get("id", ""),
                "from": from_addr,
                "subject": msg.get("subject", ""),
                "date": msg.get("receivedDateTime", ""),
                "snippet": (msg.get("bodyPreview") or "")[:200],
            })
        return {"ok": True, "provider": "outlook", "action": "read", "count": len(messages), "messages": messages}
    except Exception as e:
        return err("outlook", str(e))


# ── unified tools ─────────────────────────────────────────────────────────

def load_tools(mcp):
    @mcp.tool(
        name="send_email",
        description="Send an email via Gmail or Outlook. Provide provider='gmail' or 'outlook'",
    )
    def send_email(to: str, subject: str, body: str, provider: str = "gmail") -> dict:
        provider = provider.lower()
        if provider == "gmail":
            token, err = _gmail_access_token()
            if err:
                return err
            return _gmail_send(token, to, subject, body)
        elif provider == "outlook":
            token, err = _outlook_access_token()
            if err:
                return err
            return _outlook_send(token, to, subject, body)
        else:
            return {"ok": False, "error": f"Unknown provider: {provider}. Use 'gmail' or 'outlook'"}

    @mcp.tool(
        name="read_emails",
        description="Read recent emails from Gmail or Outlook inbox. Provide provider='gmail' or 'outlook'",
    )
    def read_emails(provider: str = "gmail", query: str = "", max_results: int = 10) -> dict:
        provider = provider.lower()
        if provider == "gmail":
            token, err = _gmail_access_token()
            if err:
                return err
            return _gmail_read(token, query, max_results)
        elif provider == "outlook":
            token, err = _outlook_access_token()
            if err:
                return err
            return _outlook_read(token, query, max_results)
        else:
            return {"ok": False, "error": f"Unknown provider: {provider}. Use 'gmail' or 'outlook'"}
