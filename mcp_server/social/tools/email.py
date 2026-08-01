
import base64
import os
from email.message import EmailMessage

import requests

from social.tools.base import env, int_env


# ── Gmail API (OAuth 2.0) ────────────────────────────────────────────────

def _gmail_access_token() -> tuple[str | None, dict | None]:
    client_id = env("GMAIL_CLIENT_ID")
    client_secret = env("GMAIL_CLIENT_SECRET")
    refresh_token = env("GMAIL_REFRESH_TOKEN")
    if not all([client_id, client_secret, refresh_token]):
        return None, {"ok": False, "provider": "gmail", "error": "GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN not set"}
    try:
        resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={"client_id": client_id, "client_secret": client_secret, "refresh_token": refresh_token, "grant_type": "refresh_token"},
            timeout=30,
        )
        payload = resp.json()
        token = payload.get("access_token") if 200 <= resp.status_code < 300 else None
        if not token:
            return None, {"ok": False, "provider": "gmail", "stage": "token_refresh", "status_code": resp.status_code, "response": payload}
        return token, None
    except Exception as e:
        return None, {"ok": False, "provider": "gmail", "stage": "token_refresh", "error": str(e)}


def _gmail_send(token: str, to: str, subject: str, body: str) -> dict:
    msg = EmailMessage()
    msg.set_content(body)
    msg["To"] = to
    msg["Subject"] = subject
    encoded = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    try:
        resp = requests.post(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"raw": encoded},
            timeout=30,
        )
        ok = 200 <= resp.status_code < 300
        return {"ok": ok, "provider": "gmail", "action": "send", "status_code": resp.status_code, "response": resp.text[:500]}
    except Exception as e:
        return {"ok": False, "provider": "gmail", "action": "send", "error": str(e)}


def _gmail_read(token: str, query: str = "", max_results: int = 10) -> dict:
    try:
        params = {"maxResults": min(max_results, 50)}
        if query:
            params["q"] = query
        resp = requests.get(
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
            detail = requests.get(
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
        return {"ok": False, "provider": "gmail", "action": "read", "error": str(e)}


# ── Microsoft Graph API (OAuth 2.0) ──────────────────────────────────────

def _outlook_access_token() -> tuple[str | None, dict | None]:
    client_id = env("OUTLOOK_CLIENT_ID")
    client_secret = env("OUTLOOK_CLIENT_SECRET")
    refresh_token = env("OUTLOOK_REFRESH_TOKEN")
    tenant = env("OUTLOOK_TENANT", "common")
    if not all([client_id, client_secret, refresh_token]):
        return None, {"ok": False, "provider": "outlook", "error": "OUTLOOK_CLIENT_ID, OUTLOOK_CLIENT_SECRET, OUTLOOK_REFRESH_TOKEN not set"}
    try:
        resp = requests.post(
            f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
                "scope": "https://graph.microsoft.com/Mail.Send https://graph.microsoft.com/Mail.Read offline_access",
            },
            timeout=30,
        )
        payload = resp.json()
        token = payload.get("access_token") if 200 <= resp.status_code < 300 else None
        if not token:
            return None, {"ok": False, "provider": "outlook", "stage": "token_refresh", "status_code": resp.status_code, "response": payload}
        return token, None
    except Exception as e:
        return None, {"ok": False, "provider": "outlook", "stage": "token_refresh", "error": str(e)}


def _outlook_send(token: str, to: str, subject: str, body: str) -> dict:
    try:
        resp = requests.post(
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
        return {"ok": False, "provider": "outlook", "action": "send", "error": str(e)}


def _outlook_read(token: str, query: str = "", max_results: int = 10) -> dict:
    try:
        params = {"$top": min(max_results, 50), "$select": "id,from,subject,receivedDateTime,bodyPreview"}
        if query:
            params["$search"] = f'"{query}"'
        resp = requests.get(
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
        return {"ok": False, "provider": "outlook", "action": "read", "error": str(e)}


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
