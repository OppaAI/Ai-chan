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

# Provider registry
_email_providers = {}


def register_email_provider(name: str, provider_impl: dict):
    """Register an email provider implementation.
    
    provider_impl should have:
    - get_client(): returns (client, error) or (client, None)
    - read_messages(client, folder, unread, max_results, query, list_only, message_id)
    - send_message(client, recipients, subject, body, cc, bcc, attachments)
    - delete_message(client, message_id)
    """
    _email_providers[name] = provider_impl


def get_email_provider(name: str):
    """Get an email provider implementation by name."""
    return _email_providers.get(name)


# Default provider from env
def _get_default_provider() -> str:
    return env("EMAIL_PROVIDER", "protonmail")


def _get_provider_client(provider_name: str):
    """Get authenticated client for the specified provider."""
    provider = get_email_provider(provider_name)
    if not provider:
        return None, {"ok": False, "error": f"email provider not registered: {provider_name}", "provider": "email"}
    return provider.get_client()


def load_tools(mcp):
    """Load generic email tools that delegate to configured provider."""
    # Ensure ProtonMail provider is registered
    from social.services.protonmail import _get_client as _proton_get_client
    from social.services.email import _email_providers
    if "protonmail" not in _email_providers:
        from social.services.protonmail import _get_client as _proton_get_client
        register_email_provider("protonmail", {
            "get_client": lambda: __import__("social.services.protonmail", fromlist=["_get_client"])._get_client(),
            "read_messages": _proton_read_messages,
            "send_message": _proton_send_message,
            "delete_message": _proton_delete_message,
        })
    
    @mcp.tool(
        name="read_email",
        description="Read email messages. If message_id is provided, returns full message body. Otherwise lists messages from specified folder with optional filters. If list_only is true, returns sender/subject/date only (no body read). Returns sender, subject, date, and snippet (or full body if message_id given).",
    )
    async def read_email(
        provider: str = "",
        message_id: str = "",
        folder: str = "inbox",
        unread: bool = True,
        query: str = "",
        max_results: int = 10,
        list_only: bool = False,
    ) -> Dict:
        provider_name = provider or _get_default_provider()
        client, err_resp = _get_provider_client(provider_name)
        if err_resp:
            return err_resp
        
        provider_impl = get_email_provider(provider_name)
        if not provider_impl or "read_messages" not in provider_impl:
            return {"ok": False, "error": f"provider {provider_name} does not support read_messages", "provider": "email"}
        
        return await provider_impl.read_messages(
            client, folder, unread, max_results, query, list_only, message_id
        )

    @mcp.tool(
        name="send_email",
        description="Send an email. Supports HTML body, CC/BCC.",
    )
    async def send_email(
        provider: str = "",
        recipients: List[str] = None,
        subject: str = "",
        body: str = "",
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        attachments: Optional[List[Dict]] = None,
    ) -> Dict:
        provider_name = provider or _get_default_provider()
        client, err_resp = _get_provider_client(provider_name)
        if err_resp:
            return err_resp
        
        provider_impl = get_email_provider(provider_name)
        if not provider_impl or "send_message" not in provider_impl:
            return {"ok": False, "error": f"provider {provider_name} does not support send_message", "provider": "email"}
        
        return await provider_impl.send_message(
            client, recipients or [], subject, body, cc or [], bcc or [], attachments or []
        )

    @mcp.tool(
        name="delete_email",
        description="Delete an email message by ID.",
    )
    async def delete_email(
        provider: str = "",
        message_id: str = "",
    ) -> Dict:
        provider_name = provider or _get_default_provider()
        client, err_resp = _get_provider_client(provider_name)
        if err_resp:
            return err_resp
        
        provider_impl = get_email_provider(provider_name)
        if not provider_impl or "delete_message" not in provider_impl:
            return {"ok": False, "error": f"provider {provider_name} does not support delete_message", "provider": "email"}
        
        return await provider_impl.delete_message(client, message_id)


def load_protonmail_provider():
    """Load the ProtonMail provider implementation."""
    from social.services.protonmail import _get_client as _proton_get_client
    
    return {
        "get_client": _proton_get_client,
        "read_messages": _proton_read_messages,
        "send_message": _proton_send_message,
        "delete_message": _proton_delete_message,
    }


async def _proton_read_messages(client, folder: str, unread: bool, max_results: int, query: str, list_only: bool, message_id: str = "") -> Dict:
    """Read messages using ProtonMail client."""
    import sys
    from social.services.protonmail import _run_client_call
    
    try:
        # Get all messages
        all_messages = await asyncio.to_thread(_run_client_call, client.get_messages)
        
        # Filter by folder (0=inbox, 3=trash, 4=spam)
        folder = folder.lower()
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
            if folder == "inbox" and is_inbox and (not unread or getattr(msg, "unread", False)):
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


async def _proton_send_message(client, recipients: List[str], subject: str, body: str, cc: List[str], bcc: List[str], attachments: List[Dict]) -> Dict:
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


async def _proton_delete_message(client, message_id: str) -> Dict:
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