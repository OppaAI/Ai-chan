from typing import Optional, List, Dict
import sys

# Provider registry (dynamically loaded)
_email_providers = {}


def register_email_provider(name: str, provider_impl: dict):
    """Register an email provider implementation.
    
    provider_impl dict should contain:
    - get_client(): returns (client, error_dict) or (client, None)
    - read_messages(client, folder, unread, max_results, query, list_only, message_id)
    - send_message(client, recipients, subject, body, cc, bcc, attachments)
    - delete_message(client, message_id)
    """
    _email_providers[name] = provider_impl
    print(f"[EMAIL] Registered provider: {name}", file=sys.stderr, flush=True)


def get_email_provider(name: str):
    """Get a registered email provider by name."""
    return _email_providers.get(name)


def _get_default_provider() -> str:
    """Get default provider from env."""
    from social.services import env
    return env("EMAIL_PROVIDER", "protonmail")


def _get_provider_client(provider_name: str):
    """Get authenticated client for specified provider."""
    provider = get_email_provider(provider_name)
    if not provider:
        return None, {"ok": False, "error": f"provider not registered: {provider_name}", "provider": provider_name}
    return provider["get_client"]()


def load_tools(mcp):
    """Load generic email tools that delegate to configured provider."""
    # Register ProtonMail provider
    from social.services import protonmail as protonmail_mod
    register_email_provider("protonmail", {
        "get_client": protonmail_mod.get_client,
        "read_messages": protonmail_mod.read_messages,
        "send_message": protonmail_mod.send_message,
        "delete_message": protonmail_mod.delete_message,
    })
    
    @mcp.tool(
        name="read_email",
        description="Read email messages from configured provider. If message_id provided, returns full body. Otherwise lists messages with optional filters. If list_only=true, returns headers only (no body read).",
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
        provider_impl = get_email_provider(provider_name)
        
        if not provider_impl:
            return {"ok": False, "error": f"provider not registered: {provider_name}", "provider": provider_name}
        
        client, err_resp = provider_impl["get_client"]()
        if err_resp:
            return err_resp
        
        return await provider_impl["read_messages"](
            client, folder, unread, max_results, query, list_only, message_id
        )

    @mcp.tool(
        name="send_email",
        description="Send an email via configured provider. Supports recipients, CC, BCC.",
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
        provider_impl = get_email_provider(provider_name)
        
        if not provider_impl:
            return {"ok": False, "error": f"provider not registered: {provider_name}", "provider": provider_name}
        
        client, err_resp = provider_impl["get_client"]()
        if err_resp:
            return err_resp
        
        return await provider_impl["send_message"](
            client, recipients or [], subject, body, cc or [], bcc or [], attachments or []
        )

    @mcp.tool(
        name="delete_email",
        description="Delete an email message by ID via configured provider.",
    )
    async def delete_email(
        provider: str = "",
        message_id: str = "",
    ) -> Dict:
        provider_name = provider or _get_default_provider()
        provider_impl = get_email_provider(provider_name)
        
        if not provider_impl:
            return {"ok": False, "error": f"provider not registered: {provider_name}", "provider": provider_name}
        
        client, err_resp = provider_impl["get_client"]()
        if err_resp:
            return err_resp
        
        return await provider_impl["delete_message"](client, message_id)