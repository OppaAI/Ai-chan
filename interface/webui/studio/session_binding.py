"""
interface/webui/studio/session_binding.py

Bind studio API requests to the logged-in webui session.

Studios are mounted sub-apps outside the chat WebSocket handshake, so
without this they'd resolve user data through env fallbacks or the guest
sentinel instead of the authenticated login. This middleware validates the
signed session_id cookie via interface.webui.auth.require_session and sets
the request-local identity contextvars for /api/* paths; unauthenticated
API calls get 401. Static assets stay public.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from system.log import get_logger
from system.userspace import (
    reset_current_display_name,
    reset_current_user_id,
    set_current_display_name,
    set_current_user_id,
)

log = get_logger(__name__)


def bind_login_session(app: FastAPI) -> None:
    """Install login-session identity binding on a studio sub-app."""

    @app.middleware("http")
    async def _bind_session_user(request: Request, call_next):
        if not request.url.path.startswith("/api"):
            return await call_next(request)
        from interface.webui.auth import require_session

        try:
            session = await require_session(request)
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        user_token = set_current_user_id(str(session.get("user_id") or ""))
        display_token = None
        try:
            display_token = set_current_display_name(session.get("username"))
        except ValueError:
            pass
        try:
            return await call_next(request)
        finally:
            reset_current_user_id(user_token)
            if display_token is not None:
                reset_current_display_name(display_token)
