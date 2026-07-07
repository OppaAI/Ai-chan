import os
import time
import secrets
from datetime import datetime, timedelta
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
import httpx
from dotenv import load_dotenv
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

load_dotenv()

app = FastAPI()

# ── cookie signing ────────────────────────────────────────────────────────────
# SECRET_KEY signs the session cookie so it can't be forged or edited client-side.
# Generate one with: python -c "import secrets; print(secrets.token_hex(32))"
# Keep it out of git, and rotating it invalidates every existing session (that's
# expected — treat it like a kill switch if a cookie ever leaks).

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY is not set. Generate one with "
        "`python -c \"import secrets; print(secrets.token_hex(32))\"` "
        "and add it to your .env."
    )

signer = URLSafeTimedSerializer(SECRET_KEY, salt="aiko-session-cookie")
SESSION_MAX_AGE_SECONDS = 86400 * 30  # 30 days, matches cookie/session TTL below

# ── OAuth client credentials ─────────────────────────────────────────────────

GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
HF_CLIENT_ID = os.getenv("HUGGINGFACE_CLIENT_ID")
HF_CLIENT_SECRET = os.getenv("HUGGINGFACE_CLIENT_SECRET")
SIMPLELOGIN_CLIENT_ID = os.getenv("SIMPLELOGIN_CLIENT_ID")
SIMPLELOGIN_CLIENT_SECRET = os.getenv("SIMPLELOGIN_CLIENT_SECRET")
DISCORD_CLIENT_ID = os.getenv("DISCORD_CLIENT_ID")
DISCORD_CLIENT_SECRET = os.getenv("DISCORD_CLIENT_SECRET")

# IMPORTANT: this must be the EXACT origin registered with every provider.
# Since AuRoRA now runs behind Tailscale Funnel, this should be:
#   REDIRECT_BASE=https://aurora.ide-chroma.ts.net
# (no trailing slash). For local dev, keep the localhost fallback.
REDIRECT_BASE = os.getenv("REDIRECT_BASE", "http://localhost:8787")

# ── allowlist — only these identities may ever get a session ────────────────
# Comma-separated in .env, e.g. ALLOWED_GITHUB_USERS=OppaAI,someOtherHandle

def _parse_allowlist(name: str) -> set[str]:
    raw = os.getenv(name, "")
    return {item.strip() for item in raw.split(",") if item.strip()}

ALLOWED_GITHUB_USERS = _parse_allowlist("ALLOWED_GITHUB_USERS")
ALLOWED_HF_USERS = _parse_allowlist("ALLOWED_HF_USERS")
ALLOWED_DISCORD_IDS = _parse_allowlist("ALLOWED_DISCORD_IDS")
# SimpleLogin's userinfo endpoint returns email (and optionally a client-scoped
# alias), so gate on email/alias rather than a numeric id.
ALLOWED_SIMPLELOGIN_EMAILS = _parse_allowlist("ALLOWED_SIMPLELOGIN_EMAILS")

# ── in-memory state ──────────────────────────────────────────────────────────
# Use redis or a DB in production; fine for solo/single-user use for now.

sessions: dict[str, dict] = {}

# oauth_states maps state -> expiry timestamp. Generated on /login, consumed
# (and deleted) on /callback. Prevents CSRF: an attacker cannot forge a valid
# callback without first having seen a state value minted for *this* browser.
oauth_states: dict[str, float] = {}
STATE_TTL_SECONDS = 300  # 5 minutes to complete the provider round-trip


def _new_state() -> str:
    state = secrets.token_urlsafe(24)
    oauth_states[state] = time.time() + STATE_TTL_SECONDS
    return state


def _consume_state(state: str | None) -> None:
    """Validate and burn a state token. Raises 400 if missing/unknown/expired."""
    if not state or state not in oauth_states:
        raise HTTPException(status_code=400, detail="Invalid or missing OAuth state")
    expiry = oauth_states.pop(state)
    if time.time() > expiry:
        raise HTTPException(status_code=400, detail="OAuth state expired — try logging in again")


def _create_session(user_id, username: str, email: str | None, provider: str) -> str:
    session_id = secrets.token_urlsafe(32)
    sessions[session_id] = {
        "user_id": user_id,
        "username": username,
        "email": email,
        "provider": provider,
        "created_at": datetime.now(),
    }
    return session_id


def _set_session_cookie(response: RedirectResponse, session_id: str) -> None:
    signed_value = signer.dumps(session_id)
    response.set_cookie(
        "session_id",
        signed_value,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=True,  # requires HTTPS — matches your mkcert/WEBUI_HTTPS setup
    )


def _callback_url(provider: str) -> str:
    """Single source of truth for building a callback URL. Keeping this in
    one place means every /login and /callback pair always agrees, and it
    must exactly match what's registered in that provider's app settings."""
    return f"{REDIRECT_BASE}/auth/{provider}/callback"


def _authorize_url(base: str, **params: str) -> str:
    """Properly URL-encode query params instead of raw f-string interpolation
    (this was the source of the broken redirect_uri — anything with special
    characters, or a REDIRECT_BASE with a trailing slash mismatch, could
    silently corrupt the query string)."""
    return f"{base}?{urlencode(params)}"


# ── dependency for gating protected routes later ─────────────────────────────

async def require_session(request: Request) -> dict:
    cookie_value = request.cookies.get("session_id")
    if not cookie_value:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        session_id = signer.loads(cookie_value, max_age=SESSION_MAX_AGE_SECONDS)
    except SignatureExpired:
        raise HTTPException(status_code=401, detail="Session expired")
    except BadSignature:
        # Cookie was tampered with or signed under a different/old SECRET_KEY
        raise HTTPException(status_code=401, detail="Invalid session")

    if session_id not in sessions:
        raise HTTPException(status_code=401, detail="Not authenticated")

    session = sessions[session_id]
    if datetime.now() - session["created_at"] > timedelta(days=30):
        del sessions[session_id]
        raise HTTPException(status_code=401, detail="Session expired")

    return session


# ── public config ────────────────────────────────────────────────────────────

@app.get("/api/auth/config")
async def get_auth_config():
    """Return public OAuth client IDs to frontend"""
    return {
        "github_id": GITHUB_CLIENT_ID,
        "hf_id": HF_CLIENT_ID,
        "simplelogin_id": SIMPLELOGIN_CLIENT_ID,
        "discord_id": DISCORD_CLIENT_ID,
    }


# ── GitHub ────────────────────────────────────────────────────────────────────

@app.get("/auth/github/login")
async def github_login():
    state = _new_state()
    url = _authorize_url(
        "https://github.com/login/oauth/authorize",
        client_id=GITHUB_CLIENT_ID,
        redirect_uri=_callback_url("github"),
        scope="read:user",
        state=state,
    )
    return RedirectResponse(url)


@app.get("/auth/github/callback")
async def github_callback(code: str, state: str | None = None):
    _consume_state(state)

    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": _callback_url("github"),
            },
            headers={"Accept": "application/json"},
        )
        token_data = token_res.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=401, detail="GitHub token exchange failed")

        user_res = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        user = user_res.json()

    username = user.get("login")
    if not username or username not in ALLOWED_GITHUB_USERS:
        raise HTTPException(status_code=403, detail="Not authorized")

    session_id = _create_session(user["id"], username, user.get("email"), "github")
    response = RedirectResponse(url="/")
    _set_session_cookie(response, session_id)
    return response


# ── Hugging Face ──────────────────────────────────────────────────────────────

@app.get("/auth/huggingface/login")
async def huggingface_login():
    state = _new_state()
    url = _authorize_url(
        "https://huggingface.co/oauth/authorize",
        client_id=HF_CLIENT_ID,
        redirect_uri=_callback_url("huggingface"),
        scope="openid profile",
        response_type="code",
        state=state,
    )
    return RedirectResponse(url)


@app.get("/auth/huggingface/callback")
async def huggingface_callback(code: str, state: str | None = None):
    _consume_state(state)

    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            "https://huggingface.co/oauth/token",
            data={
                "client_id": HF_CLIENT_ID,
                "client_secret": HF_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": _callback_url("huggingface"),
            },
        )
        token_data = token_res.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=401, detail="Hugging Face token exchange failed")

        user_res = await client.get(
            "https://huggingface.co/api/user",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        user = user_res.json()

    username = user.get("username")
    if not username or username not in ALLOWED_HF_USERS:
        raise HTTPException(status_code=403, detail="Not authorized")

    session_id = _create_session(user.get("id", username), username, user.get("email"), "huggingface")
    response = RedirectResponse(url="/")
    _set_session_cookie(response, session_id)
    return response


# ── SimpleLogin ───────────────────────────────────────────────────────────────
# Docs: https://simplelogin.io/docs/siwsl/code-flow/
# Register your app + this exact callback URL at: https://app.simplelogin.io/developer

SIMPLELOGIN_AUTHORIZE_URL = "https://app.simplelogin.io/oauth2/authorize"
SIMPLELOGIN_TOKEN_URL = "https://app.simplelogin.io/oauth2/token"
SIMPLELOGIN_USERINFO_URL = "https://app.simplelogin.io/oauth2/userinfo"


@app.get("/auth/simplelogin/login")
async def simplelogin_login():
    state = _new_state()
    url = _authorize_url(
        SIMPLELOGIN_AUTHORIZE_URL,
        client_id=SIMPLELOGIN_CLIENT_ID,
        redirect_uri=_callback_url("simplelogin"),
        response_type="code",
        scope="profile",
        state=state,
    )
    return RedirectResponse(url)


@app.get("/auth/simplelogin/callback")
async def simplelogin_callback(code: str, state: str | None = None):
    _consume_state(state)

    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            SIMPLELOGIN_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": _callback_url("simplelogin"),
                "client_id": SIMPLELOGIN_CLIENT_ID,
                "client_secret": SIMPLELOGIN_CLIENT_SECRET,
            },
        )
        token_data = token_res.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=401, detail="SimpleLogin token exchange failed")

        user_res = await client.get(
            SIMPLELOGIN_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        user = user_res.json()

    email = user.get("email")
    if not email or email not in ALLOWED_SIMPLELOGIN_EMAILS:
        raise HTTPException(status_code=403, detail="Not authorized")

    session_id = _create_session(email, user.get("name", email), email, "simplelogin")
    response = RedirectResponse(url="/")
    _set_session_cookie(response, session_id)
    return response


# ── Discord ───────────────────────────────────────────────────────────────────

@app.get("/auth/discord/login")
async def discord_login():
    state = _new_state()
    url = _authorize_url(
        "https://discord.com/api/oauth2/authorize",
        client_id=DISCORD_CLIENT_ID,
        redirect_uri=_callback_url("discord"),
        response_type="code",
        scope="identify",
        state=state,
    )
    return RedirectResponse(url)


@app.get("/auth/discord/callback")
async def discord_callback(code: str, state: str | None = None):
    _consume_state(state)

    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            "https://discord.com/api/oauth2/token",
            data={
                "client_id": DISCORD_CLIENT_ID,
                "client_secret": DISCORD_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": _callback_url("discord"),
            },
        )
        token_data = token_res.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=401, detail="Discord token exchange failed")

        user_res = await client.get(
            "https://discord.com/api/users/@me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        user = user_res.json()

    discord_id = user.get("id")
    if not discord_id or discord_id not in ALLOWED_DISCORD_IDS:
        raise HTTPException(status_code=403, detail="Not authorized")

    session_id = _create_session(discord_id, user.get("username", discord_id), user.get("email"), "discord")
    response = RedirectResponse(url="/")
    _set_session_cookie(response, session_id)
    return response


# ── session endpoints ─────────────────────────────────────────────────────────

@app.get("/api/auth/me")
async def get_me(session: dict = Depends(require_session)):
    return session


@app.post("/api/auth/logout")
async def logout(request: Request):
    cookie_value = request.cookies.get("session_id")
    if cookie_value:
        try:
            session_id = signer.loads(cookie_value, max_age=SESSION_MAX_AGE_SECONDS)
            sessions.pop(session_id, None)
        except (BadSignature, SignatureExpired):
            pass  # already invalid/expired, nothing to clean up
    response = RedirectResponse(url="/")
    response.delete_cookie("session_id")
    return response


# ── websocket bridge ──────────────────────────────────────────────────────────

aiko_web_instance = None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if aiko_web_instance is not None:
        await aiko_web_instance._ws_handler(websocket)
    else:
        await websocket.close(code=1011)