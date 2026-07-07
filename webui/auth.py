import os
import time
import secrets
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
import httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# ── OAuth client credentials ─────────────────────────────────────────────────

GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
HF_CLIENT_ID = os.getenv("HUGGINGFACE_CLIENT_ID")
HF_CLIENT_SECRET = os.getenv("HUGGINGFACE_CLIENT_SECRET")
PROTON_CLIENT_ID = os.getenv("PROTON_CLIENT_ID")
PROTON_CLIENT_SECRET = os.getenv("PROTON_CLIENT_SECRET")
DISCORD_CLIENT_ID = os.getenv("DISCORD_CLIENT_ID")
DISCORD_CLIENT_SECRET = os.getenv("DISCORD_CLIENT_SECRET")
REDIRECT_BASE = os.getenv("REDIRECT_BASE", "http://localhost:8787")

# ── allowlist — only these identities may ever get a session ────────────────
# Comma-separated in .env, e.g. ALLOWED_GITHUB_USERS=OppaAI,someOtherHandle

def _parse_allowlist(name: str) -> set[str]:
    raw = os.getenv(name, "")
    return {item.strip() for item in raw.split(",") if item.strip()}

ALLOWED_GITHUB_USERS = _parse_allowlist("ALLOWED_GITHUB_USERS")
ALLOWED_HF_USERS = _parse_allowlist("ALLOWED_HF_USERS")
ALLOWED_DISCORD_IDS = _parse_allowlist("ALLOWED_DISCORD_IDS")
ALLOWED_PROTON_USERS = _parse_allowlist("ALLOWED_PROTON_USERS")

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
    response.set_cookie(
        "session_id",
        session_id,
        max_age=86400 * 30,
        httponly=True,
        samesite="lax",
        secure=True,  # requires HTTPS — matches your mkcert/WEBUI_HTTPS setup
    )


# ── dependency for gating protected routes later ─────────────────────────────

async def require_session(request: Request) -> dict:
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
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
        "proton_id": PROTON_CLIENT_ID,
        "discord_id": DISCORD_CLIENT_ID,
    }


# ── GitHub ────────────────────────────────────────────────────────────────────

@app.get("/auth/github/login")
async def github_login():
    state = _new_state()
    url = (
        "https://github.com/login/oauth/authorize"
        f"?client_id={GITHUB_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_BASE}/auth/github/callback"
        f"&scope=read:user"
        f"&state={state}"
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
                "redirect_uri": f"{REDIRECT_BASE}/auth/github/callback",
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
    url = (
        "https://huggingface.co/oauth/authorize"
        f"?client_id={HF_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_BASE}/auth/huggingface/callback"
        f"&scope=openid%20profile"
        f"&response_type=code"
        f"&state={state}"
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
                "redirect_uri": f"{REDIRECT_BASE}/auth/huggingface/callback",
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


# ── Discord ───────────────────────────────────────────────────────────────────

@app.get("/auth/discord/login")
async def discord_login():
    state = _new_state()
    url = (
        "https://discord.com/api/oauth2/authorize"
        f"?client_id={DISCORD_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_BASE}/auth/discord/callback"
        f"&response_type=code"
        f"&scope=identify"
        f"&state={state}"
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
                "redirect_uri": f"{REDIRECT_BASE}/auth/discord/callback",
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
    session_id = request.cookies.get("session_id")
    if session_id:
        sessions.pop(session_id, None)
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
