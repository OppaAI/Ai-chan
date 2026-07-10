import os
import json
import secrets
from pathlib import Path
from datetime import timedelta
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket
from fastapi.responses import RedirectResponse
import httpx
from dotenv import load_dotenv
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from core import bioclock
from core.userspace import normalize_user_id, user_state_path

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

# ── OAuth client credentials — GitHub (owner/contributors) + Patreon (paid) ──

GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")

PATREON_CLIENT_ID = os.getenv("PATREON_CLIENT_ID")
PATREON_CLIENT_SECRET = os.getenv("PATREON_CLIENT_SECRET")
# Creator access token — needed to look up campaign membership tiers.
# Generate at https://www.patreon.com/portal/registration/register-clients
PATREON_CREATOR_ACCESS_TOKEN = os.getenv("PATREON_CREATOR_ACCESS_TOKEN")
PATREON_CAMPAIGN_ID = os.getenv("PATREON_CAMPAIGN_ID")

# IMPORTANT: this must be the EXACT origin registered with every provider.
# Since AuRoRA now runs behind Tailscale Funnel, this should be:
#   REDIRECT_BASE=https://aiko.ide-chroma.ts.net
# (no trailing slash). For local dev, keep the localhost fallback.
REDIRECT_BASE = os.getenv("REDIRECT_BASE", "http://localhost:8787")

# ── owner / contributor allowlist (GitHub) ───────────────────────────────────
# Comma-separated in .env, e.g. ALLOWED_GITHUB_USERS=OppaAI,someOtherHandle

def _parse_allowlist(name: str) -> set[str]:
    raw = os.getenv(name, "")
    return {item.strip() for item in raw.split(",") if item.strip()}

ALLOWED_GITHUB_USERS = _parse_allowlist("ALLOWED_GITHUB_USERS")
ALLOWED_PATREON_USERS = _parse_allowlist("ALLOWED_PATREON_USERS")

# Flip to true later when you're ready to accept PRs from anyone with commits
# on the repo. Until then, GitHub login only works for the owner allowlist.
CONTRIBUTORS_ENABLED = os.getenv("CONTRIBUTORS_ENABLED", "false").lower() == "true"
GITHUB_REPO = os.getenv("GITHUB_REPO", "OppaAI/Aiko-chan")

# ── terms/guidelines gate ────────────────────────────────────────────────────
# Bump TERMS_VERSION whenever you materially change the guidelines — anyone
# who accepted an older version gets re-prompted next login.
TERMS_VERSION = "2026-07-07"
TERMS_STORE_PATH = os.getenv("TERMS_STORE_PATH", "").strip()


def _terms_store_path(user_id: str | None) -> Path:
    """Per-user terms acceptance store next to profile/user.md by default."""
    if TERMS_STORE_PATH:
        override = Path(TERMS_STORE_PATH).expanduser()
        if override.is_absolute():
            return override
        if user_id:
            return user_state_path(f"profile/{TERMS_STORE_PATH}", str(user_id)).resolve()
        return override.resolve()
    if user_id:
        return user_state_path("profile/terms_acceptance.json", str(user_id)).resolve()
    return Path("terms_acceptance.json").resolve()


def _load_terms_store(user_id: str | None) -> dict:
    path = _terms_store_path(user_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            return {}
    return {}


def _save_terms_store(user_id: str | None, store: dict) -> None:
    path = _terms_store_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def _has_accepted_terms(provider: str, user_id) -> bool:
    store = _load_terms_store(str(user_id) if user_id is not None else None)
    entry = store.get(f"{provider}:{user_id}")
    return bool(entry and entry.get("version") == TERMS_VERSION)


def _record_terms_acceptance(provider: str, user_id) -> None:
    uid = str(user_id) if user_id is not None else None
    store = _load_terms_store(uid)
    store[f"{provider}:{user_id}"] = {
        "version": TERMS_VERSION,
        "accepted_at": bioclock.local_now().isoformat(),
    }
    _save_terms_store(uid, store)


# ── in-memory state ──────────────────────────────────────────────────────────
# Use redis or a DB in production; fine for solo/small-community use for now.

sessions: dict[str, dict] = {}

# oauth_states maps state -> expiry timestamp. Generated on /login, consumed
# (and deleted) on /callback. Prevents CSRF: an attacker cannot forge a valid
# callback without first having seen a state value minted for *this* browser.
oauth_states: dict[str, float] = {}
STATE_TTL_SECONDS = 300  # 5 minutes to complete the provider round-trip


def _new_state() -> str:
    state = secrets.token_urlsafe(24)
    oauth_states[state] = bioclock.monotonic_now() + STATE_TTL_SECONDS
    return state


def _consume_state(state: str | None) -> None:
    """Validate and burn a state token. Raises 400 if missing/unknown/expired."""
    if not state or state not in oauth_states:
        raise HTTPException(status_code=400, detail="Invalid or missing OAuth state")
    expiry = oauth_states.pop(state)
    if bioclock.monotonic_now() > expiry:
        raise HTTPException(status_code=400, detail="OAuth state expired — try logging in again")


def _create_session(user_id, username: str, email: str | None, provider: str) -> str:
    session_id = secrets.token_urlsafe(32)
    runtime_user_id = normalize_user_id(provider, user_id)
    sessions[session_id] = {
        "user_id": runtime_user_id,
        "provider_user_id": str(user_id),
        "username": username,
        "email": email,
        "provider": provider,
        "created_at": bioclock.local_now(),
        # gate stays closed until they check the box, unless they've already
        # accepted this exact terms version in a previous session
        "accepted_terms": (
            _has_accepted_terms(provider, runtime_user_id)
            or _has_accepted_terms(provider, user_id)
        ),
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
        secure=True,  # requires HTTPS — matches your Tailscale Funnel / Cloudflare setup
    )


def _callback_url(provider: str) -> str:
    """Single source of truth for building a callback URL. Must exactly match
    what's registered in that provider's app settings."""
    return f"{REDIRECT_BASE}/auth/{provider}/callback"


def _authorize_url(base: str, **params: str) -> str:
    """Properly URL-encode query params instead of raw f-string interpolation."""
    return f"{base}?{urlencode(params)}"


# ── dependencies for gating protected routes ─────────────────────────────────

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
    if bioclock.local_now() - session["created_at"] > timedelta(days=30):
        del sessions[session_id]
        raise HTTPException(status_code=401, detail="Session expired")

    return session


# Stricter dependency for anything that should actually talk to Aiko —
# logged in is not enough, they also need to have accepted the current terms.
async def require_accepted_session(session: dict = Depends(require_session)) -> dict:
    if not session.get("accepted_terms"):
        raise HTTPException(status_code=403, detail="Guidelines not yet accepted")
    return session


# ── public config ─────────────────────────────────────────────────────────────

@app.get("/api/auth/config")
async def get_auth_config():
    """Return public OAuth client IDs to frontend."""
    return {
        "github_id": GITHUB_CLIENT_ID,
        "patreon_id": PATREON_CLIENT_ID,
    }


# ── GitHub (owner + future contributors) ─────────────────────────────────────

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


async def _is_contributor(username: str, access_token: str) -> bool:
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/contributors",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if res.status_code != 200:
            return False
        return any(c.get("login") == username for c in res.json())


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
    is_owner = username in ALLOWED_GITHUB_USERS
    is_contrib = CONTRIBUTORS_ENABLED and await _is_contributor(username, access_token)

    if not username or not (is_owner or is_contrib):
        raise HTTPException(status_code=403, detail="Not authorized")

    session_id = _create_session(user["id"], username, user.get("email"), "github")
    response = RedirectResponse(url="/")
    _set_session_cookie(response, session_id)
    return response


# ── Patreon (paid members) ────────────────────────────────────────────────────
# Docs: https://docs.patreon.com/ — verify scopes/endpoints against current
# docs before wiring this up, Patreon's API has changed shape more than once.

PATREON_AUTHORIZE_URL = "https://www.patreon.com/oauth2/authorize"
PATREON_TOKEN_URL = "https://www.patreon.com/api/oauth2/token"
PATREON_IDENTITY_URL = "https://www.patreon.com/api/oauth2/v2/identity"


@app.get("/auth/patreon/login")
async def patreon_login():
    state = _new_state()
    url = _authorize_url(
        PATREON_AUTHORIZE_URL,
        client_id=PATREON_CLIENT_ID,
        redirect_uri=_callback_url("patreon"),
        response_type="code",
        scope="identity identity.memberships",
        state=state,
    )
    return RedirectResponse(url)


@app.get("/auth/patreon/callback")
async def patreon_callback(code: str, state: str | None = None):
    _consume_state(state)

    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            PATREON_TOKEN_URL,
            data={
                "code": code,
                "grant_type": "authorization_code",
                "client_id": PATREON_CLIENT_ID,
                "client_secret": PATREON_CLIENT_SECRET,
                "redirect_uri": _callback_url("patreon"),
            },
        )
        token_data = token_res.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=401, detail="Patreon token exchange failed")

        identity_res = await client.get(
            PATREON_IDENTITY_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            params={
                "include": "memberships",
                "fields[member]": "patron_status,currently_entitled_amount_cents",
            },
        )
        identity = identity_res.json()

    user_id = identity.get("data", {}).get("id")
    username = identity.get("data", {}).get("attributes", {}).get("full_name", user_id)
    memberships = identity.get("included", [])

    active = any(
        m.get("attributes", {}).get("patron_status") == "active_patron"
        for m in memberships
        if m.get("type") == "member"
    )

    is_owner = str(user_id) in ALLOWED_PATREON_USERS

    if not user_id or not (is_owner or active):
        raise HTTPException(status_code=403, detail="Active Patreon membership required")

    session_id = _create_session(user_id, username, None, "patreon")
    response = RedirectResponse(url="/")
    _set_session_cookie(response, session_id)
    return response


# ── terms acceptance ─────────────────────────────────────────────────────────

@app.get("/api/auth/me")
async def get_me(session: dict = Depends(require_session)):
    return {**session, "terms_version_required": TERMS_VERSION}


@app.post("/api/auth/accept-terms")
async def accept_terms(request: Request, session: dict = Depends(require_session)):
    body = await request.json()
    if body.get("accepted") is not True:
        raise HTTPException(status_code=400, detail="Must explicitly accept")

    cookie_value = request.cookies.get("session_id")
    session_id = signer.loads(cookie_value, max_age=SESSION_MAX_AGE_SECONDS)
    sessions[session_id]["accepted_terms"] = True
    _record_terms_acceptance(session["provider"], session["user_id"])
    return {"accepted": True}


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


# ── websocket bridge — gated on accepted terms, not just login ──────────────

aiko_web_instance = None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    cookie_value = websocket.cookies.get("session_id")
    try:
        session_id = signer.loads(cookie_value, max_age=SESSION_MAX_AGE_SECONDS)
        session = sessions.get(session_id)
        if not session or not session.get("accepted_terms"):
            await websocket.close(code=4403)  # custom: terms not accepted
            return
    except (BadSignature, SignatureExpired, TypeError):
        await websocket.close(code=4401)
        return

    if aiko_web_instance is not None:
        await aiko_web_instance._ws_handler(websocket)
    else:
        await websocket.close(code=1011)
