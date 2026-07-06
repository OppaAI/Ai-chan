import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
import httpx
import secrets
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
HF_CLIENT_ID = os.getenv("HUGGINGFACE_CLIENT_ID")
HF_CLIENT_SECRET = os.getenv("HUGGINGFACE_CLIENT_SECRET")
PROTON_CLIENT_ID = os.getenv("PROTON_CLIENT_ID")
PROTON_CLIENT_SECRET = os.getenv("PROTON_CLIENT_SECRET")
DISCORD_CLIENT_ID = os.getenv("DISCORD_CLIENT_ID")
DISCORD_CLIENT_SECRET = os.getenv("DISCORD_CLIENT_SECRET")
REDIRECT_BASE = os.getenv("REDIRECT_BASE", "http://localhost:8787")

sessions = {}  # Use redis or DB in production

@app.get("/api/auth/config")
async def get_auth_config():
    """Return public OAuth client IDs to frontend"""
    return {
        "github_id": GITHUB_CLIENT_ID,
        "hf_id": HF_CLIENT_ID,
        "proton_id": PROTON_CLIENT_ID,
        "discord_id": DISCORD_CLIENT_ID,
    }

@app.get("/auth/github/callback")
async def github_callback(code: str, request: Request):
    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            "https://github.com/login/oauth/access_token",
            json={"client_id": GITHUB_CLIENT_ID, "client_secret": GITHUB_CLIENT_SECRET, "code": code},
            headers={"Accept": "application/json"}
        )
        token_data = token_res.json()
        access_token = token_data.get("access_token")
        
        user_res = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        user = user_res.json()
        
        session_id = secrets.token_urlsafe(32)
        sessions[session_id] = {
            "user_id": user["id"],
            "username": user["login"],
            "email": user.get("email"),
            "provider": "github",
            "created_at": datetime.now()
        }
        
        response = RedirectResponse(url="/")
        response.set_cookie("session_id", session_id, max_age=86400*30, httponly=True, samesite="lax")
        return response

@app.get("/auth/huggingface/callback")
async def huggingface_callback(code: str):
    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            "https://huggingface.co/oauth/token",
            data={
                "client_id": HF_CLIENT_ID,
                "client_secret": HF_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": f"{REDIRECT_BASE}/auth/huggingface/callback"
            }
        )
        token_data = token_res.json()
        access_token = token_data.get("access_token")
        
        user_res = await client.get(
            "https://huggingface.co/api/user",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        user = user_res.json()
        
        session_id = secrets.token_urlsafe(32)
        sessions[session_id] = {
            "user_id": user["id"],
            "username": user["username"],
            "email": user.get("email"),
            "provider": "huggingface",
            "created_at": datetime.now()
        }
        
        response = RedirectResponse(url="/")
        response.set_cookie("session_id", session_id, max_age=86400*30, httponly=True, samesite="lax")
        return response

@app.get("/auth/discord/callback")
async def discord_callback(code: str):
    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            "https://discord.com/api/oauth2/token",
            data={
                "client_id": DISCORD_CLIENT_ID,
                "client_secret": DISCORD_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": f"{REDIRECT_BASE}/auth/discord/callback"
            }
        )
        token_data = token_res.json()
        access_token = token_data.get("access_token")
        
        user_res = await client.get(
            "https://discord.com/api/users/@me",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        user = user_res.json()
        
        session_id = secrets.token_urlsafe(32)
        sessions[session_id] = {
            "user_id": user["id"],
            "username": user["username"],
            "email": user.get("email"),
            "provider": "discord",
            "created_at": datetime.now()
        }
        
        response = RedirectResponse(url="/")
        response.set_cookie("session_id", session_id, max_age=86400*30, httponly=True, samesite="lax")
        return response

@app.get("/api/auth/me")
async def get_me(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    session = sessions[session_id]
    if datetime.now() - session["created_at"] > timedelta(days=30):
        del sessions[session_id]
        raise HTTPException(status_code=401, detail="Session expired")
    
    return session

@app.post("/api/auth/logout")
async def logout(request: Request):
    response = RedirectResponse(url="/")
    response.delete_cookie("session_id")
    return response
