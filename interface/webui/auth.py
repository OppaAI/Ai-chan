from __future__ import annotations

import logging
import os
import json
import secrets
from pathlib import Path
from datetime import timedelta
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket
from fastapi.responses import RedirectResponse
import httpx
from system.config import load_config
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from system import bioclock
from system.userspace import normalize_user_id, user_state_path

load_config()

log = logging.getLogger(__name__)

app = FastAPI()

# Mount studio backends
# LTM Graph Studio
try:
    from interface.webui.studio.memory.ltm.backend.api import app as ltm_studio_app
    app.mount("/studio/memory/ltm", ltm_studio_app)
except ImportError as e:
    log.warning(f"Could not mount LTM studio: {e}")


# STM Studio (combined STM/LTM/KB entry)
try:
    from interface.webui.studio.memory.stm.backend.api import app as stm_studio_app
    app.mount("/studio/memory/stm", stm_studio_app)
except ImportError as e:
    log.warning(f"Could not mount STM studio: {e}")



@app.get("/studio/grasp", include_in_schema=False)
@app.get("/studio/grasp/", include_in_schema=False)
async def redirect_legacy_grasp_studio():
    return RedirectResponse(url="/studio/memory/stm/", status_code=307)


@app.get("/studio/memory", include_in_schema=False)
@app.get("/studio/memory/", include_in_schema=False)
async def redirect_legacy_memory_studio():
    return RedirectResponse(url="/studio/memory/ltm/", status_code=307)

# DAG Studio
try:
    from interface.webui.studio.dag.backend.api import app as dag_studio_app
    app.mount("/studio/dag", dag_studio_app)
except ImportError as e:
    log.warning(f"Could not mount DAG studio: {e}")

# KB Storage Viewer Studio
try:
    from interface.webui.studio.memory.kb.backend.api import app as kb_studio_app
    app.mount("/studio/memory/kb", kb_studio_app)
except ImportError as e:
    log.warning(f"Could not mount KB studio: {e}")

# Approval Studio
try:
    from interface.webui.studio.approval.backend.api import app as approval_studio_app
    app.mount("/studio/approval", approval_studio_app)
except ImportError as e:
    log.warning(f"Could not mount approval studio: {e}")

# MCP Studio
try:
    from interface.webui.studio.mcp.backend.api import app as mcp_studio_app
    app.mount("/studio/mcp", mcp_studio_app)
except ImportError as e:
    log.warning(f"Could not mount MCP studio: {e}")

# Spec Studio (Layer 4)
try:
    from interface.webui.studio.spec.backend.api import app as spec_studio_app
    app.mount("/studio/spec", spec_studio_app)
except ImportError as e:
    log.warning(f"Could not mount Spec studio: {e}")

# ── cookie signing ────────────────────────────────────────────────────────────
# SECRET_KEY signs the session cookie so it can't be forged or edited client-side.
# Generate one with: python -c "import secrets; print(secrets.token_hex(32))"
# Keep it out of git, and rotating it invalidates every existing session (that's
# expected — treat it like a kill switch if a cookie ever leaks).

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_hex(32)
    log.warning(
        "SECRET_KEY not set in .env — generated ephemeral key. "
        "All sessions will be invalidated on restart. "
        "Generate a permanent one with: "
        "`python -c \"import secrets; print(secrets.token_hex(32))\"`"
    )

signer = URLSafeTimedSerializer(SECRET_KEY, salt="aiko-session-cookie")
SESSION_MAX_AGE_SECONDS = 86400 * 30  # 30 days, matches cookie/session TTL below
