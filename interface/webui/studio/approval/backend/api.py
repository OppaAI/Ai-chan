"""Approval Studio backend — review and approve daily job post drafts."""
from __future__ import annotations

import json
import mimetypes
import os
import requests
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from system.config import load_config
load_config()

app = FastAPI(title="Aiko Approval Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
app.mount("/shared", StaticFiles(directory=str(SHARED_DIR), html=True), name="studio-shared")
