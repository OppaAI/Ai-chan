"""
Shogi: play Shogi vs Aiko.

Moved out of interface/webui/lingo/ — Shogi is a game, not language
learning. Importing this package exposes the FastAPI router; auth.py
mounts it explicitly (see interface/webui/auth.py).
"""

from .games_shogi import router

__all__ = ["router"]
