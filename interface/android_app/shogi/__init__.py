"""
Shogi: play Shogi vs Aiko.

Android-app backend (moved from interface/webui/shogi/, which itself was
moved out of the lingo package — Shogi is a game, not language learning).
Importing this package exposes the FastAPI router; auth.py mounts it
explicitly (see interface/webui/auth.py).
"""

from .games_shogi import router

__all__ = ["router"]
