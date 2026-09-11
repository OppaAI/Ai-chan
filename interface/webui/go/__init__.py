"""
Go (囲碁): play Go vs Aiko.

Parallel to interface/webui/shogi/. Importing this package exposes the
FastAPI router; auth.py mounts it explicitly.
"""

from .games_go import router

__all__ = ["router"]
