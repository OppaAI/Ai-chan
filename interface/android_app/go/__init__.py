"""
Go (囲碁): play Go vs Aiko.

Android-app backend (moved from interface/webui/go/).
Parallel to interface/android_app/shogi/. Importing this package exposes the
FastAPI router; auth.py mounts it explicitly.
"""

from .games_go import router

__all__ = ["router"]
