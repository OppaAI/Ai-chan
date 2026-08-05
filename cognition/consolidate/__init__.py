"""Consolidation package façade.

The monthly consolidation implementation is grouped here instead of under
``cognition.memory`` so retention, promotion, archival, and journal helpers can
remain together.
"""

from __future__ import annotations

from .backend import *  # noqa: F401,F403
