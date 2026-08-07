"""Consolidation lifecycle, archival, and maintenance entry points."""

from __future__ import annotations

from .backend import archive_reports, maintenance_run, maybe_run_consolidation

__all__ = ["archive_reports", "maintenance_run", "maybe_run_consolidation"]
