"""
core/tools.py

Compatibility facade for Aiko's autonomous toolkit.

Keep this file even though implementations live in ``core/toolkit``: it gives
older callers and the agent loop one stable import surface while domain tools
move into focused modules. New primitive capabilities should be implemented in
``core/toolkit/<domain>.py`` and re-exported here only when the chat facade or
agent loop needs them.
"""

from __future__ import annotations

from core.toolkit.web import fetch_and_extract, deep_search, web_search, web_search_context
from core.toolkit.planning import make_plan, create_checklist, save_note, read_workspace_file, summarize_task_state
from core.toolkit.scheduling import schedule_job, list_schedule, cancel_schedule, schedule_reminder, list_reminders, cancel_reminder
from core.toolkit.photo import scan_photo_workspace, propose_photo_ingestion, write_photo_ingestion_report
from core.toolkit.architecture import repo_file_tree, repo_read_file, repo_search_text

__all__ = [
    "web_search",
    "fetch_and_extract",
    "deep_search",
    "web_search_context",
    "make_plan",
    "create_checklist",
    "save_note",
    "read_workspace_file",
    "summarize_task_state",
    "schedule_job",
    "list_schedule",
    "cancel_schedule",
    "schedule_reminder",
    "list_reminders",
    "cancel_reminder",
    "scan_photo_workspace",
    "propose_photo_ingestion",
    "write_photo_ingestion_report",
    "repo_file_tree",
    "repo_read_file",
    "repo_search_text",
]
