"""
agentic/tools.py

Compatibility facade for Aiko's autonomous toolkit.

Keep this file even though implementations live under ``agentic/toolkit/``: it gives
older callers and the agent loop one stable import surface while domain tools
move into focused modules. New primitive capabilities should be implemented in
``agentic/toolkit/<domain>.py`` and re-exported here only when the chat facade or
agent loop needs them.
"""
from __future__ import annotations

from agentic.toolkit.research import web_fetch, deep_search, deep_research, web_search, web_search_context
from agentic.toolkit.plan import make_plan, create_checklist, save_note, read_workspace_file, summarize_task_state
from agentic.toolkit.organize import schedule_job, list_schedule, cancel_schedule, schedule_reminder, list_reminders, cancel_reminder
from agentic.toolkit.photography import scan_photo_workspace, propose_photo_ingestion, write_photo_ingestion_report
from agentic.toolkit.self_improve import repo_file_tree, repo_read_file, repo_search_text
from agentic.toolkit.job_hunt import search_jobs, dedupe_postings
from agentic.toolkit.social import (
    draft_photo_social, post_photo_social,
    draft_video_social, post_video_social,
)

__all__ = [
    "cancel_reminder",
    "cancel_schedule",
    "create_checklist",
    "dedupe_postings",
    "deep_research",
    "deep_search",
    "draft_photo_social",
    "draft_video_social",
    "list_reminders",
    "list_schedule",
    "make_plan",
    "post_photo_social",
    "post_video_social",
    "propose_photo_ingestion",
    "read_workspace_file",
    "repo_file_tree",
    "repo_read_file",
    "repo_search_text",
    "save_note",
    "scan_photo_workspace",
    "schedule_job",
    "schedule_reminder",
    "search_jobs",
    "summarize_task_state",
    "web_fetch",
    "web_search",
    "web_search_context",
    "write_photo_ingestion_report",
]
