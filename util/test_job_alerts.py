#!/usr/bin/env python3
"""
Test script to verify job alert email detection from 300-char snippets
and full-body retrieval via read_protonmail_full.

Usage:
    python3 util/test_job_alerts.py

Requires:
- PROTONMAIL_USERNAME and PROTONMAIL_PASSWORD in environment
- Aiko MCP server running (or this will start it via stdio)
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load env early for config
from system.config import load_config
load_config()

from agentic.mcp_client import init_mcp_client, get_mcp_client
from system.log import get_logger

log = get_logger(__name__)

JOB_KEYWORDS = [
    "linkedin", "glassdoor", "indeed", "job alert", "job notification",
    "new job", "recommended job", "job match", "career", "hiring",
    "software engineer", "developer", "programmer", "devops", "data scientist"
]


def is_job_alert(snippet: str, subject: str, sender: str) -> bool:
    """Heuristic: does this look like a job alert email?"""
    text = f"{subject} {snippet} {sender}".lower()
    return any(kw in text for kw in JOB_KEYWORDS)


async def main():
    # Connect to MCP server (starts it if needed)
    client = init_mcp_client()
    if client is None:
        log.error("Failed to connect to MCP server")
        return 1

    # 1) List messages with 300-char snippets
    log.info("Calling read_protonmail (max_results=20)...")
    result = await client.call_tool("read_protonmail", {"max_results": 20})
    
    if not result.get("ok"):
        log.error("read_protonmail failed: %s", result.get("error"))
        return 1

    messages = result.get("messages", [])
    log.info("Got %d messages", len(messages))

    # 2) Filter for job alerts using 300-char snippets
    job_alerts = []
    for msg in messages:
        snippet = msg.get("snippet", "")
        subject = msg.get("subject", "")
        sender = msg.get("from", "")
        msg_id = msg.get("id", "")
        
        if is_job_alert(snippet, subject, sender):
            job_alerts.append(msg)
            log.info("JOB ALERT found: id=%s subject=%s sender=%s snippet=%.100s",
                     msg_id, subject, sender, snippet)

    if not job_alerts:
        log.warning("No job alerts detected in snippets")
        # Show first few for manual inspection
        for msg in messages[:5]:
            log.info("  id=%s subject=%s sender=%s snippet=%.100s",
                     msg.get("id"), msg.get("subject"), msg.get("from"), msg.get("snippet", ""))
        return 0

    # 3) For each job alert, fetch full body via read_protonmail_full
    log.info("Fetching full bodies for %d job alert(s)...", len(job_alerts))
    for alert in job_alerts:
        msg_id = alert.get("id")
        log.info("Calling read_protonmail_full for id=%s", msg_id)
        full_result = await client.call_tool("read_protonmail_full", {"message_id": msg_id})
        
        if not full_result.get("ok"):
            log.error("  read_protonmail_full failed: %s", full_result.get("error"))
            continue

        body = full_result.get("body", "")
        log.info("  Full body length: %d chars", len(body))
        log.info("  Body preview (first 500 chars):")
        log.info("  %s", body[:500])

        # Check for apply links
        urls = re.findall(r'https?://\S+', body)
        if urls:
            log.info("  Found %d URL(s) in full body:", len(urls))
            for url in urls[:5]:
                log.info("    %s", url)

    log.info("Test completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))