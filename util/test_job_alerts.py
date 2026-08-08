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
import time
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


def progress(msg):
    """Print progress message to stderr for immediate visibility."""
    print(f"[PROGRESS] {msg}", file=sys.stderr, flush=True)

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
    progress("Starting job alert test...")
    
    # Connect to MCP server (starts it if needed)
    progress("Connecting to MCP server...")
    start_time = time.time()
    client = init_mcp_client()
    if client is None:
        progress("ERROR: Failed to connect to MCP server")
        log.error("Failed to connect to MCP server")
        return 1
    connect_time = time.time() - start_time
    progress(f"MCP server connected in {connect_time:.2f}s")

    # 1) List messages with 300-char snippets
    progress("Fetching latest messages (max_results=20)...")
    result = await client.call_tool("read_protonmail", {"max_results": 20})
    
    if not result.get("ok"):
        progress("ERROR: read_protonmail failed")
        log.error("read_protonmail failed: %s", result.get("error"))
        return 1

    messages = result.get("messages", [])
    progress(f"Got {len(messages)} messages")
    log.info("Got %d messages", len(messages))

    # 2) Filter for job alerts using 300-char snippets
    progress("Scanning messages for job alerts...")
    job_alerts = []
    for i, msg in enumerate(messages):
        if i % 5 == 0 and i > 0:
            progress(f"  Checked {i}/{len(messages)} messages...")
        
        snippet = msg.get("snippet", "")
        subject = msg.get("subject", "")
        sender = msg.get("from", "")
        msg_id = msg.get("id", "")
        
        if is_job_alert(snippet, subject, sender):
            job_alerts.append(msg)
            progress(f"  JOB ALERT found: id={msg_id}")
            log.info("JOB ALERT found: id=%s subject=%s sender=%s snippet=%.100s",
                     msg_id, subject, sender, snippet)

    progress(f"Scanned all {len(messages)} messages, found {len(job_alerts)} job alert(s)")
    log.info("Scanned all %d messages, found %d job alert(s)", len(messages), len(job_alerts))

    if not job_alerts:
        progress("WARNING: No job alerts detected in snippets")
        log.warning("No job alerts detected in snippets")
        # Show first few for manual inspection
        progress("Showing first 5 messages for reference:")
        for msg in messages[:5]:
            log.info("  id=%s subject=%s sender=%s snippet=%.100s",
                     msg.get("id"), msg.get("subject"), msg.get("from"), msg.get("snippet", ""))
        return 0

    # 3) For each job alert, fetch full body via read_protonmail_full
    progress(f"Fetching full bodies for {len(job_alerts)} job alert(s)...")
    for i, alert in enumerate(job_alerts):
        msg_id = alert.get("id")
        progress(f"  [{i+1}/{len(job_alerts)}] Fetching message {msg_id}...")
        log.info("Calling read_protonmail_full for id=%s", msg_id)
        
        full_start = time.time()
        full_result = await client.call_tool("read_protonmail_full", {"message_id": msg_id})
        full_time = time.time() - full_start
        
        if not full_result.get("ok"):
            progress(f"  ERROR: read_protonmail_full failed: {full_result.get('error')}")
            log.error("  read_protonmail_full failed: %s", full_result.get("error"))
            continue

        body = full_result.get("body", "")
        progress(f"  Message {msg_id}: {len(body)} chars in {full_time:.2f}s")
        log.info("  Full body length: %d chars", len(body))
        log.info("  Body preview (first 500 chars):")
        log.info("  %s", body[:500])

        # Check for apply links
        urls = re.findall(r'https?://\S+', body)
        if urls:
            progress(f"  Found {len(urls)} URL(s) in email:")
            log.info("  Found %d URL(s) in full body:", len(urls))
            for url in urls[:5]:
                log.info("    %s", url)
                progress(f"    - {url}")

    elapsed = time.time() - start_time
    progress(f"Test completed successfully in {elapsed:.2f}s")
    log.info("Test completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))