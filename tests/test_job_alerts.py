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
import threading
import itertools
import argparse

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Skip config loading - env vars already set manually
# from system.config import load_config
# load_config()

from agentic.mcp_client import init_mcp_client, get_mcp_client
from system.log import get_logger
from system.userspace import user_state_path

log = get_logger(__name__)

# ── "senior" highlighter ────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

_SENIOR_RE = re.compile(r"senior", re.IGNORECASE)


def highlight_senior(text: str, label: str = "", context_chars: int = 80) -> bool:
    """Print `text` with 'senior' highlighted in green (with surrounding
    context), or the first 200 chars in red if no match is found.
    Returns True if a match was found."""
    tag = f"[{label}] " if label else ""
    if not text or not text.strip():
        print(f"{RED}{tag}<empty>{RESET}")
        return False

    m = _SENIOR_RE.search(text)
    if m:
        start = max(0, m.start() - context_chars)
        end = min(len(text), m.end() + context_chars)
        before = text[start:m.start()].replace("\n", " ")
        match = text[m.start():m.end()]
        after = text[m.end():end].replace("\n", " ")
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        print(f"{tag}{DIM}{prefix}{before}{RESET}{GREEN}{BOLD}{match}{RESET}{DIM}{after}{suffix}{RESET}")
        return True
    else:
        snippet = text[:200].replace("\n", " ")
        suffix = "..." if len(text) > 200 else ""
        print(f"{RED}{tag}{snippet}{suffix}{RESET}")
        return False


def progress(msg):
    """Print progress message to stderr for immediate visibility."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def spinner_thread(stop_event):
    """Display a spinner while waiting for long operations."""
    spinner = itertools.cycle(['|', '/', '-', '\\'])
    while not stop_event.is_set():
        print(f"\r[WAITING] {next(spinner)}", end='', file=sys.stderr, flush=True)
        time.sleep(0.2)
    print("\r" + " " * 40, end='\r', file=sys.stderr, flush=True)

JOB_KEYWORDS = [
    "linkedin", "glassdoor", "indeed", "job alert", "job notification",
    "new job", "recommended job", "job match", "career", "hiring",
    "software engineer", "developer", "programmer", "devops", "data scientist"
]


def is_job_alert(snippet: str, subject: str, sender: str) -> bool:
    """Heuristic: does this look like a job alert email?"""
    text = f"{subject} {snippet} {sender}".lower()
    return any(kw in text for kw in JOB_KEYWORDS)


def manual_login(session_file: str, username: str, password: str) -> int:
    """Interactively log in and save the documented ProtonMail session."""
    try:
        from protonmail import ProtonMail
        from protonmail.models import CaptchaConfig
    except ImportError as exc:
        progress(f"ERROR: ProtonMail client unavailable: {exc}")
        return 1
    progress("Starting interactive ProtonMail login...")
    progress("Follow the CAPTCHA instructions in your browser and paste the token when prompted.")
    try:
        client = ProtonMail()
        client.login(username, password, captcha_config=CaptchaConfig(type=CaptchaConfig.CaptchaType.MANUAL))
        Path(session_file).parent.mkdir(parents=True, exist_ok=True)
        client.save_session(session_file)
        progress(f"Login successful; session saved to {session_file}")
        return 0
    except Exception as exc:
        progress(f"ERROR: ProtonMail login failed: {exc}")
        return 1


class DebugState:
    """Minimal stand-in for graph_engine's execution state: toolset.py only
    touches .data (dict) and .runtime (dict)."""

    def __init__(self):
        self.data = {}
        self.runtime = {}

    def set(self, key, value):
        self.data[key] = value


def posting_text_for_highlight(posting: dict) -> str:
    parts = [
        str(posting.get("title", "")),
        str(posting.get("organization", "")),
        str(posting.get("summary", "")),
        str(posting.get("location", "")),
    ]
    return " ".join(p for p in parts if p)


def run_pipeline_steps(args) -> int:
    """Walk the same nodes build_gen_job_post_graph() wires up, node by node,
    printing what Aiko sees/does at each step with 'senior' highlighting."""
    try:
        # Importing the graph module registers its tools with the registry;
        # we call the underlying toolset functions directly so we can print
        # in between steps instead of running the graph engine opaquely.
        import agentic.workflows.job_hunt.graph  # noqa: F401  (registers tools)
        from agentic.workflows.job_hunt import toolset as jh
    except Exception as exc:
        progress(f"ERROR: failed to import job_hunt toolset: {exc}")
        return 1

    state = DebugState()

    progress("=== PIPELINE NODE: fetch_rss_and_email_into_state ===")
    fetch_raw = jh.fetch_rss_and_email_into_state(json.dumps({"max_results": 30}), state=state)
    fetch_result = json.loads(fetch_raw)
    progress(f"total_found={fetch_result.get('total_found')}  sources={fetch_result.get('sources')}")

    all_postings = state.data.get("job_all_postings", [])
    print(f"\n{BOLD}Raw postings fetched ({len(all_postings)}):{RESET}\n")
    match_count = 0
    for i, posting in enumerate(all_postings):
        src = posting.get("_source_name", posting.get("source", "?"))
        found = highlight_senior(
            posting_text_for_highlight(posting),
            label=f"{i+1}/{len(all_postings)} src={src}",
            context_chars=args.context,
        )
        match_count += int(found)
    print(f"\n{BOLD}{match_count}/{len(all_postings)} postings mention 'senior'{RESET}\n")

    if not all_postings:
        progress("No postings fetched — nothing further to walk.")
        return 0

    progress("=== PIPELINE LOOP: get_next_job -> draft_single_job -> save_single_job_draft ===")
    processed = 0
    while True:
        if args.max_jobs is not None and processed >= args.max_jobs:
            progress(f"Reached --max-jobs={args.max_jobs}, stopping loop early")
            break

        next_result = json.loads(jh.get_next_job(state=state, worker_id="debug"))
        if next_result.get("done"):
            progress(f"get_next_job: done (total_processed={next_result.get('total_processed')})")
            break

        job = next_result["job"]
        idx, total = next_result["index"], next_result["total"]
        processed += 1

        print(f"\n{BOLD}--- job {idx+1}/{total} ---{RESET}")
        highlight_senior(posting_text_for_highlight(job), label="raw", context_chars=args.context)

        draft_result = json.loads(
            jh.draft_single_job(json.dumps({"job": job}), "", client=None, model=None, state=state)
        )
        if not draft_result.get("success"):
            progress(f"  draft_single_job FAILED: {draft_result.get('reason')}")
            continue

        draft = draft_result["draft"]
        highlight_senior(draft.get("text", ""), label="formatted draft", context_chars=args.context)
        progress(f"  llm_enriched={draft.get('llm_enriched')}  category={draft.get('category')}")

        if args.save_drafts:
            save_result = json.loads(jh.save_single_job_draft("false", state=state))
            if save_result.get("success"):
                progress(f"  saved -> {save_result.get('draft_dir')}")
            else:
                progress(f"  save_single_job_draft FAILED: {save_result.get('reason')}")
        else:
            progress("  (skipped disk write; pass --save-drafts to persist)")

    progress("=== PIPELINE NODE: check_jobs_remaining ===")
    progress(f"check_jobs_remaining -> {jh.check_jobs_remaining(state=state)}")

    progress("=== PIPELINE NODE: report_job_run ===")
    report = jh.report_job_run(
        plan=fetch_raw,
        search=json.dumps({"total_found": fetch_result.get("total_found")}),
        draft="{}",
        save=json.dumps({"total_saved": processed if args.save_drafts else 0}),
    )
    print(report)
    return 0


async def main():
    parser = argparse.ArgumentParser(description="Test ProtonMail job-alert reading")
    parser.add_argument("--login", action="store_true", help="Interactively log in and save a ProtonMail session")
    parser.add_argument("--pipeline", action="store_true",
                         help="After the raw ProtonMail check, also walk the job_hunt graph nodes "
                              "(fetch_rss_and_email_into_state -> get_next_job -> draft_single_job -> "
                              "save_single_job_draft -> report_job_run) with the same highlighting")
    parser.add_argument("--max-jobs", type=int, default=None, help="Cap jobs walked in --pipeline mode")
    parser.add_argument("--save-drafts", action="store_true", help="In --pipeline mode, actually persist drafts to disk")
    parser.add_argument("--context", type=int, default=80, help="Chars of context shown around a 'senior' match")
    args = parser.parse_args()
    progress("Starting job alert test...")
    
    # Check environment setup first
    progress("Checking ProtonMail configuration...")
    username = os.environ.get("PROTONMAIL_USERNAME", "")
    password = os.environ.get("PROTONMAIL_PASSWORD", "")
    
    if not username:
        progress("ERROR: PROTONMAIL_USERNAME not set in environment")
        return 1
    if not password:
        progress("ERROR: PROTONMAIL_PASSWORD not set in environment")
        return 1
    
    # Show masked username (first 3 chars visible)
    if len(username) > 3:
        visible = username[:3] + "*" * (len(username) - 3)
    else:
        visible = username
    progress(f"Username configured: {visible}")
    
    # Show password is set (show first 2 chars)
    if len(password) > 2:
        visible = password[:2] + "*" * (len(password) - 2)
    else:
        visible = password
    progress(f"Password configured: {visible} ({len(password)} chars)")
    
    # Check session cache
    session_file = str(user_state_path("profile/protonmail_session.pickle"))
    if os.path.exists(session_file):
        size = os.path.getsize(session_file)
        progress(f"Session cache exists: {session_file} ({size} bytes)")
    else:
        progress(f"Session cache NOT found: {session_file}")
    
    if args.login:
        return manual_login(session_file, username, password)

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
    progress("Fetching messages from ProtonMail...")
    progress("  This may take 10-30 seconds on first login...")
    
    # Start spinner for long operation
    stop_spinner = threading.Event()
    spinner = threading.Thread(target=spinner_thread, args=(stop_spinner,))
    spinner.daemon = True
    spinner.start()
    
    result = client.call_tool_sync("read_protonmail", {"max_results": 20})
    
    stop_spinner.set()
    spinner.join(timeout=1)
    print("", file=sys.stderr)  # newline after spinner
    
    progress("  Received messages from ProtonMail")
    
    if not result.get("ok"):
        progress("ERROR: read_protonmail failed")
        log.error("read_protonmail failed: %s", result.get("error"))
        return 1

    messages = result.get("messages", [])
    progress(f"Got {len(messages)} messages")
    for i, msg in enumerate(messages):
        subject = msg.get("subject", "")[:50]
        sender = msg.get("from", "")[:40]
        progress(f"  [{i+1}/{len(messages)}] {sender}: {subject}")
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
            highlight_senior(f"{subject} {snippet}", label=f"snippet id={msg_id}", context_chars=args.context)

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
        full_result = client.call_tool_sync("read_protonmail_full", {"message_id": msg_id})
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
        highlight_senior(body, label=f"full body id={msg_id}", context_chars=args.context)

        # Check for apply links
        urls = re.findall(r'https?://\S+', body)
        if urls:
            progress(f"  Found {len(urls)} URL(s) in email:")
            log.info("  Found %d URL(s) in full body:", len(urls))
            for url in urls[:5]:
                log.info("    %s", url)
                progress(f"    - {url}")

    if args.pipeline:
        pipeline_rc = run_pipeline_steps(args)
        if pipeline_rc != 0:
            return pipeline_rc

    elapsed = time.time() - start_time
    progress(f"Test completed successfully in {elapsed:.2f}s")
    log.info("Test completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))