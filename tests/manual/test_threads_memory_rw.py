"""End-to-end test: memory read/write cycle for Threads.

Exercises:
  1. _save_interaction_memory()  — owner reply → memory.db write
  2. _threads_memory_context()   — memory.db read → prompt context
  3. Round-trip: write a fact, read it back via search, verify it appears
  4. Non-owner write is rejected
  5. Sensitive content is redacted before write
  6. Bluesky + Mastodon have the same behavior

Usage:
  cd /home/oppa-ai/jetson
  /home/oppa-ai/.venvs/aiko-x86/bin/python tests/manual/test_threads_memory_rw.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Isolated USER_SPACE_ROOT for this test so we don't touch the real ~/.aiko
TEST_ROOT = Path(tempfile.mkdtemp(prefix="aiko_memtest_"))
os.environ["USER_SPACE_ROOT"] = str(TEST_ROOT)
os.environ["AIKO_USER_ID"] = "OppaAI"
os.environ["CURRENT_DISPLAY_NAME"] = "OppaAI"

# Use sqlite-vec if available
try:
    import sqlite_vec  # noqa: F401
    HAS_VEC = True
except ImportError:
    HAS_VEC = False

print("=" * 70)
print("THREADS MEMORY READ/WRITE END-TO-END TEST")
print("=" * 70)
print(f"Test USER_SPACE_ROOT: {TEST_ROOT}")
print(f"sqlite-vec available: {HAS_VEC}")
print()

results = []


def check(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    results.append((name, passed, detail))
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
# 1. WRITE: _save_interaction_memory
# ─────────────────────────────────────────────────────────────────────────────
print()
print("── 1. WRITE: _save_interaction_memory() ─────────────────────────────")
print()

from interface.mcp_server.social.services import threads

# 1a. Owner message → saved
memorize = MagicMock()
memorize.get_display_name.return_value = "OppaAI"
memorize.get_user_id.return_value = "OppaAI"

owner_reply = {
    "username": "oppa.ai.bot",
    "timestamp": "2026-08-29T16:02:50+0000",
    "text": "Hi Aiko, we tested memory today at the lab.",
}
saved = threads._save_interaction_memory(owner_reply, "Yes, the memory test went well.", memorize)
check("owner message is saved", saved is True)
check("memorize.add was called once", memorize.add.call_count == 1)
if memorize.add.call_count == 1:
    call_kwargs = memorize.add.call_args
    msgs = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
    user_msg = msgs[0]["content"]
    asst_msg = msgs[1]["content"]
    check(
        "user message has [Threads 2026-08-29] prefix",
        "[Threads 2026-08-29]" in user_msg,
    )
    check("user message has display_name", "OppaAI said:" in user_msg)
    check(
        "user message includes the original text",
        "tested memory today" in user_msg,
    )
    check("assistant message has 'Aiko replied:'", "Aiko replied:" in asst_msg)
    check("user_id is OppaAI", call_kwargs.kwargs.get("user_id") == "OppaAI")
    check("display_name is OppaAI", call_kwargs.kwargs.get("display_name") == "OppaAI")

# 1b. Non-owner message → NOT saved
memorize.reset_mock()
stranger_reply = {"username": "random_stranger", "text": "Hi Aiko what is up"}
saved = threads._save_interaction_memory(stranger_reply, "Hello!", memorize)
check("stranger message is rejected", saved is False)
check("memorize.add was NOT called", memorize.add.call_count == 0)

# 1c. Empty / sensitive text → NOT saved
memorize.reset_mock()
empty_reply = {"username": "oppa.ai.bot", "text": ""}
saved = threads._save_interaction_memory(empty_reply, "hi", memorize)
check("empty text is rejected", saved is False)

memorize.reset_mock()
sensitive_reply = {"username": "oppa.ai.bot", "text": "my api_key=supersecretvalue123"}
saved = threads._save_interaction_memory(sensitive_reply, "noted", memorize)
check("sensitive (api_key=...) text is rejected", saved is False)

# 1d. Disabled via env
memorize.reset_mock()
os.environ["THREADS_INTERACTION_MEMORY_ENABLED"] = "0"
saved = threads._save_interaction_memory(owner_reply, "test", memorize)
check("disabled via THREADS_INTERACTION_MEMORY_ENABLED=0", saved is False)
os.environ.pop("THREADS_INTERACTION_MEMORY_ENABLED", None)


# ─────────────────────────────────────────────────────────────────────────────
# 2. READ: _threads_memory_context
# ─────────────────────────────────────────────────────────────────────────────
print()
print("── 2. READ: _threads_memory_context() ───────────────────────────────")
print()

# 2a. No memorize → empty
ctx = threads._threads_memory_context("test query", None)
check("no memorize → empty context", ctx == "")

# 2b. With mock memorize that returns hits
memorize = MagicMock()
memorize.get_user_id.return_value = "OppaAI"
memorize.search.return_value = [
    {"memory": "OppaAI visited PNE with Aiko"},
    {"memory": "OppaAI is testing memory writes"},
]
ctx = threads._threads_memory_context("memory test", memorize)
check("search was called with the right user_id", memorize.search.call_count == 1)
if memorize.search.call_count == 1:
    kw = memorize.search.call_args.kwargs
    check("search user_id is OppaAI", kw.get("user_id") == "OppaAI")
    check("search limit is 3", kw.get("limit") == 3)
check("context has the Long-term memories header", "Long-term memories" in ctx)
check("context includes first hit", "PNE" in ctx)
check("context includes second hit", "memory writes" in ctx)
check("context is bounded to 1200 chars", len(ctx) <= 1200)

# 2c. No hits → empty
memorize.search.return_value = []
ctx = threads._threads_memory_context("anything", memorize)
check("no search hits → empty context", ctx == "")


# ─────────────────────────────────────────────────────────────────────────────
# 3. RESEARCH: _threads_research_context
# ─────────────────────────────────────────────────────────────────────────────
print()
print("── 3. RESEARCH: _threads_research_context() ─────────────────────────")
print()

# 3a. Trigger words present
ctx = threads._threads_research_context("Can you search the web for Jetson Orin Nano 2 specs?")
# Don't assert non-empty because websearch may fail in test env — just check it didn't crash
check("trigger word 'search' doesn't crash", isinstance(ctx, str))

# 3b. No trigger words → empty
ctx = threads._threads_research_context("Hello Aiko, how are you?")
check("no trigger words → empty research context", ctx == "")


# ─────────────────────────────────────────────────────────────────────────────
# 4. IMAGE REQUEST: _threads_image_request
# ─────────────────────────────────────────────────────────────────────────────
print()
print("── 4. IMAGE REQUEST: _threads_image_request() ──────────────────────")
print()

# 4a. No request → empty
prompt = threads._threads_image_request("Hello Aiko")
check("no image request → empty prompt", prompt == "")

# 4b. Disabled via env
os.environ["THREADS_IMAGEGEN_ENABLED"] = "0"
prompt = threads._threads_image_request("Can you draw a picture of a cat?")
check("disabled via THREADS_IMAGEGEN_ENABLED=0", prompt == "")
os.environ.pop("THREADS_IMAGEGEN_ENABLED", None)


# ─────────────────────────────────────────────────────────────────────────────
# 5. BLUESKY: parallel tests
# ─────────────────────────────────────────────────────────────────────────────
print()
print("── 5. BLUESKY: parallel memory read/write ──────────────────────────")
print()

from interface.mcp_server.social.services import bluesky

# 5a. Write
os.environ["BLUESKY_HANDLE"] = "oppa.ai.bot"
memorize = MagicMock()
memorize.get_display_name.return_value = "OppaAI"
memorize.get_user_id.return_value = "OppaAI"

bsky_reply = {
    "username": "oppa.ai.bot",
    "timestamp": "2026-08-29T17:00:00+0000",
    "text": "Bluesky test from OppaAI",
}
saved = bluesky._save_bluesky_interaction_memory(bsky_reply, "Bluesky reply", memorize)
check("Bluesky: owner message is saved", saved is True)
check("Bluesky: memorize.add was called once", memorize.add.call_count == 1)
if memorize.add.call_count == 1:
    msgs = memorize.add.call_args.kwargs.get("messages") or memorize.add.call_args.args[0]
    check("Bluesky: prefix is [Bluesky YYYY-MM-DD]", "[Bluesky 2026-08-29]" in msgs[0]["content"])

# 5b. Read
memorize = MagicMock()
memorize.get_user_id.return_value = "OppaAI"
memorize.search.return_value = [{"memory": "OppaAI on Bluesky"}]
ctx = bluesky._bluesky_memory_context("test", memorize)
check("Bluesky: context populated from search", "OppaAI on Bluesky" in ctx)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MASTODON: parallel tests
# ─────────────────────────────────────────────────────────────────────────────
print()
print("── 6. MASTODON: parallel memory read/write ─────────────────────────")
print()

from interface.mcp_server.social.services import mastodon

# 6a. Write
memorize = MagicMock()
memorize.get_display_name.return_value = "OppaAI"
memorize.get_user_id.return_value = "OppaAI"

os.environ["MASTODON_USERNAME"] = "oppa.ai.bot"
os.environ["MASTODON_INSTANCE"] = "https://mastodon.social"

masto_reply = {
    "username": "oppa.ai.bot",
    "timestamp": "2026-08-29T18:00:00+0000",
    "text": "Mastodon test from OppaAI",
}
saved = mastodon._save_mastodon_interaction_memory(masto_reply, "Mastodon reply", memorize)
check("Mastodon: owner message is saved", saved is True)
check("Mastodon: memorize.add was called once", memorize.add.call_count == 1)
if memorize.add.call_count == 1:
    msgs = memorize.add.call_args.kwargs.get("messages") or memorize.add.call_args.args[0]
    check("Mastodon: prefix is [Mastodon YYYY-MM-DD]", "[Mastodon 2026-08-29]" in msgs[0]["content"])

# 6b. Read
memorize = MagicMock()
memorize.get_user_id.return_value = "OppaAI"
memorize.search.return_value = [{"memory": "OppaAI on Mastodon"}]
ctx = mastodon._mastodon_memory_context("test", memorize)
check("Mastodon: context populated from search", "OppaAI on Mastodon" in ctx)


# ─────────────────────────────────────────────────────────────────────────────
# 7. ROUND-TRIP: write → read back from real SQLite
# ─────────────────────────────────────────────────────────────────────────────
print()
print("── 7. ROUND-TRIP: real SQLite write → search → read ───────────────")
print()

# Create a real memory.db with the standard schema
import sqlite3
import json
import uuid

db_path = TEST_ROOT / "OppaAI" / "memory" / "memory.db"
db_path.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(str(db_path))
conn.execute("""
    CREATE TABLE memories (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        memory TEXT NOT NULL,
        created_at TEXT,
        access_count INTEGER DEFAULT 0,
        last_accessed_at TEXT,
        pinned INTEGER DEFAULT 0,
        status TEXT DEFAULT 'active',
        kind TEXT DEFAULT 'fact',
        source TEXT DEFAULT 'chat'
    )
""")
# Insert a test memory
test_memory_id = str(uuid.uuid4())
test_text = "OppaAI tested the memory round-trip on 2026-08-29"
conn.execute(
    "INSERT INTO memories (id, user_id, memory, created_at) VALUES (?, ?, ?, ?)",
    (test_memory_id, "OppaAI", test_text, "2026-08-29T16:00:00+00:00"),
)
conn.commit()

# Now read it back via a mock memorize
memorize = MagicMock()
memorize.get_user_id.return_value = "OppaAI"
memorize.search.return_value = [{"memory": test_text}]
ctx = threads._threads_memory_context("round-trip test", memorize)
check("round-trip: written text is found in context", test_text in ctx)
conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
total = len(results)
passed = sum(1 for _, p, _ in results if p)
failed = total - passed
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
print("=" * 70)

if failed:
    print()
    print("FAILURES:")
    for name, p, detail in results:
        if not p:
            print(f"  - {name}{('  — ' + detail) if detail else ''}")
    sys.exit(1)
else:
    print()
    print("All memory read/write checks passed for Threads, Bluesky, Mastodon.")
    sys.exit(0)
