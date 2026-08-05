"""Approval Studio backend — review and approve daily job post drafts."""
from __future__ import annotations

import json
import mimetypes
import os
import requests
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from system.config import load_config
load_config()

app = FastAPI(title="Aiko Approval Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")


def _job_post_social_root() -> Path:
    import os
    from system.userspace import user_workspace_root
    override = os.getenv("JOB_POST_SOCIAL_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (user_workspace_root() / "social" / "job_posts").resolve()


def _resolve_draft_path(draft_dir: str) -> Path:
    """Resolve a draft_dir param to a path inside the social root, or raise 403."""
    root = _job_post_social_root()
    draft_path = (root / draft_dir).resolve()
    try:
        draft_path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid draft path")
    return draft_path


def _scan_all_drafts() -> list[dict]:
    """Scan all job post drafts from the social root, including the
    rejected/ archive (tagged rejected: True, kept separate from the
    active pending/approved/posted set)."""
    root = _job_post_social_root()
    rejected_root = root / "rejected"
    drafts = []

    def _collect(base: Path, is_rejected: bool, skip_top: set[str] = frozenset()) -> None:
        for meta_path in list(base.glob("*/*/draft.json")) + list(base.glob("*/draft.json")):
            if skip_top and meta_path.relative_to(base).parts[0] in skip_top:
                continue  # e.g. root's own rejected/ subtree, collected separately below
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            draft_dir = meta_path.parent
            draft_post_path = draft_dir / "draft_post.txt"
            draft_text = ""
            if draft_post_path.exists():
                draft_text = draft_post_path.read_text(encoding="utf-8").strip()

            drafts.append({
                "draft_dir": str(draft_dir),
                "relative_path": str(draft_dir.relative_to(root)),
                "date": meta.get("date", ""),
                "category": meta.get("category", ""),
                "human_approved": meta.get("human_approved", False),
                "posted": meta.get("posted", False),
                "rejected": is_rejected,
                "created_at": meta.get("created_at", ""),
                "llm_enriched": meta.get("llm_enriched", False),
                "provider": meta.get("provider", "threads"),
                "draft_text": draft_text,
                "posting": meta.get("posting", {}),
                "postings": meta.get("postings", []),
                "meta": meta,
            })

    _collect(root, is_rejected=False, skip_top={"rejected"})
    if rejected_root.exists():
        _collect(rejected_root, is_rejected=True)

    # Sort by created_at descending
    drafts.sort(key=lambda d: d.get("created_at", ""), reverse=True)
    return drafts


@app.get("/api/drafts")
async def get_drafts(
    user_id: str | None = Query(None, description="User id (default: current_user_id)"),
    status: str | None = Query(None, description="Filter: pending, approved, posted, rejected, all"),
):
    """Get all job post drafts for review."""
    drafts = _scan_all_drafts()

    if status == "pending":
        drafts = [d for d in drafts if not d["human_approved"] and not d["posted"] and not d["rejected"]]
    elif status == "approved":
        drafts = [d for d in drafts if d["human_approved"] and not d["posted"] and not d["rejected"]]
    elif status == "posted":
        drafts = [d for d in drafts if d["posted"]]
    elif status == "rejected":
        drafts = [d for d in drafts if d["rejected"]]
    elif status == "all":
        pass
    else:
        # Default: show pending first, then approved, then posted, then rejected
        drafts = [d for d in drafts if not d["human_approved"] and not d["posted"] and not d["rejected"]] + \
                 [d for d in drafts if d["human_approved"] and not d["posted"] and not d["rejected"]] + \
                 [d for d in drafts if d["posted"]] + \
                 [d for d in drafts if d["rejected"]]

    return {"drafts": drafts, "count": len(drafts)}


@app.get("/api/drafts/{draft_dir:path}")
async def get_draft_detail(draft_dir: str):
    """Get detailed information for a specific draft."""
    draft_path = _resolve_draft_path(draft_dir)
    root = _job_post_social_root()

    meta_path = draft_path / "draft.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Draft not found")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    draft_post_path = draft_path / "draft_post.txt"
    draft_text = ""
    if draft_post_path.exists():
        draft_text = draft_post_path.read_text(encoding="utf-8").strip()

    review_md_path = draft_path / "review.md"
    review_text = ""
    if review_md_path.exists():
        review_text = review_md_path.read_text(encoding="utf-8").strip()

    return {
        "draft_dir": str(draft_path),
        "relative_path": str(draft_path.relative_to(root)),
        "draft_text": draft_text,
        "review_text": review_text,
        "meta": meta,
    }


@app.post("/api/drafts/{draft_dir:path}/toggle-approval")
async def toggle_approval(draft_dir: str, request: Request):
    """Toggle draft approval status - approve if not approved, reject if approved."""
    draft_path = _resolve_draft_path(draft_dir)

    meta_path = draft_path / "draft.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Draft not found")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if meta.get("human_approved"):
        # Currently approved, reject it
        meta["human_approved"] = False
        meta["rejected_at"] = datetime.now().isoformat()
        message = "Draft unapproved"
    else:
        # Currently not approved, approve it
        meta["human_approved"] = True
        meta["approved_at"] = datetime.now().isoformat()
        message = "Draft approved"

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {"success": True, "message": message, "meta": meta}


@app.post("/api/drafts/{draft_dir:path}/update-content")
async def update_content(draft_dir: str, request: Request):
    """Update the editable job post text for a draft (draft_post.txt)."""
    draft_path = _resolve_draft_path(draft_dir)

    meta_path = draft_path / "draft.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Draft not found")

    body = await request.json()
    new_text = body.get("draft_text")
    if new_text is None:
        raise HTTPException(status_code=400, detail="Missing draft_text")

    # A draft that's already been posted shouldn't be silently rewritten.
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("posted"):
        raise HTTPException(status_code=409, detail="Cannot edit a draft that has already been posted")

    draft_post_path = draft_path / "draft_post.txt"
    draft_post_path.write_text(new_text, encoding="utf-8")

    meta["edited_at"] = datetime.now().isoformat()
    meta["manually_edited"] = True
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {"success": True, "message": "Draft content updated", "meta": meta}


@app.post("/api/drafts/{draft_dir:path}/publish")
async def publish_draft(draft_dir: str):
    """Publish an approved draft to Meta Threads — posts immediately, no
    further confirmation step past this call. Refuses unless human_approved
    is true and the draft hasn't already been posted."""
    draft_path = _resolve_draft_path(draft_dir)

    meta_path = draft_path / "draft.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Draft not found")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("posted"):
        raise HTTPException(status_code=409, detail="Draft has already been posted")
    if not meta.get("human_approved"):
        raise HTTPException(status_code=409, detail="Draft must be approved before publishing")

    try:
        from agentic.toolkit.social import post_job_post_draft
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Could not import posting module: {e}")

    result = post_job_post_draft(draft_path)
    if not result.get("posted"):
        raise HTTPException(status_code=502, detail=result.get("error") or "Failed to post to Threads")

    updated_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else meta
    return {"success": True, "message": "Published to Threads", "result": result, "meta": updated_meta}


@app.post("/api/drafts/{draft_dir:path}/reject")
async def reject_draft(draft_dir: str):
    """Reject a draft — moves its folder into <job_post_social_root>/rejected/
    instead of deleting it, so rejected drafts stay auditable but drop out
    of the active pending/approved/posted list. No-op'd for already-posted
    drafts (those aren't rejectable)."""
    import shutil

    draft_path = _resolve_draft_path(draft_dir)
    if not draft_path.exists() or not draft_path.is_dir():
        raise HTTPException(status_code=404, detail="Draft not found")

    root = _job_post_social_root()
    meta_path = draft_path / "draft.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("posted"):
            raise HTTPException(status_code=409, detail="Cannot reject a draft that has already been posted")
        meta["rejected_at"] = datetime.now().isoformat()
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rejected_root = root / "rejected"
    relative = draft_path.relative_to(root)
    target = (rejected_root / relative).resolve()
    try:
        target.relative_to(rejected_root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid draft path")

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        # Same relative path already archived (e.g. same date+category
        # rejected twice) — disambiguate rather than clobber.
        target = target.parent / f"{target.name}_{datetime.now().strftime('%H%M%S')}"

    shutil.move(str(draft_path), str(target))
    return {"success": True, "message": "Draft rejected and moved to rejected/", "rejected_dir": str(target)}


@app.get("/api/fetch-url")
async def fetch_url(url: str = Query(..., description="URL to fetch content from")):
    """Fetch a URL and convert it to Markdown using MarkItDown.

    We fetch the page ourselves (with browser-like headers + a retry) rather
    than letting MarkItDown's convert(url) fetch it internally: several sites
    — jobbank.gc.ca among them — return 503 to MarkItDown's default request
    because it doesn't look like a real browser. Fetching with our own
    headers first and feeding the bytes into convert_stream() sidesteps that
    while still getting MarkItDown's HTML → Markdown conversion (headings,
    lists, links, tables all come through as real Markdown syntax).
    Requires: pip install markitdown
    """
    import io
    import time
    from urllib.parse import urlparse

    # Validate URL
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return {"content": "", "error": "Invalid URL"}
    except Exception:
        return {"content": "", "error": "Invalid URL"}

    try:
        from markitdown import MarkItDown
    except ImportError:
        return {
            "content": "",
            "error": "markitdown is not installed. Run: pip install markitdown",
        }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
    }

    resp = None
    last_err = ""
    for attempt in range(2):  # one retry — 503s from gov sites are often transient
        try:
            resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            if resp.status_code == 503 and attempt == 0:
                time.sleep(1.5)
                continue
            resp.raise_for_status()
            last_err = ""
            break
        except Exception as e:
            last_err = str(e)
            resp = None
            if attempt == 0:
                time.sleep(1.5)
                continue

    if resp is None:
        return {"content": "", "error": last_err or "Failed to fetch URL"}

    try:
        converter = MarkItDown()
        content_type = resp.headers.get("content-type", "").split(";", 1)[0].strip()
        ext = mimetypes.guess_extension(content_type) or ".html"
        result = converter.convert_stream(io.BytesIO(resp.content), file_extension=ext, url=url)
        content = (result.text_content or "").strip()
        if not content:
            content = "[Could not extract readable content]"
        return {"content": content, "error": None}
    except Exception as e:
        return {"content": "", "error": str(e)}


@app.get("/api/health")
async def health():
    return {"ok": True, "service": "approval-studio"}


@app.get("/")
async def serve_studio():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
