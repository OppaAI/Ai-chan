"""Approval Studio backend — review and approve daily job post drafts."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import json

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


def _job_post_social_root() -> Path:
    import os
    from system.userspace import user_workspace_root
    override = os.getenv("JOB_POST_SOCIAL_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (user_workspace_root() / "social" / "job_posts").resolve()


def _scan_all_drafts() -> list[dict]:
    """Scan all job post drafts from the social root."""
    root = _job_post_social_root()
    drafts = []
    
    for meta_path in list(root.glob("*/*/draft.json")) + list(root.glob("*/draft.json")):
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
            "created_at": meta.get("created_at", ""),
            "llm_enriched": meta.get("llm_enriched", False),
            "provider": meta.get("provider", "threads"),
            "draft_text": draft_text,
            "posting": meta.get("posting", {}),
            "postings": meta.get("postings", []),
            "meta": meta,
        })
    
    # Sort by created_at descending
    drafts.sort(key=lambda d: d.get("created_at", ""), reverse=True)
    return drafts


@app.get("/api/drafts")
async def get_drafts(
    user_id: str | None = Query(None, description="User id (default: current_user_id)"),
    status: str | None = Query(None, description="Filter: pending, approved, posted, all"),
):
    """Get all job post drafts for review."""
    drafts = _scan_all_drafts()
    
    if status == "pending":
        drafts = [d for d in drafts if not d["human_approved"] and not d["posted"]]
    elif status == "approved":
        drafts = [d for d in drafts if d["human_approved"] and not d["posted"]]
    elif status == "posted":
        drafts = [d for d in drafts if d["posted"]]
    elif status == "all":
        pass
    else:
        # Default: show pending first, then approved, then posted
        drafts = [d for d in drafts if not d["human_approved"] and not d["posted"]] + \
                 [d for d in drafts if d["human_approved"] and not d["posted"]] + \
                 [d for d in drafts if d["posted"]]
    
    return {"drafts": drafts, "count": len(drafts)}


@app.get("/api/drafts/{draft_dir:path}")
async def get_draft_detail(draft_dir: str):
    """Get detailed information for a specific draft."""
    root = _job_post_social_root()
    draft_path = (root / draft_dir).resolve()
    
    # Security: ensure path is within root
    try:
        draft_path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid draft path")
    
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


@app.post("/api/drafts/{draft_dir:path}/approve")
async def approve_draft(draft_dir: str):
    """Mark a draft as human-approved."""
    root = _job_post_social_root()
    draft_path = (root / draft_dir).resolve()
    
    try:
        draft_path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid draft path")
    
    meta_path = draft_path / "draft.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Draft not found")
    
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    
    if meta.get("human_approved"):
        return {"success": True, "message": "Already approved", "meta": meta}
    
    meta["human_approved"] = True
    meta["approved_at"] = datetime.now().isoformat()
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    
    return {"success": True, "message": "Draft approved", "meta": meta}


@app.post("/api/drafts/{draft_dir:path}/reject")
async def reject_draft(draft_dir: str, request: Request):
    """Mark a draft as rejected (remove human_approved flag)."""
    root = _job_post_social_root()
    draft_path = (root / draft_dir).resolve()
    
    try:
        draft_path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid draft path")
    
    meta_path = draft_path / "draft.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Draft not found")
    
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    
    meta["human_approved"] = False
    meta["rejected_at"] = datetime.now().isoformat()
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    
    return {"success": True, "message": "Draft rejected", "meta": meta}


@app.get("/api/health")
async def health():
    return {"ok": True, "service": "approval-studio"}


@app.get("/")
async def serve_studio():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)