from pathlib import Path
import json
import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="approval-frontend")
app.mount("/shared", StaticFiles(directory=str(SHARED_DIR), html=True), name="studio-shared")


def _job_post_social_root() -> Path:
    override = os.getenv("JOB_POST_SOCIAL_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    from system.userspace import user_workspace_root
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
        # Lane D / save_single_job_draft writes:
        #   <root>/<date>/<category>/<slug>/draft.json  (3 levels)
        # Older layouts may use 1–2 levels. rglob covers all of them.
        for meta_path in base.rglob("draft.json"):
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

    drafts.sort(key=lambda d: d.get("created_at", ""), reverse=True)
    return drafts
