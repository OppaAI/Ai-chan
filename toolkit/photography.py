"""
toolkit/photography.py

Photo-library tools for Aiko's wildlife/nature/astro workflows.

This module provides utilities for managing photo libraries:

  - scan_photo_workspace()       — scan inbox for ingestible image files
  - scan_video_workspace()       — scan inbox for postable video files
  - propose_photo_ingestion()    — suggest photos for library ingestion
  - write_photo_ingestion_report() — generate an ingestion summary report

Supports common RAW formats (CR2, CR3, NEF, ARW, ORF, RW2) and standard
image formats (JPEG, PNG, TIFF, WebP, HEIC, DNG), plus common video
containers (MP4, MOV, WebM, MKV, AVI, M4V).
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from itertools import islice
from pathlib import Path

from system.bioclock import local_now
from toolkit.common import json_block, now_stamp, safe_path, slugify, workspace_root

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
DEFAULT_PHOTO_INBOX = "photos/inbox"
DEFAULT_VIDEO_INBOX = "videos"
DEFAULT_PHOTO_REPORTS = "photos/reports"


def _files_with_extensions(root: Path, extensions: set[str], limit: int | None = None) -> list[Path]:
    if not root.exists():
        return []
    matches = (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    if limit is not None:
        matches = islice(matches, limit)
    return list(matches)


def _image_files(root: Path, limit: int | None = None) -> list[Path]:
    return _files_with_extensions(root, IMAGE_EXTENSIONS, limit)


def scan_photo_workspace(inbox: str = DEFAULT_PHOTO_INBOX, limit: int = 100) -> str:
    """Scan a workspace photo inbox for image files Aiko can ingest."""
    try:
        root = safe_path(inbox)
        ws_root = workspace_root()
        files = _image_files(root, max(1, min(limit, 1000)))
        by_ext = Counter(p.suffix.lower() for p in files)
        return json_block("photo workspace scan", {
            "inbox": str(root),
            "exists": root.exists(),
            "image_count": len(files),
            "by_extension": dict(sorted(by_ext.items())),
            "files": [str(p.relative_to(ws_root)) for p in files[:50]],
            "note": "Use propose_photo_ingestion to create a dry-run plan before moving or editing metadata.",
        })
    except Exception as e:
        return f"[photo scan failed: {e}]"


def scan_video_workspace(inbox: str = DEFAULT_VIDEO_INBOX, limit: int = 100) -> str:
    """Scan a workspace video inbox for video files Aiko can post.

    No ingestion/rating pipeline for video (unlike scan_photo_workspace) —
    this just enumerates postable files for skills/photo_social.py's
    video-posting queue.
    """
    try:
        root = safe_path(inbox)
        ws_root = workspace_root()
        files = _files_with_extensions(root, VIDEO_EXTENSIONS, max(1, min(limit, 1000)))
        by_ext = Counter(p.suffix.lower() for p in files)
        return json_block("video workspace scan", {
            "inbox": str(root),
            "exists": root.exists(),
            "video_count": len(files),
            "by_extension": dict(sorted(by_ext.items())),
            "files": [str(p.relative_to(ws_root)) for p in files[:50]],
        })
    except Exception as e:
        return f"[video scan failed: {e}]"


def propose_photo_ingestion(inbox: str = DEFAULT_PHOTO_INBOX, library_root: str = "photos/library", rating_rule: str = "manual-review-first") -> str:
    """Create a safe dry-run ingestion plan for untracked photos."""
    try:
        root = safe_path(inbox)
        ws_root = workspace_root()
        files = _image_files(root, 100)
        planned = []
        for path in files:
            rel = path.relative_to(ws_root)
            stem_slug = slugify(path.stem, fallback="photo")
            planned.append({
                "source": str(rel),
                "proposed_destination": f"{library_root.strip('/').rstrip('/')}/review/{stem_slug}{path.suffix.lower()}",
                "metadata_status": "pending VLM species/category/rating",
                "action": "dry_run_only",
            })
        return json_block("photo ingestion proposal", {
            "created_at": now_stamp(),
            "inbox": str(root),
            "library_root": library_root,
            "rating_rule": rating_rule,
            "count": len(files),
            "planned_files": planned,
            "safety": "No files were moved and no EXIF/XMP metadata was written.",
            "next_tools": ["write_photo_ingestion_report"],
        })
    except Exception as e:
        return f"[photo ingestion proposal failed: {e}]"


def write_photo_ingestion_report(title: str = "photo-ingestion", content: str = "", report_dir: str = DEFAULT_PHOTO_REPORTS) -> str:
    """Write a photo workflow report under the workspace report folder."""
    try:
        base = safe_path(report_dir)
        base.mkdir(parents=True, exist_ok=True)
        filename = f"{local_now().strftime('%Y%m%d-%H%M%S')}-{slugify(title, 'photo-ingestion')}.md"
        path = base / filename
        body = content.strip() or f"# Photo Ingestion Report\n\nCreated: {now_stamp()}\n\nNo details provided."
        path.write_text(body, encoding="utf-8")
        return json_block("photo report written", {"path": str(path), "chars": len(body)})
    except Exception as e:
        return f"[photo report failed: {e}]"
