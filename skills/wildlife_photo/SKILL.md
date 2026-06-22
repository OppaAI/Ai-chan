---
id: wildlife_photo
name: Wildlife / Nature / Astro Photo Ingestion
summary: Process untracked wildlife, nature, or astro photos from Aiko's photo inbox with a safe scan, dry-run organization proposal, metadata plan, and ingestion report.
triggers: photo inbox, wildlife photos, nature photos, astro photos, untracked photos, ingest photos, categorize photos, rate photos
tools: scan_photo_workspace, propose_photo_ingestion, write_photo_ingestion_report, save_note, schedule_job
---
# Wildlife / Nature / Astro Photo Ingestion

Use this skill when Oppa asks Aiko to process wildlife, nature, or astro photos, or when a scheduled/watch task detects untracked images in the photo inbox.

## Default paths

- Photo inbox: `workspace/photos/inbox`
- Report folder: `workspace/photos/reports`
- Dry-run library root: `photos/library`

## Workflow

1. Run `scan_photo_workspace` on the inbox to count and list candidate image files.
2. If files exist, run `propose_photo_ingestion` before any destructive action.
3. Treat the first implementation as **dry-run only**:
   - do not delete originals;
   - do not overwrite EXIF/XMP metadata;
   - do not move files unless Oppa explicitly approves a future move tool.
4. Summarize what would happen:
   - total files;
   - extension counts;
   - proposed destination pattern;
   - missing VLM/species/rating steps;
   - files needing review.
5. Write a report with `write_photo_ingestion_report`.

## Future tool slots

Aiko may later gain tools for VLM species classification, quality scoring, EXIF/XMP writing, RAW+JPEG sidecar handling, and approved file moves. Until those tools exist, she must clearly say the workflow is a safe scan/proposal/report, not final ingestion.

## Safety rules

- Never delete source photos.
- Never modify metadata unless a dedicated metadata tool exists and Oppa explicitly asks for metadata writes.
- Preserve RAW/JPEG sidecar relationships.
- If species/category confidence is unknown, mark as `review`.
- Reports should be boring, factual, and useful.
