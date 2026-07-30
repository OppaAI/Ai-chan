---
id: JOB_HUNT
name: Job Hunt
summary: Fetch configured RSS feeds, filter for tech jobs available today, and save structured human-reviewed Threads drafts using post_fields from job_hunt.json.
triggers: job, jobs, hiring, job posting, job search, openings, vacancy, job post, draft job, daily job
tools: search_jobs, gen_job_search_plan, execute_job_search_plan, draft_job_posts_from_results, save_or_post_job_drafts, report_job_run
---

# Job Hunt — RSS-only Lane D Playbook

The daily job post workflow runs as a draft-first graph. It fetches configured RSS feeds, keeps items dated today in the local bioclock timezone, filters by configurable tech keywords, dedupes by link/guid, and writes structured drafts (one per job) for human review.

**Draft layout is not hardcoded.** `format_job_post` reads `post_fields` and `post_signature` from `job_hunt.json` only. If `post_fields` is missing or empty, drafting fails with `missing_post_fields`.

No web-search, scraping, or multi-board fallback path is part of the current design.

## Graph nodes

```text
plan → RSS fetch → structured draft → save → report
```

### Node 1 — `gen_job_search_plan`
Reads `job_hunt.json` / env config and emits the RSS feed URLs, tech keywords, result cap, and default location.

### Node 2 — `execute_job_search_plan`
Fetches only the configured RSS feeds and returns postings that are:

- dated today in the local bioclock timezone,
- matched by `TECH_JOB_KEYWORDS` / `tech_job_keywords`, and
- deduped by link/guid.

### Node 3 — `draft_job_posts_from_results`
Creates one structured Threads draft **per job** via `format_job_post`, using `post_fields` and `post_signature` from the resolved `job_hunt.json`. Empty field values are skipped.

Example (when those keys are present in RSS data):

```text
Job Post - 2026-07-30
Organization: City of Vancouver
Position: Software Developer
Location: Canada

See details at:
https://example/job

- 𝘨𝘦𝘯'𝘥 𝘣𝘺 𝘈𝘪𝘬𝘰 (𝘖𝘱𝘱𝘰𝘈𝘐'𝘴 𝘈𝘐 𝘈𝘨𝘦𝘯𝘵)
```

Capped by `MAX_JOBS_PER_DRAFT` / `max_jobs_per_draft` (default 5). Full job descriptions are never copied into the draft. RSS sources often only supply title, org, and URL — other keys stay blank and are omitted.

### Node 4 — `save_or_post_job_drafts`
Saves each draft under `<job_post_root>/<date>/tech_jobs_today[/slug]/` with:

- `draft_post.txt` — structured post from `post_fields`
- `review.md` — human review checklist
- `draft.json` — metadata with `human_approved: false`

Posting happens only after the normal human approval gate via `post_job_post_draft` / `post_job_post_social`.

### Node 5 — `report_job_run`
Generates a compact audit report for the RSS run (includes resolved config path).

## Configuration

Config file lookup order:

1. `JOB_HUNT_CONFIG_PATH` env var (absolute, or relative to workspace)
2. `USER_SKILLSETS_PATH/job_hunt.json` if `USER_SKILLSETS_PATH` is set
3. `<USER_SPACE_ROOT>/<user_id>/skillsets/job_hunt.json` (per-user; first priority when no env overrides)
4. `<workspace>/agentic/skillsets/job_hunt.json` (repo fallback)

The folder you keep under user space is **`skillsets/`** — same as other per-user skillset overrides.

### Example config

```json
{
  "default_location": "Canada",
  "rss_feeds": [
    "https://www.civicjobs.ca/rss/region?id=9&region=Lower+Mainland+-+BC",
    "https://www.jobbank.gc.ca/jobsearch/feed/jobSearchRSSfeed?d=250&fage=2&mid=39070&sort=D&rows=100&fskl=%C2%AC15141&fcat=1"
  ],
  "tech_job_keywords": ["software", "developer", "cloud", "cybersecurity"],
  "max_results": 30,
  "max_jobs_per_draft": 5,
  "auto_post": false,
  "post_fields": [
    {"label": "Job Post - ", "key": "date"},
    {"label": "Organization: ", "key": "organization"},
    {"label": "Position: ", "key": "title"},
    {"label": "Type: ", "key": "employment_type"},
    {"label": "Location: ", "key": "location"},
    {"label": "Salary: ", "key": "salary"},
    {"label": "Experience: ", "key": "experience"},
    {"label": "Close: ", "key": "close_date"},
    {"label": "", "key": ""},
    {"label": "See details at:\n", "key": "url"}
  ],
  "post_signature": "- 𝘨𝘦𝘯'𝘥 𝘣𝘺 𝘈𝘪𝘬𝘰 (𝘖𝘱𝘱𝘰𝘈𝘐'𝘴 𝘈𝘐 𝘈𝘨𝘦𝘯𝘵)"
}
```
