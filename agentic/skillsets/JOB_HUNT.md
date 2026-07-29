---
id: JOB_HUNT
name: Job Hunt
summary: Fetch configured RSS feeds, filter for tech jobs available today, and save a single human-reviewed Threads teaser-list draft.
triggers: job, jobs, hiring, job posting, job search, openings, vacancy, job post, draft job, daily job
tools: search_jobs, gen_job_search_plan, execute_job_search_plan, draft_job_posts_from_results, save_or_post_job_drafts, report_job_run
---

# Job Hunt — RSS-only Lane D Playbook

The daily job post workflow runs as a draft-first graph. It fetches configured RSS feeds, keeps items dated today in the local bioclock timezone, filters by configurable tech keywords, dedupes by link/guid, and writes one teaser-list draft for human review.

No web-search, scraping, or multi-board fallback path is part of the current design.

## Graph nodes

```text
plan → RSS fetch → teaser draft → save → report
```

### Node 1 — `gen_job_search_plan`
Reads `job_hunt.json` / env config and emits the RSS feed URLs, tech keywords, result cap, and default location.

### Node 2 — `execute_job_search_plan`
Fetches only the configured RSS feeds and returns postings that are:

- dated today in the local bioclock timezone,
- matched by `TECH_JOB_KEYWORDS` / `tech_job_keywords`, and
- deduped by link/guid.

### Node 3 — `draft_job_posts_from_results`
Creates one Threads teaser-list draft:

```text
Tech jobs available today (YYYY-MM-DD):
- Title — Org: https://example/job
```

The list is capped by `MAX_JOBS_PER_DRAFT` / `max_jobs_per_draft` (default 5). Full job descriptions are never copied into the draft.

### Node 4 — `save_or_post_job_drafts`
Saves the draft under `<job_post_root>/<date>/tech_jobs_today/` with:

- `draft_post.txt` — teaser list only
- `review.md` — human review checklist
- `draft.json` — metadata with `human_approved: false`

Posting happens only after the normal human approval gate via `post_job_post_draft` / `post_job_post_social`.

### Node 5 — `report_job_run`
Generates a compact audit report for the RSS run.

## Configuration

Config file lookup order:

1. `JOB_HUNT_CONFIG_PATH` env var
2. `<user_state>/skillsets/job_hunt.json` (per-user)
3. `<workspace>/agentic/skillsets/job_hunt.json` (fallback)

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
  "auto_post": false
}
```
