---
id: JOB_HUNT
name: Job Hunt
summary: Fetch configured RSS feeds, filter for tech jobs available today, optionally enrich post_fields with an LLM from title/summary, and save structured human-reviewed Threads drafts using post_fields from job_hunt.json.
triggers: job, jobs, hiring, job posting, job search, openings, vacancy, job post, draft job, daily job
tools: search_jobs, fetch_all_sources_into_state, check_jobs_remaining, get_next_job, draft_single_job, save_single_job_draft, report_job_run
---

# Job Hunt — RSS-only Lane D Playbook

The daily job post workflow runs as a draft-first graph. It fetches configured RSS feeds, keeps items dated today in the local bioclock timezone, filters by configurable tech keywords, dedupes by link/guid, optionally enriches sparse fields with an LLM, and writes structured drafts (one per job) for human review.

**Draft layout is not hardcoded.** `format_job_post` reads `post_fields` and `post_signature` from `job_hunt.json` only. If `post_fields` is missing or empty, drafting fails with `missing_post_fields`.

**LLM field fill (optional).** When the graph executor injects `client`/`model` into `draft_single_job`, empty fillable keys (`organization`, `title`, `employment_type`, `location`, `salary`, `experience`, `close_date`) are extracted from each posting’s title + RSS summary. The model must not invent facts; unsupported keys stay blank and are omitted from the draft. Without an LLM, behavior is pure key mapping (RSS-only).

No web-search, scraping, or multi-board fallback path is part of the current design.

## Graph nodes

The playbook runs a fan-out DAG (`max_workers` parallel worker chains, default 2). Workers share a single fetch/state pass and pull jobs from shared state via `worker_id`.

```text
fetch_all → check_more ─→ get_job → draft_one → save_one ─┐
                      (\wN)  ↕ for each worker N            → report
                      (loops while "more")                  ↙
```

### Node 1 — `fetch_all_sources_into_state`
Fetches configured RSS feeds and optional email alerts, keeps items dated today in the local bioclock timezone, filters by `TECH_JOB_KEYWORDS` / `tech_job_keywords`, dedupes by link/guid, and persists the postings batch (with fetch timestamp) into graph state for the worker chains.

Same-day reruns reuse this disk cache for `cache_fetch_minutes` (default 30; override `JOB_HUNT_FETCH_CACHE_MINUTES`). The cache is auto-cleared when the playbook run finishes.

### Node — `check_jobs_remaining` / `get_next_job` (per worker)

Each worker N owns a `check_more_/get_job_/draft_one_/save_one_` chain:

- `check_jobs_remaining` — "more" if state still holds getabled postings else "done" (with a `max_visits` safety cap).
- `get_next_job` (`worker_id: wN`) — pops the next unfetched posting for its worker.
- `draft_single_job` (`job_json: $result:get_job_N`) — formats **one** structured Threads draft via `format_job_post` using `post_fields` / `post_signature` from the resolved `job_hunt.json`. Empty field values are skipped; LLM enrichment runs per job when `client`/`model` are present.
- `save_single_job_draft` (`draft_json: $result:draft_one_N`) — persists the draft dir.

Example (when those keys are present after RSS + optional LLM):

```text
Job Post - 2026-07-30
Organization: City of Vancouver
Position: Software Developer
Location: Canada

See details at:
https://example/job

- 𝘨𝘦𝘯'𝘥 𝘣𝘺 𝘈𝘪𝘬𝘰 (𝘖𝘱𝘱𝘰𝘈𝘐'𝘴 𝘈𝘐 𝘈𝘨𝘦𝘯𝘵)
```

Capped per source by `JOB_HUNT_MAX_RSS_POSTS` / `max_rss_posts` and `JOB_HUNT_MAX_EMAIL_POSTS` / `max_email_posts` (default 10 each). Full job descriptions are never copied into the draft. RSS sources often only supply title, org, and URL — other keys stay blank and are omitted unless the LLM can extract them from the summary.

### Node — `report_job_run`
Runs after all worker save-nodes complete. Generates a compact audit report for the run (covers resolved config path, fetched, drafted, saved counts).

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
  "max_rss_posts": 10,
  "max_email_posts": 10,
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
