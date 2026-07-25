---
id: JOB_HUNT
name: Job Hunt
summary: Search configured job boards, filter, format, and save/post job listings through a graph playbook. Each step is a separate node — inspect, modify, or reorder independently.
triggers: job, jobs, hiring, job posting, job search, find me a job, openings, vacancy, job post, draft job, daily job
tools: search_searxng, parse_jobs, filter_jobs, format_job_post, dedupe_postings, gen_job_search_plan, execute_job_search_plan, draft_job_posts_from_results, save_or_post_job_drafts, report_job_run
---

# Job Hunt — Graph Playbook

The daily job post workflow runs as a **6-node graph** defined in the `daily_job_post` playbook (`agentic/schema.py`). Each node is a tool registered in the graph executor.

## Graph nodes

```
plan → search → draft → save → report
         │
     (parallel fan-out per query category)
```

### Node 1 — `gen_job_search_plan`
Reads `job_hunt.json` config + user prompt, emits a structured plan as JSON.

**Config keys used:**
- `queries` — list of `{category, query, job_type}` dicts
- `default_location` — fallback location
- `min_salary_hourly`, `min_salary_annual` — salary floors
- `max_age_days` — posting age limit
- `max_results` — max postings per search
- `post_template` — format string (see below)
- `auto_post` — whether to post without human review

**Prompt overrides** (parsed automatically):
- `in <city>` / `near <city>` / `vicinity of <city>` → location
- `higher than $X/hr` / `>$X/yr` → salary floor
- `auto post` / `no review` / `human review` → auto_post flag

### Node 2 — `execute_job_search_plan`
Takes the plan, searches each query category through SearXNG in parallel (one thread per category), applies configured salary/age/specialty filters, deduplicates, and returns structured results.

Performs fallback to default `JOB_SITES` (Greenhouse/Lever/Ashby/RemoteOK/WeWorkRemotely/Wellfound) when config sites return zero results.

### Node 3 — `draft_job_posts_from_results`
Formats each valid posting using the configured `post_template`. Template supports these placeholders:
`{date}`, `{organization}`, `{title}`, `{employment_type}`, `{location}`, `{salary}`, `{experience}`, `{close_date}`, `{url}`

Default template (bilingual Chinese + English):
```
Job Post - {date}
機構：{organization}
職位：{title}
類別：{employment_type}
地區：{location}
薪金：{salary}
經驗：{experience}
截止日期：{close_date}

*請入以下連結參看詳情
{url}
```

### Node 4 — `save_or_post_job_drafts`
Saves each draft under `<job_post_root>/<date>/<category>/` with:
- `draft_post.txt` — the formatted text
- `review.md` — human review checklist
- `draft.json` — metadata (including `human_approved: false`)

If `auto_post` is true, attempts to post to Meta Threads immediately.

### Node 5 — `report_job_run`
Generates a detailed audit report covering all 4 preceding steps, including errors.

## Configuration

Config file lookup order:
1. `JOB_HUNT_CONFIG_PATH` env var
2. `<user_state>/skillsets/job_hunt.json` (per-user)
3. `<workspace>/agentic/skillsets/job_hunt.json` (fallback)

### Example config in `<user_state>/skillsets/job_hunt.json`

```json
{
  "default_location": "Vancouver, BC, Canada",
  "nearby_locations": ["Burnaby, BC", "Richmond, BC"],
  "queries": [
    {"category": "tech", "query": "software engineer developer", "job_type": ""},
    {"category": "admin", "query": "administrative assistant office", "job_type": ""},
    {"category": "food_qa", "query": "food quality assurance inspector", "job_type": ""}
  ],
  "job_sites": [
    "site:boards.greenhouse.io",
    "site:ca.indeed.com"
  ],
  "min_salary_hourly": 20,
  "min_salary_annual": 45000,
  "max_age_days": 30,
  "max_results": 30,
  "include_remote": true,
  "auto_post": false,
  "post_template": "Job Post - {date}\n機構：{organization}\n職位：{title}\n..."
}
```

## Primitives (building blocks in `agentic/toolkit/job_hunt.py`)

| Function | What it does |
|---|---|
| `search_searxng(query)` | Bare SearXNG search, returns raw results |
| `parse_jobs(raw_results, location)` | Parse raw results into structured postings |
| `filter_jobs(postings, max_age, min_hr, min_yr)` | Filter by age/salary/specialty |
| `format_job_post(posting, template)` | Format posting as social text |
| `dedupe_postings(postings)` | Collapse near-duplicates by URL/title |
