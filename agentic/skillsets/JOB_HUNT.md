---
id: JOB_HUNT
name: Job Hunt
summary: Search configured scrape-friendly job boards for postings matching a role and location, dedupe, and save a structured report.
triggers: job, jobs, hiring, job posting, job search, find me a job, openings, vacancy
tools: search_jobs, dedupe_postings, save_note
---
# Job Hunt

Find current job postings for a role, filtered by location, and save a
structured report. Defaults live in `<workspace>/agentic/skillsets/job_hunt.json`.

## Configuration

The `job_hunt.json` config file is looked up in this order:

1. `JOB_HUNT_CONFIG_PATH` environment variable (absolute or relative to workspace)
2. `<user_state>/skillsets/job_hunt.json` (per-user configuration)
3. `<workspace>/agentic/skillsets/job_hunt.json` (fallback)

You can configure:
- `default_location` - where to search (default: "San Francisco, CA, USA" for fallback)
- `nearby_locations` - list of nearby cities to include
- `radius_km` - search radius around locations
- `max_results` - max postings per search
- `max_age_days` - how far back to search (default: 30)
- `fallback_max_age_days` - fallback search window if no results (default: 60)
- `job_sites` - which job boards to search
- `default_job_type` - optional filter (full-time, contract, etc.)
- `include_remote` - include remote jobs (default: true)

## Steps

1. Extract the role/query from the user's prompt. Job type should come from
   the prompt when possible (for example full-time, contract, remote, junior).
2. If the user specified a location, pass it. Otherwise use the configured
   default location from `<user_state>/skillsets/job_hunt.json` (`Vancouver, BC, Canada`) rather
   than asking.
3. Call `search_jobs(query, location, max_results, max_age_days, job_type)`.
   This already runs `dedupe_postings` internally - do not call it again.
4. If zero results: widen `max_age_days` once using
   `fallback_max_age_days` from `<user_state>/skillsets/job_hunt.json` before telling the user
   nothing was found. Do not silently drop the location filter.
5. Format results as a table: Title, Organization, Employment Type,
   Salary, Location, Experience, Close Date, Posted, URL. Leave blank
   fields as "-", do not invent values.
6. Save the table with `save_note` (title: "jobs-\u003crole\u003e-\u003clocation\u003e-\u003cdate\u003e").
7. In `final_answer`, state the count found, the location/date filter
   used, and the note path. Do not paste the full table into the spoken
   answer if it's long - summarize top 3-5 and point to the saved note.

## Notes

- `close_date` is rarely published by these boards; it will usually be
  blank. Do not guess a close date.
- The default location is San Francisco, CA, USA in the fallback config.
  Your custom config at `<user_state>/skillsets/job_hunt.json` uses Vancouver, BC, Canada.
- Sources are configured in the config file and default to
  Greenhouse/Lever/Ashby/RemoteOK/WeWorkRemotely/Wellfound - chosen
  specifically because they do not block scripted access. Never substitute
  LinkedIn/Indeed/Glassdoor scraping for this workflow unless the tool is
  changed to use an approved API or user-provided export.
