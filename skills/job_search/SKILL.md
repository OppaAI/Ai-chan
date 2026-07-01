---
id: job_search
name: Job Search
summary: Search scrape-friendly job boards for postings matching a role and location, dedupe, and save a structured report.
triggers: job, jobs, hiring, job posting, job search, find me a job, openings, vacancy
tools: search_jobs, dedupe_postings, save_note
---
# Job Search

Find current job postings for a role, filtered by location, and save a
structured report. Location is the most important filter — always confirm
it before searching if the user didn't give one.

## Steps

1. If the user did not specify a location, ask before searching. Do not
   default silently — a nationwide/remote search returns noise.
2. Call `search_jobs(query, location, max_results=30, max_age_days=30)`.
   This already runs `dedupe_postings` internally — do not call it again.
3. If zero results: widen `max_age_days` once (e.g. to 60) before telling
   the user nothing was found. Do not silently drop the location filter.
4. Format results as a table: Title, Organization, Employment Type,
   Salary, Location, Experience, Close Date, Posted, URL. Leave blank
   fields as "—", do not invent values.
5. Save the table with `save_note` (title: "jobs-<role>-<location>-<date>").
6. In `final_answer`, state the count found, the location/date filter
   used, and the note path. Do not paste the full table into the spoken
   answer if it's long — summarize top 3-5 and point to the saved note.

## Notes

- `close_date` is rarely published by these boards; it will usually be
  blank. Do not guess a close date.
- Sources are Greenhouse/Lever/Ashby/RemoteOK/WeWorkRemotely/Wellfound —
  chosen specifically because they don't block scripted access. Never
  substitute LinkedIn/Indeed/Glassdoor scraping for this workflow.
