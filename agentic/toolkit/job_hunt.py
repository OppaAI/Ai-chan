"""
toolkit/job_hunt.py

Job search primitive blocks and graph-level tools.

Primitives (building blocks used by graph nodes):
  - search_searxng()     — bare SearXNG search, returns raw results
  - parse_jobs()         — convert raw SearXNG results into structured postings
  - filter_jobs()        — filter by age, salary floor, specialty exclusion
  - format_job_post()    — format a posting as social-media text
  - dedupe_postings()    — collapse near-duplicate postings by URL/similarity

Graph-level tools (registered in schema._tool_map + agentic tools):
  - gen_job_search_plan()          — read config + prompt → job search plan
  - execute_job_search_plan()      — plan → search + parse + filter results
  - draft_job_posts_from_results() — results → formatted social drafts
  - save_or_post_job_drafts()      — save drafts / auto-post / flag for review
  - report_job_run()               — detailed audit text

Configuration lives in agentic/skillsets/job_hunt.json (or per-user override).
"""

from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from agentic.registry import tool
from system.bioclock import local_now
from agentic.toolkit.websurf import MAX_RESULTS, web_fetch, fetch_search_results

_RELATIVE_RE = re.compile(
    r"(?P<num>\d+)\s*(?P<unit>hour|day|week|month)s?\s+ago", re.IGNORECASE,
)
_TODAY_RE = re.compile(r"\b(today|just posted|new)\b", re.IGNORECASE)
_SALARY_RE = re.compile(
    r"[\$€£¥][\d,]+(?:\.\d+)?\s*(?:[-–to]+\s*[\$€£¥]?[\d,]+(?:\.\d+)?)?\s*(?:/(?:yr|year|hr|hour|wk|week|mo|month))?",
    re.IGNORECASE,
)

_YEARS_EXP_RE = re.compile(
    r"(\d+[\+]?(?:\s*[-–to]+\s*\d+)?)\s*(?:year|yr)s?\b",
    re.IGNORECASE,
)
_FTPT_RE = re.compile(
    r"\b(full[\s-]?time|part[\s-]?time|contract|internship|temporary|freelance|remote)\b",
    re.IGNORECASE,
)
_EXP_RE = re.compile(
    r"\b(entry[\s-]?level|junior|mid[\s-]?level|senior|lead|principal|\d\+?\s*years?)\b",
    re.IGNORECASE,
)
_SPECIALTY_RE = re.compile(
    r"\b(senior|lead|principal|manager|director|nurse|doctor|engineer iii|security clearance)\b",
    re.IGNORECASE,
)

JOB_SITES = [
    "site:boards.greenhouse.io",
    "site:jobs.lever.co",
    "site:jobs.ashbyhq.com",
    "site:linkedin.com/jobs",
    "site:indeed.com",
    "site:glassdoor.com",
    "site:ziprecruiter.com",
    "site:simplyhired.com",
    "site:craigslist.org",
    "site:remoteok.com",
    "site:weworkremotely.com",
    "site:wellfound.com",
]


# ── Config ──

def _job_config_path() -> Path:
    env_path = os.getenv("JOB_HUNT_CONFIG_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_absolute():
            return p
        return Path(__file__).resolve().parents[2] / p
    try:
        from system.userspace import user_state_dir
        user_path = user_state_dir() / "skillsets" / "job_hunt.json"
        if user_path.exists():
            return user_path
    except Exception:
        pass
    return Path(__file__).resolve().parents[2] / "agentic" / "skillsets" / "job_hunt.json"


def _job_config() -> dict[str, Any]:
    config_path = _job_config_path()
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


# ── Internal helpers ──

def _search_locations(config: dict[str, Any], location: str) -> list[str]:
    primary = (location or config.get("default_location") or "").strip()
    locations = [primary] if primary else []
    locations.extend(str(item).strip() for item in config.get("nearby_locations", []) if str(item).strip())
    if config.get("include_remote", True):
        locations.append("remote")
    seen: set[str] = set()
    unique: list[str] = []
    for item in locations:
        key = item.casefold()
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _parse_relative_date(text: str) -> datetime | None:
    if _TODAY_RE.search(text):
        return local_now()
    match = _RELATIVE_RE.search(text)
    if not match:
        return None
    num = int(match.group("num"))
    unit = match.group("unit").lower()
    delta = {
        "hour": timedelta(hours=num),
        "day": timedelta(days=num),
        "week": timedelta(weeks=num),
        "month": timedelta(days=30 * num),
    }[unit]
    return local_now() - delta


def _extract_field(pattern: re.Pattern, text: str) -> str:
    match = pattern.search(text)
    return match.group(0).strip() if match else ""


def _location_matches(target: str, candidate: str, aliases: list[str]) -> bool:
    target = target.lower().strip()
    candidate = candidate.lower()
    if target in ("remote", "anywhere"):
        return "remote" in candidate
    checks = [target, *(alias.lower().strip() for alias in aliases)]
    return any(check and check in candidate for check in checks)


def _guess_org_from_url(url: str) -> str:
    match = re.search(r"(?:greenhouse\.io|lever\.co|ashbyhq\.com)/([^/]+)", url)
    return match.group(1).replace("-", " ").title() if match else ""


_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _similarity(a: str, b: str) -> float:
    words_a, words_b = _normalize(a), _normalize(b)
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


# ── Primitives ──

def search_searxng(query: str, max_results: int = 10) -> list[dict]:
    """Search SearXNG and return raw result dicts."""
    raw_results, _err = fetch_search_results(query, max_results)
    return raw_results or []


def parse_jobs(raw_results: list[dict], location: str = "",
               loc_aliases: list[str] | None = None) -> list[dict]:
    """Convert raw SearXNG results into structured job posting dicts."""
    if not raw_results:
        return []
    aliases = loc_aliases or []
    postings: list[dict] = []
    for result in raw_results:
        url = result.get("url", "")
        if not url:
            continue
        snippet = result.get("content", "") or result.get("snippet", "") or ""
        title = result.get("title", "Unknown title")
        blob = f"{title} {snippet}"
        if location and not _location_matches(location, blob, aliases):
            try:
                page = web_fetch(url)
                page_text = page[:1500]
            except Exception:
                page_text = ""
            if not _location_matches(location, page_text, aliases):
                continue
            blob += " " + page_text

        posted = _parse_relative_date(blob)
        postings.append({
            "title": title,
            "organization": result.get("company", "") or _guess_org_from_url(url),
            "employment_type": _extract_field(_FTPT_RE, blob),
            "salary": _extract_field(_SALARY_RE, blob),
            "location": location or "",
            "experience": _extract_field(_YEARS_EXP_RE, blob),
            "close_date": "",
            "posted_date": posted.isoformat() if posted else "",
            "url": url,
        })
    return postings


def filter_jobs(postings: list[dict], max_age_days: int = 30,
                min_salary_hourly: float = 20.0,
                min_salary_annual: float = 45000.0) -> list[dict]:
    """Filter postings by recency, salary floor, and specialty exclusion."""
    if not postings:
        return []
    now = local_now()
    cutoff = now - timedelta(days=max_age_days)
    kept: list[dict] = []
    for p in postings:
        posted_str = p.get("posted_date") or ""
        if posted_str:
            try:
                posted = datetime.fromisoformat(posted_str)
                if posted.tzinfo is None:
                    posted = posted.replace(tzinfo=now.tzinfo)
                if posted < cutoff:
                    continue
            except (ValueError, TypeError):
                pass
        salary = str(p.get("salary") or "")
        nums = [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", salary)]
        if nums:
            high = max(nums)
            text = salary.lower()
            if any(unit in text for unit in ("/hr", "hour", "hr")):
                if high < min_salary_hourly:
                    continue
            elif high > 1000 and high < min_salary_annual:
                continue
        blob = " ".join(str(p.get(k, "")) for k in ("title", "experience"))
        if _SPECIALTY_RE.search(blob):
            continue
        kept.append(p)
    return kept


_POST_FIELDS = [
    {"label": "Job Post - ", "key": "date"},
    {"label": "機構：", "key": "organization"},
    {"label": "職位：", "key": "title"},
    {"label": "類別：", "key": "employment_type"},
    {"label": "地區：", "key": "location"},
    {"label": "薪金：", "key": "salary"},
    {"label": "經驗：", "key": "experience"},
    {"label": "截止日期：", "key": "close_date"},
    {"label": "", "key": ""},
    {"label": "*請入以下連結參看詳情\n", "key": "url"},
]

_POST_SIGNATURE = ""


def format_job_post(posting: dict, template: str | None = None, date_text: str | None = None) -> str:
    """Format a single job posting, skipping empty fields."""
    config = _job_config()
    if date_text is None:
        date_text = local_now().strftime("%Y-%m-%d")

    fields = config.get("post_fields", _POST_FIELDS)
    signature = config.get("post_signature", _POST_SIGNATURE)

    def value_for(key: str) -> str:
        if key == "date":
            return date_text
        if key == "":
            return "\n"
        return str(posting.get(key, "") or "").strip()

    lines = []
    for fd in fields:
        key = fd.get("key", "")
        label = fd.get("label", "")
        val = value_for(key)
        if key == "":
            lines.append("")
        elif val:
            lines.append(f"{label}{val}")

    if signature:
        lines.append("")
        lines.append(signature)

    return "\n".join(lines).rstrip("\n")


def dedupe_postings(postings: list[dict], title_threshold: float = 0.7) -> list[dict]:
    """Collapse near-duplicate postings by normalized URL or org/title similarity."""
    kept: list[dict] = []
    for posting in postings:
        posting_url = posting.get("url", "").split("?")[0].rstrip("/").lower()
        posting_org = (posting.get("organization") or "").strip().lower()
        posting_title = posting.get("title", "")
        duplicate = False
        for existing in kept:
            existing_url = existing.get("url", "").split("?")[0].rstrip("/").lower()
            if posting_url and posting_url == existing_url:
                duplicate = True
                break
            existing_org = (existing.get("organization") or "").strip().lower()
            if posting_org and posting_org == existing_org:
                if _similarity(posting_title, existing.get("title", "")) >= title_threshold:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(posting)
    return kept


# ── Composed convenience (backwards-compatible) ──

@tool(
    name="search_jobs",
    description="Search configured job boards for a role. If location is omitted, uses the job_hunt skill default location. Deduped automatically.",
    props={"query": {"type": "string"}, "location": {"type": "string", "description": "Optional override. Defaults to the job_hunt skill location."}, "max_results": {"type": "integer"}, "max_age_days": {"type": "integer"}, "job_type": {"type": "string", "description": "Optional employment type filter from the user prompt, e.g. full-time, contract, remote."}},
    required=["query"],
    domain="jobs",
    react=False,
    graph=True,
)
def search_jobs(
    query: str,
    location: str = "",
    max_results: int | None = None,
    max_age_days: int | None = None,
    job_type: str = "",
) -> list[dict]:
    """Search configured job boards — convenience wrapper around primitives."""
    config = _job_config()
    max_results = int(max_results or config.get("max_results", 30))
    max_age_days = int(max_age_days or config.get("max_age_days", 30))
    job_type = (job_type or config.get("default_job_type") or "").strip()
    search_locations = _search_locations(config, location)
    aliases = search_locations[1:]
    sites = config.get("job_sites") or JOB_SITES
    seen_urls: set[str] = set()
    all_raw: list[dict] = []

    def _search(site_list: list[str]) -> None:
        for site in site_list:
            for sloc in search_locations:
                sq = " ".join(p for p in [str(site), query, job_type, sloc] if p)
                for r in search_searxng(sq, MAX_RESULTS):
                    u = r.get("url", "")
                    if u and u not in seen_urls:
                        seen_urls.add(u)
                        all_raw.append(r)

    _search(sites)
    if len(seen_urls) < max_results and sites is not JOB_SITES:
        _search(JOB_SITES)

    postings = parse_jobs(all_raw, search_locations[0] if search_locations else "", aliases)
    from datetime import datetime as _dt
    epoch_floor = _dt.min.replace(tzinfo=local_now().tzinfo)
    postings.sort(key=lambda p: (datetime.fromisoformat(p["posted_date"]) if p.get("posted_date") else epoch_floor), reverse=True)
    return dedupe_postings(postings)[:max_results]


# ── Graph-level tools ──

def gen_job_search_plan(prompt: str = "", config_source: str = "") -> str:
    """Node 1: Read config + prompt → structured job search plan JSON."""
    config = _job_config()
    prompt_lower = prompt.lower()
    queries = list(config.get("queries", []))
    location = config.get("default_location", "")
    for tok in ["in ", "near ", "around ", "vicinity of "]:
        if tok in prompt_lower:
            idx = prompt_lower.index(tok) + len(tok)
            remainder = prompt_lower[idx:].split(",")[0].strip()
            if remainder:
                location = remainder.title()

    min_salary_hourly = float(config.get("min_salary_hourly") or 0)
    min_salary_annual = float(config.get("min_salary_annual") or 0)
    salary_match = re.search(r"(?:higher than|gt|>|above|min)\s*\$?([\d,]+)\s*(k|/hr|/hour|/yr|/year)?", prompt_lower)
    if salary_match:
        val = float(salary_match.group(1).replace(",", ""))
        unit = (salary_match.group(2) or "").lower()
        if unit in ("/hr", "/hour"):
            min_salary_hourly = val
        elif unit in ("k",):
            min_salary_annual = val * 1000
        elif val > 1000:
            min_salary_annual = val
        else:
            min_salary_hourly = val

    auto_post = bool(config.get("auto_post", False))
    if "auto post" in prompt_lower or "autopost" in prompt_lower or "no review" in prompt_lower:
        auto_post = True
    if "human review" in prompt_lower or "review first" in prompt_lower or "approve" in prompt_lower:
        auto_post = False

    return json.dumps({
        "location": location,
        "nearby": config.get("nearby_locations", []),
        "queries": queries,
        "min_salary_hourly": min_salary_hourly,
        "min_salary_annual": min_salary_annual,
        "max_age_days": int(config.get("max_age_days", 30)),
        "max_results": int(config.get("max_results", 30)),
        "post_template": config.get("post_template", ""),
        "auto_post": auto_post,
    }, ensure_ascii=False)


# ── Direct job site search (bypasses SearXNG) ──

def _parse_job_html(html: str, domain: str) -> list[dict]:
    """Parse job listing URLs from a search results HTML page."""
    results: list[dict] = []
    for m in re.finditer(
        r'<a[^>]*href="(/[^"]*job[^"]*|[^"]*(?:career|job|position|posting|opening)[^"]*)"[^>]*>(.*?)</a>',
        html, re.IGNORECASE,
    ):
        href = m.group(1)
        title = re.sub(r'<[^>]+>', '', m.group(2)).strip()
        if title and len(title) > 10:
            url = f"https://{domain}{href}" if href.startswith("/") else href
            results.append({"title": title, "url": url, "content": "", "company": ""})
    return results


def _search_greenhouse(query: str, location: str, max_results: int) -> list[dict]:
    """Search Greenhouse jobs via their public JSON embed API."""
    try:
        resp = requests.get(
            "https://boards.greenhouse.io/embed/jobs",
            params={"title": query, "location": location},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    results = []
    for job in data.get("jobs", [])[:max_results]:
        results.append({
            "title": job.get("title", ""),
            "url": (job.get("absolute_url") or "").strip(),
            "content": job.get("location", {}).get("name", ""),
            "company": data.get("company", {}).get("name", ""),
        })
    return results


def _search_indeed(query: str, location: str, max_results: int) -> list[dict]:
    """Search Indeed Canada via HTML scraping."""
    ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    try:
        resp = requests.get(
            "https://ca.indeed.com/jobs",
            params={"q": query, "l": location},
            timeout=15, headers={"User-Agent": ua},
        )
        resp.raise_for_status()
        html = resp.text
    except Exception:
        return []
    return _parse_indeed_html(html, max_results)


def _parse_indeed_html(html: str, max_results: int) -> list[dict]:
    """Extract job listings from Indeed HTML."""
    results: list[dict] = []
    base = "https://ca.indeed.com"
    # Extract job cards via regex — matches Indeed's structure
    for m in re.finditer(
        r'data-jk="([^"]+)"[^>]*>.*?<a[^>]*class="[^"]*jcs-JobTitle[^"]*"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        html, re.DOTALL | re.IGNORECASE,
    ):
        jk = m.group(1)
        href = m.group(2)
        title = re.sub(r'<[^>]+>', '', m.group(3)).strip()
        url = base + href if href.startswith("/") else href
        results.append({"title": title, "url": url, "content": "", "company": ""})
        if len(results) >= max_results:
            break
    if not results:
        return _parse_job_html(html, "ca.indeed.com")
    return results


def _search_jobbank(query: str, location: str, max_results: int) -> list[dict]:
    """Search Job Bank Canada via HTML scraping."""
    ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    try:
        resp = requests.get(
            "https://jobbank.gc.ca/jobsearch/jobsearch.aspx",
            params={"Keywords": query, "LocationMulti": location, "f": "true"},
            timeout=15, headers={"User-Agent": ua},
        )
        resp.raise_for_status()
        html = resp.text
    except Exception:
        return []
    results: list[dict] = []
    for m in re.finditer(
        r'<a[^>]*href="(/jobsearch/jobdetails[^"]*)"[^>]*class="[^"]*job-title[^"]*"[^>]*>\s*(.*?)\s*</a>',
        html, re.DOTALL | re.IGNORECASE,
    ):
        href = m.group(1)
        title = re.sub(r'<[^>]+>', '', m.group(2)).strip()
        url = f"https://jobbank.gc.ca{href}" if href.startswith("/") else href
        results.append({"title": title, "url": url, "content": "", "company": ""})
        if len(results) >= max_results:
            break
    if not results:
        return _parse_job_html(html, "jobbank.gc.ca")
    return results


_DIRECT_JOB_HANDLERS: dict[str, callable] = {
    "boards.greenhouse.io": _search_greenhouse,
    "indeed.com": _search_indeed,
    "ca.indeed.com": _search_indeed,
    "jobbank.gc.ca": _search_jobbank,
}


def _direct_job_search(site: str, query: str, location: str, max_results: int) -> list[dict]:
    """Try to search a known job site directly, bypassing SearXNG."""
    domain = site[5:] if site.startswith("site:") else site
    for pattern, handler in _DIRECT_JOB_HANDLERS.items():
        if pattern in domain:
            return handler(query, location, max_results)
    return []


def execute_job_search_plan(plan_json: str) -> str:
    """Node 2: Execute a job search plan → search + parse + filter results JSON."""
    plan = json.loads(plan_json)
    location = plan.get("location", "")
    min_hr = float(plan.get("min_salary_hourly", 20))
    min_yr = float(plan.get("min_salary_annual", 45000))
    max_age = int(plan.get("max_age_days", 30))
    max_results = int(plan.get("max_results", 30))
    config = _job_config()
    config_default = config.get("default_location", "")
    config_nearby = config.get("nearby_locations", [])
    config_known = [config_default] + [str(n) for n in config_nearby]
    loc_lower = location.lower().strip()
    location_matches_config = False
    if loc_lower:
        for known in config_known:
            kl = known.lower()
            if kl and (kl in loc_lower or loc_lower in kl):
                location_matches_config = True
                break
    sites = config.get("job_sites") or JOB_SITES
    if not location_matches_config and loc_lower:
        sites = JOB_SITES
    search_locs = _search_locations(config, location)
    aliases = search_locs[1:] if len(search_locs) > 1 else []
    seen_urls: set[str] = set()
    all_raw: list[dict] = []
    lock = threading.Lock()
    query_list = plan.get("queries", [])

    def search_one(cat: str, query_text: str, job_type: str) -> None:
        local_raw: list[dict] = []
        for site in sites:
            for sloc in search_locs:
                raw = _direct_job_search(site, query_text, sloc, MAX_RESULTS)
                if not raw:
                    sq = " ".join(p for p in [str(site), query_text, job_type, sloc] if p)
                    raw = search_searxng(sq, MAX_RESULTS)
                for r in raw:
                    u = r.get("url", "")
                    if u:
                        with lock:
                            if u not in seen_urls:
                                seen_urls.add(u)
                                local_raw.append(r)
        with lock:
            all_raw.extend(local_raw)

    with ThreadPoolExecutor(max_workers=min(len(query_list), 8)) as pool:
        futures = {pool.submit(search_one, q.get("category", ""), q.get("query", ""), q.get("job_type", "")): q for q in query_list}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception:
                pass

    if not seen_urls and sites is not JOB_SITES:
        sites = JOB_SITES
        for q in query_list:
            search_one(q.get("category", ""), q.get("query", ""), q.get("job_type", ""))

    postings = parse_jobs(all_raw, search_locs[0] if search_locs else location, aliases)
    postings = filter_jobs(postings, max_age_days=max_age, min_salary_hourly=min_hr, min_salary_annual=min_yr)
    from datetime import datetime as _dt
    epoch_floor = _dt.min.replace(tzinfo=local_now().tzinfo)
    postings.sort(key=lambda p: (datetime.fromisoformat(p["posted_date"]) if p.get("posted_date") else epoch_floor), reverse=True)
    postings = dedupe_postings(postings)[:max_results]

    result = {
        "location": location,
        "total_found": len(postings),
        "queries_executed": [q.get("category") for q in query_list],
        "postings": postings,
    }
    return json.dumps(result, ensure_ascii=False)


def draft_job_posts_from_results(results_json: str, template: str = "") -> str:
    """Node 3: Format results into social media draft posts."""
    results = json.loads(results_json)
    postings = results.get("postings", [])
    if not postings:
        return json.dumps({"success": False, "reason": "no_postings", "drafts": []}, ensure_ascii=False)

    loc = results.get("location", "")
    today = local_now().strftime("%Y-%m-%d")
    drafts = []
    for posting in postings:
        text = format_job_post(posting, date_text=today)
        drafts.append({
            "text": text,
            "posting": posting,
            "category": posting.get("_category", "general"),
        })
    return json.dumps({
        "success": True,
        "total_drafts": len(drafts),
        "location": loc,
        "date": today,
        "drafts": drafts,
    }, ensure_ascii=False)


def save_or_post_job_drafts(drafts_json: str, auto_post: str = "false") -> str:
    """Node 4: Save drafts to disk; auto-post if flag set, else flag for human review."""
    # Lazy import to avoid circular dependency (social.py imports from this module)
    from agentic.toolkit.social import job_post_social_root, _post_threads

    drafts_data = json.loads(drafts_json)
    if not drafts_data.get("drafts"):
        return json.dumps({"success": False, "reason": "no_drafts", "saved": []}, ensure_ascii=False)

    auto = str(auto_post).lower() in {"true", "1", "yes", "on"}
    date_str = drafts_data.get("date", local_now().strftime("%Y-%m-%d"))
    base_dir = job_post_social_root() / date_str
    saved = []

    for i, draft in enumerate(drafts_data.get("drafts", [])):
        cat = draft.get("category", f"post_{i}")
        draft_dir = base_dir / cat
        draft_dir.mkdir(parents=True, exist_ok=True)
        text = draft.get("text", "")

        draft_post_path = draft_dir / "draft_post.txt"
        draft_post_path.write_text(text.strip() + "\n", encoding="utf-8")

        review_path = draft_dir / "review.md"
        review_path.write_text(
            f"# Job Post Draft — {date_str} ({cat})\n\n"
            f"## Draft post\n\n{text}\n\n"
            "## Review checklist\n\n"
            "- [ ] Job details look correct\n"
            "- [ ] Salary is acceptable\n"
            "- [ ] Approved to post to Meta Threads\n",
            encoding="utf-8",
        )

        meta = {
            "success": True,
            "draft_dir": str(draft_dir),
            "provider": "threads",
            "posting": draft.get("posting"),
            "created_at": datetime.now().isoformat(),
            "posted": False,
            "human_approved": False,
        }

        if auto:
            try:
                result = _post_threads(text, None)
                if result.get("ok"):
                    meta["posted"] = True
                    meta["post_results"] = [result]
                    (draft_dir / "posted.json").write_text(
                        json.dumps({"posted": True, "results": [result]}, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
            except Exception as e:
                meta["post_error"] = str(e)

        meta_path = draft_dir / "draft.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        saved.append({"category": cat, "draft_dir": str(draft_dir), "auto_posted": meta.get("posted", False)})

    return json.dumps({
        "success": True,
        "total_saved": len(saved),
        "auto_posted": auto,
        "saved": saved,
    }, ensure_ascii=False)


def report_job_run(plan: str = "", search: str = "", draft: str = "", save: str = "") -> str:
    """Node 5: Generate a detailed step-by-step audit report."""
    lines = ["# Job Post Run Report", ""]

    plan_data = json.loads(plan) if plan else {}
    search_data = json.loads(search) if search else {}
    draft_data = json.loads(draft) if draft else {}
    save_data = json.loads(save) if save else {}

    lines.append("## Step 1 — Generate Search Plan")
    if plan_data:
        lines.append(f"- Location: {plan_data.get('location', 'N/A')}")
        lines.append(f"- Queries: {len(plan_data.get('queries', []))}")
        for q in plan_data.get("queries", []):
            lines.append(f"  - {q.get('category')}: {q.get('query')}")
        lines.append(f"- Min salary: ${plan_data.get('min_salary_hourly', '?')}/hr or ${plan_data.get('min_salary_annual', '?')}/yr")
        lines.append(f"- Auto-post: {plan_data.get('auto_post', False)}")
    else:
        lines.append("- ERROR: No plan data")
    lines.append("")

    lines.append("## Step 2 — Execute Search")
    if search_data:
        lines.append(f"- Total found: {search_data.get('total_found', 0)}")
        lines.append(f"- Queries executed: {search_data.get('queries_executed', [])}")
        for p in search_data.get("postings", [])[:5]:
            lines.append(f"  - {p.get('title')} @ {p.get('organization')} ({p.get('location')})")
        if len(search_data.get("postings", [])) > 5:
            lines.append(f"  - ... and {len(search_data['postings']) - 5} more")
    else:
        lines.append("- ERROR: No search results")
    lines.append("")

    lines.append("## Step 3 — Draft Posts")
    if draft_data:
        lines.append(f"- Total drafts: {draft_data.get('total_drafts', 0)}")
    else:
        lines.append("- ERROR: No drafts generated")
    lines.append("")

    lines.append("## Step 4 — Save / Post")
    if save_data:
        lines.append(f"- Saved: {save_data.get('total_saved', 0)}")
        lines.append(f"- Auto-posted: {save_data.get('auto_posted', False)}")
        for s in save_data.get("saved", []):
            lines.append(f"  - {s.get('category')}: {s.get('draft_dir')} (posted: {s.get('auto_posted')})")
    else:
        lines.append("- ERROR: No save data")
    lines.append("")

    errors: list[str] = []
    if not plan_data:
        errors.append("Step 1 failed: no plan")
    if not search_data or not search_data.get("postings"):
        errors.append("Step 2: no postings found")
    if not draft_data or not draft_data.get("drafts"):
        errors.append("Step 3: no drafts created")
    if not save_data or not save_data.get("saved"):
        errors.append("Step 4: nothing saved")

    if errors:
        lines.append("## Errors")
        for e in errors:
            lines.append(f"- {e}")

    lines.append("")
    lines.append("---")
    lines.append("*Report generated by job_hunt.report_job_run*")
    return "\n".join(lines)
