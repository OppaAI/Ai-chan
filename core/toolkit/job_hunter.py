"""
core/toolkit/job_hunter.py

Job hunt: query configured scrape-friendly boards, filter by location +
recency, and return structured postings.

This module provides tools for searching and aggregating job postings
from configured job boards:

  - search_jobs()    — search configured job boards with location/type filters
  - dedupe_postings() — collapse near-duplicate listings by URL or similarity

Configuration lives in skills/job_hunt/job_hunt.json.

Supported boards: Greenhouse, Lever, Ashby, RemoteOK, We Work Remotely, Wellfound.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from core.bioclock import local_now
from core.toolkit.researcher import web_fetch, web_search

_RELATIVE_RE = re.compile(
    r"(?P<num>\d+)\s*(?P<unit>hour|day|week|month)s?\s+ago", re.IGNORECASE
)
_TODAY_RE = re.compile(r"\b(today|just posted|new)\b", re.IGNORECASE)
_SALARY_RE = re.compile(
    r"\$[\d,]+(?:\.\d+)?\s*(?:[-–to]+\s*\$?[\d,]+(?:\.\d+)?)?\s*(?:/(?:yr|hr|hour|year|mo|month))?",
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

JOB_SITES = [
    "site:boards.greenhouse.io",
    "site:jobs.lever.co",
    "site:jobs.ashbyhq.com",
    "site:remoteok.com",
    "site:weworkremotely.com",
    "site:wellfound.com",
]
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "skills" / "job_hunt" / "job_hunt.json"


@dataclass
class JobPosting:
    title: str
    organization: str
    employment_type: str
    salary: str
    location: str
    experience: str
    close_date: str
    posted_date: datetime | None
    url: str

    def to_row(self) -> dict:
        row = asdict(self)
        row["posted_date"] = self.posted_date.isoformat() if self.posted_date else ""
        return row


def _job_config() -> dict[str, Any]:
    default = {
        "default_location": "Vancouver, BC, Canada",
        "radius_km": 50,
        "nearby_locations": [],
        "max_results": 30,
        "max_age_days": 30,
        "fallback_max_age_days": 60,
        "job_sites": JOB_SITES,
        "default_job_type": "",
        "include_remote": True,
    }
    try:
        data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    if isinstance(data, dict):
        default.update(data)
    return default


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


def search_jobs(
    query: str,
    location: str = "",
    max_results: int | None = None,
    max_age_days: int | None = None,
    job_type: str = "",
) -> list[dict]:
    """
    Search configured job boards for `query`.

    Defaults come from `skills/job_hunt/job_hunt.json`, including
    Vancouver-area location aliases, result count, posting age, and source sites.
    """
    config = _job_config()
    max_results = int(max_results or config.get("max_results", 30))
    max_age_days = int(max_age_days or config.get("max_age_days", 30))
    job_type = (job_type or config.get("default_job_type") or "").strip()
    search_locations = _search_locations(config, location)
    aliases = search_locations[1:]
    sites = config.get("job_sites") or JOB_SITES

    postings: list[JobPosting] = []
    seen_urls: set[str] = set()

    for site in sites:
        for search_location in search_locations:
            search_query = " ".join(part for part in [str(site), query, job_type, search_location] if part)
            for result in web_search(search_query):
                url = result.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                snippet = result.get("snippet", "") or ""
                title = result.get("title", "Unknown title")
                blob = f"{title} {snippet}"

                if not _location_matches(search_location, blob, aliases):
                    try:
                        page = web_fetch(url)
                        page_text = page.get("text", "")[:1500]
                    except Exception:
                        page_text = ""
                    if not _location_matches(search_location, page_text, aliases):
                        continue
                    blob += " " + page_text

                posted = _parse_relative_date(blob)
                if posted is not None and posted < local_now() - timedelta(days=max_age_days):
                    continue

                postings.append(JobPosting(
                    title=title,
                    organization=result.get("company", "") or _guess_org_from_url(url),
                    employment_type=_extract_field(_FTPT_RE, blob),
                    salary=_extract_field(_SALARY_RE, blob),
                    location=search_location if search_location != "remote" else "Remote",
                    experience=_extract_field(_EXP_RE, blob),
                    close_date="",
                    posted_date=posted,
                    url=url,
                ))

    from datetime import datetime as _dt
    epoch_floor = _dt.min.replace(tzinfo=local_now().tzinfo)
    postings.sort(key=lambda posting: posting.posted_date or epoch_floor, reverse=True)
    rows = dedupe_postings([posting.to_row() for posting in postings])
    return rows[:max_results]


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


def dedupe_postings(postings: list[dict], title_threshold: float = 0.7) -> list[dict]:
    """Collapse near-duplicate postings by normalized URL or org/title similarity."""
    kept: list[dict] = []

    def norm_url(url: str) -> str:
        return url.split("?")[0].rstrip("/").lower()

    for posting in postings:
        posting_url = norm_url(posting.get("url", ""))
        posting_org = (posting.get("organization") or "").strip().lower()
        posting_title = posting.get("title", "")

        duplicate = False
        for existing in kept:
            existing_url = norm_url(existing.get("url", ""))
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
