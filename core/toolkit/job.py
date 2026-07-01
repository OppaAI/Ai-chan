"""
core/toolkit/jobs.py
Job search: query scrape-friendly boards, filter by location + recency,
return structured postings.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from core.toolkit.web import web_search, fetch_and_extract

_RELATIVE_RE = re.compile(
    r"(?P<num>\d+)\s*(?P<unit>hour|day|week|month)s?\s+ago", re.IGNORECASE
)
_TODAY_RE = re.compile(r"\b(today|just posted|new)\b", re.IGNORECASE)
_SALARY_RE = re.compile(
    r"\$[\d,]+(?:\.\d+)?\s*(?:[-–to]+\s*\$?[\d,]+(?:\.\d+)?)?\s*(?:/(?:yr|hr|hour|year|mo|month))?",
    re.IGNORECASE,
)
_FTPT_RE = re.compile(
    r"\b(full[\s-]?time|part[\s-]?time|contract|internship|temporary|freelance)\b",
    re.IGNORECASE,
)
_EXP_RE = re.compile(
    r"\b(entry[\s-]?level|junior|mid[\s-]?level|senior|lead|principal|\d\+?\s*years?)\b",
    re.IGNORECASE,
)

# No hostile bot defenses; direct-hosted or search-engine-surfaced.
JOB_SITES = [
    "site:boards.greenhouse.io",
    "site:jobs.lever.co",
    "site:jobs.ashbyhq.com",
    "site:remoteok.com",
    "site:weworkremotely.com",
    "site:wellfound.com",
]


@dataclass
class JobPosting:
    title: str
    organization: str
    employment_type: str  # FT/PT/Contract/etc, "" if unknown
    salary: str            # "" if unlisted
    location: str
    experience: str        # "" if unlisted
    close_date: str        # "" if unlisted — most boards don't publish this
    posted_date: datetime | None
    url: str

    def to_row(self) -> dict:
        d = asdict(self)
        d["posted_date"] = self.posted_date.isoformat() if self.posted_date else ""
        return d


def _parse_relative_date(text: str) -> datetime | None:
    if _TODAY_RE.search(text):
        return datetime.now()
    m = _RELATIVE_RE.search(text)
    if not m:
        return None
    num = int(m.group("num"))
    unit = m.group("unit").lower()
    delta = {
        "hour": timedelta(hours=num),
        "day": timedelta(days=num),
        "week": timedelta(weeks=num),
        "month": timedelta(days=30 * num),
    }[unit]
    return datetime.now() - delta


def _extract_field(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text)
    return m.group(0).strip() if m else ""


def _location_matches(target: str, candidate: str) -> bool:
    """Loose match: target substring in candidate, or 'remote' if target says remote."""
    t = target.lower().strip()
    c = candidate.lower()
    if t in ("remote", "anywhere"):
        return "remote" in c
    return t in c


def search_jobs(
    query: str,
    location: str,
    max_results: int = 30,
    max_age_days: int = 30,
) -> list[dict]:
    """
    Search job boards for `query`, filtered to `location` (required — most
    important filter per user), keep postings within `max_age_days`,
    return up to `max_results` sorted newest-first.

    Fields returned per posting: title, organization, employment_type,
    salary, location, experience, close_date, posted_date, url.
    """
    postings: list[JobPosting] = []
    seen_urls: set[str] = set()

    for site in JOB_SITES:
        q = f"{site} {query} {location}"
        results = web_search(q)  # list[dict]: url/title/snippet expected

        for r in results:
            url = r.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            snippet = r.get("snippet", "") or ""
            title = r.get("title", "Unknown title")
            blob = f"{title} {snippet}"

            if not _location_matches(location, blob):
                # snippet too thin to tell -> fetch page before discarding
                try:
                    page = fetch_and_extract(url)
                    page_text = page.get("text", "")[:1500]
                except Exception:
                    page_text = ""
                if not _location_matches(location, page_text):
                    continue
                blob += " " + page_text

            posted = _parse_relative_date(blob)
            if posted is not None and posted < datetime.now() - timedelta(days=max_age_days):
                continue

            postings.append(
                JobPosting(
                    title=title,
                    organization=r.get("company", "") or _guess_org_from_url(url),
                    employment_type=_extract_field(_FTPT_RE, blob),
                    salary=_extract_field(_SALARY_RE, blob),
                    location=location,
                    experience=_extract_field(_EXP_RE, blob),
                    close_date="",  # rarely published; left for manual fill-in
                    posted_date=posted,
                    url=url,
                )
            )

    postings.sort(key=lambda p: p.posted_date or datetime.min, reverse=True)
    rows = dedupe_postings([p.to_row() for p in postings])
    return rows[:max_results]


def _guess_org_from_url(url: str) -> str:
    """boards.greenhouse.io/<org>/... or jobs.lever.co/<org>/... -> org name."""
    m = re.search(r"(?:greenhouse\.io|lever\.co|ashbyhq\.com)/([^/]+)", url)
    return m.group(1).replace("-", " ").title() if m else ""


_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _similarity(a: str, b: str) -> float:
    """Jaccard similarity on word sets. 0..1."""
    wa, wb = _normalize(a), _normalize(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def dedupe_postings(
    postings: list[dict],
    title_threshold: float = 0.7,
) -> list[dict]:
    """
    Collapse near-duplicate postings — same role re-posted across boards,
    or same URL with tracking params changed. Keeps the earliest-seen
    (i.e. first in list, since search_jobs already sorts newest-first)
    entry of each duplicate cluster.

    Two postings are duplicates if:
      - normalized URLs match, OR
      - organization matches (case-insensitive) AND title similarity
        >= title_threshold
    """
    kept: list[dict] = []

    def _norm_url(u: str) -> str:
        return u.split("?")[0].rstrip("/").lower()

    for posting in postings:
        p_url = _norm_url(posting.get("url", ""))
        p_org = (posting.get("organization") or "").strip().lower()
        p_title = posting.get("title", "")

        is_dup = False
        for existing in kept:
            e_url = _norm_url(existing.get("url", ""))
            if p_url and p_url == e_url:
                is_dup = True
                break
            e_org = (existing.get("organization") or "").strip().lower()
            if p_org and p_org == e_org:
                if _similarity(p_title, existing.get("title", "")) >= title_threshold:
                    is_dup = True
                    break

        if not is_dup:
            kept.append(posting)

    return kept
