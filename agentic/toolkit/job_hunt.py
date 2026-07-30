"""
toolkit/job_hunt.py

RSS-only Lane D tech-job drafting.

Lane D fetches configured CivicJobs.ca / Job Bank Canada RSS feeds, keeps
items dated today in the local bioclock timezone, filters by tech keywords,
dedupes by link/guid, and produces one teaser-list Threads draft for human
review. No web search or scraping path is kept in this module.
"""

from __future__ import annotations

import email.utils
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from defusedxml import ElementTree as ET

from agentic.registry import TOOLS, tool
from system.bioclock import local_now
from system.log import get_logger

log = get_logger(__name__)

DEFAULT_TECH_JOB_FEEDS = [
    "https://www.civicjobs.ca/rss/region?id=9&region=Lower+Mainland+-+BC",
    "https://www.jobbank.gc.ca/jobsearch/feed/jobSearchRSSfeed?d=250&fage=2&mid=39070&sort=D&rows=100&fskl=%C2%AC15141&fcat=1",
]
DEFAULT_TECH_JOB_KEYWORDS = [
    "software", "developer", "programmer", "engineer", "devops", "cloud",
    "data", "database", "systems", "network", "cybersecurity", "security",
    "it ", "information technology", "web", "frontend", "backend", "full stack",
    "qa", "quality assurance", "technical support",
]


def _job_config_path() -> Path:
    env_path = os.getenv("JOB_HUNT_CONFIG_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        return p if p.is_absolute() else Path(__file__).resolve().parents[2] / p
    try:
        from system.userspace import user_state_dir
        user_path = user_state_dir() / "skillsets" / "job_hunt.json"
        if user_path.exists():
            return user_path
    except Exception:
        log.warning("job_hunt: failed to resolve per-user config path")
    return Path(__file__).resolve().parents[2] / "agentic" / "skillsets" / "job_hunt.json"


def _job_config() -> dict[str, Any]:
    try:
        data = json.loads(_job_config_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _config_list(config: dict[str, Any], key: str, env_key: str, default: list[str]) -> list[str]:
    env = os.getenv(env_key, "").strip()
    if env:
        return [item.strip() for item in env.split(",") if item.strip()]
    value = config.get(key)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [item.strip() for item in value.split(",") if item.strip()]
    return default


def _max_days_back(config: dict[str, Any]) -> int:
    raw = os.getenv("JOB_HUNT_DATE_RANGE_DAYS", config.get("date_range_days", 1))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _max_jobs_per_draft(config: dict[str, Any]) -> int:
    raw = os.getenv("MAX_JOBS_PER_DRAFT", config.get("max_jobs_per_draft", 5))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 5


def _parse_rss_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=local_now().tzinfo)
    return dt.astimezone(local_now().tzinfo)


def _rss_text(element: ET.Element, names: tuple[str, ...]) -> str:
    for name in names:
        found = element.find(name) or element.find(f"{{*}}{name}")
        if found is not None and found.text:
            return found.text.strip()
    return ""


def _rss_link(entry: ET.Element) -> str:
    link = _rss_text(entry, ("link",))
    if link:
        return link
    link_el = entry.find("{*}link")
    return link_el.attrib.get("href", "").strip() if link_el is not None else ""


def _dedupe_key(link: str, guid: str) -> tuple[str, str]:
    return (link or guid).split("?", 1)[0].rstrip("/").casefold(), guid.casefold()


def fetch_today_tech_jobs_from_rss(config: dict[str, Any] | None = None) -> list[dict]:
    """Fetch configured RSS feeds, keeping tech postings from the last N days.

    N is controlled by JOB_HUNT_DATE_RANGE_DAYS env var or the date_range_days
    config key (default: 1 = today only). Widening this lets you catch jobs
    from the past week when a day's feed has no matches.
    """
    config = config or _job_config()
    feeds = _config_list(config, "rss_feeds", "TECH_JOB_RSS_FEEDS", DEFAULT_TECH_JOB_FEEDS)
    keywords = [kw.casefold() for kw in _config_list(config, "tech_job_keywords", "TECH_JOB_KEYWORDS", DEFAULT_TECH_JOB_KEYWORDS)]
    today = local_now().date()
    kept: list[dict] = []
    seen_ids: set[str] = set()
    for feed_url in feeds:
        try:
            resp = requests.get(feed_url, timeout=30, headers={"User-Agent": "Aiko-chan job RSS/1.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as e:
            log.warning("Lane D RSS feed fetch/parse failed for %s: %s", feed_url, e)
            continue
        entries = list(root.findall(".//item")) or list(root.findall(".//{*}entry"))
        for entry in entries:
            title = re.sub(r"\s+", " ", _rss_text(entry, ("title",))).strip()
            link = _rss_link(entry)
            guid = _rss_text(entry, ("guid", "id")) or link
            summary = _rss_text(entry, ("description", "summary"))
            org = _rss_text(entry, ("author", "creator"))
            posted = _parse_rss_datetime(_rss_text(entry, ("pubDate", "published", "updated")))
            max_days = _max_days_back(config)
            if not posted or posted.date() < today - timedelta(days=max_days - 1):
                continue
            if keywords and not any(kw in f"{title} {summary}".casefold() for kw in keywords):
                continue
            link_key, guid_key = _dedupe_key(link, guid)
            if link_key in seen_ids or guid_key in seen_ids:
                continue
            seen_ids.update({link_key, guid_key})
            kept.append({
                "title": title or "Untitled role",
                "organization": org,
                "url": link,
                "guid": guid,
                "posted_date": posted.isoformat(),
                "source_feed": feed_url,
                "_category": "tech_jobs_today",
            })
    return kept


@tool(TOOLS["search_jobs"])
def search_jobs(
    query: str = "",
    location: str = "",
    max_results: int | None = None,
    max_age_days: int | None = None,
    job_type: str = "",
) -> list[dict]:
    """Return today's tech jobs from configured RSS feeds; query args are ignored by design."""
    config = _job_config()
    limit = int(max_results or config.get("max_results", 30))
    return fetch_today_tech_jobs_from_rss(config)[:limit]


def gen_job_search_plan(prompt: str = "", config_source: str = "") -> str:
    """Node 1: Read RSS config into a Lane D execution plan."""
    config = _job_config()
    return json.dumps({
        "location": config.get("default_location", "Canada"),
        "queries": [{"category": "tech_jobs_today", "query": "tech jobs available today", "job_type": ""}],
        "max_results": int(config.get("max_results", 30)),
        "rss_feeds": _config_list(config, "rss_feeds", "TECH_JOB_RSS_FEEDS", DEFAULT_TECH_JOB_FEEDS),
        "tech_job_keywords": _config_list(config, "tech_job_keywords", "TECH_JOB_KEYWORDS", DEFAULT_TECH_JOB_KEYWORDS),
        "auto_post": bool(config.get("auto_post", False)),
    }, ensure_ascii=False)


def execute_job_search_plan(plan_json: str) -> str:
    """Node 2: Execute the Lane D RSS-only tech job search plan."""
    plan = json.loads(plan_json)
    config = _job_config()
    postings = fetch_today_tech_jobs_from_rss(config)
    max_results = int(plan.get("max_results") or config.get("max_results") or 30)
    result = {
        "location": plan.get("location", config.get("default_location", "Canada")),
        "total_found": len(postings[:max_results]),
        "queries_executed": ["rss_today_tech_jobs"],
        "sources": _config_list(config, "rss_feeds", "TECH_JOB_RSS_FEEDS", DEFAULT_TECH_JOB_FEEDS),
        "postings": postings[:max_results],
    }
    return json.dumps(result, ensure_ascii=False)


def draft_job_posts_from_results(results_json: str, template: str = "") -> str:
    """Node 3: Build one teaser-list Threads draft for today's tech RSS jobs."""
    results = json.loads(results_json)
    config = _job_config()
    postings = results.get("postings", [])
    if not postings:
        return json.dumps({"success": False, "reason": "no_tech_jobs_today", "drafts": []}, ensure_ascii=False)

    today = local_now().strftime("%Y-%m-%d")
    max_jobs = _max_jobs_per_draft(config)
    selected = postings[:max_jobs]
    date_str = today
    count = len(selected)

    template_str = template or config.get("draft_template") or os.getenv(
        "JOB_HUNT_DRAFT_TEMPLATE",
        "Tech jobs available today ({date}):\n{items}",
    )
    template_str = template_str.replace("\\n", "\n")

    item_lines = []
    for posting in selected:
        title = str(posting.get("title") or "Untitled role").strip()
        org = str(posting.get("organization") or "").strip()
        link = str(posting.get("url") or "").strip()
        label = f"{title} — {org}" if org else title
        item_lines.append(f"- {label}: {link}" if link else f"- {label}")

    if len(postings) > len(selected):
        item_lines.append(f"(+{len(postings) - len(selected)} more in this week's RSS results)")

    items = "\n".join(item_lines)
    text = template_str.format(date=date_str, items=items, count=count)
    return json.dumps({
        "success": True,
        "total_drafts": 1,
        "draft_policy": "tech_jobs_available_today",
        "max_jobs_per_draft": max_jobs,
        "location": results.get("location", ""),
        "date": today,
        "drafts": [{"text": text, "postings": selected, "category": "tech_jobs_today"}],
    }, ensure_ascii=False)


def save_or_post_job_drafts(drafts_json: str, auto_post: str = "false") -> str:
    """Node 4: Save one Lane D draft for human review; posting happens after approval."""
    from agentic.toolkit.social import job_post_social_root

    drafts_data = json.loads(drafts_json)
    if not drafts_data.get("drafts"):
        return json.dumps({"success": False, "reason": "no_drafts", "saved": []}, ensure_ascii=False)

    auto_requested = str(auto_post).lower() in {"true", "1", "yes", "on"}
    date_str = drafts_data.get("date", local_now().strftime("%Y-%m-%d"))
    base_dir = job_post_social_root() / date_str
    saved = []

    for i, draft in enumerate(drafts_data.get("drafts", [])):
        cat = draft.get("category", f"post_{i}")
        draft_dir = base_dir / cat
        draft_dir.mkdir(parents=True, exist_ok=True)
        text = draft.get("text", "").strip()
        (draft_dir / "draft_post.txt").write_text(text + "\n", encoding="utf-8")
        (draft_dir / "review.md").write_text(
            f"# Job Post Draft — {date_str} ({cat})\n\n"
            f"## Draft post\n\n{text}\n\n"
            "## Review checklist\n\n"
            "- [ ] Job teaser list is accurate\n"
            "- [ ] Links open to the source postings\n"
            "- [ ] Approved to post to Meta Threads\n",
            encoding="utf-8",
        )
        meta = {
            "success": True,
            "draft_dir": str(draft_dir),
            "provider": "threads",
            "postings": draft.get("postings"),
            "created_at": datetime.now().isoformat(),
            "posted": False,
            "human_approved": False,
        }
        (draft_dir / "draft.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        saved.append({"category": cat, "draft_dir": str(draft_dir), "auto_posted": False})

    return json.dumps({"success": True, "total_saved": len(saved), "auto_posted": False, "auto_post_requested": auto_requested, "saved": saved}, ensure_ascii=False)


def report_job_run(plan: str = "", search: str = "", draft: str = "", save: str = "") -> str:
    """Node 5: Generate an RSS Lane D audit report."""
    plan_data = json.loads(plan) if plan else {}
    search_data = json.loads(search) if search else {}
    draft_data = json.loads(draft) if draft else {}
    save_data = json.loads(save) if save else {}
    lines = ["# Job Post Run Report", "", "## RSS Lane D", ""]
    lines.append(f"- Feeds: {len(plan_data.get('rss_feeds', []))}")
    lines.append(f"- Results found today: {search_data.get('total_found', 0)}")
    lines.append(f"- Draft policy: {draft_data.get('draft_policy', draft_data.get('reason', 'n/a'))}")
    lines.append(f"- Drafts saved: {save_data.get('total_saved', 0)}")
    return "\n".join(lines)
