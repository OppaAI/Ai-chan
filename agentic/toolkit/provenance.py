"""
toolkit/provenance.py

Domain-authority + recency scoring for fetched web content.

Design intent
-------------
websurf.py already has one "trust" signal: _apply_corroboration_bonus,
which boosts a chunk when a SECOND independent domain says the same thing.
This module adds the other two signals a careful human researcher uses:

  - AUTHORITY: is this domain a primary/official source for this topic
    (gov, standards bodies, academia, wire services) or an "official"
    site for the specific thing being asked about (the query names a
    company/product and the domain matches it)?
  - RECENCY: for queries that smell time-sensitive ("current", "latest",
    "2026", "still", a fast-moving topic), how old is this page?

Both return small additive bonuses in the same [-0.2, +0.35] range that
RESEARCH_AGREEMENT_BONUS already uses, so they compose the same way:

    adjusted = min(1.0, base_relevance + corroboration_bonus
                                        + authority_bonus
                                        + recency_bonus)

Nothing here does a second embedding pass or LLM call — it's all cheap
string/regex work, so it's fine to run on every candidate URL, including
ones that never end up being fetched (i.e. it can also be used to RANK
which URLs are worth fetching in the first place, not just to re-score
chunks after the fact).
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import importlib.util

# ── domain authority tiers ───────────────────────────────────────────────────

TIER1_SUFFIXES = (".gov", ".mil", ".int")
TIER1_DOMAINS = {
    "arxiv.org", "who.int", "un.org", "nist.gov", "sec.gov", "fda.gov",
    "cdc.gov", "europa.eu", "oecd.org",
}
TIER1_BONUS = 0.30

TIER2_SUFFIXES = (".edu",)
TIER2_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nature.com",
    "science.org", "nytimes.com", "wsj.com", "bloomberg.com",
    "washingtonpost.com", "theguardian.com", "npr.org", "economist.com",
}
TIER2_BONUS = 0.18

TIER3_DOMAINS = {
    "wikipedia.org", "github.com", "stackoverflow.com", "docs.python.org",
    "developer.mozilla.org",
}
TIER3_BONUS = 0.08

# Cheap "this looks like an SEO listicle farm" signal — a weak negative,
# not a blocklist. Real bad-source filtering should still happen upstream
# (harmful_content_safety-style checks), this is purely a quality nudge.
LOW_QUALITY_URL_PATTERNS = (
    r"/top-?\d+-", r"/best-\d+-", r"listicle", r"/\d+-best-",
)
LOW_QUALITY_BONUS = -0.12

AUTHORITY_BONUS_CAP = 0.32


def _registrable_domain(netloc: str) -> str:
    """Strip 'www.' and return the bare host — good enough for our purposes
    without a full public-suffix-list dependency."""
    netloc = netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _base_name(domain: str) -> str:
    """'openai.com' -> 'openai', 'docs.anthropic.com' -> 'anthropic'."""
    parts = domain.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return parts[0]


def _looks_like_official_site(domain: str, query: str) -> bool:
    """Heuristic 'official company/product site' detector: does the
    query mention a token matching this domain's base name? Deliberately
    generic (no hardcoded company list) — 'OpenAI pricing' boosts
    openai.com, 'Anthropic API docs' boosts anthropic.com or
    docs.anthropic.com, without enumerating every possible brand.

    Guardrails against false positives: the base name must be at least
    4 characters (so short/common words like 'io' or 'ai' as a base name
    don't trigger on unrelated queries) and must appear as a whole word
    in the query, not a substring match.
    """
    base = _base_name(domain)
    if len(base) < 4:
        return False
    return bool(re.search(rf"\b{re.escape(base)}\b", query, flags=re.IGNORECASE))


def authority_bonus(url: str, query: str = "") -> float:
    """Domain-authority bonus for one URL. Independent of page content —
    safe to call before fetching, e.g. to rank candidate URLs."""
    try:
        domain = _registrable_domain(urlparse(url).netloc)
    except Exception:
        return 0.0
    if not domain:
        return 0.0

    bonus = 0.0
    if domain in TIER1_DOMAINS or domain.endswith(TIER1_SUFFIXES):
        bonus = TIER1_BONUS
    elif domain in TIER2_DOMAINS or domain.endswith(TIER2_SUFFIXES):
        bonus = TIER2_BONUS
    elif domain in TIER3_DOMAINS:
        bonus = TIER3_BONUS

    if query and _looks_like_official_site(domain, query):
        bonus = max(bonus, TIER2_BONUS)  # official-for-this-query beats generic tiering

    path = urlparse(url).path.lower()
    if any(re.search(p, path) for p in LOW_QUALITY_URL_PATTERNS):
        bonus += LOW_QUALITY_BONUS

    return max(-0.2, min(AUTHORITY_BONUS_CAP, bonus))


# ── recency ───────────────────────────────────────────────────────────────

FRESHNESS_KEYWORDS = (
    "current", "currently", "latest", "still", "today", "now",
    "recent", "recently", "new", "update", "updated", "price", "pricing",
    "release", "version",
)

FRESHNESS_HALF_LIFE_DAYS = 45      # fast-moving query: news/prices/current status
STABLE_HALF_LIFE_DAYS = 720        # everything else: gentle preference for newer, not urgent
RECENCY_BONUS_CAP = 0.20


def query_looks_time_sensitive(query: str) -> bool:
    q = query.lower()
    if re.search(r"\b20\d{2}\b", q):
        return True
    return any(kw in q for kw in FRESHNESS_KEYWORDS)


def _parse_date(raw: str | None):
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y/%m/%d", "%Y"):
        try:
            dt = datetime.strptime(raw[: len(fmt.replace("%z", "+0000"))] if "%z" in fmt else raw[: len(fmt)], fmt)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except ValueError:
            continue
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def fetch_published_date(url: str, timeout: int = 6):
    """Best-effort page publish-date lookup via trafilatura's metadata
    extractor. Separate from research.py's web_fetch (which only returns
    body text) — this is an extra, optional call, meant to be used
    sparingly (e.g. only for the handful of URLs you're about to fetch
    in full, not for every search snippet). Returns a timezone-aware
    datetime or None if unavailable."""
    if importlib.util.find_spec("trafilatura") is None:
        return None
    trafilatura = __import__("trafilatura")
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        meta = trafilatura.extract_metadata(downloaded)
        raw_date = getattr(meta, "date", None) if meta else None
        return _parse_date(raw_date)
    except Exception:
        return None


def recency_bonus(published: "datetime | None", freshness_bias: bool) -> float:
    """Exponential decay bonus. freshness_bias=True (query_looks_time_sensitive)
    uses a short half-life so a month-old page scores meaningfully lower than
    a day-old one; otherwise a long half-life gives only a gentle nudge
    toward newer material without punishing evergreen sources."""
    if published is None:
        return 0.0
    age_days = max(0.0, (datetime.now(timezone.utc) - published).total_seconds() / 86400)
    half_life = FRESHNESS_HALF_LIFE_DAYS if freshness_bias else STABLE_HALF_LIFE_DAYS
    decay = 0.5 ** (age_days / half_life)
    return round(RECENCY_BONUS_CAP * decay, 4)


def source_quality_bonus(url: str, query: str, published=None, freshness_bias: bool | None = None) -> float:
    """Combined authority + recency bonus for one URL, capped like the
    corroboration bonus so no single signal can dominate the relevance score."""
    if freshness_bias is None:
        freshness_bias = query_looks_time_sensitive(query)
    total = authority_bonus(url, query) + recency_bonus(published, freshness_bias)
    return max(-0.25, min(0.45, total))


def rank_urls_by_quality(urls: list[str], query: str) -> list[str]:
    """Cheap pre-fetch ranking (authority only, no recency lookup — that
    needs a network call per URL, save it for the URLs you actually fetch).
    Use this to decide WHICH top-N candidates from a search-result page
    are worth spending a fetch on."""
    return sorted(urls, key=lambda u: authority_bonus(u, query), reverse=True)