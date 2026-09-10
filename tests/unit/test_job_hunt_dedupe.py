"""Lane D dedupe/idempotency regression tests.

Covers the 2026-09-09 job-hunt fixes:
- _dedupe_key keeps significant query params (CivicJobs ?id=) while
  stripping trackers/fragments,
- board fetchers (Greenhouse/Lever/Ashby) consult the cross-run ledger,
- save_single_job_draft is idempotent and refuses empty text,
- same-run merge dedupe collapses cross-posted roles.
"""
from __future__ import annotations

import json

import pytest

from agentic.workflows.job_hunt import toolset


@pytest.fixture(autouse=True)
def _user_env(tmp_path, monkeypatch):
    state_root = tmp_path / ".aiko"
    (state_root / "github_205369547" / "agentic" / "workflows" / "job_hunt").mkdir(parents=True)
    monkeypatch.setenv("USER_SPACE_ROOT", str(state_root))
    monkeypatch.setenv("AIKO_USER_ID", "github_205369547")


def test_dedupe_key_keeps_significant_query_params():
    lk, _ = toolset._dedupe_key("http://www.civicjobs.ca/jobs?id=116034", "")
    assert lk == "http://www.civicjobs.ca/jobs?id=116034"
    lk2, _ = toolset._dedupe_key("http://www.civicjobs.ca/jobs?id=116032", "")
    assert lk != lk2


def test_dedupe_key_strips_trackers_and_fragments():
    lk, _ = toolset._dedupe_key(
        "https://example.com/jobs/1?utm_source=linkedin&fbclid=ABC#section", "")
    assert lk == "https://example.com/jobs/1"
    lk2, _ = toolset._dedupe_key("https://example.com/jobs/1", "")
    assert lk == lk2


def _fake_board_response(payload):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    return Resp()


def _ashby_payload(job_id="abc-1", title="Software Engineer"):
    return {"jobs": [{
        "id": job_id, "title": title, "jobUrl": f"https://jobs.ashbyhq.com/x/{job_id}",
        "locationName": "Vancouver, BC", "publishedAt": "2026-09-09T10:00:00Z",
        "descriptionPlain": "software engineering python",
    }]}


def test_ashby_consults_ledger_and_saves(monkeypatch):
    cfg = {"ashby_source": {"company_tokens": ["x"]}, "job_keywords": [],
           "date_range_days": 30, "dedup_days": 3}
    monkeypatch.setattr(toolset, "_http_get_with_tls_fallback",
                        lambda *a, **k: _fake_board_response(_ashby_payload()))
    first = toolset.fetch_today_jobs_from_ashby(cfg, filter_keywords=False,
                                                filter_date=False)
    assert len(first) == 1
    # Second run sees the ledger and drops it.
    second = toolset.fetch_today_jobs_from_ashby(cfg, filter_keywords=False,
                                                 filter_date=False)
    assert second == []
    # ... unless dedup is explicitly off.
    third = toolset.fetch_today_jobs_from_ashby(cfg, filter_keywords=False,
                                                filter_date=False, filter_dedup=False)
    assert len(third) == 1


def test_save_draft_skips_existing_and_empty(monkeypatch):
    from agentic.toolkit import social

    class S:
        def __init__(self):
            self.data = {}
            self.runtime = {}

    posting = {"title": "Software Engineer",
               "url": "https://jobs.ashbyhq.com/x/abc-1", "guid": "ashby:x:abc-1"}
    state = S()
    state.data["job_drafts_list"] = [{"text": "hello", "posting": posting,
                                      "postings": [posting], "category": "software_engineer"}]
    first = json.loads(toolset.save_single_job_draft("false", state=state))
    assert first["success"] and not first.get("deduped")

    # Same posting again -> skipped, success, no new directory.
    state.data["job_drafts_list"] = [{"text": "hello", "posting": posting,
                                      "postings": [posting], "category": "software_engineer"}]
    n_before = len(list(social.job_post_social_root().rglob("draft.json")))
    second = json.loads(toolset.save_single_job_draft("false", state=state))
    n_after = len(list(social.job_post_social_root().rglob("draft.json")))
    assert second["success"] and second.get("deduped") is True
    assert n_after == n_before

    # Empty text is refused, never persisted.
    state.data["job_drafts_list"] = [{"text": "   ", "posting": posting,
                                      "postings": [posting], "category": "software_engineer"}]
    third = json.loads(toolset.save_single_job_draft("false", state=state))
    assert third["success"] is False and third["reason"] == "empty_draft_text"


def test_merge_dedupe_collapses_cross_posted_role():
    a = {"url": "https://jobs.ashbyhq.com/x/1", "guid": "ashby:x:1", "title": "SWE"}
    b = {"url": "https://boards.greenhouse.io/y/jobs/9", "guid": "greenhouse:y:9", "title": "SWE"}
    dup = {"url": "https://jobs.ashbyhq.com/x/1?utm_source=x", "guid": "other", "title": "SWE"}
    out = toolset._merge_dedupe_postings([a, b, dup])
    assert out == [a, b]
