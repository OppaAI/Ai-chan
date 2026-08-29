"""Tests for Lane D per-user config priority and sender-org refinement."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _user_env(tmp_path, monkeypatch):
    """Point USER_SPACE_ROOT at a temp dir with a per-user job_hunt config."""
    state_root = tmp_path / ".aiko"
    user_dir = state_root / "github_205369547"
    cfg_dir = user_dir / "agentic" / "workflows" / "job_hunt"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.json").write_text(json.dumps({
        "rss_feeds": ["https://example.test/civic.xml"],
        "job_keywords": ["software"],
        "topic_tag": "溫哥華溫哥華溫哥華",
        "post_fields": [
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
        ],
    }, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("USER_SPACE_ROOT", str(state_root))
    monkeypatch.setenv("AIKO_USER_ID", "github_205369547")
    return cfg_dir


def test_graph_build_prefers_user_config(_user_env, monkeypatch):
    from agentic.workflows.job_hunt.graph import build_gen_job_post_graph

    graph = build_gen_job_post_graph()
    synth = next(n for n in graph.nodes if n.id == "synth")
    cfg = json.loads(synth.args["config_json"])
    assert cfg["topic_tag"] == "溫哥華溫哥華溫哥華"
    org_field = next(f for f in cfg["post_fields"] if f.get("key") == "organization")
    assert org_field["label"] == "機構："


def test_graph_build_falls_back_to_repo_config(tmp_path, monkeypatch):
    monkeypatch.delenv("USER_SPACE_ROOT", raising=False)
    monkeypatch.setenv("USER_SPACE_ROOT", str(tmp_path / "empty"))
    monkeypatch.setenv("AIKO_USER_ID", "github_205369547")
    from agentic.workflows.job_hunt.graph import build_gen_job_post_graph

    graph = build_gen_job_post_graph()
    synth = next(n for n in graph.nodes if n.id == "synth")
    cfg = json.loads(synth.args["config_json"])
    assert cfg["topic_tag"] == "California"


def test_job_cfg_override_applies_user_config(_user_env):
    from agentic.workflows.common import execution as ex

    baked = {"post_fields": [{"label": "Organization: ", "key": "organization"}], "topic_tag": "California"}
    merged = ex._job_cfg_override(baked)
    assert merged["topic_tag"] == "溫哥華溫哥華溫哥華"
    org_field = next(f for f in merged["post_fields"] if f.get("key") == "organization")
    assert org_field["label"] == "機構："


def test_job_cfg_override_ignores_non_job_workflows():
    from agentic.workflows.common import execution as ex

    aurora = {"workflow_id": "aurora_forecast", "summary": "x"}
    assert ex._job_cfg_override(aurora) is aurora


def test_missing_fillable_treats_sender_org_as_missing():
    from agentic.workflows.job_hunt.toolset import _missing_fillable

    keys = ["organization", "title", "salary"]
    posting = {"organization": "jobalerts-noreply", "title": "AI Engineer", "salary": ""}
    missing = _missing_fillable(posting, keys)
    assert "organization" in missing
    assert "salary" in missing
    assert "title" not in missing


def test_missing_fillable_real_org_not_missing():
    from agentic.workflows.job_hunt.toolset import _missing_fillable

    keys = ["organization", "title"]
    posting = {"organization": "City of Vancouver", "title": "Clerk"}
    assert _missing_fillable(posting, keys) == []


def test_email_extraction_clears_sender_placeholder_org(_user_env):
    from agentic.workflows.job_hunt.toolset import _extract_jobs_from_cleaned_email

    body = (
        "[AI Engineer, Entry Level (Canada)](https://www.linkedin.com/comm/jobs/view/4454204127/)\n"
        "Remote\n$80-$90 / hour\n"
    )
    jobs = _extract_jobs_from_cleaned_email(
        body, sender="jobalerts-noreply@linkedin.com", subject="Job alert", config={}
    )
    assert jobs, "expected at least one extracted job"
    assert jobs[0]["organization"] == ""



def test_greenhouse_board_tokens_parse_config_and_urls():
    from agentic.workflows.job_hunt.toolset import _greenhouse_board_tokens

    cfg = {
        "greenhouse_source": {
            "board_tokens": [
                "greenhouse",
                "https://boards.greenhouse.io/exampleco",
                "https://boards-api.greenhouse.io/v1/boards/acme/jobs?content=true",
            ]
        }
    }

    assert _greenhouse_board_tokens(cfg) == ["greenhouse", "exampleco", "acme"]


def test_fetch_one_greenhouse_board_uses_rss_style_config(monkeypatch, tmp_path):
    from agentic.workflows.job_hunt import toolset

    monkeypatch.setattr(toolset, "_job_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(toolset, "_cache_is_fresh_simple", lambda date_str, config: False)
    monkeypatch.setattr(toolset, "fetch_today_jobs_from_greenhouse", lambda cfg, filter_keywords, filter_date: [
        {"title": "Software Engineer", "summary": "Python APIs", "url": "https://example.test/1"},
        {"title": "Store Manager", "summary": "Retail", "url": "https://example.test/2"},
    ])

    postings, info, failure = toolset._fetch_one_greenhouse_board(
        "2026-08-28",
        {"job_keywords": ["software"], "max_rss_posts": 10},
        False,
        0,
        "greenhouse",
    )

    assert failure is None
    assert info["type"] == "greenhouse"
    assert info["matched_count"] == 1
    assert postings[0]["_source_name"] == "greenhouse_0"
    assert (tmp_path / "fetch_2026-08-28_greenhouse_0.jsonl").exists()


def test_greenhouse_api_workers_isolate_configured_tokens(monkeypatch, tmp_path):
    from agentic.workflows.job_hunt import toolset

    configured_tokens = ["configured-one", "configured-two"]
    config = {
        "greenhouse_source": {"board_tokens": configured_tokens},
        "job_keywords": ["software"],
        "max_greenhouse_posts": 10,
    }
    received_tokens = []

    def fetcher(cfg, filter_keywords, filter_date, board_tokens=None):
        received_tokens.append(board_tokens)
        token = board_tokens[0]
        return [{
            "title": "Software Engineer",
            "summary": f"Role at {token}",
            "url": f"https://example.test/{token}",
        }]

    monkeypatch.setenv("GREENHOUSE_BOARD_TOKENS", "environment-one,environment-two")
    monkeypatch.setattr(toolset, "_job_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(toolset, "_cache_is_fresh_simple", lambda date_str, cfg: False)

    results = [
        toolset._fetch_one_api_board(
            "2026-08-29", config, False, "greenhouse", idx, token, fetcher
        )
        for idx, token in enumerate(configured_tokens)
    ]

    assert received_tokens == [["configured-one"], ["configured-two"]]
    assert [postings[0]["_source_name"] for postings, _, _ in results] == [
        "greenhouse_0",
        "greenhouse_1",
    ]
    assert [(info["index"], info["token"]) for _, info, _ in results] == [
        (0, "configured-one"),
        (1, "configured-two"),
    ]
    assert all(failure is None for _, _, failure in results)
    assert (tmp_path / "fetch_2026-08-29_greenhouse_0.jsonl").exists()
    assert (tmp_path / "fetch_2026-08-29_greenhouse_1.jsonl").exists()


def test_job_board_tokens_parse_lever_and_ashby_urls():
    from agentic.workflows.job_hunt.toolset import _job_board_tokens

    cfg = {
        "lever_source": {"company_tokens": ["https://jobs.lever.co/acme/role-id"]},
        "ashby_source": {"company_tokens": ["https://jobs.ashbyhq.com/example/role-id"]},
    }

    assert _job_board_tokens(cfg, "lever_source", ("NOPE",)) == ["acme"]
    assert _job_board_tokens(cfg, "ashby_source", ("NOPE",)) == ["example"]


def test_extract_linkedin_digest_cards_from_email_image_shape():
    from agentic.workflows.job_hunt.toolset import _extract_jobs_from_cleaned_email

    body = """
Your job alert for System Engineer
Systems Engineer - Infrastructure Operations
Electronic Arts (EA) · Vancouver, BC (Hybrid)
Actively recruiting
AI Solutions Engineer
Thales · Ottawa, ON (Hybrid)
Actively recruiting
System Administrator
Insight Global · Vancouver, BC (Hybrid)
CA$91K-CA$113K / year
"""

    jobs = _extract_jobs_from_cleaned_email(body, sender="jobs-noreply@linkedin.com", subject="Your job alert", config={})

    assert len(jobs) >= 3
    assert jobs[0]["title"] == "Systems Engineer - Infrastructure Operations"
    assert jobs[0]["organization"] == "Electronic Arts (EA)"
    assert jobs[0]["location"] == "Vancouver, BC (Hybrid)"
    assert jobs[2]["salary"] == "CA$91K-CA$113K / year"


def test_extract_glassdoor_digest_cards_from_email_image_shape():
    from agentic.workflows.job_hunt.toolset import _extract_jobs_from_cleaned_email

    body = """
Jobs for Aiko
W3Global 3.8 ★
QA tester (Manual/Functional)
Vancouver
$50K - $54K (Employer Est.)
Easy Apply
Goodly Foods
Quality Assurance Technician
Vancouver
$19 - $20 (Employer Est.)
3d
WorkSafeBC 3.5 ★
Support Analyst I
Richmond
$32 - $40 (Employer Est.)
5d
"""

    jobs = _extract_jobs_from_cleaned_email(body, sender="alerts@glassdoor.com", subject="Jobs for Aiko", config={})

    assert len(jobs) >= 3
    assert jobs[0]["title"] == "QA tester (Manual/Functional)"
    assert jobs[0]["organization"] == "W3Global 3.8 ★"
    assert jobs[0]["location"] == "Vancouver"
    assert jobs[1]["salary"] == "$19 - $20 (Employer Est.)"
