"""Tests for deterministic, least-privilege Needle worker orchestration."""
from __future__ import annotations

import pytest

from agentic.needle import NeedleCall, NeedleError, NeedleResponse
from agentic.needle_orchestrator import NeedleOrchestrator, load_needle_workers


def _tools(*names: str):
    return [{"type": "function", "function": {"name": name, "parameters": {}}} for name in names]


class _Client:
    calls = []

    def __init__(self, base_url, *, timeout, confidence_threshold):
        self.base_url = base_url
        self.timeout = timeout
        self.confidence_threshold = confidence_threshold

    def complete(self, prompt, tools):
        self.calls.append((self.base_url, prompt, tools))
        name = tools[0]["function"]["name"]
        return NeedleResponse("call", (NeedleCall(name, {"worker": self.base_url}),), 0.99, "", "")


def test_load_needle_workers_requires_least_privilege_tool_lists():
    with pytest.raises(NeedleError, match="allowed_tools"):
        load_needle_workers('[{"id":"a","role":"r","base_url":"http://a"}]', default_timeout=1, default_confidence_threshold=0.8)


def test_load_needle_workers_enforces_worker_limit_and_finite_limits():
    with pytest.raises(NeedleError, match="worker limit"):
        load_needle_workers(
            '[{"id":"a","role":"r","base_url":"http://a","allowed_tools":["search"]},'
            '{"id":"b","role":"r","base_url":"http://b","allowed_tools":["search"]}]',
            default_timeout=1,
            default_confidence_threshold=0.8,
            max_workers=1,
        )
    with pytest.raises(NeedleError, match="between 0 and 1"):
        load_needle_workers(
            '[{"id":"a","role":"r","base_url":"http://a","allowed_tools":["search"],"confidence_threshold":2}]',
            default_timeout=1,
            default_confidence_threshold=0.8,
        )


def test_orchestrator_fans_out_with_each_workers_tool_intersection():
    _Client.calls = []
    workers = load_needle_workers(
        '[{"id":"research","role":"researcher","base_url":"http://a","allowed_tools":["search"]},'
        '{"id":"notes","role":"note taker","base_url":"http://b","allowed_tools":["save_note"]}]',
        default_timeout=3,
        default_confidence_threshold=0.8,
    )
    results = NeedleOrchestrator(workers, client_factory=_Client).complete("find and save", _tools("search", "save_note", "post"))

    assert [result.worker_id for result in results] == ["research", "notes"]
    assert [result.response.calls[0].name for result in results if result.response] == ["search", "save_note"]
    assert {call[2][0]["function"]["name"] for call in _Client.calls} == {"search", "save_note"}
    assert all("Role:" in call[1] for call in _Client.calls)


def test_orchestrator_reports_ineligible_workers_without_calling_them():
    workers = load_needle_workers(
        '[{"id":"research","role":"researcher","base_url":"http://a","allowed_tools":["search"]}]',
        default_timeout=3,
        default_confidence_threshold=0.8,
    )
    with pytest.raises(NeedleError, match="ineligible"):
        NeedleOrchestrator(workers, client_factory=_Client).complete("save", _tools("save_note"))
