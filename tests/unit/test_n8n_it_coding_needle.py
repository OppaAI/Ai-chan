"""
tests/unit/test_n8n_it_coding_needle.py

n8n-style DAG studio + IT/everyday + coding-agent + Needle-team coverage.

Hermetic: no network, no LLM, no docker. sys_health takes ~1s (psutil
interval); everything else is instant.
"""
from __future__ import annotations

import json

from system.config import load_config

load_config()

import agentic.tools  # noqa: F401 — register all toolkit tools
from agentic.registry import registry


def _payload(text: str) -> dict:
    assert text.startswith("["), text[:120]
    tag_end = text.index("]")
    return json.loads(text[tag_end + 1:].strip() or "{}")


class TestItOpsTools:
    def test_sys_health_shape(self):
        from agentic.toolkit.it_ops import sys_health

        data = _payload(sys_health())
        assert data["ok"] is True
        assert data["status"] in {"healthy", "warning", "critical"}
        assert 0 <= data["cpu_percent"] <= 100

    def test_unit_convert_km_mi(self):
        from agentic.toolkit.it_ops import unit_convert

        data = _payload(unit_convert(value=10, from_unit="km", to_unit="mi"))
        assert data["ok"] is True
        assert abs(data["result"] - 6.213712) < 0.001

    def test_unit_convert_temp(self):
        from agentic.toolkit.it_ops import unit_convert

        data = _payload(unit_convert(value=0, from_unit="c", to_unit="f"))
        assert data["result"] == 32.0

    def test_port_check_closed(self):
        from agentic.toolkit.it_ops import port_check

        data = _payload(port_check(host="127.0.0.1", port=1))
        assert data["ok"] is True
        assert data["open"] is False

    def test_dns_localhost(self):
        from agentic.toolkit.it_ops import dns_lookup

        data = _payload(dns_lookup(hostname="localhost"))
        assert data["ok"] is True
        assert data["addresses"]

    def test_pass_gen_length(self):
        from agentic.toolkit.it_ops import pass_gen

        data = _payload(pass_gen(length=20, symbols=False))
        assert data["ok"] is True and len(data["password"]) == 20

    def test_registered_graph_react(self):
        for name in ("sys_health", "log_triage", "port_check", "http_check",
                     "text_summarize", "unit_convert", "pass_gen"):
            spec = registry.get(name)
            assert spec is not None, name
            assert spec.graph and spec.react, name


class TestCodingTools:
    def test_diff_preview(self):
        from agentic.toolkit.coding import code_diff_preview

        data = _payload(code_diff_preview(
            relative_path="a.py", old_text="x=1\n", new_text="x=2\n"))
        assert data["ok"] is True and "-x=1" in data["diff"]

    def test_plan_heuristic_no_llm(self):
        from agentic.toolkit.coding import code_plan

        data = _payload(code_plan(goal="fix off-by-one in pager"))
        assert data["ok"] is True and len(data["steps"]) >= 3

    def test_apply_rejects_secrets(self):
        from agentic.toolkit.coding import code_apply_patch

        data = _payload(code_apply_patch(
            relative_path=".env", old_text="A", new_text="B"))
        assert data["ok"] is False

    def test_lint_self(self):
        from agentic.toolkit.coding import code_lint

        data = _payload(code_lint(relative_path="agentic/toolkit/it_ops.py"))
        assert data["ok"] is True

    def test_write_tools_need_approval(self):
        assert registry.get("code_apply_patch").needs_approval is True
        assert registry.get("repo_write_file").needs_approval is True
        assert registry.get("repo_replace_text").needs_approval is True


class TestNeedleTeam:
    def test_status_unconfigured(self, monkeypatch):
        from agentic.toolkit.needle_team import needle_team_status

        monkeypatch.setenv("NEEDLE_WORKERS", "")
        data = _payload(needle_team_status())
        assert data["ok"] is True and data["configured"] is False

    def test_run_unconfigured(self, monkeypatch):
        from agentic.toolkit.needle_team import needle_team_run

        monkeypatch.setenv("NEEDLE_WORKERS", "")
        data = _payload(needle_team_run(task="hello"))
        assert data["ok"] is False and "fallback" in data


class TestDefaultPlaybooks:
    def test_new_ids_present(self):
        from agentic.graph_engine import _default_playbooks

        ids = {p["id"] for p in _default_playbooks()}
        for pid in ("it_syscheck", "it_log_triage", "it_endpoint_check",
                    "it_disk_cleanup_preview", "everyday_summarize",
                    "everyday_translate_save", "everyday_morning_brief",
                    "code_explain", "code_small_fix", "needle_team_research"):
            assert pid in ids, pid

    def test_it_syscheck_routable_by_keyword(self):
        from agentic.graph_engine import plan_from_master

        g = plan_from_master("run a system health check please")
        assert g is not None and g.id == "it_syscheck"

    def test_code_small_fix_has_approval_node(self):
        from agentic.graph_engine import _default_playbooks

        pb = next(p for p in _default_playbooks() if p["id"] == "code_small_fix")
        apply = next(n for n in pb["nodes"] if n["id"] == "apply")
        assert apply.get("needs_approval") is True


class TestDagStudioApi:
    def _client(self):
        from fastapi.testclient import TestClient
        from interface.webui.studio.dag.backend.api import app

        return TestClient(app)

    def test_clean_nodes_rejects_cycle(self):
        from interface.webui.studio.dag.backend.api import _clean_nodes

        _, errors = _clean_nodes([
            {"id": "a", "tool": "sys_health", "args": {}, "depends_on": ["b"]},
            {"id": "b", "tool": "sys_health", "args": {}, "depends_on": ["a"]},
        ])
        assert any("cycle" in e for e in errors)

    def test_clean_nodes_rejects_unknown_dep(self):
        from interface.webui.studio.dag.backend.api import _clean_nodes

        _, errors = _clean_nodes([{"id": "a", "tool": "sys_health", "args": {}, "depends_on": ["ghost"]}])
        assert any("unknown dependency" in e for e in errors)

    def test_crud_validate_run(self):
        c = self._client()
        assert c.post("/api/playbooks", json={
            "id": "tmp_n8n_ut", "name": "ut",
            "nodes": [{"id": "a", "tool": "sys_health", "args": {}}],
        }).status_code == 200
        assert c.post("/api/playbooks/tmp_n8n_ut/validate").json()["ok"] is True
        run = c.post("/api/playbooks/tmp_n8n_ut/run",
                     json={"prompt": "ut", "timeout_s": 30}).json()
        assert run["ok"] is True
        assert run["nodes"][0]["ok"] is True
        assert c.post("/api/playbooks/tmp_n8n_ut/duplicate",
                      json={"new_id": "tmp_n8n_ut2"}).status_code == 200
        assert c.delete("/api/playbooks/tmp_n8n_ut").json()["ok"] is True
        assert c.delete("/api/playbooks/tmp_n8n_ut2").json()["ok"] is True

    def test_builtin_delete_protected(self):
        c = self._client()
        r = c.delete("/api/playbooks/it_syscheck")
        assert r.status_code == 400

    def test_tools_palette_lists_new_domains(self):
        c = self._client()
        groups = c.get("/api/tools").json()["groups"]
        for domain in ("it_ops", "everyday", "coding", "multi_agent"):
            assert domain in groups, domain
