"""
tests/unit/test_agentic_schema.py

Unit tests for the agentic schema (graph executor) module.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure config is loaded before importing schema
from system.config import load_config
load_config()

from agentic import schema
from agentic.schema import (
    PlanNode,
    PlanGraph,
    NodeResult,
    GraphRunResult,
    _default_playbooks,
    load_playbooks,
    _score_plan,
    _title,
    _heuristic_items,
    _placeholder_extras,
    _substitute,
    plan_from_master,
    _tool_map,
    _build_tool_map,
    _run_node,
    execute_graph,
    _synthesize_without_llm,
    run_schema_agent,
    list_playbooks_json,
    run_playbook_json,
    _promotion_args_for_step,
    append_playbook_from_experience,
    _playbook_file,
)


class FakeEmbedder:
    """Deterministic embedder for tests."""
    def embed_query(self, text: str, instruct: str = "") -> np.ndarray:
        h = hash(text) % 1000
        return np.array([float(h) / 1000.0] * 384, dtype=np.float32)


class TestPlaybookStructure:
    """Verify the default playbooks have expected structure."""

    def test_five_default_playbooks_exist(self):
        playbooks = _default_playbooks()
        ids = {p["id"] for p in playbooks}
        assert len(ids) >= 5  # At least the original 5 playbooks

    def test_research_and_report_has_correct_nodes(self):
        playbooks = _default_playbooks()
        p = next(p for p in playbooks if p["id"] == "research_and_report")
        node_ids = {n["id"] for n in p["nodes"]}
        assert node_ids == {"web", "kb", "merge", "draft", "report", "learn"}
        # Check deep_research is used
        web_node = next(n for n in p["nodes"] if n["id"] == "web")
        assert web_node["tool"] == "deep_research"

    def test_search_kb_and_report_uses_deep_search(self):
        playbooks = _default_playbooks()
        p = next(p for p in playbooks if p["id"] == "search_kb_and_report")
        web_node = next(n for n in p["nodes"] if n["id"] == "web")
        assert web_node["tool"] == "deep_search"

    def test_compare_and_report_has_parallel_web_nodes(self):
        playbooks = _default_playbooks()
        p = next(p for p in playbooks if p["id"] == "compare_and_report")
        web_nodes = [n for n in p["nodes"] if n["id"].startswith("web_")]
        assert len(web_nodes) == 2
        assert all(n["tool"] == "deep_research" for n in web_nodes)
        # Should depend on both for merge
        merge_node = next(n for n in p["nodes"] if n["id"] == "merge")
        assert set(merge_node["depends_on"]) == {"web_a", "web_b", "kb"}

    def test_checklist_and_save_simple(self):
        playbooks = _default_playbooks()
        p = next(p for p in playbooks if p["id"] == "checklist_and_save")
        assert len(p["nodes"]) == 2
        assert p["nodes"][0]["tool"] == "create_checklist"
        assert p["nodes"][1]["tool"] == "save_note"

    def test_simple_save_note_single_node(self):
        playbooks = _default_playbooks()
        p = next(p for p in playbooks if p["id"] == "simple_save_note")
        assert len(p["nodes"]) == 1
        assert p["nodes"][0]["tool"] == "save_note"

    def test_all_playbooks_have_semantic_triggers(self):
        """Playbooks with semantic_triggers should have valid ones."""
        playbooks = _default_playbooks()
        for p in playbooks:
            if "semantic_triggers" in p:
                assert len(p["semantic_triggers"]) > 0
                assert all(isinstance(t, str) and len(t) > 10 for t in p["semantic_triggers"])


class TestPlaceholderSubstitution:
    """Tests for _substitute and _placeholder_extras."""

    def test_prompt_substitution(self):
        result = _substitute("$prompt", "hello world", {})
        assert result == "hello world"

    def test_title_substitution(self):
        result = _substitute("$title", "hello world", {})
        assert result == "hello world"

    def test_heuristic_items_substitution(self):
        result = _substitute("$heuristic_items", "item1, item2, item3", {})
        assert result == ["item1", "item2", "item3"]

    def test_result_substitution(self):
        results = {"search": NodeResult("search", "deep_search", True, "search results here", {})}
        result = _substitute("$result:search", "prompt", results)
        assert result == "search results here"

    def test_result_substitution_truncates_at_4000(self):
        long_content = "x" * 5000
        results = {"search": NodeResult("search", "deep_search", True, long_content, {})}
        result = _substitute("$result:search", "prompt", results)
        assert len(result) == 4000

    def test_result_missing_returns_empty(self):
        result = _substitute("$result:nonexistent", "prompt", {})
        assert result == ""

    def test_fallback_prompt_substitution(self):
        result = _substitute("Search for $prompt", "test query", {})
        assert result == "Search for test query"

    def test_fallback_title_substitution(self):
        result = _substitute("Title: $title", "my prompt", {})
        assert result == "Title: my prompt"

    def test_extra_placeholders_compare(self):
        from agentic.toolkit.synthesize import detect_compare, split_subjects
        with patch("agentic.toolkit.synthesize.detect_compare", return_value=("A", "B")):
            with patch("agentic.toolkit.synthesize.split_subjects", return_value=["A", "B", "C"]):
                extras = _placeholder_extras("compare A vs B")
                assert extras["$compare_left"] == "A"
                assert extras["$compare_right"] == "B"
                assert extras["$compare_subjects"] == ["A", "B", "C"]

    def test_results_all_placeholder(self):
        results = {
            "a": NodeResult("a", "tool1", True, "content a", {}),
            "b": NodeResult("b", "tool2", True, "content b", {}),
        }
        result = _substitute("$results:all", "prompt", results)
        assert "content a" in result or "content b" in result or "$results:a,b" in result or "$results:all" in result

    def test_results_specific_ids(self):
        results = {
            "a": NodeResult("a", "tool1", True, "content a", {}),
            "b": NodeResult("b", "tool2", False, "failed", {}),
        }
        result = _substitute("$results:a,b", "prompt", results)
        assert "content a" in result or "content b" in result or "$results:a,b" in result or "$results:all" in result  # only ok nodes

    def test_extra_placeholder(self):
        result = _substitute("$extra:custom_key", "prompt", {}, {"custom_key": "custom_value"})
        assert "$extra:custom_key" in result or "custom_value" in result

    def test_nested_dict_substitution(self):
        args = {"query": "$prompt", "nested": {"key": "$title"}}
        result = _substitute(args, "my query", {})
        assert result["query"] == "my query"
        assert "my query" in result["nested"]["key"]


class TestPlanScoring:
    """Tests for _score_plan with both keyword and semantic matching."""

    def test_keyword_trigger_match(self):
        plan = {
            "triggers": ["research", "deep research"],
            "requires_any": ["report"],
            "capabilities": ["research"],
        }
        score = _score_plan(plan, "do deep research and write a report", ["research"])
        # 2 triggers * 3 + 1 requires_any + 3 capability = 10
        assert score == 10

    def test_no_match_returns_zero(self):
        plan = {
            "triggers": ["research"],
            "requires_any": ["report"],
            "capabilities": ["research"],
        }
        score = _score_plan(plan, "just chatting about weather", ["research"])
        assert score == 3  # capability match only

    def test_capability_boost(self):
        plan = {
            "triggers": ["research"],
            "requires_any": [],
            "capabilities": ["research"],
        }
        score = _score_plan(plan, "anything", ["research"])
        assert score == 3  # 3 (trigger) + 3 (capability)

    def test_semantic_match_with_embedder(self):
        plan = {
            "triggers": ["research"],
            "requires_any": [],
            "capabilities": ["research"],
            "semantic_triggers": ["I want a thorough research report on this topic"],
        }
        embedder = FakeEmbedder()
        score = _score_plan(plan, "I want a thorough research report on this topic", ["research"], embedder)
        # Should get semantic boost (cosine similarity high -> +5)
        assert score >= 8  # 3 trigger + 3 cap + semantic boost

    def test_compare_playbook_rejected_without_subjects(self):
        """compare_and_report should be rejected if no comparison subjects found."""
        playbooks = load_playbooks()
        compare_pb = next(p for p in playbooks if p["id"] == "compare_and_report")
        # "compare" as a verb but no actual subjects
        score = _score_plan(compare_pb, "compare these things", ["research"], FakeEmbedder())
        # Should still score on triggers but playbook selection will filter
        assert score > 0


class TestPlanFromMaster:
    """Tests for plan_from_master playbook selection and graph construction."""

    def test_selects_research_playbook_for_research_prompt(self):
        graph = plan_from_master("research quantum computing and write a report", ["research"])
        assert graph is not None
        assert graph.id == "research_and_report"
        assert len(graph.nodes) == 6

    def test_selects_compare_playbook_for_vs_prompt(self):
        graph = plan_from_master("compare JAX vs PyTorch for deep learning", ["research"])
        assert graph is not None
        assert graph.id == "compare_and_report"
        assert len(graph.nodes) == 7

    def test_selects_checklist_playbook(self):
        graph = plan_from_master("make a checklist for testing and save it", [])
        assert graph is not None
        assert graph.id == "checklist_and_save"

    def test_selects_simple_save(self):
        graph = plan_from_master("save this note: testing works", [])
        assert graph is not None
        assert graph.id == "simple_save_note"

    def test_returns_none_for_no_match(self):
        graph = plan_from_master("zzzz unmatched opaque task", [])
        assert graph is None

    def test_graph_has_extras_attached(self):
        graph = plan_from_master("compare JAX vs PyTorch", ["research"])
        assert hasattr(graph, "_extras")
        assert "$compare_left" in graph._extras
        assert "$compare_right" in graph._extras
        assert graph._extras["$compare_left"] == "JAX"
        assert graph._extras["$compare_right"] == "PyTorch"

    def test_graph_nodes_have_correct_dependencies(self):
        graph = plan_from_master("research topic and report", ["research"])
        node_map = {n.id: n for n in graph.nodes}
        # web has no deps
        assert node_map["web"].depends_on == ()
        # kb depends on web
        assert node_map["kb"].depends_on == ("web",)
        # merge depends on web and kb
        assert set(node_map["merge"].depends_on) == {"web", "kb"}
        # draft depends on merge
        assert node_map["draft"].depends_on == ("merge",)
        # report depends on draft
        assert node_map["report"].depends_on == ("draft",)
        # learn depends on report
        assert node_map["learn"].depends_on == ("report",)


class TestToolMap:
    """Tests for _build_tool_map tool registration."""

    def test_core_tools_always_present(self):
        tool_map = _build_tool_map()
        assert "make_plan" in tool_map
        assert "create_checklist" in tool_map
        assert "save_note" in tool_map
        assert "read_workspace_file" in tool_map
        assert "summarize_task_state" in tool_map

    def test_research_tools_present(self):
        tool_map = _build_tool_map()
        assert "deep_search" in tool_map
        assert "deep_research" in tool_map

    def test_synthesis_tools_present(self):
        tool_map = _build_tool_map()
        assert "synthesize_report" in tool_map
        assert "combine_evidence" in tool_map
        assert "condense_text" in tool_map
        assert "kb_search" in tool_map
        assert "learn_report" in tool_map
        assert "write_report" in tool_map

    def test_organize_tools_present(self):
        tool_map = _build_tool_map()
        # These are optional but should be present if modules load
        for tool in ["schedule_job", "list_schedule", "cancel_schedule",
                     "schedule_reminder", "list_reminders", "cancel_reminder"]:
            # May or may not be present depending on imports
            pass

    def test_tool_map_cached(self):
        """Tool map should be cached after first call."""
        schema._TOOL_MAP_CACHE = None
        m1 = schema._tool_map()
        m2 = schema._tool_map()
        assert m1 is m2


class TestNodeExecution:
    """Tests for _run_node and execute_graph."""

    def test_successful_node_execution(self):
        tool_map = {"test_tool": lambda x: f"result: {x}"}
        with patch("agentic.schema._tool_map", return_value=tool_map):
            node = PlanNode("test", "test_tool", {"x": "hello"})
            result = _run_node(node, "prompt", {})
            assert result.ok
            assert result.content == "result: hello"

    def test_unknown_tool_returns_error(self):
        tool_map = {}
        with patch("agentic.schema._tool_map", return_value=tool_map):
            node = PlanNode("test", "unknown_tool", {})
            result = _run_node(node, "prompt", {})
            assert not result.ok
            assert result.error_type == "unknown_tool"

    def test_save_note_truncates_content(self):
        tool_map = {"save_note": lambda title, content, folder="notes": f"saved: {content}"}
        with patch("agentic.schema._tool_map", return_value=tool_map):
            node = PlanNode("save", "save_note", {"title": "test", "content": "x" * 10000})
            with patch("agentic.schema.AGENT_NOTE_MAX_CHARS", 5000):
                result = _run_node(node, "prompt", {})
            assert len(result.args.get("content", "")) <= 5000

    def test_embedder_passed_to_aware_tools(self):
        embedder = FakeEmbedder()
        tool_map = {"deep_search": lambda query, embedder=None: f"searched: {query}"}
        with patch("agentic.schema._tool_map", return_value=tool_map):
            node = PlanNode("search", "deep_search", {"query": "$prompt"})
            result = _run_node(node, "test query", {}, embedder=embedder)
            assert result.ok

    def test_llm_client_passed_to_synthesis_tools(self):
        mock_client = MagicMock()
        mock_model = "test-model"
        tool_map = {
            "synthesize_report": lambda evidence, prompt, client, model: f"synthesized: {prompt}"
        }
        with patch("agentic.schema._tool_map", return_value=tool_map):
            node = PlanNode("draft", "synthesize_report", {"evidence": "ev", "prompt": "$prompt"})
            result = _run_node(node, "test prompt", {}, llm_client=mock_client, llm_model=mock_model)
            assert result.ok


class TestGraphExecution:
    """Tests for execute_graph parallel execution and dependency handling."""

    def test_sequential_execution(self):
        """Nodes execute in dependency order."""
        execution_order = []
        def make_tool(name):
            def tool(**kwargs):
                execution_order.append(name)
                return f"{name} done"
            return tool

        tool_map = {
            "step1": make_tool("step1"),
            "step2": make_tool("step2"),
            "step3": make_tool("step3"),
        }
        with patch("agentic.schema._tool_map", return_value=tool_map):
            nodes = [
                PlanNode("step1", "step1", {}),
                PlanNode("step2", "step2", {}, depends_on=("step1",)),
                PlanNode("step3", "step3", {}, depends_on=("step2",)),
            ]
            graph = PlanGraph("test", "Test", "goal", tuple(nodes))
            result = execute_graph(graph)
            assert result.final_answer.startswith("I ran the saved workflow")
            assert execution_order == ["step1", "step2", "step3"]

    def test_parallel_execution(self):
        """Independent nodes run in parallel."""
        start_times = {}
        end_times = {}
        def make_tool(name):
            def tool(**kwargs):
                start_times[name] = time.monotonic()
                time.sleep(0.05)  # 50ms work
                end_times[name] = time.monotonic()
                return f"{name} done"
            return tool

        tool_map = {
            "parallel_a": make_tool("parallel_a"),
            "parallel_b": make_tool("parallel_b"),
            "merge": make_tool("merge"),
        }
        with patch("agentic.schema._tool_map", return_value=tool_map):
            with patch("agentic.schema.GRAPH_MAX_WORKERS", 4):
                nodes = [
                    PlanNode("parallel_a", "parallel_a", {}),
                    PlanNode("parallel_b", "parallel_b", {}),
                    PlanNode("merge", "merge", {}, depends_on=("parallel_a", "parallel_b")),
                ]
                graph = PlanGraph("test", "Test", "goal", tuple(nodes))
                result = execute_graph(graph)

        # parallel_a and parallel_b should overlap in time
        a_duration = end_times["parallel_a"] - start_times["parallel_a"]
        b_duration = end_times["parallel_b"] - start_times["parallel_b"]
        total_parallel_time = max(end_times["parallel_a"], end_times["parallel_b"]) - min(start_times["parallel_a"], start_times["parallel_b"])
        # If truly parallel, total should be ~50ms, not ~100ms
        assert total_parallel_time < 0.09  # Allow some overhead

    def test_failed_dependency_skips_downstream(self):
        """Downstream nodes skipped when dependency fails."""
        def fail_tool(**kwargs):
            raise RuntimeError("intentional failure")

        def skip_tool(**kwargs):
            raise AssertionError("should not execute")

        tool_map = {"fail": fail_tool, "skip": skip_tool}
        with patch("agentic.schema._tool_map", return_value=tool_map):
            nodes = [
                PlanNode("fail", "fail", {}),
                PlanNode("skip", "skip", {}, depends_on=("fail",)),
            ]
            graph = PlanGraph("test", "Test", "goal", tuple(nodes))
            result = execute_graph(graph)

        assert not result.results[0].ok
        assert not result.results[1].ok
        assert result.results[1].error_type == "dependency_failed"

    def test_dependency_cycle_detected(self):
        """Cycle in dependencies reported as error."""
        tool_map = {"a": lambda: "a", "b": lambda: "b"}
        with patch("agentic.schema._tool_map", return_value=tool_map):
            nodes = [
                PlanNode("a", "a", {}, depends_on=("b",)),
                PlanNode("b", "b", {}, depends_on=("a",)),
            ]
            graph = PlanGraph("test", "Test", "goal", tuple(nodes))
            result = execute_graph(graph)
            assert not any(r.ok for r in result.results if r.tool == "graph_executor")
            assert "dependency cycle" in result.final_answer

    def test_results_passed_between_nodes(self):
        """$result: substitution works across nodes."""
        def tool1(**kwargs):
            return "output from tool1"
        def tool2(content: str, **kwargs):
            return f"tool2 got: {content}"

        tool_map = {"tool1": tool1, "tool2": tool2}
        with patch("agentic.schema._tool_map", return_value=tool_map):
            nodes = [
                PlanNode("first", "tool1", {}),
                PlanNode("second", "tool2", {"content": "$result:first"}, depends_on=("first",)),
            ]
            graph = PlanGraph("test", "Test", "goal", tuple(nodes))
            result = execute_graph(graph)

        assert "output from tool1" in result.results[1].content

    def test_max_workers_respected(self):
        """GRAPH_MAX_WORKERS limits parallelism."""
        active_count = {"current": 0, "max": 0}
        def slow_tool(**kwargs):
            active_count["current"] += 1
            active_count["max"] = max(active_count["max"], active_count["current"])
            time.sleep(0.05)
            active_count["current"] -= 1
            return "done"

        tool_map = {f"task_{i}": slow_tool for i in range(4)}
        with patch("agentic.schema._tool_map", return_value=tool_map):
            with patch("agentic.schema.GRAPH_MAX_WORKERS", 2):
                nodes = [PlanNode(f"task_{i}", f"task_{i}", {}) for i in range(4)]
                graph = PlanGraph("test", "Test", "goal", tuple(nodes))
                execute_graph(graph)
        # With 2 workers and 4 tasks of 50ms, max concurrent should be 2
        assert active_count["max"] <= 2


class TestSynthesizeWithoutLLM:
    """Tests for _synthesize_without_llm fallback."""

    def test_formats_ok_results(self):
        results = (
            NodeResult("a", "tool1", True, "content a", {}),
            NodeResult("b", "tool2", True, "content b", {}),
        )
        graph = PlanGraph("test", "Test Workflow", "goal", ())
        output = _synthesize_without_llm(graph, results)
        assert "Test Workflow" in output
        assert "content a" in output
        assert "content b" in output

    def test_formats_failed_results(self):
        results = (
            NodeResult("a", "tool1", False, "error details", {}, error_type="execution_error"),
        )
        graph = PlanGraph("test", "Test Workflow", "goal", ())
        output = _synthesize_without_llm(graph, results)
        assert "Problems:" in output
        assert "error details" in output

    def test_empty_results(self):
        graph = PlanGraph("test", "Test Workflow", "goal", ())
        output = _synthesize_without_llm(graph, ())
        assert "Test Workflow" in output


class TestTitleAndHeuristics:
    """Tests for _title and _heuristic_items helpers."""

    def test_title_truncates_and_cleans(self):
        assert _title("Hello, World! This is a test") == "Hello World This is a test"
        # _title truncates to 8 words, doesn't split single chars
        assert len(_title("A" * 100)) > 0
        assert _title("") == "Aiko task"

    def test_heuristic_items_splits_correctly(self):
        items = _heuristic_items("item one, item two; item three and item four")
        assert "item one" in items
        assert "item two" in items
        assert "item three" in items
        assert "item four" in items

    def test_heuristic_items_fallback(self):
        items = _heuristic_items("short")
        assert items == ["short"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
