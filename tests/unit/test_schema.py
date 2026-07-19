from agentic import schema


def test_playbook_prefers_checklist_workflow():
    graph = schema.plan_from_master("make a checklist for Needle testing and save it", [])
    assert graph is not None
    assert graph.id == "checklist_and_save"


def test_graph_executes_save_note_without_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    graph = schema.plan_from_master("save this as a note: Needle practice works", [])
    assert graph is not None
    result = schema.execute_graph(graph)
    assert result.final_answer.startswith("I ran the saved workflow")
    assert any(r.tool == "save_note" and r.ok for r in result.results)


def test_dependent_graph_skips_downstream_when_dependency_fails(monkeypatch):
    def fail_plan(**_kwargs):
        raise RuntimeError("forced planning failure")

    def unexpected_save(**_kwargs):
        raise AssertionError("save_note should not run after a failed dependency")

    monkeypatch.setattr(schema, "_TOOL_MAP_CACHE", {
        "make_plan": fail_plan,
        "save_note": unexpected_save,
    })
    # The new plan playbook is no longer a default; test with checklist instead
    # which also uses make_plan internally? Actually checklist doesn't use make_plan.
    # Let's test with a simple save_note that will fail upstream.
    # We'll just test the logic with the checklist_and_save playbook and a
    # mock fail_checklist tool.
    def fail_checklist(**_kwargs):
        raise RuntimeError("forced checklist failure")

    monkeypatch.setattr(schema, "_TOOL_MAP_CACHE", {
        "create_checklist": fail_checklist,
        "save_note": unexpected_save,
    })
    graph = schema.plan_from_master("make a checklist for testing and save it", [])
    assert graph is not None
    assert graph.id == "checklist_and_save"

    result = schema.execute_graph(graph)

    by_id = {r.node_id: r for r in result.results}
    assert by_id["checklist"].ok is False
    assert by_id["save"].ok is False
    assert by_id["save"].error_type == "dependency_failed"


def test_schema_observation_lists_playbooks():
    data = schema.list_playbooks_json()
    assert "playbooks" in data
    assert "checklist_and_save" in data


def test_run_playbook_json_reports_no_match():
    data = schema.run_playbook_json("zzzz unmatched opaque task")
    assert "no_matching_playbook" in data


def test_new_playbooks_exist():
    """The four new default playbooks should be loaded."""
    playbooks = schema.load_playbooks()
    ids = {p["id"] for p in playbooks}
    assert "research_and_report" in ids
    assert "search_kb_and_report" in ids
    assert "compare_and_report" in ids
    assert "checklist_and_save" in ids
    assert "simple_save_note" in ids
    # Old playbooks should be gone
    assert "plan_and_save_note" not in ids
    assert "research_and_save" not in ids
