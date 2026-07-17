from agentic import schema


def test_master_plan_prefers_checklist_workflow():
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
    graph = schema.plan_from_master("make a plan and save it as a note", [])
    assert graph is not None
    assert graph.id == "plan_and_save_note"

    result = schema.execute_graph(graph)

    by_id = {r.node_id: r for r in result.results}
    assert by_id["plan"].ok is False
    assert by_id["save"].ok is False
    assert by_id["save"].error_type == "dependency_failed"

def test_schema_observation_lists_master_plans():
    data = schema.list_master_plans_json()
    assert "master_plans" in data
    assert "checklist_and_save" in data


def test_run_master_plan_json_reports_no_match():
    data = schema.run_master_plan_json("zzzz unmatched opaque task")
    assert "no_matching_master_plan" in data
