from skills import graph_agent


def test_master_plan_prefers_checklist_workflow():
    graph = graph_agent.plan_from_master("make a checklist for Needle testing and save it", [])
    assert graph is not None
    assert graph.id == "checklist_and_save"


def test_graph_executes_save_note_without_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    graph = graph_agent.plan_from_master("save this as a note: Needle practice works", [])
    assert graph is not None
    result = graph_agent.execute_graph(graph)
    assert result.final_answer.startswith("I ran the saved workflow")
    assert any(r.tool == "save_note" and r.ok for r in result.results)


def test_graph_agent_schema_observation_lists_master_plans():
    data = graph_agent.list_master_plans_json()
    assert "master_plans" in data
    assert "checklist_and_save" in data


def test_run_master_plan_json_reports_no_match():
    data = graph_agent.run_master_plan_json("zzzz unmatched opaque task")
    assert "no_matching_master_plan" in data
