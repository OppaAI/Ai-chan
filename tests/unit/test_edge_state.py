from cognition.memory.edge_state import EdgeCognitiveState


def test_edge_state_is_bounded_and_retrieves_current_focus():
    state = EdgeCognitiveState()
    for i in range(12):
        state.record(f"remember task {i}", f"Noted item {i}.")
    block = state.context("task 11")
    assert "task 11" in block
    assert "<edge_cognitive_state>" in block
    assert len(state._events) == 7
    assert len(state._open_loops) == 3


def test_edge_state_has_no_context_before_first_turn():
    assert EdgeCognitiveState().context("hello") == ""
