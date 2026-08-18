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


def test_detects_recent_contradiction():
    state = EdgeCognitiveState()
    state.record("I like coffee", "")
    state.record("I do not like coffee", "")
    assert state.snapshot()["contradictions"]


def test_prioritizes_goal_relevant_memory():
    state = EdgeCognitiveState()
    state.record("I want to finish the release", "")
    rows = [{"memory": "unrelated note"}, {"memory": "release checklist", "access_count": 3}]
    ranked = state.prioritize_memories("release", rows)
    assert ranked[0]["memory"] == "release checklist"


def test_requires_confirmation_for_memory_update():
    state = EdgeCognitiveState()
    memories = [{"id": "old-1", "memory": "User likes coffee"}]
    query = "Actually I no longer like coffee"
    assert state.memory_resolution_guidance(query, memories)
    state.situation_context(query, memories)
    assert state.confirm_memory_update("No, keep the old fact") == []
    state.situation_context(query, memories)
    pending = state.confirm_memory_update("Yes, that changed")
    assert pending and pending[0]["memory_id"] == "old-1"


def test_promotes_repeated_feedback_to_semantic_rule():
    state = EdgeCognitiveState()
    state.record("Please explain this", "Long answer")
    state.record("This is too verbose", "Short answer")
    state.record("That was too long, be concise", "Short answer")
    assert any("Prefer concise responses." in lesson for lesson in state.snapshot()["durable_lessons"])


def test_adaptive_guidance_and_reflection_include_state():
    state = EdgeCognitiveState()
    state.record("I want to finish the release", "")
    state.record_perception("voice", prosody={"rms": 0.1, "pause_density": 0.7, "words_per_second": 1.5})
    assert "calm" in state.adaptive_response_guidance()
    assert "Active priority" in state.reflection_summary()


def test_response_review_flags_unverified_current_claim():
    state = EdgeCognitiveState()
    review = state.review_response("What is the latest status?", "It is definitely complete.")
    assert any("verification" in flag for flag in review["flags"])


def test_preference_guidance_is_actionable():
    state = EdgeCognitiveState()
    state.record("From now on, be concise and ask before using tools", "")
    guidance = state.preference_guidance()
    assert "concise" in guidance
    assert "Ask before consequential external actions" in guidance


def test_adaptive_tts_rate_responds_to_voice_cues():
    state = EdgeCognitiveState()
    state.record_perception("voice", prosody={"pause_density": 0.8, "words_per_second": 1.2})
    assert state.adaptive_tts_rate() == 0.9


def test_confirmed_memory_conflict_is_consumed_once():
    state = EdgeCognitiveState()
    memories = [{"id": "m1", "memory": "User likes tea"}]
    query = "Actually I no longer like tea"
    state.situation_context(query, memories)
    assert state.confirm_memory_update("Yes, that changed")
    assert state.confirm_memory_update("Yes, that changed") == []


def test_attention_keeps_a_readable_focus_phrase():
    state = EdgeCognitiveState()
    state.record("Can you do todays job post?", "")
    assert state.snapshot()["attention"] == "job post"


def test_completed_task_closes_matching_open_loop_and_goal():
    state = EdgeCognitiveState()
    state.record("I need to finish the release", "")
    state.record("Can you check the release?", "")
    state.record("The release is completed", "")
    snap = state.snapshot()
    assert not snap["goals"]
    assert not snap["open_loops"]


def test_reconstructive_recall_uses_active_context_and_reports_confidence():
    state = EdgeCognitiveState()
    state.record("I want to finish the release", "")
    rows = [
        {"memory": "The release checklist is ready", "valence_score": 1},
        {"memory": "A recipe for soup", "valence_score": 1},
    ]
    ranked = state.prioritize_memories("what is next", rows)
    assert ranked[0]["memory"] == "The release checklist is ready"
    assert ranked[0]["_reconstruction_confidence"] in {"moderate", "high"}
    assert "_reconstruction_confidence" not in rows[0]
