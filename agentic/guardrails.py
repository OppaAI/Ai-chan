"""Composable guardrails for Aiko's lightweight agentic runtime.

Guardrails are deliberately tiny, pure callables. They inspect the planned
operation or candidate answer and return ``GuardResult`` only when they want to
block/repair; returning ``None`` means the check passed. Keeping these outside
``agentic.py`` makes policy gates pluggable without adopting a heavier agent
framework.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class GuardResult:
    """A guardrail failure or repair request."""

    error_type: str
    content: str
    retryable: bool = False
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class PreToolGuard(Protocol):
    def __call__(self, name: str, args: dict[str, Any], state: Any) -> GuardResult | None: ...


class PostAnswerGuard(Protocol):
    def __call__(self, answer: str, state: Any, user_input: str) -> GuardResult | None: ...


SOCIAL_POST_TOOLS = frozenset({"post_job_post_social", "post_photo_social", "post_video_social"})
RESEARCH_TOOLS = frozenset({"adaptive_search", "deep_research", "deep_read"})
SCHEDULE_TOOLS = frozenset({"schedule_job", "schedule_reminder"})

_EXTERNAL_ACTION_RE = re.compile(r"\b(send|sent|email|post|posted|buy|bought|book|booked|order|ordered|delete|deleted)\b", re.IGNORECASE)
_LOCAL_ARTIFACT_RE = re.compile(r"\b(saved|created|scheduled|cancelled|path|id|draft|note|workspace)\b", re.IGNORECASE)
_DISCLOSURE_RE = re.compile(r"\b(could not|couldn't|failed|unable|not able|blocked|limitation|didn't|did not)\b", re.IGNORECASE)


def _successful_tools(state: Any) -> set[str]:
    return {str(step.get("tool")) for step in getattr(state, "steps", []) if step.get("ok")}


def _failed_tools(state: Any) -> list[str]:
    return [str(f.tool) for f in getattr(state, "failures", [])]


def social_post_requires_review_bundle(name: str, args: dict[str, Any], state: Any) -> GuardResult | None:
    """Block social post tools unless the model supplies a review bundle path.

    The deeper toolkit still enforces human approval and duplicate-post checks;
    this earlier guard gives the LLM a clearer repair target before dispatch.
    """
    if name in SOCIAL_POST_TOOLS and not str(args.get("draft_dir") or "").strip():
        return GuardResult(
            error_type="missing_args",
            content="Social post tools require draft_dir pointing to a human-reviewed draft bundle.",
            retryable=True,
        )
    return None


def research_budget_guard(max_calls: int) -> PreToolGuard:
    def guard(name: str, args: dict[str, Any], state: Any) -> GuardResult | None:
        if name not in RESEARCH_TOOLS:
            return None
        used = sum(1 for step in getattr(state, "steps", []) if step.get("tool") in RESEARCH_TOOLS and step.get("ok"))
        if used >= max_calls:
            return GuardResult(
                error_type="research_limit_reached",
                content=(
                    f"adaptive_search/deep_research/deep_read have already been used {max_calls} time(s) "
                    "in this agentic workflow. Do not search again; use the evidence already "
                    "gathered to plan, summarize, save, or answer."
                ),
                retryable=False,
                metadata={"max_calls": max_calls, "used": used},
            )
        return None
    return guard


def saved_note_path_guard(answer: str, state: Any, user_input: str) -> GuardResult | None:
    if "save_note" not in _successful_tools(state):
        return None
    lowered = (answer or "").lower()
    if "path" in lowered or "workspace" in lowered or ".md" in lowered:
        return None
    return GuardResult(
        error_type="missing_saved_path",
        content="A saved note was created, but the final answer does not mention where it was saved.",
    )


def schedule_confirmation_guard(answer: str, state: Any, user_input: str) -> GuardResult | None:
    if not (_successful_tools(state) & SCHEDULE_TOOLS):
        return None
    lowered = (answer or "").lower()
    if "scheduled" in lowered or "reminder" in lowered or "alarm" in lowered:
        return None
    return GuardResult(
        error_type="missing_schedule_confirmation",
        content="A schedule/reminder tool succeeded, but the final answer does not confirm it.",
    )


def external_action_claim_guard(answer: str, state: Any, user_input: str) -> GuardResult | None:
    stripped = (answer or "").strip()
    posted_for_real = bool(_successful_tools(state) & SOCIAL_POST_TOOLS)
    if _EXTERNAL_ACTION_RE.search(user_input or "") and not _LOCAL_ARTIFACT_RE.search(stripped) and not posted_for_real:
        return GuardResult(
            error_type="unsupported_external_action_claim",
            content="The answer may imply an unsupported external action instead of a local draft/staged artifact.",
        )
    return None


def unresolved_failure_disclosure_guard(answer: str, state: Any, user_input: str) -> GuardResult | None:
    stripped = (answer or "").strip()
    failures = _failed_tools(state)
    if failures and not _DISCLOSURE_RE.search(stripped):
        failed = ", ".join(failures[-3:])
        return GuardResult(
            error_type="undisclosed_tool_failure",
            content=f"Unresolved tool failure(s) were not disclosed: {failed}.",
        )
    return None


def empty_answer_guard(answer: str, state: Any, user_input: str) -> GuardResult | None:
    if not (answer or "").strip():
        return GuardResult(error_type="empty_final_answer", content="The final answer is empty.")
    return None


DEFAULT_POST_ANSWER_GUARDRAILS: tuple[PostAnswerGuard, ...] = (
    empty_answer_guard,
    unresolved_failure_disclosure_guard,
    saved_note_path_guard,
    schedule_confirmation_guard,
    external_action_claim_guard,
)


def default_pre_tool_guardrails(max_research_calls: int) -> tuple[PreToolGuard, ...]:
    return (
        research_budget_guard(max_research_calls),
    )
