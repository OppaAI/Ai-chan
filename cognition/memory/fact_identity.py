"""Fact-identity helpers — merged into cognition.memory.entity.

Re-exports the most-used symbols so existing imports continue to work."""

from __future__ import annotations

from cognition.memory.entity import (
    DEFAULT_ASSISTANT_NAME,
    DEFAULT_USER_NAME,
    IDENTITY_PROMPT_RULES,
    fix_fact_identity,
    format_identity_prompt_rules,
    should_skip_misattributed_fact,
    sanitize_extracted_facts,
    sanitize_fact_score_pairs,
)

__all__ = [
    "DEFAULT_ASSISTANT_NAME",
    "DEFAULT_USER_NAME",
    "fix_fact_identity",
    "format_identity_prompt_rules",
    "should_skip_misattributed_fact",
    "sanitize_extracted_facts",
    "sanitize_fact_score_pairs",
]
