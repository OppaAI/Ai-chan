"""Compatibility wrapper for attention gate helpers.

Gate logic now lives with edge cognitive state in :mod:`cognition.attention`.
"""
from cognition.attention import (  # noqa: F401
    capability_from_outcomes,
    is_critical_task,
    is_time_sensitive,
    should_attempt,
    soft_user_prompt,
)
