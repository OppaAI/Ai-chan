"""Compatibility alias for :mod:`cognition.attention`."""
import sys

from cognition import attention as _attention

sys.modules[__name__] = _attention
