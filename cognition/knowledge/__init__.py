"""Public knowledge-store façade.

Implementation details are grouped by responsibility in companion modules while
this file preserves the existing import path, including private helpers used by
legacy tests and diagnostics.
"""

from __future__ import annotations

from . import backend as _backend

globals().update({name: getattr(_backend, name) for name in dir(_backend) if not name.startswith("__")})
__all__ = [name for name in globals() if not name.startswith("__") and name != "_backend"]
