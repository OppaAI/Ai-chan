"""Backward-compat shim: go backend moved to interface.android_app.go.

New code should import from interface.android_app.go directly.
"""

import warnings

warnings.warn(
    "interface.webui.go is deprecated; use interface.android_app.go",
    DeprecationWarning,
    stacklevel=2,
)

from interface.android_app.go import router  # noqa: F401
