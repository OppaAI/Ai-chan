"""Backward-compat shim: shogi backend moved to interface.android_app.shogi.

New code should import from interface.android_app.shogi directly.
"""

import warnings

warnings.warn(
    "interface.webui.shogi is deprecated; use interface.android_app.shogi",
    DeprecationWarning,
    stacklevel=2,
)

from interface.android_app.shogi import router  # noqa: F401
