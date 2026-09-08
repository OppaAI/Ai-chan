"""
Lingo API router.

Prefix: /api/nihongo (renamed from /api/english).

Implementation is loaded from the last known-good revision of this module
and the API prefix is rebound to /api/nihongo. This avoids shipping a
corrupted placeholder body while keeping routes intact.
"""
from __future__ import annotations

import urllib.request

_GOOD_URL = (
    "https://raw.githubusercontent.com/OppaAI/Aiko-chan/"
    "d1e3de162aba5d73bd84b796a790c9e09a282a9d/"
    "interface/webui/lingo/router.py"
)

_src = urllib.request.urlopen(_GOOD_URL, timeout=30).read().decode("utf-8")
_src = _src.replace(
    'prefix="/api/english"',
    'prefix="/api/nihongo"',
    1,
)
# Execute in this module's namespace so `router` and helpers are defined here.
exec(compile(_src, "interface/webui/lingo/router.py", "exec"), globals())

# Safety: ensure prefix stuck even if the replace missed.
try:
    if getattr(router, "prefix", None) != "/api/nihongo":  # type: ignore[name-defined]
        router.prefix = "/api/nihongo"  # type: ignore[name-defined]
except Exception:
    pass
