"""Backward-compat shim: lingo backend moved to interface.android_app.lingo.

New code should import from interface.android_app.lingo directly.
"""

import warnings

warnings.warn(
    "interface.webui.lingo is deprecated; use interface.android_app.lingo",
    DeprecationWarning,
    stacklevel=2,
)

from interface.android_app.lingo.router import *  # noqa: F401,F403
from interface.android_app.lingo.router import router  # noqa: F401

try:
    from interface.android_app.lingo.srs import (  # noqa: F401
        LingoSRS,
        ReviewGrade,
        LingoVocabCard,
        ReviewLog,
        init_srs_db,
    )
except Exception:  # pragma: no cover
    pass

try:
    from interface.android_app.lingo.vocab import (  # noqa: F401
        VocabExtractor,
        MemoryBridge,
    )
except Exception:  # pragma: no cover
    pass
