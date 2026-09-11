# TEMPORARILY BROKEN in this branch.
# Restore before merge:
#   git checkout dev -- interface/webui/auth.py
#
# Shogi does NOT require auth.py changes — it mounts from lingo/__init__.py.
raise RuntimeError(
    "auth.py was truncated during PR prep. "
    "Restore with: git checkout dev -- interface/webui/auth.py"
)
