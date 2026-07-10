"""core package bootstrap.

Importing any core module should see config/*.yaml and local secrets loaded
before module-level os.getenv constants are evaluated. Individual entrypoints
still call load_config() explicitly; this package-level safety net keeps direct
imports consistent.
"""

from core.config import load_config

load_config()
