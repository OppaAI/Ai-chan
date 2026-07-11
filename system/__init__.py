"""system package bootstrap.

Importing any system module should see config/*.yaml and local secrets loaded
before module-level os.getenv constants are evaluated. Individual entrypoints
still call load_config() explicitly; this package-level safety net keeps direct
imports consistent.
"""

from system.config import load_config

load_config()
