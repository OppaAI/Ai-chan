"""
main.py

Aiko-chan — thin entry point.

Usage:
    python main.py               # browser WebUI (default) — full voice, ASR + TTS
    python main.py --text        # WebUI, keyboard input + TTS/ASR toggled off
    python main.py --no-asr      # WebUI, keyboard input but keep TTS on
    python main.py --cli         # plain no-curses CLI, for local testing only
    python main.py --debug       # show memory debug info each turn
    python main.py --clear-mem   # wipe all stored memories and exit
    python main.py --logout      # clear stored CLI (GitHub OAuth) auth token and exit

This module only parses arguments and dispatches to the right front end:
    - interface/webui/webui.py  -> run_webui(args)   (default)
    - interface/cli/cli.py      -> run_cli(args)     (--cli)
Both front ends converge on the same shared boot/turn-loop logic in
system/orchestrate.py:run_session(ui, args) — see that module for the
actual session orchestration (subsystem boot, main loop, commands,
proactive idle check-ins, karaoke typewriter, latency/debug accounting).

Front-end imports are deferred into main() rather than done at module load,
so that --clear-mem and --logout (which don't need FastAPI, uvicorn,
websockets, or any voice subsystem) stay fast and don't require those
dependencies to be installed at all.

Removed in this pass (dead code found while splitting main.py up):
    The old main.py built a second, never-served `FastAPI()` +
    `app.include_router(auth_app.router)` at module level. AikoWeb._run_http
    (interface/webui/webui.py) always serves `auth_app` directly via its own
    uvicorn.Config — that wrapper object was never mounted, never passed to
    uvicorn, and never reachable. If some external deploy script runs
    `uvicorn main:app`, it was already pointing at a dead app with no
    routes actually serving traffic; that call needs to target
    `interface.webui.auth:app` (or wherever the live app now lives) instead.
    Flagging this explicitly since it's an external-facing assumption I
    can't verify from here — please check your deploy config before pulling
    this in.
"""
from __future__ import annotations

from system.config import load_config
load_config()

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

from system.log import get_logger, silent_stderr

log = get_logger(__name__)

with silent_stderr():
    from memory.memorize import AikoMemorize


def parse_args():
    """Parse and return the CLI argument namespace for Aiko-chan's launch options."""
    p = argparse.ArgumentParser(description="Aiko-chan")
    p.add_argument("--text",      action="store_true",
                   help="keyboard input + TTS/ASR initially off; both subsystems still load for /voice and /listen toggles")
    p.add_argument("--no-asr",    action="store_true",
                   help="keyboard input but keep TTS on; ASR still loads for /listen")
    p.add_argument("--debug",     action="store_true",
                   help="show memory hits each turn")
    p.add_argument("--cli",       action="store_true",
                   help="use the plain no-curses CLI instead of the WebUI — for local testing only")
    p.add_argument("--clear-mem", action="store_true",
                   help="wipe all stored memories and exit")
    p.add_argument("--logout",   action="store_true",
                   help="clear stored CLI auth token and exit")
    p.add_argument("--name",     type=str, default="",
                   help="set your display name for CLI mode (skips GitHub OAuth name)")
    return p.parse_args()


def main():
    """Primary entry point for the Aiko-chan CLI."""
    args = parse_args()

    if args.clear_mem:
        log.info("Clearing all memories...")
        m = AikoMemorize()
        m.clear()
        sys.exit(0)

    if args.logout:
        from interface.cli.cli import handle_logout
        handle_logout()
        sys.exit(0)

    if args.cli:
        from interface.cli.cli import run_cli
        run_cli(args)
    else:
        from interface.webui.webui import run_webui
        run_webui(args)


if __name__ == '__main__':
    main()