"""
main.py

Aiko-chan — thin entry point.

Usage:
    python main.py               # browser WebUI (default) — full voice, ASR + TTS
    python main.py --text        # WebUI, keyboard input + TTS/ASR toggled off
    python main.py --no-asr      # WebUI, keyboard input but keep TTS on
    python main.py --cli         # plain no-curses CLI, for local testing only
    # Two-way messenger adapters run beside WebUI/CLI when AIKO_MESSENGER_ADAPTERS is set.
    python main.py --debug       # show memory debug info each turn + verbose console logging
    python main.py --clear-mem   # wipe all stored memories and exit
    python main.py --logout      # clear stored CLI (GitHub OAuth) auth token and exit
    python main.py --name <name> # set CLI display name (only when GitHub OAuth isn't configured)

This module only parses arguments and dispatches to the right front end:
    - interface/webui/webui.py  -> run_webui(args)   (default)
    - interface/cli/cli.py      -> run_cli(args)     (--cli)

main.py does NOT call AikoWakeup().boot() itself — each front end owns its
own boot timing, because the two have genuinely different requirements:
    - WebUI (interface/webui/webui.py, run_webui()): boot runs to completion
      BEFORE the HTTP/WS server opens (constructed with defer_servers=True),
      so browsers never see a half-booted Aiko. Post-login work (memory
      cleanup, playbook/social seeding) runs in PARALLEL after the first
      authenticated connect via system/prepare.run_post_auth().
    - CLI (interface/cli/cli.py, run_cli()): boot happens inside
      system/orchestrate.py's run_session(ui, args), using AikoSimpleCLI's
      own step_loading/step_done/step_skip methods as the boot callbacks —
      there's no separate browser to keep responsive, so a single blocking
      boot before the prompt appears is the right tradeoff there.
Both paths converge on system/orchestrate.py:run_session(ui, args) for the
actual turn loop (main loop, commands, proactive idle check-ins, karaoke
typewriter, latency/debug accounting) — see that module for details.

Flow:

                                      parse_args()
                                          │
        ┌────────────────┼────────────────┼─────────────────┐
        ▼                ▼                ▼                 ▼
   --clear-mem       --logout          --cli           (default)
        │                │                │                 │
        ▼                ▼                ▼                 ▼
  AikoMemorize()    handle_logout()    run_cli(args)  run_webui(args)
     .clear()            │                │                 │
        │                ▼                ▼                 ▼
        ▼           SystemExit(0)  boot inside      AikoWeb(defer_servers=True)
   SystemExit(0)                   run_session(),    boot runs to completion,
                                    then turn loop    THEN server opens; post-auth
                                                      init via system/prepare.py

Front-end imports are deferred into main() rather than done at module load,
so that --clear-mem and --logout (which don't need FastAPI, uvicorn,
websockets, or any voice subsystem) stay fast and don't require those
dependencies to be installed at all. The heavy AikoMemorize memory stack is
likewise deferred into the --clear-mem branch only, so normal WebUI/CLI
launches never pay for it at import time.

Argument parsing happens BEFORE logging setup (not after) specifically so
--debug can flip LOG_CONSOLE/LOG_LEVEL in the environment before
system.log's root-logger configuration runs — see the --debug handling
in main() below and system/log.py's module docstring for why import-time
resolution would be too late here.
"""
from __future__ import annotations            # evaluates type annotations later

# Public libraries
import warnings                               # for filtering out the warning messages
# Suppress transformers FutureWarning unconditionally; cheap global call, acceptable even with deferred imports
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import argparse                               # for parsing CLI arguments
import os as _os                              # for intercepting hard exits
import traceback as _tb                       # for logging exit origins
from importlib.metadata import PackageNotFoundError, version as _pkg_version  # for --version, single source of truth

_real_os_exit = _os._exit                     # keep the real hard-exit handle

__all__ = ["parse_args", "main"]              # external API — internal defs keep leading _


def _resolve_version() -> str:
    """Read the installed package version from pyproject.toml metadata.

    Avoids hardcoding the version string a second time in argparse (which
    drifts from pyproject.toml the moment one of the two is bumped and the
    other isn't). Falls back to a placeholder if the package metadata isn't
    installed/discoverable (e.g. running straight from a checkout without
    `pip install -e .`).
    """
    try:
        return _pkg_version("Aiko-chan")       # must match [project].name in pyproject.toml
    except PackageNotFoundError:
        return "0.0.0-dev"


def _setup_exit_logging(log) -> None:  # type: ignore[no-untyped-def]
    """Apply os._exit() wrapper for diagnostic logging (only if AIKO_TRACE_EXIT=1)."""
    if _os.environ.get("AIKO_TRACE_EXIT") != "1":
        return

    def _logged_os_exit(code):                # os._exit() cannot be caught by try/except,
        try:
            log.error("[main] os._exit(%s) called from:\n%s",  # so wrap it to log WHO called it before dying
                      code, "".join(_tb.format_stack()))
        except Exception:                     # if logging itself fails (e.g., during shutdown),
            pass                              # don't let traceback formatting block the actual exit
        finally:
            _real_os_exit(code)               # then still perform the hard exit

    _os._exit = _logged_os_exit               # patch applied; any code saving _os._exit before this bypasses logging


def _run_trapped(log, label, fn) -> None:  # type: ignore[no-untyped-def]
    """Run a function with fatal-error logging; re-raise on exception."""
    try:
        fn()
    except Exception:
        log.exception("[main] fatal error in %s", label)
        raise


def parse_args() -> argparse.Namespace:
    """Parse and return the CLI argument namespace for Aiko-chan's launch options."""
    p = argparse.ArgumentParser(description="Aiko-chan")          # create argument object for declaring arguments
    p.add_argument("--text",      action="store_true",            # text (keyboard) input only
                   help="keyboard input + TTS/ASR initially off; both subsystems still load for /voice and /listen toggles")
    p.add_argument("--no-asr",    action="store_true",            # disable ASR
                   help="keyboard input but keep TTS on; ASR still loads for /listen")
    p.add_argument("--debug",     action="store_true",            # debug mode
                   help="show memory hits each turn, and enable verbose console logging (sets LOG_CONSOLE=1, LOG_LEVEL=DEBUG)")
    p.add_argument("--cli",       action="store_true",            # launch in CLI
                   help="use the plain no-curses CLI instead of the WebUI — for local testing only")
    g = p.add_mutually_exclusive_group()      # prevent conflicting exits (industrial: --clear-mem vs --logout)
    g.add_argument("--clear-mem", action="store_true",            # wipe out all memory and exit
                   help="WARNING: irreversibly wipes all stored memories, then exits")
    g.add_argument("--logout",   action="store_true",             # logout user session
                   help="clear stored CLI auth token and exit")
    p.add_argument("--name",     type=str, default="",            # for use in CLI mode without OAuth setup
                   help="set display name (CLI mode only, ignored with GitHub OAuth)")
    p.add_argument("--version", action="version", version=f"%(prog)s {_resolve_version()}")  # reads pyproject.toml via importlib.metadata
    args = p.parse_args()                                         # return namespace of the arguments
    if args.name and not args.cli:            # validate display name only meaningful in CLI (industrial: early fail)
        p.error("--name requires --cli")
    return args


def _handle_clear_mem(log) -> int:  # type: ignore[no-untyped-def]
    """Handle --clear-mem branch (extracted to reduce main() complexity C901)."""
    try:
        confirm = input("WARNING: This will permanently erase all memories. Continue? [y/N]: ").strip().lower()  # prompt for user confirm memory wiping
    except (EOFError, KeyboardInterrupt):           # Ctrl-D or Ctrl-C during prompt
        print("\nAborted.")                         # quiet abort message
        return 0                                 # exit code 0
    if confirm != "y":                              # anything other than explicit 'y' aborts
        print("Aborted memory clear.")              # user-facing message
        return 0                                 # exit code 0 (user chose to abort, not an error)
    log.info("Clearing all memories...")            # log success info
    from cognition.memory.memorize import AikoMemorize  # deferred — heavy memory stack, only needed for --clear-mem

    def do_wipe():
        """Initialize memory system and clear all stored memories."""
        mem = AikoMemorize()                        # load memory system
        mem.clear()                                 # wipe out memory

    _run_trapped(log, "memory wipe (--clear-mem)", do_wipe)
    log.info("Memory cleared.")                     # log completion
    return 0                                     # exit code 0


def _handle_logout(log) -> int:  # type: ignore[no-untyped-def]
    """Handle --logout branch (extracted to reduce main() complexity)."""
    try:
        from interface.cli.cli import handle_logout  # load CLI logout handler (may fail if CLI deps missing)
    except ImportError as e:
        log.error("Could not load CLI logout handler (missing dependencies?): %s", e)
        return 1
    _run_trapped(log, "handle_logout()", handle_logout)
    return 0                                     # exit code 0


def main() -> int:
    """Primary entry point for the Aiko-chan application."""
    # Parse args FIRST — --debug needs to set LOG_CONSOLE/LOG_LEVEL in the
    # environment before system.log's root logger is configured below.
    # (system.log resolves its config on the first get_logger() call, not
    # at import time, precisely so this ordering works.)
    args = parse_args()

    # Load config early, before any subsystem init (but after filters, before logging setup)
    from system.config import load_config
    load_config()

    if args.debug:                                      # explicit CLI flag overrides whatever .env set
        _os.environ["LOG_CONSOLE"] = "1"
        _os.environ["LOG_LEVEL"] = "DEBUG"
        # Auto-enable the per-step brain tracer. --debug now implies a
        # full pipeline trace (system/brain_trace.py) — env-only knob so
        # normal launches are unaffected.
        _os.environ.setdefault("AIKO_TRACE_BRAIN", "1")

    # Set up logging and exit tracing
    from system.log import get_logger
    log = get_logger(__name__)
    _setup_exit_logging(log)

    if args.clear_mem:                                  # if clear memory argument set
        return _handle_clear_mem(log)

    if args.logout:                                     # if logout argument set
        return _handle_logout(log)

    try:                                                # one shared fatal-error trap for both front ends:
        if args.cli:                                    # SystemExit in the main thread exits SILENTLY (no traceback),
            from interface.cli.cli import run_cli       # so log WHO escaped before re-raising; BaseException catch-all
            run_cli(args)                               # covers KeyboardInterrupt and anything else unexpected.
        else:
            from interface.webui.webui import run_webui
            run_webui(args)
    except SystemExit as e:                             # silent-killer trap: SystemExit in the main thread
        if e.code not in (0, None):                     # non-zero exit = abnormal shutdown
            log.exception("[main] SystemExit(%r) escaped the session loop", e.code)
        else:                                           # exit(0) or exit() = clean shutdown
            log.info("[main] clean exit")
        raise                                           # preserve original exit behavior
    except KeyboardInterrupt:                           # graceful interrupt handling
        log.info("[main] KeyboardInterrupt")
        raise
    except Exception:                                   # any other fatal error (not BaseException to avoid catching asyncio cancels)
        log.exception("[main] fatal error escaped the session loop")  # full traceback to aiko.log
        raise                                           # re-raise after logging
    return 0


if __name__ == "__main__":
    raise SystemExit(main())                                              # start the entry point
