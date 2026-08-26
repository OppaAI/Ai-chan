"""
main.py

Aiko-chan — thin entry point.

Usage:
    python main.py               # browser WebUI (default) — full voice, ASR + TTS
    python main.py --text        # WebUI, keyboard input + TTS/ASR toggled off
    python main.py --no-asr      # WebUI, keyboard input but keep TTS on
    python main.py --cli         # plain no-curses CLI, for local testing only
    # Two-way messenger adapters run beside WebUI/CLI when AIKO_MESSENGER_ADAPTERS is set.
    python main.py --debug       # show memory debug info each turn
    python main.py --clear-mem   # wipe all stored memories and exit
    python main.py --logout      # clear stored CLI (GitHub OAuth) auth token and exit

This module only parses arguments and dispatches to the right front end:
    - interface/webui/webui.py  -> run_webui(args)   (default)
    - interface/cli/cli.py      -> run_cli(args)     (--cli)

main.py does NOT call AikoWakeup().boot() itself — each front end owns its
own boot timing, because the two have genuinely different requirements:
    - WebUI (interface/webui/webui.py, run_webui()): AikoWeb's __init__
      starts the HTTP/WS server immediately (frontend loads and shows
      "Initializing..." right away), then boot runs on a background thread;
      AikoWeb.set_boot_result() broadcasts a "ready" event to connected
      browsers once it finishes. Login is gated on that event, not on
      main.py's control flow.
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
        ▼           sys.exit(0)   boot inside      AikoWeb() starts server
   sys.exit(0)                    run_session(),    instantly; boot runs on
                                   then turn loop    a background thread,
                                                      then turn loop

Front-end imports are deferred into main() rather than done at module load,
so that --clear-mem and --logout (which don't need FastAPI, uvicorn,
websockets, or any voice subsystem) stay fast and don't require those
dependencies to be installed at all.
"""
from __future__ import annotations            # evaluates type annotations later

# Public libraries
import warnings                               # for filtering out the warning messages
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import argparse                               # for parsing CLI arguments
import sys                                    # for assigning exit code

# Aiko's components
from system.config import load_config                             # load user configs
load_config()
from system.log import get_logger                                 # assign logging to universal logger
log = get_logger(__name__)

from cognition.memory.memorize import AikoMemorize                # load memory system for --clear-mem


def parse_args():
    """Parse and return the CLI argument namespace for Aiko-chan's launch options."""
    p = argparse.ArgumentParser(description="Aiko-chan")          # create argument object for declaring arguments
    p.add_argument("--text",      action="store_true",            # text (keyboard) input only
                   help="keyboard input + TTS/ASR initially off; both subsystems still load for /voice and /listen toggles")
    p.add_argument("--no-asr",    action="store_true",            # disable ASR
                   help="keyboard input but keep TTS on; ASR still loads for /listen")
    p.add_argument("--debug",     action="store_true",            # debug mode
                   help="show memory hits each turn")
    p.add_argument("--cli",       action="store_true",            # launch in CLI
                   help="use the plain no-curses CLI instead of the WebUI — for local testing only")
    p.add_argument("--clear-mem", action="store_true",            # wipe out all memory and exit
                   help="WARNING: irreversibly wipes all stored memories, then exits")
    p.add_argument("--logout",   action="store_true",             # logout user session
                   help="clear stored CLI auth token and exit")
    p.add_argument("--name",     type=str, default="",            # for use in CLI mode without OAuth setup
                   help="set your display name for CLI mode (only used when GitHub OAuth isn't configured)")
    return p.parse_args()                                         # return namespace of the arguments


def main():
    """Primary entry point for the Aiko-chan application."""
    args = parse_args()                                 # assign argument namespace to check which ones are set

    if args.clear_mem:                                  # if clear memory argument set
        confirm = input("WARNING: This will permanently erase all memories. Continue? [y/N]: ").strip().lower()  # prompt for user confirm memory wiping
        if confirm != "y":                              # anything other than explicit 'y' aborts
            log.info("Aborted memory clear.")           # log abort info
            sys.exit(1)                                 # exit code 1 (aborted, not an error but not success either)
        log.info("Clearing all memories...")            # log success info
        mem = AikoMemorize()                            # load memory system
        mem.clear()                                     # wipe out memory
        sys.exit(0)                                     # exit code 0

    if args.logout:                                     # if logout argument set
        from interface.cli.cli import handle_logout     # load CLI logout handler
        handle_logout()                                 # logout user session
        sys.exit(0)                                     # exit code 0

    if args.cli:                                        # if CLI argument set
        from interface.cli.cli import run_cli           # load CLI with set arguments
        run_cli(args)                                   # launch CLI — boots internally via run_session()
    else:                                               # otherwise, launch WebUI
        from interface.webui.webui import run_webui     # load WebUI
        run_webui(args)                                 # launch WebUI — server starts instantly, boots in background


if __name__ == "__main__":
    main()                                              # start the entry point
