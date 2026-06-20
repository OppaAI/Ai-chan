"""
tui/tui.py
Aiko-chan's full-screen curses TUI visual cortex.

Exports:
    AikoTUI         — the main TUI class
    init_colours()  — curses colour pair registration
    ARCH_SECTIONS   — static neural architecture display data
    INIT_STEPS      — boot step label/detail registry
    SPINNER         — animation frame sequence

Layout:
    ╔══════════════════════════════════════════════════════════╗
    ║                      BANNER                              ║
    ╠══════════════════╦═══════════════════════════════════════╣
    ║  ASCII ART       ║  INIT LOG  /  ARCH INFO               ║
    ║  (62 wide)       ╠═══════════════════════════════════════╣
    ║                  ║  CHAT MESSAGES                        ║
    ║                  ╠═══════════════════════════════════════╣
    ║                  ║  VITALS BAR                           ║
    ║                  ╠═══════════════════════════════════════╣
    ║                  ║  INPUT                                ║
    ╚══════════════════╩═══════════════════════════════════════╝
"""

import curses
import os
from pathlib import Path
import textwrap
import threading
import time

from tui.identity import (
    BANNER_LINES, BANNER_H,
    ANIME_ART_LINES, ANIME_ART_COLORS,
    ART_W, ART_H,
)
from core.health import _ram_used_str, _db_size_str, _fmt_uptime

# ── env ───────────────────────────────────────────────────────────────────────

LLM_BASE_URL  = os.getenv("LLM_BASE_URL", "unknown")
LLM_MODEL     = os.getenv("LLM_MODEL", "unknown")
ASR_MODEL     = os.getenv("ASR_MODEL", os.getenv("ASR_MODE", "csukuangfj/reazonspeech-k2-v2-ja-en"))
MIOTTS_MODEL  = os.getenv("MIOTTS_MODEL", "MioTTS 0.4B")
SEARXNG_URL   = os.getenv("SEARXNG_URL",   "localhost:8080")
AI_NAME       = os.getenv("AI_NAME", "Aiko")
USER_ID       = os.getenv("USER_ID", "")
LEFT_W        = ART_W + 2

# ── colour pairs ──────────────────────────────────────────────────────────────

CP_PINK    = 1
CP_CYAN    = 2
CP_PURPLE  = 3
CP_MAUVE   = 4
CP_DIM     = 5
CP_WHITE   = 6
CP_SBARBG  = 7
CP_INPUTBG = 8
CP_ART_BASE= 8

ART_PALETTE = [None, 236, 239, 243, 246, 249, 253, 95, 138, 181]


def init_colours() -> None:
    """
    Awaken the terminal's chromatic pathways by registering all colour pairs
    used across the TUI's visual cortex.

    Initialises curses colour mode with transparent backgrounds (-1) for all
    standard UI pairs, plus the art palette block starting at CP_ART_BASE.
    Must be called once inside the curses wrapper before any drawing begins.
    """
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(CP_PINK,    198,  -1)
    curses.init_pair(CP_CYAN,     51,  -1)
    curses.init_pair(CP_PURPLE,  135,  -1)
    curses.init_pair(CP_MAUVE,   177,  -1)
    curses.init_pair(CP_DIM,     245,  -1)
    curses.init_pair(CP_WHITE,    15,  -1)
    curses.init_pair(CP_SBARBG,   16, 198)
    curses.init_pair(CP_INPUTBG,  51,  -1)
    for idx, color256 in enumerate(ART_PALETTE):
        if color256 is not None:
            curses.init_pair(CP_ART_BASE + idx, color256, -1)


def _art_attr(palette_idx: int) -> int:
    """
    Resolve an art palette index to a curses attribute ready for rendering.

    Palette indices 5, 6, and 9 correspond to the brightest tones in the
    portrait and receive A_BOLD to punch them forward on dim terminals.
    Index 0 is the transparency sentinel; it returns -1 to signal the caller
    to render a space with no attribute.

    Args:
        palette_idx: Index into ART_PALETTE (0–9).

    Returns:
        A curses attribute int, or -1 for transparent cells.
    """
    if palette_idx == 0:
        return -1
    attr = curses.color_pair(CP_ART_BASE + palette_idx)
    if palette_idx in (5, 6, 9):
        attr |= curses.A_BOLD
    return attr


# ── display config ────────────────────────────────────────────────────────────

ARCH_SECTIONS = [
    ("MEMORY SYSTEMS", [
        ("Long-term store", "SQLite-vec DB"),
        ("Embedding model", "BGE-base-en-v1.5  (768d)"),
        ("Short-term ctx",  f"Rolling {os.getenv('CONTEXT_WINDOW_TURNS', '20')}-turn window"),
    ]),
    ("COGNITION", [
        ("Inference engine", "Ollama  (local, offline)"),
        ("Active model",     LLM_MODEL),
        ("Web search",       SEARXNG_URL),
    ]),
    ("VOICE ENGINE", [
        ("TTS backend",   MIOTTS_MODEL),
        ("Voice preset",  os.getenv("MIOTTS_PRESET", "jp_female")),
        ("ASR model",     ASR_MODEL),
    ]),
]

ARCH_ROWS = sum(1 + len(items) for _, items in ARCH_SECTIONS) + 2
db_path = os.getenv("SQLITE_MEMORY_PATH", str(Path.home() / ".aiko" / "memory.db"))

INIT_STEPS = {
    'think_start':      ('Inference Engine', f'Spawning Ollama worker  ·  {LLM_MODEL}'),
    'think_warmup':     ('Model Warm-up',    'Loading weights, running prefill pass …'),
    'mem_sqlite_vec':   ('Vector Database',  f'Connecting to SQLite-vec  ·  {db_path}'),
    'mem_embed':        ('Embedding Model',  'Loading BGE-base-en-v1.5  ·  768-dim vectors'),
    'mem_cleanup':      ('Memory Lifecycle', 'Pruning decayed memories …'),
    'mem_ready':        ('Memory Cortex',    'OppaAI custom-build  ·  long-term recall online'),
    'speak_miotts':     ('TTS Engine',       f'Initializing MioTTS  ·  {os.getenv("MIOTTS_PRESET", "jp_female")}'),
    'speak_ready':      ('Voice Output',     'Audio pipeline ready  ·  24 kHz'),
    'speak_skip':       ('Voice Output',     'TTS disabled  (--text mode)'),
    'listen_ready':     ('Speech Input',     f'ReazonSpeech ready  ·  {ASR_MODEL}'),
    'listen_skip':      ('Speech Input',     'ASR disabled  (--text mode)'),
}

SPINNER = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']


# ═════════════════════════════════════════════════════════════════════════════
# AikoTUI
# ═════════════════════════════════════════════════════════════════════════════

class AikoTUI:
    """
    The visual cortex of Aiko-chan's terminal presence.

    Manages the full-screen curses layout, rendering all UI regions — banner,
    left-column portrait, right-column init log / arch panel, chat area, vitals
    bar, and input line — and exposes a thread-safe API for the cognitive
    subsystems to inject messages, stream tokens, and update session metrics.
    """

    INPUT_PROMPT = "  ❯  "

    def __init__(self, stdscr, no_voice: bool = False, debug: bool = False):
        """
        Initialise the TUI's sensory and display apparatus.

        Args:
            stdscr:   The curses window object provided by curses.wrapper.
            no_voice: When True, ASR and TTS are both dormant (--text mode).
            debug:    When True, memory retrieval hits are surfaced each turn.
        """
        self._scr      = stdscr
        self._no_voice = no_voice
        self._debug    = debug
        self._lock     = threading.Lock()
        self._ts       = time.time()

        self._messages: list[tuple[str, str]] = []
        self._scroll    = 0
        self._streaming = ''

        self._init_log: list[tuple[str, str, str]] = []
        self._frame     = 0
        self._phase     = 'init'
        self._input_buf: list[str] = []

        self._stats: dict = {
            'tokens':     0,
            'turn_tok':   0,
            'turn_start': None,
            'tok_s':      0.0,
            'asr_on':     not no_voice,
            'tts_on':     not no_voice,
        }

        init_colours()
        curses.curs_set(0)
        self._scr.nodelay(False)

    # ── layout ────────────────────────────────────────────────────────────────

    @property
    def _pt(self):
        """Row index where the left and right columns begin (below the banner border)."""
        return BANNER_H + 2

    @property
    def _arch_sep_row(self):
        """Row index of the horizontal separator between the arch/init panel and chat area."""
        return self._pt + ARCH_ROWS

    @property
    def _chat_top(self):
        """First row of the chat message region."""
        return self._arch_sep_row + 1

    def _dims(self):
        """Return the current terminal dimensions as (height, width)."""
        return self._scr.getmaxyx()

    def _chat_bot(self, h):
        """Return the last row of the chat message region given terminal height h."""
        return h - 6

    # ── low-level write ───────────────────────────────────────────────────────

    def _wr(self, y, x, text, attr=0):
        """
        Safely emit a string to the terminal at the given coordinates.

        Clips text to the available column width and uses insstr on the final
        cell of the last row to avoid the curses bottom-right corner exception.
        Silently swallows all curses errors so a resize mid-draw never crashes.
        """
        h, w = self._dims()
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        avail = w - x
        if avail <= 0:
            return
        try:
            if y == h - 1 and x + len(text) >= w:
                self._scr.addstr(y, x, text[:w - x - 1], attr)
                try:
                    self._scr.insstr(y, w - 1, text[w - x - 1:w - x], attr)
                except curses.error:
                    pass
            else:
                self._scr.addstr(y, x, text[:avail], attr)
        except curses.error:
            pass

    # ── banner ────────────────────────────────────────────────────────────────

    def _draw_banner(self, h, w):
        """Render the top banner region: decorative header, identity lines, clock, version."""
        pk = curses.color_pair(CP_PINK)  | curses.A_BOLD
        wh = curses.color_pair(CP_WHITE)
        cy = curses.color_pair(CP_CYAN)  | curses.A_BOLD

        self._wr(0, 0, '╔' + '═' * (w - 2) + '╗', pk)

        for i, line in enumerate(BANNER_LINES):
            row = 1 + i
            pad = max(1, (w - len(line)) // 2)
            self._wr(row, 1,   ' ' * (w - 2), 0)
            self._wr(row, 0,   '║', pk)
            self._wr(row, w-1, '║', pk)
            self._wr(row, pad, line, pk)

        clock_str = f" {time.strftime('%B %d, %Y  %I:%M:%S %p')} "
        self._wr(1, w - 1 - len(clock_str), clock_str, wh)

        ver_str = f" {AI_NAME} v0.1.99 "
        self._wr(2, w - 1 - len(ver_str), ver_str, cy)

        self._wr(BANNER_H + 1, 0,
            '╠' + '═' * (LEFT_W - 1) + '╦' + '═' * (w - LEFT_W - 2) + '╣', pk)

    def _draw_clock_only(self):
        """Refresh only the clock and cursor position without redrawing the full frame."""
        with self._lock:
            h, w = self._dims()
            wh = curses.color_pair(CP_WHITE) | curses.A_BOLD
            clock_str = f" {time.strftime('%B %d, %Y  %I:%M:%S %p')} "
            self._wr(1, w - 1 - len(clock_str), clock_str, wh)   # row 1 = banner row
            rx    = LEFT_W + 1
            inp_r = self._chat_bot(h) + 4
            content = self.INPUT_PROMPT + ''.join(self._input_buf)
            cx = min(rx + len(content), w - 2)
            try:    self._scr.move(inp_r, cx)
            except: pass
            self._scr.refresh()

    # ── left column ───────────────────────────────────────────────────────────

    def _draw_left_col(self, h, w):
        """Paint the left column with Aiko's ASCII art portrait."""
        pk  = curses.color_pair(CP_PINK) | curses.A_BOLD
        pt  = self._pt
        bot = h - 2

        for row in range(pt, bot + 1):
            art_idx = row - pt
            self._wr(row, 0, '║', pk)

            if art_idx < ART_H and art_idx < len(ANIME_ART_LINES):
                art_line = ANIME_ART_LINES[art_idx]
                clr_row  = ANIME_ART_COLORS[art_idx] if art_idx < len(ANIME_ART_COLORS) else []
                x = 1
                for col_idx, ch in enumerate(art_line):
                    if x >= LEFT_W:
                        break
                    pidx = clr_row[col_idx] if col_idx < len(clr_row) else 0
                    attr = _art_attr(pidx)
                    if attr < 0 or ch == ' ':
                        try:    self._scr.addch(row, x, ' ')
                        except: pass
                    else:
                        try:    self._scr.addstr(row, x, ch, attr)
                        except: pass
                    x += 1
                while x < LEFT_W:
                    try:    self._scr.addch(row, x, ' ')
                    except: pass
                    x += 1
            else:
                try:    self._scr.addstr(row, 1, ' ' * (LEFT_W - 1))
                except: pass

        self._wr(h - 1, 0, '╚', pk)

    # ── right column ──────────────────────────────────────────────────────────

    def _draw_right_col(self, h, w, buf):
        """Render the entire right column: borders, info panel, chat, vitals, input."""
        pk        = curses.color_pair(CP_PINK)   | curses.A_BOLD
        sbar_attr = curses.color_pair(CP_SBARBG) | curses.A_BOLD
        inp_attr  = curses.color_pair(CP_INPUTBG)

        pt  = self._pt
        sep = self._arch_sep_row
        cb  = self._chat_bot(h)
        rx  = LEFT_W + 1
        rw  = w - LEFT_W - 2

        for row in range(pt, h - 1):
            self._wr(row, LEFT_W, '║', pk)
            self._wr(row, w - 1,  '║', pk)

        for row in range(pt, sep):
            try:    self._scr.addstr(row, rx, ' ' * (rw - 1))
            except: pass

        if self._phase == 'init':
            self._draw_right_init(pt, sep, rx, rw)
        else:
            self._draw_right_arch(pt, sep, rx, rw)

        self._wr(sep, LEFT_W, '╠' + '═' * (w - LEFT_W - 2) + '╣', pk)
        self._draw_chat_area(sep + 1, cb, rx, rw, w)

        sbar_sep = cb + 1
        self._wr(sbar_sep, LEFT_W, '╠', pk)
        self._wr(sbar_sep, w - 1,  '╣', pk)
        try:    self._scr.addstr(sbar_sep, LEFT_W + 1, '═' * rw,
                                 curses.color_pair(CP_PINK) | curses.A_BOLD)
        except: pass

        self._draw_vitals(sbar_sep + 1, rx, rw, sbar_attr)

        ir = sbar_sep + 2
        self._wr(ir, LEFT_W, '╠', pk)
        self._wr(ir, w - 1,  '╣', pk)
        try:    self._scr.addstr(ir, LEFT_W + 1, '═' * rw,
                                 curses.color_pair(CP_PINK) | curses.A_BOLD)
        except: pass

        inp_r   = ir + 1
        content = self.INPUT_PROMPT + ''.join(buf)
        vis_start = max(0, len(content) - rw + 1)
        line = content[vis_start:].ljust(rw)
        try:    self._scr.addstr(inp_r, rx, line, inp_attr)
        except: pass

        bot_r = inp_r + 1
        if bot_r < h:
            self._wr(bot_r, LEFT_W, '╩', pk)
            self._wr(bot_r, w - 1,  '╝', pk)
            try:    self._scr.addstr(bot_r, LEFT_W + 1, '═' * rw,
                                     curses.color_pair(CP_PINK) | curses.A_BOLD)
            except: pass
        self._wr(bot_r, 0, '╚' + '═' * (LEFT_W - 1), pk)

        cx = min(rx + len(content), w - 2)
        try:    self._scr.move(inp_r, cx)
        except: pass

    # ── vitals bar ────────────────────────────────────────────────────────────

    def _draw_vitals(self, row: int, rx: int, rw: int, attr):
        """Render the live session vitals as a single status line."""
        s = self._stats

        mode_parts = []
        mode_parts.append("☊ ASR" if s['asr_on'] else "⌨  TXT")
        mode_parts.append("🎙 TTS" if s['tts_on'] else "🔇 TTS")
        mode_str = "  ".join(mode_parts)

        ram_str  = f"RAM {_ram_used_str()}"
        db_str   = f"DB {_db_size_str()}"
        tok_str  = f"{s['tokens']:,} tokens"
        toks_str = f"{s['tok_s']:.1f} t/s" if s['tok_s'] > 0 else "— t/s"
        up_str   = f"↑ {_fmt_uptime(time.time() - self._ts)}"

        segments = [mode_str, ram_str, db_str, tok_str, toks_str, up_str]
        sep      = "  │  "
        built    = ""
        for seg in segments:
            candidate = (built + sep + seg) if built else ("  " + seg)
            if len(candidate) + 2 <= rw:
                built = candidate

        bar = built.ljust(rw - 1)
        try:    self._scr.addstr(row, rx, bar[:rw - 1], attr)
        except: pass

    # ── init log ──────────────────────────────────────────────────────────────

    def _draw_right_init(self, pt, pb, rx, rw):
        """Render the boot sequence log in the upper right panel during the init phase."""
        pk  = curses.color_pair(CP_PINK)  | curses.A_BOLD
        cy  = curses.color_pair(CP_CYAN)
        dim = curses.color_pair(CP_DIM)
        wh  = curses.color_pair(CP_WHITE)

        self._wr(pt,     rx, " INITIALIZING NEURAL SYSTEMS", pk)
        self._wr(pt + 1, rx, ' ─' * min(14, (rw - 1) // 2), pk)

        row = pt + 2
        for key, state, detail in self._init_log:
            if row >= pb:
                break
            lbl, dflt = INIT_STEPS.get(key, (key, detail))
            txt = detail if detail else dflt
            if state == 'loading':
                sp = SPINNER[self._frame]
                self._wr(row, rx,      f"  {sp} ", wh)
                self._wr(row, rx + 4,  f"{lbl:<20}", wh)
                self._wr(row, rx + 24, txt[:rw - 25], dim)
            elif state == 'done':
                self._wr(row, rx,      "  ✓ ", cy)
                self._wr(row, rx + 4,  f"{lbl:<20}", wh)
                self._wr(row, rx + 24, txt[:rw - 25], dim)
            elif state == 'skip':
                self._wr(row, rx,      "  – ", dim)
                self._wr(row, rx + 4,  f"{lbl:<20}", dim)
                self._wr(row, rx + 24, txt[:rw - 25], dim)
            elif state == 'error':
                self._wr(row, rx,      "  ✗ ", wh)
                self._wr(row, rx + 4,  f"{lbl:<20}", wh)
                self._wr(row, rx + 24, txt[:rw - 25], curses.color_pair(CP_MAUVE))
            row += 1

        all_fin = (len(self._init_log) > 0 and
                   all(s in ('done', 'skip', 'error') for _, s, _ in self._init_log))
        if all_fin and row < pb - 1:
            self._wr(row + 1, rx, "  [ ALL SYSTEMS ONLINE ]",
                     curses.color_pair(CP_CYAN) | curses.A_BOLD)

    # ── arch panel ────────────────────────────────────────────────────────────

    def _draw_right_arch(self, pt, pb, rx, rw):
        """Render the static neural architecture summary in the upper right panel."""
        pk = curses.color_pair(CP_PINK)  | curses.A_BOLD
        cy = curses.color_pair(CP_CYAN)  | curses.A_BOLD
        mv = curses.color_pair(CP_MAUVE)
        wh = curses.color_pair(CP_WHITE)

        self._wr(pt,     rx, " NEURAL ARCHITECTURE", pk)
        self._wr(pt + 1, rx, ' ─' * min(10, (rw - 1) // 2), pk)

        row = pt + 2
        for section, items in ARCH_SECTIONS:
            if row >= pb - 1:
                break
            self._wr(row, rx, f"  {section}", cy)
            row += 1
            for name, val in items:
                if row >= pb:
                    break
                self._wr(row, rx,      f"    {name:<18}", wh)
                self._wr(row, rx + 22, val[:rw - 23],    mv)
                row += 1

    # ── chat area ─────────────────────────────────────────────────────────────

    def _render_lines(self, rw):
        """
        Flatten the message log and active streaming buffer into a list of
        typed display lines suitable for the chat viewport.
        """
        avail = rw - 2
        out   = []
        for sender, text in self._messages:
            if sender == 'you':
                pre = f" {USER_ID}: "
                ind = " " * len(pre)
                wrp = textwrap.wrap(text, avail - len(pre)) or [""]
                out.append(('Y', pre + wrp[0]))
                for l in wrp[1:]: out.append(('Y', ind + l))
                out.append(('S', ''))
            elif sender == 'aiko':
                pre = f" {AI_NAME}: "
                ind = " " * len(pre)
                wrp = textwrap.wrap(text, avail - len(pre)) or [""]
                out.append(('A', pre + wrp[0]))
                for l in wrp[1:]: out.append(('A', ind + l))
                out.append(('S', ''))
            elif sender == 'sys':
                out.append(('S', f"  ◈  {text}"))
        if self._streaming:
            pre = f" {AI_NAME}: "
            ind = " " * len(pre)
            wrp = textwrap.wrap(self._streaming, avail - len(pre)) or [""]
            out.append(('A', pre + wrp[0]))
            for l in wrp[1:]: out.append(('A', ind + l))
        return out

    def _draw_chat_area(self, ct, cb, rx, rw, w):
        """Render the visible slice of the chat history into the chat viewport."""
        pk  = curses.color_pair(CP_PINK)   | curses.A_BOLD
        cy  = curses.color_pair(CP_CYAN)
        mv  = curses.color_pair(CP_MAUVE)
        dim = curses.color_pair(CP_DIM)
        pu  = curses.color_pair(CP_PURPLE)

        ch       = cb - ct
        rendered = self._render_lines(rw)
        total    = len(rendered)
        self._scroll = max(0, min(self._scroll, max(0, total - ch)))
        start    = max(0, total - ch - self._scroll)
        visible  = rendered[start:start + ch]

        for i, (kind, line) in enumerate(visible):
            row = ct + i
            if row >= cb:
                break
            if kind == 'Y':
                prefix    = f" {USER_ID}: "
                bold_attr = cy | curses.A_BOLD
                text_attr = cy
            elif kind == 'A':
                prefix    = f" {AI_NAME}: "
                bold_attr = mv | curses.A_BOLD
                text_attr = mv
            else:
                prefix    = ""
                bold_attr = dim
                text_attr = dim

            try:    self._scr.addstr(row, rx, ' ' * (rw - 1))
            except: pass

            if prefix and line.startswith(prefix):
                self._wr(row, rx, prefix, bold_attr)
                self._wr(row, rx + len(prefix),
                         line[len(prefix):][:rw - len(prefix) - 1], text_attr)
            else:
                self._wr(row, rx, line[:rw - 1], text_attr)
            self._wr(row, w - 1, '║', pk)

        for row in range(ct + len(visible), cb):
            try:    self._scr.addstr(row, rx, ' ' * (rw - 1))
            except: pass
            self._wr(row, w - 1, '║', pk)

        if self._scroll > 0:
            hint = f" ↑{self._scroll}  PgDn↓ "
            self._wr(ct, w - len(hint) - 2, hint, pu)

    # ── master draw ───────────────────────────────────────────────────────────

    def _draw(self, buf=None):
        """Execute a full repaint of the entire TUI surface under the draw lock."""
        with self._lock:
            h, w = self._dims()
            min_h = BANNER_H + 2 + ARCH_ROWS + 8
            min_w = LEFT_W + 40
            if h < min_h or w < min_w:
                self._scr.clear()
                self._wr(0, 0, f"Terminal too small: {w}x{h}  (need {min_w}x{min_h})", 0)
                self._scr.refresh()
                return
            if buf is None:
                buf = []
            self._draw_banner(h, w)
            self._draw_left_col(h, w)
            self._draw_right_col(h, w, buf)
            self._scr.refresh()

    # ── init API ──────────────────────────────────────────────────────────────

    def _upsert_step(self, key, state, detail=''):
        """Insert or update a boot step entry in the init log under the draw lock."""
        with self._lock:
            for i, (k, s, d) in enumerate(self._init_log):
                if k == key:
                    self._init_log[i] = (key, state, detail)
                    return
            self._init_log.append((key, state, detail))

    def step_loading(self, key, detail=''):
        """Mark a boot step as actively loading (spinner state)."""
        self._upsert_step(key, 'loading', detail)

    def step_done(self, key, detail=''):
        """Mark a boot step as successfully completed (✓ state)."""
        self._upsert_step(key, 'done',    detail)

    def step_skip(self, key, detail=''):
        """Mark a boot step as intentionally skipped (– state)."""
        self._upsert_step(key, 'skip',    detail)

    def step_error(self, key, detail=''):
        """Mark a boot step as failed (✗ state)."""
        self._upsert_step(key, 'error',   detail)

    def status_finish(self):
        """Transition the TUI from the init phase to the active chat phase."""
        with self._lock:
            self._phase = 'chat'

    # ── chat API ──────────────────────────────────────────────────────────────

    def add_message(self, sender, text):
        """Commit a completed message to the conversation log."""
        with self._lock:
            self._messages.append((sender, text))
            self._scroll = 0

    def stream_token(self, token):
        """Ingest an incoming token into the live streaming buffer."""
        with self._lock:
            self._streaming += token
            count = len(token)
            self._stats['tokens']   += count
            self._stats['turn_tok'] += count
            if self._stats['turn_start'] is None:
                self._stats['turn_start'] = time.time()

    def stream_commit(self):
        """Finalise the active streaming turn and commit the buffered response."""
        with self._lock:
            if self._streaming:
                self._messages.append(('aiko', self._streaming))
                self._streaming = ''
                self._scroll    = 0
            if self._stats['turn_start'] is not None:
                elapsed = time.time() - self._stats['turn_start']
                self._stats['tok_s'] = (
                    self._stats['turn_tok'] / elapsed if elapsed > 0 else 0.0)
            self._stats['turn_tok']   = 0
            self._stats['turn_start'] = None

    def turn_start(self):
        """Signal the beginning of a new cognitive turn."""
        with self._lock:
            self._stats['turn_start'] = time.time()
            self._stats['turn_tok']   = 0

    # ── text input ────────────────────────────────────────────────────────────

    def get_input(self):
        """
        Enter a blocking text input loop, collecting keystrokes until the user
        submits with Enter or interrupts with Ctrl-C / Ctrl-D.
        """
        buf = []
        self._input_buf = buf
        curses.curs_set(1)
        self._scr.nodelay(True)

        stop_tick = threading.Event()
        def _ticker():
            while not stop_tick.is_set():
                self._draw_clock_only()
                stop_tick.wait(1.0)
        tick_t = threading.Thread(target=_ticker, daemon=True)
        tick_t.start()

        try:
            while True:
                try:
                    ch = self._scr.get_wch()
                except curses.error:
                    time.sleep(0.05)
                    continue

                h, w = self._dims()

                if ch in ('\n', '\r', curses.KEY_ENTER):
                    break
                elif ch in (curses.KEY_BACKSPACE, '\x7f', '\b'):
                    if buf: buf.pop()
                elif ch == curses.KEY_PPAGE:
                    with self._lock:
                        rw       = w - LEFT_W - 2
                        rendered = self._render_lines(rw)
                        ch_h     = self._chat_bot(h) - self._chat_top
                        self._scroll = min(
                            self._scroll + max(1, ch_h - 2),
                            max(0, len(rendered) - ch_h))
                elif ch == curses.KEY_NPAGE:
                    with self._lock:
                        ch_h = self._chat_bot(h) - self._chat_top
                        self._scroll = max(0, self._scroll - max(1, ch_h - 2))
                elif ch in ('\x03', '\x04'):
                    curses.curs_set(0)
                    stop_tick.set()
                    tick_t.join()
                    raise KeyboardInterrupt
                elif isinstance(ch, str) and ch.isprintable():
                    buf.append(ch)

                self._draw(buf=buf)
        finally:
            stop_tick.set()
            tick_t.join()
            curses.curs_set(0)

        return ''.join(buf).strip()

    # ── voice input ───────────────────────────────────────────────────────────

    def get_voice_input(self, listen, speak=None, wait_fn=None):
        """
        Capture a voice utterance via the ASR pipeline and return the transcribed text.

        Passes speak= through to listen.listen() so it can call
        speak.wait_or_barge_in() for interruptible playback. Falls back to
        wait_fn for text-mode / TTS-toggled-off sessions.

        Status tokens handled:
            __WAITING__      — Aiko is speaking; waiting for her to finish or
                               for the user to barge in
            __LISTENING__    — mic is open, capturing audio
            __TRANSCRIBING__ — ASR is processing the recorded audio
            __IDLE__         — pipeline idle, nothing captured

        Args:
            listen:  AikoListen instance.
            speak:   AikoSpeak instance, or None. When provided and is_playing()
                     is True, listen.listen() waits interruptibly via
                     speak.wait_or_barge_in(_barge_in_event).
            wait_fn: Legacy blocking callable; used when speak is None and TTS
                     is disabled at runtime. Ignored when speak is provided.
        """
        self._input_buf = []
        result_holder   = [None]
        done_event      = threading.Event()

        def _status_cb(token):
            if token == '__WAITING__':
                with self._lock:
                    self._input_buf = list("⏸  Waiting …")
            elif token == '__LISTENING__':
                with self._lock:
                    self._input_buf = list("🎤  Listening …")
            elif token == '__TRANSCRIBING__':
                with self._lock:
                    self._input_buf = list("⚙  Transcribing …")
            elif token == '__IDLE__':
                with self._lock:
                    self._input_buf = []
            self._draw(buf=list(self._input_buf))

        def _run():
            result_holder[0] = listen.listen(
                status_callback=_status_cb,
                speak=speak,
                wait_fn=wait_fn,
            )
            done_event.set()

        threading.Thread(target=_run, daemon=True).start()

        while not done_event.is_set():
            self._draw_clock_only()
            done_event.wait(1.0)

        self._input_buf = []
        self._draw(buf=[])
        return result_holder[0] or ""

    # ── spin loop ─────────────────────────────────────────────────────────────

    def spin_loop(self, stop_event):
        """Drive the init-phase animation loop from a dedicated thread."""
        while not stop_event.is_set():
            self._draw()
            with self._lock:
                self._frame = (self._frame + 1) % len(SPINNER)
            stop_event.wait(0.08)
        self._draw()
