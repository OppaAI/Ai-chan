"""
Minimal pure-Python Go board for rules + legal moves.

Supports 9×9 / 13×13 / 19×19, captures, simple ko, pass.
No external dependency — KataGo is optional for strong AI only.
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

EMPTY, BLACK, WHITE = 0, 1, 2

# GTP letters skip 'I'
_GTP_COLS = "ABCDEFGHJKLMNOPQRST"


def color_name(c: int) -> str:
    if c == BLACK:
        return "black"
    if c == WHITE:
        return "white"
    return "empty"


def opponent(c: int) -> int:
    return WHITE if c == BLACK else BLACK


def parse_gtp_move(move: str, size: int) -> Optional[Tuple[int, int]]:
    """'D4' / 'pass' / 'resign' → (row, col) with row 0 = top (GTP rank size).
    Returns None for pass. Raises ValueError for resign or invalid.
    """
    m = (move or "").strip().upper()
    if not m:
        raise ValueError("empty move")
    if m in ("PASS", "PA"):
        return None
    if m in ("RESIGN", "R"):
        raise ValueError("resign")
    col_letter = m[0]
    if col_letter not in _GTP_COLS:
        raise ValueError(f"bad column: {col_letter}")
    col = _GTP_COLS.index(col_letter)
    try:
        rank = int(m[1:])
    except ValueError as e:
        raise ValueError(f"bad rank: {m}") from e
    if col < 0 or col >= size or rank < 1 or rank > size:
        raise ValueError(f"off board: {m}")
    # GTP: rank 1 = bottom; our row 0 = top = rank size
    row = size - rank
    return row, col


def to_gtp(row: int, col: int, size: int) -> str:
    return f"{_GTP_COLS[col]}{size - row}"


class GoBoard:
    def __init__(self, size: int = 9):
        if size not in (9, 13, 19):
            raise ValueError("size must be 9, 13, or 19")
        self.size = size
        self.grid = [[EMPTY for _ in range(size)] for _ in range(size)]
        self.turn = BLACK
        self.ko: Optional[Tuple[int, int]] = None  # square opponent may not retake
        self.passes = 0
        self.history: List[str] = []  # GTP moves
        self.captured = {BLACK: 0, WHITE: 0}
        self.status = "playing"  # playing | finished | resigned

    def copy(self) -> "GoBoard":
        b = GoBoard(self.size)
        b.grid = [row[:] for row in self.grid]
        b.turn = self.turn
        b.ko = self.ko
        b.passes = self.passes
        b.history = list(self.history)
        b.captured = dict(self.captured)
        b.status = self.status
        return b

    def _neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        out = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                out.append((nr, nc))
        return out

    def _group_on(
        self,
        grid: List[List[int]],
        r: int,
        c: int,
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """Group + liberties on an arbitrary grid (supports trial stones)."""
        color = grid[r][c]
        if color == EMPTY:
            return set(), set()
        stones: Set[Tuple[int, int]] = set()
        libs: Set[Tuple[int, int]] = set()
        stack = [(r, c)]
        seen = {(r, c)}
        while stack:
            cr, cc = stack.pop()
            stones.add((cr, cc))
            for nr, nc in self._neighbors(cr, cc):
                v = grid[nr][nc]
                if v == EMPTY:
                    libs.add((nr, nc))
                elif v == color and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        return stones, libs

    def _group(self, r: int, c: int) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        return self._group_on(self.grid, r, c)

    def is_legal(self, r: Optional[int], c: Optional[int] = None) -> bool:
        """Pass is always legal while playing. Point must be empty, not suicide, not ko.

        Non-mutating: builds a local trial grid so concurrent /state and
        /legal-moves readers never observe a phantom stone.
        """
        if self.status != "playing":
            return False
        if r is None:
            return True  # pass
        assert c is not None
        if not (0 <= r < self.size and 0 <= c < self.size):
            return False
        if self.grid[r][c] != EMPTY:
            return False
        if self.ko is not None and (r, c) == self.ko:
            return False

        color = self.turn
        opp = opponent(color)
        # Shallow-copy rows so we do not touch self.grid
        trial = [row[:] for row in self.grid]
        trial[r][c] = color

        captured_any = False
        for nr, nc in self._neighbors(r, c):
            if trial[nr][nc] == opp:
                _stones, libs = self._group_on(trial, nr, nc)
                if not libs:
                    captured_any = True
                    break
        _stones, libs = self._group_on(trial, r, c)
        suicide = not libs and not captured_any
        return not suicide

    def legal_moves_gtp(self) -> List[str]:
        moves = ["pass"]
        for r in range(self.size):
            for c in range(self.size):
                if self.is_legal(r, c):
                    moves.append(to_gtp(r, c, self.size))
        return moves

    def play_gtp(self, move: str) -> None:
        m = (move or "").strip().upper()
        if self.status != "playing":
            raise ValueError(f"game over: {self.status}")
        if m in ("RESIGN", "R"):
            self.status = "resigned"
            self.history.append("resign")
            return
        if m in ("PASS", "PA"):
            self._apply_pass()
            return
        rc = parse_gtp_move(m, self.size)
        assert rc is not None
        r, c = rc
        if not self.is_legal(r, c):
            raise ValueError(f"illegal: {m}")
        self._place(r, c)

    def _apply_pass(self) -> None:
        self.ko = None
        self.passes += 1
        self.history.append("pass")
        self.turn = opponent(self.turn)
        if self.passes >= 2:
            self.status = "finished"

    def _place(self, r: int, c: int) -> None:
        color = self.turn
        opp = opponent(color)
        self.grid[r][c] = color
        # One group may touch the play point on multiple sides — dedupe.
        captured: Set[Tuple[int, int]] = set()
        for nr, nc in self._neighbors(r, c):
            if self.grid[nr][nc] == opp:
                stones, libs = self._group(nr, nc)
                if not libs:
                    captured.update(stones)
        for sr, sc in captured:
            self.grid[sr][sc] = EMPTY
        self.captured[color] += len(captured)

        # Simple ko: single-stone capture that leaves one empty at the capture point
        self.ko = None
        if len(captured) == 1:
            stones, libs = self._group(r, c)
            if len(stones) == 1 and len(libs) == 1:
                self.ko = next(iter(captured))

        self.passes = 0
        self.history.append(to_gtp(r, c, self.size))
        self.turn = opponent(self.turn)

    def stones_list(self) -> List[dict]:
        out = []
        for r in range(self.size):
            for c in range(self.size):
                v = self.grid[r][c]
                if v != EMPTY:
                    out.append(
                        {
                            "color": color_name(v),
                            "row": r,
                            "col": c,
                            "gtp": to_gtp(r, c, self.size),
                        }
                    )
        return out

    def ascii(self) -> str:
        lines = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                v = self.grid[r][c]
                row.append("." if v == EMPTY else ("X" if v == BLACK else "O"))
            lines.append("".join(row))
        return "\n".join(lines)
