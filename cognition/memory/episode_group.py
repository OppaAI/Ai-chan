"""EMC-6: coherent episode formation helpers (staging → one episode).

No LLM. Missing 5W fields stay NULL — never invent.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from cognition.memory.env import env_bool, env_int

EMC_GROUP_ENABLED = env_bool("EMC_GROUP_ENABLED", "1")
EMC_GROUP_MAX_GAP_SEC = max(0, env_int("EMC_GROUP_MAX_GAP_SEC", 900))
EMC_GROUP_MAX_TURNS = max(1, env_int("EMC_GROUP_MAX_TURNS", 6))
EMC_GROUP_MAX_CHARS = max(200, env_int("EMC_GROUP_MAX_CHARS", 2000))


def parse_ts(value: Any) -> float | None:
    s = str(value or "").strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        if "." in s:
            head, tail = s.split(".", 1)
            digits = "".join(c for c in tail if c.isdigit())
            tz = "".join(c for c in tail if not c.isdigit())
            s = head + "." + digits[:6] + tz
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def group_staging_rows(rows: list) -> list[list]:
    if not rows:
        return []
    groups: list[list] = [[rows[0]]]
    for row in rows[1:]:
        cur = groups[-1]
        prev = cur[-1]
        same_session = (prev[10] or None) == (row[10] or None)
        gap_ok = True
        t_prev = parse_ts(prev[2])
        t_cur = parse_ts(row[2])
        if t_prev is not None and t_cur is not None and EMC_GROUP_MAX_GAP_SEC > 0:
            gap_ok = abs(t_cur - t_prev) <= float(EMC_GROUP_MAX_GAP_SEC)
        turns_ok = len(cur) < EMC_GROUP_MAX_TURNS
        existing = sum(len(str(r[4] or "")) for r in cur)
        added = len(str(row[4] or ""))
        chars_ok = (existing + added + 8 * len(cur)) <= EMC_GROUP_MAX_CHARS
        if same_session and gap_ok and turns_ok and chars_ok:
            cur.append(row)
        else:
            groups.append([row])
    return groups


def merge_staging_group(group: list) -> tuple:
    if len(group) == 1:
        return group[0]
    first = group[0]
    traces = [str(r[4] or "").strip() for r in group if str(r[4] or "").strip()]
    trace = "\n\n".join(traces)
    valence_tag = None
    for r in group:
        if r[5] is not None and str(r[5]).strip():
            valence_tag = r[5]
            break
    arousal_vals = [float(r[6]) for r in group if r[6] is not None]
    arousal_score = max(arousal_vals) if arousal_vals else None
    salience_vals = [float(r[7]) for r in group if r[7] is not None]
    salience_score = max(salience_vals) if salience_vals else None
    ents: list[str] = []
    seen: set[str] = set()
    for r in group:
        raw = r[8]
        if raw is None or raw == "":
            continue
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(data, list):
            for e in data:
                s = str(e).strip()
                key = s.casefold()
                if s and key not in seen:
                    seen.add(key)
                    ents.append(s)
    entities = json.dumps(ents, ensure_ascii=False) if ents else None
    source = None
    for r in group:
        if r[9] is not None and str(r[9]).strip():
            source = r[9]
            break
    if source is None:
        source = "emc_group"
    session_id = None
    for r in group:
        if r[10] is not None and str(r[10]).strip():
            session_id = r[10]
            break
    return (
        first[0], first[1], first[2], first[3], trace,
        valence_tag, arousal_score, salience_score,
        entities, source, session_id,
    )
