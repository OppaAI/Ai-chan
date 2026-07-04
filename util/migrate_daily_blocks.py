"""
scripts/migrate_daily_blocks.py
One-off: breaks old monolithic pinned daily-summary/day-record blocks
into individual atomic pinned entries tagged "[YYYY-MM-DD] fact".
"""
from __future__ import annotations
import argparse
import re
from datetime import datetime

from core.config import load_config
load_config()

from core.log import get_logger
from core.memorize import AikoMemorize, USER_ID
from core.reflect import _generate_daily_facts

log = get_logger(__name__)

_SUMMARY_RE = re.compile(r"^Daily experience summary for (\d{4}-\d{2}-\d{2}):\s*(.*)", re.DOTALL)
_RECORD_RE  = re.compile(r"^Day record for (\d{4}-\d{2}-\d{2}):\s*(.*)", re.DOTALL)


def _split_bullets(body: str) -> list[str]:
    facts = []
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("- ") and line[2:].strip() != "no memories recorded.":
            facts.append(line[2:].strip())
    return facts


def migrate(dry_run: bool = True, user_id: str = USER_ID) -> dict:
    memorize = AikoMemorize()
    all_mems = memorize.get_all(user_id=user_id)
    stats = {"summary_blocks": 0, "record_blocks": 0, "facts_created": 0, "blocks_deleted": 0}

    for m in all_mems:
        text, mem_id = m.get("memory") or "", m.get("id")
        if not mem_id:
            continue

        record_match  = _RECORD_RE.match(text)
        summary_match = _SUMMARY_RE.match(text)
        date_str, facts = None, []

        if record_match:
            date_str, body = record_match.groups()
            facts = _split_bullets(body)
            stats["record_blocks"] += 1
        elif summary_match:
            date_str, prose = summary_match.groups()
            date = datetime.strptime(date_str, "%Y-%m-%d")
            try:
                facts = _generate_daily_facts(prose.strip(), snippets=[], date=date)
            except Exception as e:
                log.warning(f"Fact extraction failed for {date_str}: {e}")
            stats["summary_blocks"] += 1
        else:
            continue

        if not facts:
            log.warning(f"No facts extracted for {mem_id} ({date_str}) — leaving original in place.")
            continue

        date_tag = f"[{date_str}]"
        log.info(f"{'(dry-run) ' if dry_run else ''}{date_str}: {len(facts)} facts from {mem_id}")
        if dry_run:
            for f in facts:
                log.info(f"  (dry-run) would pin: {date_tag} {f}")
            continue

        created = sum(1 for f in facts if memorize.add_raw(f"{date_tag} {f}", pinned=True))
        stats["facts_created"] += created

        if created > 0:
            memorize.delete(mem_id)
            stats["blocks_deleted"] += 1
        else:
            log.warning(f"Nothing persisted for {mem_id} ({date_str}) — keeping original.")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--user-id", default=USER_ID)
    args = parser.parse_args()
    log.info(f"Result: {migrate(dry_run=args.dry_run, user_id=args.user_id)}")