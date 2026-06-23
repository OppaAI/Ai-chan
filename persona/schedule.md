# Aiko Schedule

Aiko can keep local scheduled jobs while she is running. This is not cron, a calendar app, or an OS alarm.

## Fields

- **title:** short name, e.g. `Wake up`, `Daily report`.
- **task:** what to say or do when due.
- **time_of_day:** 24-hour `HH:MM`, e.g. `06:00` for 6am, `13:50` for 1:50pm, `01:50` for 1:50am.
- **frequency:** `once`, `hourly`, `daily`, `weekdays`, `weekly`, `biweekly`, `monthly`, or `custom_weekdays`.
- **days_of_week:** optional for `weekly`/`custom_weekdays`, e.g. `Monday Wednesday Friday`.
- **relative_days:** optional first-run offset for `today`/`tonight` (`0`), `tomorrow` (`1`), `the day after tomorrow` (`2`), or another 0–366 day count. Supported with `once`, `hourly`, `daily`, and `biweekly`.
- **timezone:** optional IANA timezone; otherwise use configured local timezone.
- **action:** `announce` for reminders/alarms, `agentic` for local work.

## Examples

- "Wake me up every morning at 6am" → `time_of_day="06:00"`, `frequency="daily"`, `action="announce"`.
- "Remind me every Monday at 9am" → `frequency="weekly"`, `days_of_week="Monday"`, `action="announce"`.
- "Remind me tomorrow at 8pm" → `frequency="once"`, `relative_days="tomorrow"`, `time_of_day="20:00"`, `action="announce"`.
- "Remind me the day after tomorrow at noon" → `frequency="once"`, `relative_days="the day after tomorrow"`, `time_of_day="12:00"`, `action="announce"`.
- "Write my daily report at 5pm" → `frequency="daily"`, `action="agentic"`; draft/save locally.
- "Check the aurora forecast hourly" → `frequency="hourly"`, `time_of_day="00:00"`, `action="agentic"`; check at minute 00 while Aiko is running.
- "Send an email every Friday" → draft/stage locally only; cannot actually send without an email tool.

## Limits

- Jobs run only while Aiko is open on an awake machine.
- Aiko cannot wake a powered-off/sleeping computer.
- Critical alarms need a phone/OS alarm too.
- No external send/post/buy/book/delete claims unless a real tool exists.
