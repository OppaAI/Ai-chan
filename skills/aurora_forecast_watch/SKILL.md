---
id: aurora_forecast_watch
name: Aurora Forecast Watch
summary: Monitor source-backed aurora/Kp forecasts on a schedule and alert Oppa when the configured threshold is met.
triggers: aurora forecast, northern lights, Kp index, KP > 4, KP > 5, geomagnetic storm, space weather, NOAA SWPC, aurora alert, aurora watch
tools: deep_search, deep_research, schedule_job, save_note
---
# Aurora Forecast Watch

Use this skill when Oppa asks Aiko to watch aurora conditions, Kp forecasts, geomagnetic storms, or northern lights visibility windows.

## Data Sources

- Prefer NOAA Space Weather Prediction Center (SWPC) products and JSON/text data because they are public official sources for Kp and aurora forecasts.
- Useful NOAA/SWPC endpoints include:
  - `https://services.swpc.noaa.gov/json/planetary_k_index_1m.json`
  - `https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json`
  - `https://services.swpc.noaa.gov/text/3-day-forecast.txt`
  - `https://www.swpc.noaa.gov/communities/aurora-dashboard-experimental`
- If Oppa says "NOCC," infer he probably means NOAA/SWPC unless he names a different provider. Verify the source before relying on it.

## Workflow

1. Clarify the alert threshold only if missing. Default to Kp >= 5 for geomagnetic-storm alerts; use Kp >= 4 for early heads-up aurora watches if Oppa asks for sensitive alerts.
2. Clarify the notification path only if missing:
   - `announce`: local Aiko reminder/notification while Aiko is running;
   - `draft_email`: draft/stage email text locally only;
   - `email`: only if a real email-sending tool/account is configured.
3. Schedule an hourly `agentic` job with `schedule_job` when Oppa asks Aiko to keep checking.
4. In each check, use `deep_research` to fetch current/forecast Kp from NOAA/SWPC or another verified source; use `deep_search` only for quick source discovery/snippets.
5. If current or forecast Kp meets/exceeds the threshold, produce a concise alert with:
   - current or forecast Kp value;
   - forecast time/window and timezone;
   - source used;
   - practical note: clouds, moonlight, darkness, and location still matter.
6. Save a note only if Oppa asks for a log/report or if the scheduled job needs a local record of checks.

## Notification Rules

- Aiko can announce locally only while she is running on an awake machine.
- Aiko must not claim she sent an email unless a real email tool completed the send.
- Without an email tool/account, Aiko may draft the email body or create a local reminder, but she cannot actually send it.
- To send email later, Aiko needs a dedicated email tool plus configured credentials/account settings, such as SMTP or a mail API. Do not store secrets in persona or skill files.

## Safety and Accuracy

- Kp is global and does not guarantee visibility at Oppa's exact location.
- Aurora visibility also depends on latitude, local weather, cloud cover, light pollution, moon phase, darkness, and horizon view.
- Distinguish observed Kp from forecast Kp.
- Cite the source or mention the endpoint used in reports.
