"""
agentic/workflows/aurora_watch/toolset.py

Hourly aurora-viewing forecast for Vancouver, BC.

Combines:
  - NOAA SWPC OVATION Aurora model (30-90 min forecast, ~1 deg grid)
  - NOAA SWPC Planetary Kp index (geomagnetic storm context)
  - Open-Meteo cloud cover + day/night (NOAA's own forecast API is US-only,
    so it can't cover Vancouver -- Open-Meteo is free/no-key and global)

Design notes for wiring into Aiko:
  - Mirrors the shape of job_hunt/toolset.py: async fns, a dataclass per
    data source, a config JSON, a single `tool_*` entrypoint for the
    agentic/MCP registry.
  - Swap the `tool_check_aurora` signature/decorator for whatever your
    capability.py / graph_engine.py registration pattern expects -- this
    file intentionally has zero framework-specific imports so you can drop
    it in and wire it up in one place.
  - Suggested schedule: hourly cron via your existing scheduler.py, same
    pattern as the A1 weekly auto-post lane. OVATION itself refreshes every
    ~5 min, but hourly is plenty for a "should I go outside" check --
    tighten to every 15-30 min only during an active storm (see
    `kp_index` in the result) if you want faster reaction time.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

OVATION_URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
KP_INDEX_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

DEFAULT_CONFIG = {
    "location_name": "Vancouver, BC",
    "latitude": 49.2827,
    "longitude": -123.1207,
    "poll_interval_hours": 1,
    # OVATION probability at 49N is near 0 almost all the time -- this is
    # NOT "will there be an aurora anywhere", it's "will Vancouver
    # specifically see one". Kp 6+ storms can push this into double digits.
    "min_aurora_probability_pct": 15,
    "max_cloud_cover_pct": 50,
    "min_kp_for_alert": 6.0,
}

CONFIG_PATH = Path(__file__).parent / "aurora_watch.json"


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------

@dataclass
class AuroraReading:
    probability_pct: int
    grid_lat: float
    grid_lon: float
    observation_time: str
    forecast_time: str


@dataclass
class CloudReading:
    cloud_cover_pct: int
    is_night: bool
    time: str


@dataclass
class AuroraReport:
    location_name: str
    aurora_probability_pct: int
    cloud_cover_pct: int
    kp_index: Optional[float]
    is_night: bool
    viewable: bool
    summary: str
    checked_at: str


# --------------------------------------------------------------------------
# Fetchers
# --------------------------------------------------------------------------

def _lon_to_0_360(lon: float) -> float:
    """OVATION uses 0-360 longitude; convert from standard -180..180."""
    return lon % 360


async def fetch_ovation_aurora(
    lat: float, lon: float, client: httpx.AsyncClient
) -> AuroraReading:
    resp = await client.get(OVATION_URL, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    target_lon = _lon_to_0_360(lon)
    coords = data["coordinates"]  # list of [lon(0-360), lat(-90..90), value]

    # Grid is ~1 deg spacing -> nearest-neighbour is plenty accurate.
    best = min(coords, key=lambda c: (c[0] - target_lon) ** 2 + (c[1] - lat) ** 2)

    return AuroraReading(
        probability_pct=int(best[2]),
        grid_lat=best[1],
        grid_lon=best[0],
        observation_time=data.get("Observation Time", ""),
        forecast_time=data.get("Forecast Time", ""),
    )


async def fetch_kp_index(client: httpx.AsyncClient) -> Optional[float]:
    """Planetary Kp index -- gives storm-level context for the raw %."""
    try:
        resp = await client.get(KP_INDEX_URL, timeout=15)
        resp.raise_for_status()
        rows = resp.json()  # rows[0] is a header row
        if len(rows) < 2:
            return None
        latest = rows[-1]
        return float(latest[1])
    except Exception:
        logger.exception("Kp index fetch failed; continuing without it")
        return None


async def fetch_cloud_forecast(
    lat: float, lon: float, client: httpx.AsyncClient
) -> CloudReading:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "cloudcover,is_day",
        "forecast_days": 1,
        "timezone": "America/Vancouver",
    }
    resp = await client.get(OPEN_METEO_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    hourly = data["hourly"]

    # Pick the hourly slot closest to "now" rather than blindly index [0]
    # (Open-Meteo returns the full day; [0] is midnight local, not "now").
    now = datetime.now().strftime("%Y-%m-%dT%H:00")
    try:
        idx = hourly["time"].index(now)
    except ValueError:
        idx = 0  # fallback

    return CloudReading(
        cloud_cover_pct=int(hourly["cloudcover"][idx]),
        is_night=not bool(hourly["is_day"][idx]),
        time=hourly["time"][idx],
    )


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------

def _build_summary(
    aurora: AuroraReading, cloud: CloudReading, kp: Optional[float],
    cfg: dict, viewable: bool,
) -> str:
    lines = [
        f"🌌 Aurora Watch — {cfg['location_name']}",
        f"Aurora probability: {aurora.probability_pct}%",
        f"Cloud cover: {cloud.cloud_cover_pct}%",
    ]
    if kp is not None:
        lines.append(f"Planetary Kp index: {kp:g}")
    lines.append("Sky: " + ("Dark ✅" if cloud.is_night else "Daylight ❌ (nothing to see yet)"))
    lines.append(
        "Verdict: " + ("👀 Worth stepping outside tonight!" if viewable
                        else "Not likely visible right now.")
    )
    return "\n".join(lines)


async def run_aurora_check(config_path: Optional[str] = None) -> AuroraReport:
    cfg = DEFAULT_CONFIG.copy()
    p = Path(config_path) if config_path else CONFIG_PATH
    if p.exists():
        cfg.update(json.loads(p.read_text()))

    async with httpx.AsyncClient() as client:
        aurora, cloud, kp = await asyncio.gather(
            fetch_ovation_aurora(cfg["latitude"], cfg["longitude"], client),
            fetch_cloud_forecast(cfg["latitude"], cfg["longitude"], client),
            fetch_kp_index(client),
        )

    viewable = (
        cloud.is_night
        and aurora.probability_pct >= cfg["min_aurora_probability_pct"]
        and cloud.cloud_cover_pct <= cfg["max_cloud_cover_pct"]
    )

    summary = _build_summary(aurora, cloud, kp, cfg, viewable)
    logger.info(summary.replace("\n", " | "))

    return AuroraReport(
        location_name=cfg["location_name"],
        aurora_probability_pct=aurora.probability_pct,
        cloud_cover_pct=cloud.cloud_cover_pct,
        kp_index=kp,
        is_night=cloud.is_night,
        viewable=viewable,
        summary=summary,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


# --------------------------------------------------------------------------
# Agentic tool entrypoint
# --------------------------------------------------------------------------
# TODO: swap this for your actual registration pattern, e.g.:
#   from agentic.capability import Capability
#   AURORA_CHECK_CAPABILITY = Capability(name="check_aurora", handler=tool_check_aurora, ...)
# or however job_hunt's toolset functions get exposed to the router/scheduler.

async def tool_check_aurora(params: Optional[dict] = None) -> dict:
    """Agentic tool entrypoint: check aurora + cloud forecast for Vancouver.

    Returns a JSON-serializable dict so it can flow straight into whatever
    downstream drafting/posting step you want (e.g. a Threads draft, same
    idea as the job_hunt formatter, or just a WebUI notification).
    """
    report = await run_aurora_check(config_path=params.get("config_path") if params else None)
    return asdict(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(run_aurora_check())
    print(result.summary)
