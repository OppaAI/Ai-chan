"""
agentic/workflows/aurora_forecast/toolset.py

Hourly aurora-viewing forecast (default: Vancouver, BC).

Sources:
  - NOAA SWPC OVATION Aurora model
  - NOAA SWPC Planetary Kp index
  - Open-Meteo cloud cover + day/night

Graph steps (see graph.py):
  check_aurora → store_aurora_forecast → notify_aurora

Storage/notify use agentic.workflows.common (shared with job_hunt patterns).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

from agentic.workflows.common.config import load_workflow_config, resolve_config_value
from agentic.workflows.common.notify import maybe_post_threads, notify_email
from agentic.workflows.common.store import append_record, prune_records

logger = logging.getLogger(__name__)

OVATION_URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
KP_INDEX_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

WORKFLOW_ID = "aurora_forecast"
_WORKFLOW_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG = {
    "location_name": "Vancouver, BC",
    "latitude": 49.2827,
    "longitude": -123.1207,
    "min_aurora_probability_pct": 15,
    "max_cloud_cover_pct": 50,
    "min_kp_for_alert": 4.0,
    "min_kp_for_threads": 4.0,
    "retain_days": 3,
    "email_on_check": True,
    "email_only_when_interesting": True,
}


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
    level: str  # high | medium | low
    summary: str
    explanation: str
    checked_at: str


def _load_cfg(config_path: str = "") -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(load_workflow_config(_WORKFLOW_DIR))
    if config_path:
        p = Path(config_path)
        if p.is_file():
            try:
                loaded = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    cfg.update(loaded)
            except (OSError, json.JSONDecodeError):
                pass
    return cfg


def _lon_to_0_360(lon: float) -> float:
    return lon % 360


async def fetch_ovation_aurora(lat: float, lon: float, client: httpx.AsyncClient) -> AuroraReading:
    resp = await client.get(OVATION_URL, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    target_lon = _lon_to_0_360(lon)
    coords = data["coordinates"]
    best = min(coords, key=lambda c: (c[0] - target_lon) ** 2 + (c[1] - lat) ** 2)
    return AuroraReading(
        probability_pct=int(best[2]),
        grid_lat=best[1],
        grid_lon=best[0],
        observation_time=data.get("Observation Time", ""),
        forecast_time=data.get("Forecast Time", ""),
    )


async def fetch_kp_index(client: httpx.AsyncClient) -> Optional[float]:
    try:
        resp = await client.get(KP_INDEX_URL, timeout=15)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return None
        # NOAA has changed the payload format over time:
        #  - Old: list-of-lists with header row, e.g. [["time_tag","kp",...], ["2024-...", 2.0]]
        #  - New: list-of-dicts with "Kp" field, e.g. [{"time_tag":"2026-...","Kp":2.67,...}]
        # Handle both (and case variants) without raising KeyError:1.
        last = rows[-1]
        if isinstance(last, dict):
            for key in ("Kp", "kp", "KP", "kP", "Kp_index", "kp_index"):
                if key in last:
                    try:
                        return float(last[key])
                    except (TypeError, ValueError):
                        continue
            # Dict but Kp under different casing or nested? Try case-insensitive.
            for k, v in last.items():
                if k.lower() == "kp":
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        continue
            # No Kp field — maybe wrapped payload {"data": [...]} or empty
            return None
        if isinstance(last, (list, tuple)):
            # Old header format: header may still be present at rows[0]
            # If payload is list-of-lists, last[1] is Kp.
            if len(last) > 1:
                try:
                    return float(last[1])  # type: ignore[index]
                except (TypeError, ValueError, KeyError, IndexError):
                    pass
            # Fallback: scan last row for first float-like value
            for v in last:
                try:
                    fv = float(v)  # type: ignore[arg-type]
                    # Kp is 0-9, so reasonable sanity check
                    if 0 <= fv <= 9:
                        return fv
                except (TypeError, ValueError):
                    continue
            return None
        # Unexpected top-level type (e.g. dict wrapping {"kp":...})
        if isinstance(rows, dict):
            for key in ("Kp", "kp", "KP"):
                if key in rows:
                    try:
                        return float(rows[key])  # type: ignore[index]
                    except (TypeError, ValueError):
                        continue
        return None
    except Exception:
        logger.exception("Kp index fetch failed")
        return None


async def fetch_cloud_forecast(lat: float, lon: float, client: httpx.AsyncClient) -> CloudReading:
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
    now = datetime.now().strftime("%Y-%m-%dT%H:00")
    try:
        idx = hourly["time"].index(now)
    except ValueError:
        idx = 0
    return CloudReading(
        cloud_cover_pct=int(hourly["cloudcover"][idx]),
        is_night=not bool(hourly["is_day"][idx]),
        time=hourly["time"][idx],
    )


def _score_level(aurora_pct: int, cloud_pct: int, kp: Optional[float], is_night: bool, cfg: dict) -> tuple[str, str]:
    """Return (level, explanation)."""
    reasons = []
    if not is_night:
        return "low", "Daylight — aurora not visible regardless of activity."

    min_p = int(cfg.get("min_aurora_probability_pct", 15))
    max_c = int(cfg.get("max_cloud_cover_pct", 50))
    min_kp = float(cfg.get("min_kp_for_alert", 5.0))

    if aurora_pct >= min_p:
        reasons.append(f"OVATION probability {aurora_pct}% ≥ {min_p}%")
    else:
        reasons.append(f"OVATION probability {aurora_pct}% below {min_p}%")

    if cloud_pct <= max_c:
        reasons.append(f"cloud cover {cloud_pct}% ≤ {max_c}%")
    else:
        reasons.append(f"cloud cover {cloud_pct}% too high (max {max_c}%)")

    if kp is not None:
        reasons.append(f"Kp={kp:g} (alert threshold {min_kp:g})")
        if kp >= min_kp + 1:
            storm = "high"
        elif kp >= min_kp:
            storm = "medium"
        else:
            storm = "low"
    else:
        storm = "medium" if aurora_pct >= min_p else "low"
        reasons.append("Kp unavailable")

    viewable = aurora_pct >= min_p and cloud_pct <= max_c
    if viewable and storm == "high":
        level = "high"
    elif viewable:
        level = "medium"
    else:
        level = "low"

    return level, "; ".join(reasons)


def _is_valid_aurora_report(report: dict[str, Any]) -> bool:
    """Validate that a dict represents a valid AuroraReport (not an error object)."""
    if not isinstance(report, dict):
        return False
    # Must not be an error object
    if "error" in report and not any(k in report for k in ("location_name", "aurora_probability_pct", "checked_at")):
        return False
    # Required fields for a valid Aurora report
    required = {"location_name", "aurora_probability_pct", "cloud_cover_pct", "is_night", "viewable", "level", "summary", "explanation", "checked_at"}
    return required.issubset(report.keys())


def _build_summary(report_bits: dict[str, Any], cfg: dict, viewable: bool) -> str:
    lines = [
        f"Aurora Watch — {cfg['location_name']}",
        f"Level: {report_bits['level']}",
        f"Aurora probability: {report_bits['aurora_probability_pct']}%",
        f"Cloud cover: {report_bits['cloud_cover_pct']}%",
    ]
    kp = report_bits.get("kp_index")
    if kp is not None:
        lines.append(f"Planetary Kp index: {kp:g}")
    lines.append("Sky: " + ("Dark" if report_bits["is_night"] else "Daylight"))
    lines.append("Verdict: " + ("Worth a look outside." if viewable else "Not likely visible right now."))
    lines.append(f"Why: {report_bits['explanation']}")
    return "\n".join(lines)


async def run_aurora_check(config_path: str = "") -> AuroraReport:
    cfg = _load_cfg(config_path)
    async with httpx.AsyncClient() as client:
        aurora, cloud, kp = await asyncio.gather(
            fetch_ovation_aurora(cfg["latitude"], cfg["longitude"], client),
            fetch_cloud_forecast(cfg["latitude"], cfg["longitude"], client),
            fetch_kp_index(client),
        )

    viewable = (
        cloud.is_night
        and aurora.probability_pct >= int(cfg["min_aurora_probability_pct"])
        and cloud.cloud_cover_pct <= int(cfg["max_cloud_cover_pct"])
    )
    level, explanation = _score_level(
        aurora.probability_pct, cloud.cloud_cover_pct, kp, cloud.is_night, cfg,
    )
    bits = {
        "level": level,
        "aurora_probability_pct": aurora.probability_pct,
        "cloud_cover_pct": cloud.cloud_cover_pct,
        "kp_index": kp,
        "is_night": cloud.is_night,
        "explanation": explanation,
    }
    summary = _build_summary(bits, cfg, viewable)
    logger.info(summary.replace("\n", " | "))

    return AuroraReport(
        location_name=cfg["location_name"],
        aurora_probability_pct=aurora.probability_pct,
        cloud_cover_pct=cloud.cloud_cover_pct,
        kp_index=kp,
        is_night=cloud.is_night,
        viewable=viewable,
        level=level,
        summary=summary,
        explanation=explanation,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


def check_aurora(config_path: str = "", *, state=None) -> str:
    """Graph step: fetch + score; returns JSON report."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        report = asyncio.run(run_aurora_check(config_path=config_path))
    else:
        import concurrent.futures

        def _run_isolated():  # type: ignore[no-untyped-def]
            return asyncio.run(run_aurora_check(config_path=config_path))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            report = pool.submit(_run_isolated).result(timeout=120)
    payload = asdict(report)
    if state is not None and hasattr(state, "data"):
        state.data["aurora_report"] = payload
    return json.dumps(payload, ensure_ascii=False)


def store_aurora_forecast(report_json: str = "", *, state=None) -> str:
    """Graph step: persist report and prune old rows."""
    cfg = _load_cfg()
    retain = int(resolve_config_value(cfg, "retain_days", default=3, as_type="int"))

    report: dict[str, Any] = {}
    if report_json:
        try:
            report = json.loads(report_json)
        except (TypeError, json.JSONDecodeError):
            report = {}
    if not report and state is not None and hasattr(state, "data"):
        report = dict(state.data.get("aurora_report") or {})

    if not report:
        return json.dumps({"ok": False, "error": "no_report"})

    # Validate that report is a valid Aurora report, not an error object
    if not _is_valid_aurora_report(report):
        return json.dumps({"ok": False, "error": "invalid_report"})

    path, append_ok = append_record(WORKFLOW_ID, report)
    if not append_ok:
        return json.dumps({"ok": False, "error": "append_failed"})

    kept, prune_ok = prune_records(WORKFLOW_ID, days=retain)
    if not prune_ok:
        return json.dumps({"ok": False, "error": "prune_failed"})

    return json.dumps({"ok": True, "path": str(path), "retained": kept, "retain_days": retain})


def notify_aurora(report_json: str = "", *, state=None) -> str:
    """Graph step: email summary; Threads when Kp >= threshold."""
    cfg = _load_cfg()
    report: dict[str, Any] = {}
    if report_json:
        try:
            report = json.loads(report_json)
        except (TypeError, json.JSONDecodeError):
            report = {}
    if not report and state is not None and hasattr(state, "data"):
        report = dict(state.data.get("aurora_report") or {})

    if not report:
        return json.dumps({"ok": False, "error": "no_report"})

    # Validate that report is a valid Aurora report, not an error object
    if not _is_valid_aurora_report(report):
        return json.dumps({"ok": False, "error": "invalid_report"})

    summary = str(report.get("summary") or "")
    level = str(report.get("level") or "low")
    kp = report.get("kp_index")
    try:
        kp_f = float(kp) if kp is not None else None
    except (TypeError, ValueError):
        kp_f = None

    min_kp_threads = float(cfg.get("min_kp_for_threads", 5.0))
    min_kp_alert = float(cfg.get("min_kp_for_alert", 5.0))
    interesting = bool(report.get("viewable")) or (kp_f is not None and kp_f >= min_kp_alert) or level in {"high", "medium"}

    email_result: dict[str, Any] = {"skipped": True}
    email_on = bool(cfg.get("email_on_check", True))
    only_interesting = bool(cfg.get("email_only_when_interesting", True))
    if email_on and (interesting or not only_interesting):
        email_result = notify_email(
            subject=f"Aurora {level.upper()} — {report.get('location_name', '')}",
            body=summary,
        )

    threads_enabled = kp_f is not None and kp_f >= min_kp_threads
    threads_result = maybe_post_threads(
        summary,
        enabled=threads_enabled,
        reason=f"kp={kp_f} threshold={min_kp_threads}" if kp_f is not None else "no_kp",
    )

    return json.dumps(
        {
            "ok": True,
            "interesting": interesting,
            "email": email_result,
            "threads": threads_result,
        },
        ensure_ascii=False,
    )


async def tool_check_aurora(params: Optional[dict] = None) -> dict:
    """Agentic/async entrypoint."""
    report = await run_aurora_check(config_path=(params or {}).get("config_path") or "")
    return asdict(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(asyncio.run(run_aurora_check()).summary)
