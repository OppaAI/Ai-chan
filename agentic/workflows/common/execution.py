"""Layer 0 shared workflow nodes.

Nodes are stable execution units. They call registry/MCP tools; they are
not the tools themselves.

  ingest_data → store_data → synthesis_data → verify_results → output_user_results

Per-workflow graph.py only arranges these (plus trigger outside the graph).
Domain details come from config.json / node args.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agentic.workflows.common.notify import maybe_post_threads, notify_email
from agentic.workflows.common.store import append_record, load_records, prune_records

log = logging.getLogger(__name__)


def _loads(raw: str | dict | list | None, default: Any = None) -> Any:
    if raw is None or raw == "":
        return default if default is not None else {}
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default if default is not None else {}


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _state_config(state) -> dict[str, Any]:
    if state is None:
        return {}
    data = getattr(state, "data", None)
    if isinstance(data, dict):
        cfg = data.get("config")
        if isinstance(cfg, dict):
            return cfg
    return {}


def _merge_config(config_json: str = "", state=None) -> dict[str, Any]:
    cfg = dict(_state_config(state))
    extra = _loads(config_json, {})
    if isinstance(extra, dict):
        cfg.update(extra)
    return cfg


# ── Node 1: ingest_data ───────────────────────────────

def ingest_data(
    sources_json: str = "[]",
    filters_json: str = "{}",
    parallel: str = "true",
    max_items: str = "50",
    config_json: str = "{}",
    *,
    state=None,
) -> str:
    """Ingest from configured sources into normalized items.

    sources_json: list of {type, id, ...} where type is rss|email|http_json|adapter
    filters_json: optional filter tree (and/or, field, op, value)
    parallel: "true"|"false"
    max_items: cap on returned items

    Returns: {"ok": bool, "items": [...], "meta": {...}}

    Layer 0: dispatches known source types; unknown types are reported in meta.
    Domain adapters (RSS parse, OVATION grid) may live under each workflow until
    fully generic.
    """
    sources = _loads(sources_json, [])
    filters = _loads(filters_json, {})
    cfg = _merge_config(config_json, state)
    if isinstance(sources, dict):
        sources = sources.get("sources") or []
    if not isinstance(sources, list):
        sources = []

    try:
        limit = max(1, int(max_items or 50))
    except (TypeError, ValueError):
        limit = 50
    run_parallel = str(parallel).lower() in {"1", "true", "yes", "on"}

    items: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen_types: list[str] = []

    for src in sources:
        if not isinstance(src, dict):
            continue
        stype = str(src.get("type") or "").strip().lower()
        sid = str(src.get("id") or stype or "src")
        seen_types.append(stype or "unknown")

        if stype == "http_json":
            url = str(src.get("url") or "").strip()
            if not url:
                errors.append({"id": sid, "error": "missing_url"})
                continue
            try:
                import httpx

                resp = httpx.get(url, timeout=float(src.get("timeout") or 15))
                resp.raise_for_status()
                payload = resp.json()
                items.append({
                    "id": sid,
                    "source": sid,
                    "type": "http_json",
                    "url": url,
                    "raw": payload,
                    "text": "",
                })
            except Exception as e:
                errors.append({"id": sid, "error": str(e)})
        elif stype in {"rss", "email", "adapter"}:
            errors.append({
                "id": sid,
                "error": f"source_type_{stype}_requires_workflow_adapter",
                "hint": "Call workflow-specific ingest until adapter is registered",
            })
        else:
            errors.append({"id": sid, "error": f"unknown_source_type:{stype}"})

    if filters and items:
        items = _apply_filters(items, filters)

    items = items[:limit]
    result = {
        "ok": not errors or bool(items),
        "items": items,
        "meta": {
            "source_types": seen_types,
            "parallel": run_parallel,
            "max_items": limit,
            "errors": errors,
            "config_keys": sorted(cfg.keys())[:20],
        },
    }
    if state is not None and hasattr(state, "data") and isinstance(state.data, dict):
        state.data["ingest_items"] = items
        state.data["ingest_meta"] = result["meta"]
    return _dumps(result)


def _apply_filters(items: list[dict[str, Any]], filters: dict[str, Any]) -> list[dict[str, Any]]:
    rules = filters.get("and") if isinstance(filters.get("and"), list) else None
    if not rules:
        return items

    def match(item: dict, rule: dict) -> bool:
        field = str(rule.get("field") or "")
        op = str(rule.get("op") or "eq").lower()
        want = rule.get("value")
        got = item.get(field)
        if got is None and field:
            raw = item.get("raw")
            if isinstance(raw, dict):
                got = raw.get(field)
        if op in {"eq", "="}:
            return got == want
        if op == "contains" and got is not None:
            return str(want).casefold() in str(got).casefold()
        if op == "contains_any" and isinstance(want, list):
            text = str(got or "").casefold()
            return any(str(w).casefold() in text for w in want)
        if op in {">=", "gte"}:
            try:
                return float(got) >= float(want)
            except (TypeError, ValueError):
                return False
        if op in {"<=", "lte"}:
            try:
                return float(got) <= float(want)
            except (TypeError, ValueError):
                return False
        return True

    out = []
    for it in items:
        if all(isinstance(r, dict) and match(it, r) for r in rules):
            out.append(it)
    return out


def store_data(
    workflow_id: str = "",
    items_json: str = "",
    mode: str = "append",
    retain_days: str = "3",
    config_json: str = "{}",
    *,
    state=None,
) -> str:
    cfg = _merge_config(config_json, state)
    wid = (workflow_id or cfg.get("workflow_id") or "default").strip() or "default"
    try:
        days = max(1, int(retain_days or cfg.get("retain_days") or 3))
    except (TypeError, ValueError):
        days = 3

    items = _loads(items_json, None)
    if items is None and state is not None and hasattr(state, "data"):
        items = state.data.get("ingest_items") or state.data.get("synth_results")
    if isinstance(items, dict) and "items" in items:
        items = items["items"]
    if not isinstance(items, list):
        items = [items] if items else []

    paths = []
    all_appends_ok = True
    for row in items:
        if not isinstance(row, dict):
            row = {"value": row}
        path, append_ok = append_record(wid, row)
        paths.append(str(path))
        if not append_ok:
            all_appends_ok = False

    kept, prune_ok = prune_records(wid, days=days)
    if state is not None and hasattr(state, "data") and isinstance(state.data, dict):
        state.data["stored_count"] = len(items)
        state.data["workflow_id"] = wid

    ok = all_appends_ok and prune_ok
    return _dumps({
        "ok": ok,
        "workflow_id": wid,
        "stored": len(items),
        "retained": kept,
        "retain_days": days,
        "mode": mode,
        "paths": paths[:5],
    })


def synthesis_data(
    items_json: str = "",
    template: str = "",
    llm_enriched: str = "false",
    per_item: str = "true",
    config_json: str = "{}",
    *,
    client=None,
    model: str | None = None,
    state=None,
) -> str:
    """Fill a template from items; job_hunt uses post_fields when present."""
    cfg = _merge_config(config_json, state)
    items = _loads(items_json, None)
    if items is None and state is not None and hasattr(state, "data"):
        items = state.data.get("ingest_items") or load_records(
            str(state.data.get("workflow_id") or cfg.get("workflow_id") or ""),
            days=int(cfg.get("retain_days") or 3),
        )
    if isinstance(items, dict) and "items" in items:
        items = items["items"]
    if not isinstance(items, list):
        items = [items] if items else []

    tmpl = template or str(cfg.get("template") or "{summary}")
    do_per = str(per_item).lower() in {"1", "true", "yes", "on"}
    want_llm = str(llm_enriched).lower() in {"1", "true", "yes", "on"}
    post_fields = cfg.get("post_fields")
    is_job = isinstance(post_fields, list) and bool(post_fields)

    results: list[dict[str, Any]] = []

    def fill_one(item: dict[str, Any]) -> dict[str, Any]:
        row = dict(item)
        text = tmpl
        enriched_note = None
        if is_job:
            try:
                from agentic.workflows.job_hunt import toolset as jh
                posting = dict(row)
                if want_llm and client is not None and model:
                    keys = jh._field_keys_from_config(cfg)
                    url = str(posting.get("url") or "").strip()
                    if url:
                        try:
                            posting["page_content"] = jh._fetch_job_page_text(url, config=cfg)
                        except Exception:
                            pass
                    posting = jh.enrich_posting_fields_with_llm(
                        posting, keys, client=client, model=model, state=state
                    )
                    posting.pop("page_content", None)
                    enriched_note = "llm"
                text = jh.format_job_post(posting, config=cfg)
                row = posting
            except Exception as exc:
                log.warning("synthesis job format failed: %s", exc)
                for key, val in row.items():
                    if isinstance(val, (str, int, float, bool)):
                        text = text.replace("{" + key + "}", str(val))
        else:
            for key, val in row.items():
                if isinstance(val, (str, int, float, bool)):
                    text = text.replace("{" + key + "}", str(val))
            text = text.replace("{summary}", str(row.get("summary") or row.get("text") or ""))
            text = text.replace("{title}", str(row.get("title") or ""))
            if want_llm and client and model:
                enriched_note = "llm_requested_generic"
        return {"text": text, "source": row.get("source") or row.get("id"), "item": row, "llm_enriched": enriched_note}

    if do_per:
        for it in items:
            if isinstance(it, dict):
                results.append(fill_one(it))
    else:
        blob = {
            "summary": "\n".join(str(it.get("summary") or it.get("text") or it) for it in items if it),
            "count": len(items),
            "items": items,
        }
        results.append(fill_one(blob))

    if state is not None and hasattr(state, "data") and isinstance(state.data, dict):
        state.data["synth_results"] = results

    return _dumps({"ok": True, "results": results, "count": len(results)})


def verify_results(
    results_json: str = "",
    human_in_the_loop: str = "false",
    llm_verify: str = "false",
    auto_pass_json: str = "{}",
    config_json: str = "{}",
    *,
    state=None,
) -> str:
    """Mark results verified or pending human review; HITL saves job drafts."""
    cfg = _merge_config(config_json, state)
    results = _loads(results_json, None)
    if results is None and state is not None and hasattr(state, "data"):
        results = state.data.get("synth_results")
    if isinstance(results, dict) and "results" in results:
        results = results["results"]
    if not isinstance(results, list):
        results = [results] if results else []

    hitl = str(human_in_the_loop or cfg.get("human_in_the_loop")).lower() in {
        "1", "true", "yes", "on",
    }
    rule = _loads(auto_pass_json, {})
    if not rule and isinstance(cfg.get("auto_pass_if"), dict):
        rule = cfg["auto_pass_if"]

    verified = []
    draft_paths: list[str] = []
    for r in results:
        if not isinstance(r, dict):
            r = {"text": str(r)}
        row = dict(r)
        item = row.get("item") if isinstance(row.get("item"), dict) else row
        if hitl:
            row["status"] = "pending_approval"
            row["verified"] = False
            if row.get("text") or item.get("title"):
                try:
                    import re as _re
                    from agentic.workflows.job_hunt.toolset import save_single_job_draft
                    if state is not None and hasattr(state, "data") and isinstance(state.data, dict):
                        drafts = state.data.get("job_drafts_list")
                        if not isinstance(drafts, list):
                            drafts = []
                        # Unique category per job so save_single_job_draft cannot
                        # overwrite earlier drafts under the same date/post dir.
                        slug_src = str(
                            item.get("title") or item.get("id") or item.get("url") or "job"
                        )
                        category = (
                            _re.sub(r"[^a-z0-9]+", "_", slug_src.casefold()).strip("_")[:48]
                            or f"job_{len(drafts) + 1}"
                        )
                        drafts.append({
                            "text": row.get("text") or "",
                            "posting": item,
                            "postings": [item],
                            "human_approved": False,
                            "category": category,
                            "topic_tag": str(cfg.get("topic_tag") or ""),
                            "source_name": str(item.get("source") or ""),
                            "source_type": str(item.get("type") or "job"),
                            "llm_enriched": bool(row.get("llm_enriched")),
                        })
                        state.data["job_drafts_list"] = drafts
                        save_raw = save_single_job_draft(auto_post="false", state=state)
                        draft_paths.append(str(save_raw)[:200])
                except Exception as exc:
                    log.debug("verify HITL draft save skipped: %s", exc)
        else:
            ok = True
            if rule:
                field = str(rule.get("field") or "")
                op = str(rule.get("op") or "eq").lower()
                want = rule.get("value")
                got = row.get(field)
                if got is None:
                    got = item.get(field)
                if op in {">=", "gte"}:
                    try:
                        ok = float(got) >= float(want)
                    except (TypeError, ValueError):
                        ok = False
                elif op == "in" and isinstance(want, list):
                    ok = got in want
                elif op in {"eq", "="}:
                    ok = got == want
            row["status"] = "passed" if ok else "rejected"
            row["verified"] = bool(ok)
        verified.append(row)

    if state is not None and hasattr(state, "data") and isinstance(state.data, dict):
        state.data["verified_results"] = verified

    return _dumps({
        "ok": True,
        "human_in_the_loop": hitl,
        "results": verified,
        "passed": sum(1 for r in verified if r.get("verified")),
        "pending": sum(1 for r in verified if r.get("status") == "pending_approval"),
        "draft_saves": draft_paths[:10],
    })


def output_user_results(
    results_json: str = "",
    email_json: str = "{}",
    social_json: str = "[]",
    config_json: str = "{}",
    *,
    state=None,
) -> str:
    cfg = _merge_config(config_json, state)
    results = _loads(results_json, None)
    if results is None and state is not None and hasattr(state, "data"):
        results = state.data.get("verified_results") or state.data.get("synth_results")
    if isinstance(results, dict) and "results" in results:
        results = results["results"]
    if not isinstance(results, list):
        results = [results] if results else []

    email_cfg = _loads(email_json, {})
    if not email_cfg and isinstance(cfg.get("email"), dict):
        email_cfg = cfg["email"]
    social_cfg = _loads(social_json, [])
    if not social_cfg and isinstance(cfg.get("social"), list):
        social_cfg = cfg["social"]

    deliverable = [
        r for r in results
        if isinstance(r, dict) and r.get("status") != "pending_approval" and r.get("verified", True)
    ]

    texts = [str(r.get("text") or "") for r in deliverable if r.get("text")]
    body = "\n\n".join(texts).strip()
    actions: list[dict[str, Any]] = []

    if email_cfg.get("enabled", False) and body:
        when = str(email_cfg.get("when") or "always").lower()
        interesting = any(
            (r.get("item") or {}).get("viewable") or r.get("level") in {"high", "medium"}
            for r in deliverable
        )
        if when == "always" or interesting:
            subject_tmpl = str(email_cfg.get("subject") or cfg.get("email_subject") or "Aiko workflow")
            subject = subject_tmpl
            if deliverable and "{" in subject_tmpl:
                first_result = deliverable[0]
                item = first_result.get("item") if isinstance(first_result.get("item"), dict) else {}
                all_fields = {**item, **first_result}
                for key, val in all_fields.items():
                    if isinstance(val, (str, int, float, bool)):
                        subject = subject.replace("{" + key + "}", str(val))
            actions.append({
                "channel": "email",
                "result": notify_email(subject=subject, body=body, to=email_cfg.get("to")),
            })

    for soc in social_cfg if isinstance(social_cfg, list) else []:
        if not isinstance(soc, dict):
            continue
        platform = str(soc.get("platform") or "threads").lower()
        when = soc.get("when") if isinstance(soc.get("when"), dict) else None
        enabled = True
        if when:
            field = str(when.get("field") or "")
            op = str(when.get("op") or ">=").lower()
            want = when.get("value")
            enabled = False
            for r in deliverable:
                item = r.get("item") if isinstance(r.get("item"), dict) else r
                got = item.get(field)
                try:
                    if op in {">=", "gte"} and float(got) >= float(want):
                        enabled = True
                        break
                except (TypeError, ValueError):
                    continue
        if platform in {"threads", "meta_threads"} and body:
            actions.append({
                "channel": "threads",
                "result": maybe_post_threads(body, enabled=enabled, reason=str(when or "always")),
            })

    return _dumps({
        "ok": True,
        "delivered_count": len(deliverable),
        "skipped_pending": len(results) - len(deliverable),
        "actions": actions,
    })


__all__ = [
    "ingest_data",
    "store_data",
    "synthesis_data",
    "verify_results",
    "output_user_results",
]
