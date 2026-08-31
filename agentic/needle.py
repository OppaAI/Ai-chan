"""Optional adapter for a local Needle 2 structured tool-calling server.

Needle is deliberately a *worker*, not an authority: callers must validate and
execute its proposed calls through Aiko's existing registry and approval gates.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen



class NeedleError(RuntimeError):
    """Needle returned an unusable response or its local service was unavailable."""


class NeedleLowConfidence(NeedleError):
    """Needle's calibrated confidence is below the configured action threshold."""


@dataclass(frozen=True)
class NeedleCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class NeedleResponse:
    kind: str
    calls: tuple[NeedleCall, ...]
    confidence: float | None
    reasoning: str
    content: str


def needle_tools(openai_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Aiko's OpenAI function schemas into Needle's schema contract."""
    converted: list[dict[str, Any]] = []
    for entry in openai_tools:
        function = entry.get("function", {}) if isinstance(entry, dict) else {}
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        parameters = function.get("parameters")
        converted.append({
            "name": name,
            "description": str(function.get("description") or ""),
            "parameters": parameters if isinstance(parameters, dict) else {"type": "object", "properties": {}},
        })
    return converted


class NeedleClient:
    """Small synchronous client for Needle's local ``POST /complete`` server."""

    def __init__(self, base_url: str, *, timeout: float = 15.0, confidence_threshold: float = 0.85):
        self.url = f"{base_url.rstrip('/')}/complete"
        self.timeout = timeout
        self.confidence_threshold = confidence_threshold

    def complete(self, prompt: str, tools: list[dict[str, Any]]) -> NeedleResponse:
        # Needle's documented HTTP server loads its catalogue at startup. Keep
        # this request to its documented input contract and enforce Aiko's
        # capability-filtered subset again after the response arrives.
        payload = json.dumps({"input": prompt}).encode("utf-8")
        request = Request(self.url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, ValueError) as exc:
            raise NeedleError(f"Needle request failed: {exc}") from exc
        allowed_names = {tool["name"] for tool in needle_tools(tools)}
        return self._parse(raw, allowed_names)

    def _parse(self, raw: Any, allowed_names: set[str]) -> NeedleResponse:
        if not isinstance(raw, dict):
            raise NeedleError("Needle response was not a JSON object")
        if raw.get("success") is False:
            raise NeedleError(str(raw.get("error") or raw.get("error_code") or "Needle reported failure"))

        confidence_value = raw.get("confidence")
        try:
            confidence = float(confidence_value) if isinstance(confidence_value, (int, float)) else None
        except OverflowError as exc:
            raise NeedleError("Needle confidence must be finite") from exc
        if confidence is not None and not math.isfinite(confidence):
            raise NeedleError("Needle confidence must be finite")
        if confidence is not None and confidence < self.confidence_threshold:
            raise NeedleLowConfidence(
                f"Needle confidence {confidence:.2f} is below threshold {self.confidence_threshold:.2f}"
            )

        calls: list[NeedleCall] = []
        for item in raw.get("function_calls", []) or []:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise NeedleError("Needle returned an invalid function call")
            if item["name"] not in allowed_names:
                raise NeedleError(f"Needle selected a tool outside Aiko's allowed subset: {item['name']}")
            arguments = item.get("arguments", {})
            if not isinstance(arguments, dict):
                raise NeedleError(f"Needle arguments for {item['name']} were not an object")
            calls.append(NeedleCall(item["name"], arguments))

        kind = str(raw.get("type") or "call")
        content = next(
            (str(raw[key]) for key in ("response", "content", "answer", "text") if isinstance(raw.get(key), str)),
            "",
        )
        return NeedleResponse(
            kind=kind,
            calls=tuple(calls),
            confidence=confidence,
            reasoning=str(raw.get("reasoning") or ""),
            content=content,
        )
