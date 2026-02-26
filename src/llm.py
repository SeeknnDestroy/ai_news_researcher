from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class XAIConfig:
    model: str = "grok-4-1-fast-reasoning"
    temperature: float = 0.2
    reasoning_effort: Optional[str] = "low"
    api_key: Optional[str] = None
    base_url: str = "https://api.x.ai/v1"
    timeout_s: int = 60
    user_agent: str = "ai-news-researcher/0.1"
    force_reasoning_effort: bool = False


def generate_json(config: XAIConfig, system: str, user: str) -> Dict[str, Any]:
    raw = complete(config=config, system=system, user=user, response_format="json")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        extracted = _extract_json_object(raw)
        if extracted is None:
            raise LLMError("LLM did not return valid JSON.")
        return json.loads(extracted)


def generate_text(config: XAIConfig, system: str, user: str) -> str:
    return complete(config=config, system=system, user=user, response_format=None)


def complete(
    config: XAIConfig,
    system: str,
    user: str,
    response_format: Optional[str] = None,
) -> str:
    api_key = config.api_key or os.getenv("XAI_API_KEY")
    if not api_key:
        raise LLMError("Missing XAI_API_KEY environment variable.")

    payload: Dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": config.temperature,
    }

    if response_format == "json":
        payload["response_format"] = {"type": "json_object"}

    reasoning_effort = _select_reasoning_effort(config)
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": config.user_agent,
    }

    request = urllib.request.Request(
        url=f"{config.base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8") if exc.fp else str(exc)
        # Retry once without response_format or reasoning_effort if the API rejects them.
        if _should_retry_without_optional_params(detail, payload):
            payload.pop("response_format", None)
            payload.pop("reasoning_effort", None)
            return _retry_request(config, api_key, payload, headers)
        raise LLMError(f"xAI API error: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LLMError(f"xAI API request failed: {exc}") from exc

    data = json.loads(body)
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("Unexpected xAI response format.") from exc


def _retry_request(config: XAIConfig, api_key: str, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
    request = urllib.request.Request(
        url=f"{config.base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8") if exc.fp else str(exc)
        raise LLMError(f"xAI API error: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LLMError(f"xAI API request failed: {exc}") from exc

    data = json.loads(body)
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("Unexpected xAI response format.") from exc


def _select_reasoning_effort(config: XAIConfig) -> Optional[str]:
    if config.force_reasoning_effort:
        return config.reasoning_effort

    if not config.reasoning_effort:
        return None

    if "non-reasoning" in config.model:
        return None

    if config.model == "grok-4-1-fast-reasoning":
        return config.reasoning_effort

    if config.model == "grok-3-mini":
        return config.reasoning_effort

    return None


def _should_retry_without_optional_params(detail: str, payload: Dict[str, Any]) -> bool:
    lowered = detail.lower()
    if "reasoning_effort" in lowered or "response_format" in lowered:
        return True

    # If we sent optional params and got a 400/403 style error, try once without them.
    if "reasoning_effort" in payload or "response_format" in payload:
        return "error" in lowered or "invalid" in lowered or "not supported" in lowered

    return False


def _extract_json_object(text: str) -> Optional[str]:
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else None
