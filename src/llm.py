from __future__ import annotations

import json
import os
import re
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .config import settings

class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class XAIConfig:
    model: str = field(default_factory=lambda: settings.xai_model)
    temperature: float = field(default_factory=lambda: settings.xai_temperature)
    reasoning_effort: Optional[str] = "low"
    api_key: Optional[str] = field(default_factory=lambda: settings.xai_api_key)
    base_url: str = field(default_factory=lambda: settings.xai_base_url)
    timeout_s: int = field(default_factory=lambda: settings.xai_timeout_s)
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


async def generate_json_async(config: XAIConfig, system: str, user: str) -> Dict[str, Any]:
    raw = await complete_async(config=config, system=system, user=user, response_format="json")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        extracted = _extract_json_object(raw)
        if extracted is None:
            raise LLMError("LLM did not return valid JSON.")
        return json.loads(extracted)


def generate_text(config: XAIConfig, system: str, user: str) -> str:
    return complete(config=config, system=system, user=user, response_format=None)


async def generate_text_async(config: XAIConfig, system: str, user: str) -> str:
    return await complete_async(config=config, system=system, user=user, response_format=None)


def _build_request_kwargs(config: XAIConfig, system: str, user: str, response_format: Optional[str]) -> tuple[str, dict, dict]:
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

    return f"{config.base_url}/chat/completions", headers, payload


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    reraise=True
)
def _do_sync_request(url: str, headers: dict, payload: dict, timeout: int) -> dict:
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    reraise=True
)
async def _do_async_request(url: str, headers: dict, payload: dict, timeout: int) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


def complete(
    config: XAIConfig,
    system: str,
    user: str,
    response_format: Optional[str] = None,
) -> str:
    url, headers, payload = _build_request_kwargs(config, system, user, response_format)

    try:
        data = _do_sync_request(url, headers, payload, config.timeout_s)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        if _should_retry_without_optional_params(detail, payload):
            payload.pop("response_format", None)
            payload.pop("reasoning_effort", None)
            data = _do_sync_request(url, headers, payload, config.timeout_s)
        else:
            raise LLMError(f"xAI API error: {detail}") from exc
    except httpx.RequestError as exc:
        raise LLMError(f"xAI API request failed: {exc}") from exc

    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("Unexpected xAI response format.") from exc


async def complete_async(
    config: XAIConfig,
    system: str,
    user: str,
    response_format: Optional[str] = None,
) -> str:
    url, headers, payload = _build_request_kwargs(config, system, user, response_format)

    try:
        data = await _do_async_request(url, headers, payload, config.timeout_s)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        if _should_retry_without_optional_params(detail, payload):
            payload.pop("response_format", None)
            payload.pop("reasoning_effort", None)
            data = await _do_async_request(url, headers, payload, config.timeout_s)
        else:
            raise LLMError(f"xAI API error: {detail}") from exc
    except httpx.RequestError as exc:
        raise LLMError(f"xAI API request failed: {exc}") from exc

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
