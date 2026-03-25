from __future__ import annotations

import asyncio
import json
import os
import random
import re
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import Any, Optional

import httpx
import tiktoken
from pydantic import BaseModel

from .config import settings


class LLMError(RuntimeError):
    pass


def _empty_usage_bucket() -> dict[str, int]:
    return {
        "request_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }


class UsageCollector:
    def __init__(self) -> None:
        self._lock = Lock()
        self._totals = _empty_usage_bucket()
        self._by_task: dict[str, dict[str, int]] = {}
        self._by_role: dict[str, dict[str, int]] = {}

    def record_attempt(self, task_name: str) -> None:
        role_name = _role_for_task(task_name)
        with self._lock:
            self._totals["request_count"] += 1
            self._bucket_for(self._by_task, task_name)["request_count"] += 1
            self._bucket_for(self._by_role, role_name)["request_count"] += 1

    def record_usage(self, task_name: str, usage: dict[str, int]) -> None:
        role_name = _role_for_task(task_name)
        with self._lock:
            self._merge_usage(self._totals, usage)
            self._merge_usage(self._bucket_for(self._by_task, task_name), usage)
            self._merge_usage(self._bucket_for(self._by_role, role_name), usage)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "totals": dict(self._totals),
                "by_task": {key: dict(value) for key, value in self._by_task.items()},
                "by_role": {key: dict(value) for key, value in self._by_role.items()},
            }

    def _bucket_for(self, container: dict[str, dict[str, int]], key: str) -> dict[str, int]:
        if key not in container:
            container[key] = _empty_usage_bucket()
        return container[key]

    def _merge_usage(self, target: dict[str, int], usage: dict[str, int]) -> None:
        for field_name in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
            "total_tokens",
        ):
            target[field_name] += usage.get(field_name, 0)


class RateLimiter:
    def __init__(self, *, rpm_limit: int, tpm_limit: int, tpd_limit: int) -> None:
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.tpd_limit = tpd_limit
        self._lock = asyncio.Lock()
        self._request_times: deque[float] = deque()
        self._minute_tokens: deque[tuple[float, int]] = deque()
        self._day_tokens: deque[tuple[float, int]] = deque()
        self._minute_token_sum = 0
        self._day_token_sum = 0

    async def acquire(self, estimated_tokens: int) -> None:
        reserved_tokens = max(1, estimated_tokens)

        while True:
            async with self._lock:
                now = monotonic()
                self._prune(now)

                request_wait = self._request_wait(now)
                minute_wait = self._token_wait(
                    queue=self._minute_tokens,
                    current_total=self._minute_token_sum,
                    now=now,
                    incoming_tokens=reserved_tokens,
                    window_s=60.0,
                    limit=self.tpm_limit,
                )
                day_wait = self._token_wait(
                    queue=self._day_tokens,
                    current_total=self._day_token_sum,
                    now=now,
                    incoming_tokens=reserved_tokens,
                    window_s=86_400.0,
                    limit=self.tpd_limit,
                )
                wait_s = max(request_wait, minute_wait, day_wait)

                if wait_s <= 0:
                    self._request_times.append(now)
                    self._minute_tokens.append((now, reserved_tokens))
                    self._day_tokens.append((now, reserved_tokens))
                    self._minute_token_sum += reserved_tokens
                    self._day_token_sum += reserved_tokens
                    return

            await asyncio.sleep(wait_s)

    def _prune(self, now: float) -> None:
        while self._request_times and now - self._request_times[0] >= 60.0:
            self._request_times.popleft()

        while self._minute_tokens and now - self._minute_tokens[0][0] >= 60.0:
            _, token_count = self._minute_tokens.popleft()
            self._minute_token_sum -= token_count

        while self._day_tokens and now - self._day_tokens[0][0] >= 86_400.0:
            _, token_count = self._day_tokens.popleft()
            self._day_token_sum -= token_count

    def _request_wait(self, now: float) -> float:
        if len(self._request_times) < self.rpm_limit:
            return 0.0
        oldest_request = self._request_times[0]
        return max(0.0, 60.0 - (now - oldest_request))

    def _token_wait(
        self,
        *,
        queue: deque[tuple[float, int]],
        current_total: int,
        now: float,
        incoming_tokens: int,
        window_s: float,
        limit: int,
    ) -> float:
        if current_total + incoming_tokens <= limit:
            return 0.0

        tokens_to_free = current_total + incoming_tokens - limit
        released_tokens = 0
        for timestamp, token_count in queue:
            released_tokens += token_count
            if released_tokens >= tokens_to_free:
                return max(0.0, window_s - (now - timestamp))
        return window_s


@dataclass(frozen=True)
class OpenAIConfig:
    model: str = field(default_factory=lambda: settings.openai_model)
    temperature: float = field(default_factory=lambda: settings.openai_temperature)
    reasoning_effort: Optional[str] = field(default_factory=lambda: settings.openai_reasoning_effort)
    api_key: Optional[str] = field(default_factory=lambda: settings.openai_api_key)
    base_url: str = field(default_factory=lambda: settings.openai_base_url)
    timeout_s: int = field(default_factory=lambda: settings.openai_timeout_s)
    user_agent: str = "ai-news-researcher/0.1"
    verbosity: str = field(default_factory=lambda: settings.openai_verbosity)
    max_output_tokens: int = field(default_factory=lambda: settings.openai_max_output_tokens)
    rpm_limit: int = field(default_factory=lambda: settings.openai_rpm_limit)
    tpm_limit: int = field(default_factory=lambda: settings.openai_tpm_limit)
    tpd_limit: int = field(default_factory=lambda: settings.openai_tpd_limit)
    store: bool = False
    rate_limiter: RateLimiter = field(
        default_factory=lambda: RateLimiter(
            rpm_limit=settings.openai_rpm_limit,
            tpm_limit=settings.openai_tpm_limit,
            tpd_limit=settings.openai_tpd_limit,
        )
    )
    usage_collector: UsageCollector = field(default_factory=UsageCollector)


XAIConfig = OpenAIConfig


async def generate_json_async(
    config: OpenAIConfig,
    system: str,
    user: str,
    *,
    schema: type[BaseModel],
    task_name: str,
) -> dict[str, Any]:
    raw = await complete_async(
        config=config,
        system=system,
        user=user,
        task_name=task_name,
        schema=schema,
    )
    return _parse_json_response(raw)


async def generate_text_async(
    config: OpenAIConfig,
    system: str,
    user: str,
    *,
    task_name: str,
) -> str:
    return await complete_async(
        config=config,
        system=system,
        user=user,
        task_name=task_name,
        schema=None,
    )


async def complete_async(
    config: OpenAIConfig,
    system: str,
    user: str,
    *,
    task_name: str,
    schema: type[BaseModel] | None,
) -> str:
    url, headers, payload, estimated_tokens = _build_request(config, system, user, task_name, schema)
    config.usage_collector.record_attempt(task_name)

    for attempt_index in range(6):
        await config.rate_limiter.acquire(estimated_tokens)

        try:
            response = await _post_request(url=url, headers=headers, payload=payload, timeout_s=config.timeout_s)
        except httpx.RequestError as exc:
            if attempt_index == 5:
                raise LLMError(f"OpenAI API request failed: {exc}") from exc
            await asyncio.sleep(_backoff_delay(attempt_index))
            continue

        if response.status_code == 429:
            if attempt_index == 5:
                raise LLMError(f"OpenAI rate limit error: {_response_detail(response)}")
            await asyncio.sleep(_retry_delay(response.headers, attempt_index))
            continue

        if response.status_code >= 500:
            if attempt_index == 5:
                raise LLMError(f"OpenAI server error: {_response_detail(response)}")
            await asyncio.sleep(_backoff_delay(attempt_index))
            continue

        if response.status_code >= 400:
            raise LLMError(f"OpenAI API error: {_response_detail(response)}")

        data = response.json()
        config.usage_collector.record_usage(task_name, _extract_usage(data))
        return _extract_output_text(data)

    raise LLMError("OpenAI request retry loop exhausted unexpectedly.")


async def _post_request(*, url: str, headers: dict[str, str], payload: dict[str, Any], timeout_s: int) -> httpx.Response:
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        return await client.post(url, headers=headers, json=payload)


def _build_request(
    config: OpenAIConfig,
    system: str,
    user: str,
    task_name: str,
    schema: type[BaseModel] | None,
) -> tuple[str, dict[str, str], dict[str, Any], int]:
    api_key = config.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("Missing OPENAI_API_KEY environment variable.")

    text_format = _text_format(task_name, schema)
    payload: dict[str, Any] = {
        "model": config.model,
        "instructions": system,
        "input": user,
        "max_output_tokens": config.max_output_tokens,
        "store": config.store,
        "text": {
            "verbosity": config.verbosity,
            "format": text_format,
        },
    }

    if config.reasoning_effort:
        payload["reasoning"] = {"effort": config.reasoning_effort}

    if _supports_temperature(config):
        payload["temperature"] = config.temperature

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": config.user_agent,
    }

    estimated_input_tokens = _estimate_request_tokens(system, user, text_format, config.model)
    estimated_tokens = max(estimated_input_tokens, config.max_output_tokens)
    return f"{config.base_url}/responses", headers, payload, estimated_tokens


def _text_format(task_name: str, schema: type[BaseModel] | None) -> dict[str, Any]:
    if schema is None:
        return {"type": "text"}

    return {
        "type": "json_schema",
        "name": task_name,
        "schema": schema.model_json_schema(),
        "strict": True,
    }


def _supports_temperature(config: OpenAIConfig) -> bool:
    if not config.model.startswith("gpt-5.4"):
        return False
    return config.reasoning_effort in (None, "none")


def _estimate_request_tokens(system: str, user: str, text_format: dict[str, Any], model_name: str) -> int:
    format_text = json.dumps(text_format, ensure_ascii=False, sort_keys=True)
    combined_text = "\n".join(part for part in (system, user, format_text) if part)
    return max(1, _estimate_text_tokens(combined_text, model_name))


def _estimate_text_tokens(text: str, model_name: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _extract_output_text(data: dict[str, Any]) -> str:
    output_items = data.get("output") or []
    text_parts: list[str] = []

    for item in output_items:
        if item.get("type") != "message":
            continue
        for content_part in item.get("content") or []:
            if content_part.get("type") == "output_text":
                text_parts.append(content_part.get("text") or "")

    if text_parts:
        return "".join(text_parts).strip()

    raise LLMError("Unexpected OpenAI response format.")


def _extract_usage(data: dict[str, Any]) -> dict[str, int]:
    usage = data.get("usage") or {}
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cached_input_tokens": int(input_details.get("cached_tokens") or 0),
        "reasoning_tokens": int(output_details.get("reasoning_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def _parse_json_response(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        extracted = _extract_json_object(raw)
        if extracted is None:
            raise LLMError("LLM did not return valid JSON.")
        return json.loads(extracted)


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text
    return json.dumps(payload, ensure_ascii=False)


def _retry_delay(headers: httpx.Headers, attempt_index: int) -> float:
    candidate_delays = [
        _parse_retry_after(headers.get("retry-after")),
        _parse_duration(headers.get("x-ratelimit-reset-requests")),
        _parse_duration(headers.get("x-ratelimit-reset-tokens")),
    ]
    valid_delays = [delay for delay in candidate_delays if delay is not None]
    base_delay = max(valid_delays) if valid_delays else _backoff_delay(attempt_index)
    return base_delay + random.uniform(0.05, 0.25)


def _backoff_delay(attempt_index: int) -> float:
    capped_attempt = min(attempt_index, 5)
    return min(60.0, (2**capped_attempt) + random.uniform(0.0, 0.5))


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return _parse_duration(value)


def _parse_duration(value: str | None) -> float | None:
    if not value:
        return None

    total_seconds = 0.0
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h|d)", value):
        numeric_value = float(amount)
        if unit == "ms":
            total_seconds += numeric_value / 1000.0
        elif unit == "s":
            total_seconds += numeric_value
        elif unit == "m":
            total_seconds += numeric_value * 60.0
        elif unit == "h":
            total_seconds += numeric_value * 3600.0
        elif unit == "d":
            total_seconds += numeric_value * 86_400.0

    if total_seconds > 0:
        return total_seconds
    return None


def _extract_json_object(text: str) -> Optional[str]:
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else None


def _role_for_task(task_name: str) -> str:
    if task_name in {"article_summary", "newsletter_split"}:
        return "researcher"
    if task_name == "judge_evaluation":
        return "judge"
    if task_name in {"draft_outline", "final_report_theme"}:
        return "writer"
    return "other"
