from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from ..llm import LLMError, OpenAIConfig, generate_json_async, generate_text_async

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredOutputError(LLMError):
    pass


@dataclass(slots=True)
class ValidationFailure:
    task: str
    message: str


class LLMClient(Protocol):
    async def generate_text(self, *, system: str, user: str) -> str:
        ...

    async def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[SchemaT],
        task_name: str,
    ) -> SchemaT:
        ...


class OpenAILLMClient:
    def __init__(self, config: OpenAIConfig) -> None:
        self.config = config

    async def generate_text(self, *, system: str, user: str) -> str:
        return await generate_text_async(
            config=self.config,
            system=system,
            user=user,
            task_name="final_report_theme",
        )

    async def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[SchemaT],
        task_name: str,
    ) -> SchemaT:
        try:
            payload = await generate_json_async(
                config=self.config,
                system=system,
                user=user,
                schema=schema,
                task_name=task_name,
            )
        except LLMError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            raise StructuredOutputError(f"{task_name} failed before validation: {exc}") from exc

        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            raise StructuredOutputError(f"{task_name} returned invalid structured output: {exc}") from exc

    def get_usage_summary(self) -> dict[str, object]:
        return self.config.usage_collector.snapshot()


XAILLMClient = OpenAILLMClient
