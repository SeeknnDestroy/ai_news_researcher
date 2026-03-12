from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any, Protocol


@dataclass(slots=True)
class PipelineEvent:
    stage: str
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    elapsed_s: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


class EventSink(Protocol):
    def emit(self, event: PipelineEvent) -> None:
        ...


class NullEventSink:
    def emit(self, event: PipelineEvent) -> None:
        del event


class ConsoleEventSink:
    def __init__(self) -> None:
        self._started_at = perf_counter()

    def emit(self, event: PipelineEvent) -> None:
        elapsed = event.elapsed_s if event.elapsed_s is not None else perf_counter() - self._started_at
        timestamp = event.timestamp.strftime("%H:%M:%S")
        payload = f"[{timestamp}] {event.stage}"
        if event.message:
            payload += f" | {event.message}"
        payload += f" (+{elapsed:.1f}s)"
        print(payload, flush=True)


class CompositeEventSink:
    def __init__(self, *sinks: EventSink) -> None:
        self._sinks = [sink for sink in sinks if sink is not None]

    def emit(self, event: PipelineEvent) -> None:
        for sink in self._sinks:
            sink.emit(event)
