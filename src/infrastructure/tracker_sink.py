from __future__ import annotations

from .events import PipelineEvent


class TrackerEventSink:
    def __init__(self, tracker_state) -> None:
        self._tracker_state = tracker_state

    def emit(self, event: PipelineEvent) -> None:
        self._tracker_state.update(
            stage=event.stage,
            message=event.message,
            details=event.details,
        )
