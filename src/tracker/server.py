import asyncio
import json
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
from pathlib import Path

app = FastAPI()

class TrackerState:
    def __init__(self):
        self.stage = "STARTING"
        self.message = "Initializing pipeline..."
        self.logs = []
        self._new_data_event = asyncio.Event()

    def snapshot(self) -> dict:
        return {
            "stage": self.stage,
            "message": self.message,
        }

    def update(self, stage: str, message: str, details: dict | None = None):
        self.stage = stage
        self.message = message
        self.logs.append({"stage": stage, "message": message, "details": details or {}})
        self._new_data_event.set()

    async def wait_for_data(self):
        await self._new_data_event.wait()
        self._new_data_event.clear()

state = TrackerState()

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = Path(__file__).parent / "index.html"
    return index_path.read_text(encoding="utf-8")

@app.get("/stream")
async def message_stream(request: Request):
    async def event_generator() -> AsyncGenerator[dict, None]:
        yield {
            "event": "update",
            "data": json.dumps(state.snapshot(), ensure_ascii=False),
        }
        while True:
            if await request.is_disconnected():
                break
            await state.wait_for_data()
            yield {
                "event": "update",
                "data": json.dumps(state.snapshot(), ensure_ascii=False),
            }
            
    return EventSourceResponse(event_generator())

def start_server_in_background():
    import uvicorn
    import threading

    global state
    state = TrackerState()

    def run():
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return state
