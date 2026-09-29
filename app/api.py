"""FastAPI service: SSE streaming briefing endpoint + demo page."""
from __future__ import annotations

import json
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agents import MAX_INPUT_CHARS, stream_briefing
from .llm import get_llm

try:
    from sse_starlette.sse import EventSourceResponse

    _HAS_SSE = True
except ImportError:  # fallback: plain StreamingResponse with manual SSE framing
    _HAS_SSE = False

app = FastAPI(title="sentinel-agent", version="0.1.0")


class BriefRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)


def _event_payloads(question: str):
    """Yield one JSON string per pipeline event (no SSE framing)."""
    for event in stream_briefing(question):
        yield json.dumps(event)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_mode": os.getenv("LLM_MODE", "mock")}


@app.get("/stream")
def stream(question: str = Query(..., min_length=1, max_length=MAX_INPUT_CHARS)):
    """Server-sent events stream of the briefing pipeline (EventSource-ready)."""
    get_llm()  # fail fast on bad LLM config before streaming starts
    if _HAS_SSE:
        # EventSourceResponse applies the `data:` framing itself.
        return EventSourceResponse(_event_payloads(question))

    def gen():
        for payload in _event_payloads(question):
            yield f"data: {payload}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/brief")
def brief(payload: BriefRequest) -> dict:
    """Non-streaming variant: run to completion, return the final briefing."""
    final_answer = ""
    events = []
    for event in stream_briefing(payload.question):
        events.append(event)
        if event["type"] == "done":
            final_answer = event["data"]["final_answer"]
        elif event["type"] == "error":
            raise HTTPException(status_code=500, detail=event["data"])
    return {"final_answer": final_answer, "events": events}


WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
