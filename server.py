"""
FastAPI Server
==============
Exposes the four-system pipeline as a REST API.

Endpoints:
  POST /analyze         — Full pipeline run, returns complete result
  POST /analyze/stream  — Full pipeline with Server-Sent Events for real-time progress
  GET  /health          — Health check
  POST /classify        — Classification only (fast, for frontend pre-flight)

NEXT STEPS:
  - Add authentication (API key or OAuth2) before any public deployment.
  - Add rate limiting (slowapi) to prevent abuse.
  - Add request ID tracking and structured logging (JSON logs to stdout)
    for production observability.
  - Add a database layer to persist results and enable history/replay.
  - Add input validation: minimum description length, profanity/abuse filter.
  - Add CORS configuration for your specific frontend domain.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue as stdlib_queue
import re
import threading
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pipeline import run_full_pipeline, PipelineResult
from utils.llm_client import call_llm_chat

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Biotech Navigator API",
    description=(
        "AI-powered regulatory pathway classification, testing roadmap generation, "
        "IP exposure analysis, and smart materials optimization for early-stage "
        "biotech and medtech teams."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    description: str = Field(
        ...,
        min_length=20,
        max_length=5000,
        description="Plain-language description of the medical device or biotech product.",
        examples=[
            "A resorbable bone screw made from PLGA polymer for fixation of small bone fractures in the hand. "
            "The screw is fully absorbed within 18-24 months, eliminating the need for hardware removal surgery."
        ],
    )
    run_ip_radar: bool = Field(
        default=True,
        description="Whether to run the IP radar (adds ~15-30s to response time).",
    )
    run_materials_optimization: bool = Field(
        default=True,
        description="Whether to run materials optimization (adds ~10-20s to response time).",
    )


class HealthResponse(BaseModel):
    status: str
    api_key_configured: bool
    version: str


class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


class ChatResponse(BaseModel):
    reply: str
    ready: bool = False
    description: Optional[str] = None


_CHAT_SYSTEM = """\
You are a regulatory intelligence assistant for compl_AI, helping early-stage biotech \
and medtech researchers describe their product for FDA regulatory analysis.

Your job: have a warm, encouraging conversation to understand their product well enough \
to generate a full regulatory analysis. You are looking for:
1. What the product is and what clinical problem it solves
2. Materials or active components
3. How it contacts the body (on the skin, implanted, ingested, inhaled, blood contact, etc.)
4. Duration of body contact (minutes? months? permanent?)
5. Target patient population

Guidelines:
- Encourage tentative or half-baked ideas — imprecision is fine, you will help shape it
- Ask 1–2 focused questions per turn, not a laundry list
- After 2–3 exchanges with reasonable information, naturally offer to run the full analysis
- Be concise; this is a quick intake, not a consultation

When you have enough information (even if incomplete), end your message with EXACTLY this \
block and nothing after it:

<READY>{"description": "your synthesized product description in 20–200 words"}</READY>

The description should be a clear clinical paragraph combining all gathered info, \
suitable for FDA regulatory analysis.\
"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        api_key_configured=bool(os.getenv("ANTHROPIC_API_KEY")),
        version="0.1.0",
    )


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    """
    Full pipeline: classify → roadmap → IP radar → materials optimization.
    Returns the complete analysis result.

    Typical response time: 45-90 seconds (dominated by LLM calls and patent API fetches).
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not configured. Set this environment variable to use the API.",
        )

    logger.info("Received analyze request (description length=%d)", len(request.description))

    result = run_full_pipeline(request.description)

    if not result.success:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Pipeline failed to complete minimum required steps.",
                "errors": result.errors,
            },
        )

    response_dict = result.to_dict()

    # Optionally strip sections the caller didn't request
    if not request.run_ip_radar:
        response_dict.pop("ip_radar", None)
    if not request.run_materials_optimization:
        response_dict.pop("materials_optimization", None)

    return response_dict


@app.post("/analyze/stream")
async def analyze_stream(request: AnalyzeRequest):
    """
    Full pipeline with Server-Sent Events for real-time progress updates.

    Streams JSON events as each pipeline stage completes:
      {"type": "progress", "step": "classification", "message": "...", "status": "running"}
      {"type": "progress", "step": "roadmap", "message": "...", "status": "done"}
      {"type": "result", "data": { ...full pipeline result... }}
      {"type": "error", "message": "..."}

    Typical total time: 45-90 seconds.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not configured. Set this environment variable to use the API.",
        )

    logger.info(
        "Received analyze/stream request (description length=%d)", len(request.description)
    )

    # Thread-safe queue for progress events; None is the sentinel value.
    progress_q: stdlib_queue.Queue = stdlib_queue.Queue()

    def progress_callback(event: dict) -> None:
        progress_q.put(event)

    def run_pipeline() -> None:
        try:
            result = run_full_pipeline(
                request.description,
                progress_callback=progress_callback,
            )
            if not result.success:
                progress_q.put({
                    "type": "error",
                    "message": "Pipeline failed to complete minimum required steps.",
                    "errors": result.errors,
                })
            else:
                response = result.to_dict()
                if not request.run_ip_radar:
                    response.pop("ip_radar", None)
                if not request.run_materials_optimization:
                    response.pop("materials_optimization", None)
                progress_q.put({"type": "result", "data": response})
        except Exception as exc:
            logger.error("Stream pipeline failed: %s", exc, exc_info=True)
            progress_q.put({"type": "error", "message": str(exc)})
        finally:
            progress_q.put(None)  # Sentinel: stream is complete

    # Run pipeline in a background daemon thread so it doesn't block the event loop.
    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    async def event_stream():
        while True:
            # Poll the thread-safe queue without blocking the async event loop.
            try:
                event = progress_q.get_nowait()
            except stdlib_queue.Empty:
                await asyncio.sleep(0.1)
                continue

            if event is None:
                # Sentinel received — pipeline is complete.
                break

            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable Nginx buffering if behind a proxy
            "Connection": "keep-alive",
        },
    )


@app.post("/classify")
def classify_only(request: AnalyzeRequest):
    """
    Classification only — fast endpoint for pre-flight checks.
    Returns device class, pathway, and confidence without running
    the full testing roadmap or IP analysis.

    Typical response time: 5-15 seconds.
    """
    from systems.classification_engine import classify_device

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured.")

    try:
        result = classify_device(request.description)
        return result.model_dump()
    except Exception as e:
        logger.error("Classification failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Conversational intake endpoint.
    Accepts a message history and returns Claude's next reply.
    When Claude has gathered enough product information it appends a
    <READY>{...}</READY> block; the endpoint strips that block, sets
    ready=True, and returns the synthesised description separately.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured.")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    try:
        reply_text = call_llm_chat(_CHAT_SYSTEM, messages, max_tokens=600)
    except Exception as exc:
        logger.error("Chat LLM failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    ready_match = re.search(r"<READY>\s*(\{.*?\})\s*</READY>", reply_text, re.DOTALL)
    if ready_match:
        try:
            data = json.loads(ready_match.group(1))
            description = data.get("description", "")
            visible_reply = reply_text[: ready_match.start()].strip()
            return ChatResponse(reply=visible_reply, ready=True, description=description)
        except Exception:
            pass  # malformed JSON — fall through and return as plain message

    return ChatResponse(reply=reply_text, ready=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
