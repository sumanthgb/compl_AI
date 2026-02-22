"""
FastAPI Server
==============
Exposes the four-system pipeline as a REST API.

Endpoints:
  POST /analyze       — Full pipeline run, returns complete result
  GET  /health        — Health check
  GET  /classify      — Classification only (fast, for frontend pre-flight)

NEXT STEPS:
  - Add authentication (API key or OAuth2) before any public deployment.
  - Add rate limiting (slowapi) to prevent abuse.
  - Add a POST /analyze/stream endpoint using Server-Sent Events (SSE)
    so the frontend can show progress as each system completes.
    FastAPI supports this natively with StreamingResponse.
  - Add request ID tracking and structured logging (JSON logs to stdout)
    for production observability.
  - Add a database layer to persist results and enable history/replay.
  - Add input validation: minimum description length, profanity/abuse filter.
  - Add CORS configuration for your specific frontend domain.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import run_full_pipeline, PipelineResult

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
        min_length=50,
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
