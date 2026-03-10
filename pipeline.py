"""
Pipeline Orchestrator
======================
Wires all four systems together into a single coherent flow.

The orchestrator handles:
  - Sequential system calls with clean data handoffs
  - Parallel execution where systems are independent (IP radar runs in parallel)
  - Error handling and partial result recovery
  - Logging and timing
  - Optional progress callbacks for SSE streaming

Usage:
    from pipeline import run_full_pipeline
    result = run_full_pipeline("A resorbable bone screw made from PLGA...")

NEXT STEPS:
  - Add a job queue (Celery + Redis) so long-running analyses don't block HTTP
    requests. The full pipeline can take 30-90s due to LLM calls + API fetches.
  - Add result persistence: store every pipeline run in PostgreSQL with the
    user's description, all intermediate results, and the final output.
    This builds your dataset for future fine-tuning.
  - Add a "resume" capability: if a pipeline run fails partway through,
    allow resumption from the last successful step using cached intermediates.
  - Add A/B testing: route 50% of requests through an alternative prompt set
    to test classification accuracy improvements.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

from systems.classification_engine import classify_device
from systems.roadmap_generator import generate_roadmap
from systems.ip_radar import run_ip_radar
from systems.materials_engine import optimize_materials
from utils.models import (
    ClassificationResult,
    IPRadarResult,
    MaterialsOptimizationResult,
    RoadmapResult,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Aggregated output of the full four-system pipeline.
    All fields except classification are Optional — partial results
    are better than a hard failure.
    """
    raw_description: str
    elapsed_seconds: float = 0.0

    classification: Optional[ClassificationResult] = None
    roadmap: Optional[RoadmapResult] = None
    ip_radar: Optional[IPRadarResult] = None
    materials_optimization: Optional[MaterialsOptimizationResult] = None

    errors: dict[str, str] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.classification is not None and self.roadmap is not None

    def to_dict(self) -> dict:
        """
        Serialize to a JSON-safe dict for API responses.
        Uses Pydantic's .model_dump() for nested models.
        """
        result: dict = {
            "raw_description": self.raw_description,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "success": self.success,
            "errors": self.errors,
        }
        if self.classification:
            result["classification"] = self.classification.model_dump()
        if self.roadmap:
            result["roadmap"] = self.roadmap.model_dump()
        if self.ip_radar:
            result["ip_radar"] = self.ip_radar.model_dump()
        if self.materials_optimization:
            result["materials_optimization"] = self.materials_optimization.model_dump()
        return result


def run_full_pipeline(
    raw_description: str,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> PipelineResult:
    """
    Execute all four systems in the correct order with parallelism where possible.

    Execution order:
      1. Classification (System 1) — must complete first; everything depends on it
      2. Roadmap generation (System 2) — depends on classification
         IP Radar (System 3) — depends only on ProductProfile, can run in parallel with System 2
      3. Materials optimization (System 4) — depends on roadmap

    Systems 2 and 3 run in parallel using a thread pool.

    Args:
        raw_description: Plain-language product description.
        progress_callback: Optional callable that receives progress event dicts.
            Called from both the main thread and worker threads — must be thread-safe.
            Event format: {"type": "progress", "step": str, "message": str, "status": str}
            where status is "running" | "done" | "error".
    """

    def emit(step: str, message: str, status: str = "running") -> None:
        """Thread-safe progress event emitter. Never raises."""
        if progress_callback is None:
            return
        try:
            progress_callback({
                "type": "progress",
                "step": step,
                "message": message,
                "status": status,
            })
        except Exception as cb_err:
            logger.warning("[Pipeline] Progress callback raised: %s", cb_err)

    result = PipelineResult(raw_description=raw_description)
    start_time = time.time()

    # ---- Step 1: Classification ----
    emit("classification", "Extracting product attributes from description...")
    logger.info("[Pipeline] Step 1: Classification")
    try:
        emit("classification", "Querying FDA classification database...")
        result.classification = classify_device(raw_description)
        logger.info(
            "[Pipeline] Classification complete: %s / %s (confidence=%.2f)",
            result.classification.device_class,
            result.classification.regulatory_pathway,
            result.classification.confidence,
        )
        emit(
            "classification",
            f"Classified: {result.classification.device_class} → "
            f"{result.classification.regulatory_pathway} "
            f"(confidence: {result.classification.confidence:.0%})",
            "done",
        )
    except Exception as e:
        logger.error("[Pipeline] Classification failed: %s", e, exc_info=True)
        result.errors["classification"] = str(e)
        emit("classification", f"Classification failed: {e}", "error")
        result.elapsed_seconds = time.time() - start_time
        return result  # Cannot continue without classification

    # ---- Step 2 + 3: Roadmap and IP Radar in parallel ----
    logger.info("[Pipeline] Step 2+3: Roadmap generation and IP radar (parallel)")
    emit("roadmap", "Applying ISO 10993-1:2018 biocompatibility matrix...")
    emit("ip_radar", "Generating patent search queries...")

    def run_roadmap():
        try:
            emit("roadmap", "Building testing dependency graph and critical path...")
            roadmap = generate_roadmap(result.classification)
            logger.info(
                "[Pipeline] Roadmap complete: %d tests, $%s–$%s, %s–%s weeks",
                len(roadmap.tests),
                f"{roadmap.total_cost_usd_low:,}",
                f"{roadmap.total_cost_usd_high:,}",
                roadmap.total_weeks_low,
                roadmap.total_weeks_high,
            )
            emit(
                "roadmap",
                f"Roadmap built: {len(roadmap.tests)} tests, "
                f"${roadmap.total_cost_usd_low:,}–${roadmap.total_cost_usd_high:,}, "
                f"{roadmap.total_weeks_low}–{roadmap.total_weeks_high} weeks",
                "done",
            )
            return roadmap
        except Exception as e:
            logger.error("[Pipeline] Roadmap generation failed: %s", e, exc_info=True)
            result.errors["roadmap"] = str(e)
            emit("roadmap", f"Roadmap generation failed: {e}", "error")
            return None

    def run_ip_radar_task():
        try:
            emit("ip_radar", "Searching USPTO patent database...")
            ip_result = run_ip_radar(result.classification.product_profile)
            red_count = sum(1 for p in ip_result.patents if p.relevance.value == "red")
            logger.info(
                "[Pipeline] IP radar complete: %d patents, %d red flags",
                len(ip_result.patents),
                red_count,
            )
            emit(
                "ip_radar",
                f"IP analysis complete: {len(ip_result.patents)} patents found, "
                f"{red_count} high-risk",
                "done",
            )
            return ip_result
        except Exception as e:
            logger.error("[Pipeline] IP radar failed: %s", e, exc_info=True)
            result.errors["ip_radar"] = str(e)
            emit("ip_radar", f"IP radar failed: {e}", "error")
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        roadmap_future = executor.submit(run_roadmap)
        ip_future = executor.submit(run_ip_radar_task)
        result.roadmap = roadmap_future.result()
        result.ip_radar = ip_future.result()

    # ---- Step 4: Materials optimization ----
    if result.roadmap:
        emit("materials", "Evaluating material substitution candidates...")
        logger.info("[Pipeline] Step 4: Materials optimization")
        try:
            emit("materials", "Simulating alternative testing roadmaps...")
            result.materials_optimization = optimize_materials(result.roadmap)
            rec_count = len(result.materials_optimization.recommendations)
            logger.info("[Pipeline] Materials optimization complete: %d recommendations", rec_count)
            emit(
                "materials",
                f"Material optimization complete: {rec_count} "
                f"recommendation{'s' if rec_count != 1 else ''} identified",
                "done",
            )
        except Exception as e:
            logger.error("[Pipeline] Materials optimization failed: %s", e, exc_info=True)
            result.errors["materials_optimization"] = str(e)
            emit("materials", f"Materials optimization failed: {e}", "error")

    result.elapsed_seconds = time.time() - start_time
    logger.info(
        "[Pipeline] Complete in %.1fs. Success=%s. Errors=%s",
        result.elapsed_seconds,
        result.success,
        list(result.errors.keys()) or "none",
    )
    return result
