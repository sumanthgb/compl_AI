"""
Shared LLM client wrapper.

Centralises all calls to the Anthropic API so that:
  - Retry logic lives in one place
  - Model name is configured once
  - Structured JSON extraction is consistent
  - Token usage can be logged/monitored centrally

NEXT STEPS:
  - Add a caching layer (Redis or sqlite) so identical product descriptions
    don't re-hit the API during demos or repeated testing.
  - Swap claude-3-5-sonnet for a fine-tuned model once you have labeled
    classification data — even 500 examples will improve accuracy meaningfully.
  - Add prompt versioning: store prompt templates in a DB with version IDs so
    you can A/B test prompt changes without redeploying.
"""

from __future__ import annotations

import json
import os
import re
import logging
from typing import Any, Type, TypeVar

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"          # Upgrade to Opus for production classification
MAX_TOKENS = 4096

T = TypeVar("T")


def _get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in environment.")
    return anthropic.Anthropic(api_key=api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(
    system_prompt: str,
    user_message: str,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """
    Call Claude and return the raw text response.
    Retries up to 3 times with exponential backoff on transient errors.
    """
    client = _get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def call_llm_for_json(
    system_prompt: str,
    user_message: str,
    max_tokens: int = MAX_TOKENS,
) -> dict[str, Any]:
    """
    Call Claude expecting a JSON response.
    Strips markdown code fences if present, then parses.
    Raises ValueError if the response is not valid JSON.
    """
    raw = call_llm(
        system_prompt=system_prompt + "\n\nYou MUST respond with valid JSON only. No preamble, no explanation, no markdown fences.",
        user_message=user_message,
        max_tokens=max_tokens,
    )

    # Strip markdown code fences if the model adds them anyway
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("JSON parse failed. Raw response:\n%s", raw)
        raise ValueError(f"LLM returned non-JSON output: {e}") from e
