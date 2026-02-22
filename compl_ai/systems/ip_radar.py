"""
System 3: IP Radar
===================
Searches for patents that may conflict with the user's device description
and returns plain-language relevance explanations with traffic-light ratings.

Architecture:
  1. Generate diverse search queries from the product profile
  2. Query USPTO PatentsView API (v1) in parallel across queries
  3. Deduplicate results
  4. Run each patent through an LLM relevance assessment
  5. Assign traffic-light flags and generate a plain-English IP landscape summary

THIS IS NOT A FREEDOM-TO-OPERATE (FTO) TOOL.
This tool provides prior art radar to flag patents that should be reviewed
by a registered patent attorney. It does not and cannot provide legal advice.

FIXES in v1.1:
  - Old endpoint (api.patentsview.org/patents/query) returned HTTP 410 Gone —
    it has been retired. Updated to new PatentSearch API v1:
    https://search.patentsview.org/api/v1/patent/
  - Updated query syntax: new API uses GET with URL-encoded q parameter,
    and pagination uses "size" instead of "per_page".
  - Updated field names to match v1 response schema:
    patent_title → patent_title (unchanged)
    patent_abstract → patent_abstract (unchanged)
    assignee_organization → assignees.assignee_organization (nested)
    app_date → applications.app_date (nested)
  - Updated response parsing to match new nested structure.

NEXT STEPS:
  - Add semantic patent search via Semantic Scholar or the
    USPTO Patent Examination Data System (PEDS).
  - Add claim-level parsing: fetch and parse full claim text (independent
    claims especially) for more accurate relevance scoring.
  - Add continuation/family tracking.
  - Add prosecution history lookup (file wrapper).
  - For a premium tier, integrate with Derwent, PatSnap, or Lens.org.
  - Add a "design around" feature.
  - Add expiration date calculation (20 years from priority date,
    minus terminal disclaimers, plus patent term adjustments).
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from typing import Optional

import httpx

from utils.llm_client import call_llm, call_llm_for_json
from utils.models import (
    IPRadarResult,
    PatentRelevance,
    PatentResult,
    ProductProfile,
)

logger = logging.getLogger(__name__)

# Updated to PatentSearch API v1 (old api.patentsview.org endpoint is 410 Gone)
PATENTSVIEW_API = "https://search.patentsview.org/api/v1/patent/"

MAX_PATENTS_TO_ANALYZE = 8    # LLM calls are expensive; cap the deep analysis
MAX_SEARCH_RESULTS = 15       # Raw results to fetch before LLM ranking


# ---------------------------------------------------------------------------
# Step 1: Search query generation
# ---------------------------------------------------------------------------

QUERY_GEN_SYSTEM_PROMPT = """
You are a patent search specialist with expertise in medical devices and biotech.

Given a product profile, generate exactly 4 patent search queries that together
provide broad coverage of potential prior art. Each query should target a
different angle:
  1. Device type + indication (broad)
  2. Mechanism of action (specific)
  3. Key material(s) + function
  4. Novel aspect or unique combination described

Keep each query to 3-6 keywords. Do NOT use Boolean operators or quoted phrases.

Return JSON: {"queries": ["query1", "query2", "query3", "query4"]}
"""


def generate_search_queries(profile: ProductProfile) -> list[str]:
    """
    Use LLM to generate diverse patent search queries from the product profile.
    """
    profile_text = (
        f"Intended use: {profile.intended_use}\n"
        f"Indication: {profile.indication}\n"
        f"Mechanism: {profile.mechanism_of_action}\n"
        f"Materials: {', '.join(profile.materials) or 'unspecified'}\n"
        f"Description: {profile.raw_description[:500]}\n"
    )

    try:
        data = call_llm_for_json(
            system_prompt=QUERY_GEN_SYSTEM_PROMPT,
            user_message=profile_text,
        )
        queries = data.get("queries", [])
        if isinstance(queries, list) and queries:
            return queries[:4]
    except Exception as e:
        logger.warning("Query generation failed: %s", e)

    # Fallback: construct queries from profile fields
    fallback = []
    if profile.intended_use:
        fallback.append(profile.intended_use[:60])
    if profile.indication:
        fallback.append(f"{profile.indication} device")
    if profile.materials:
        fallback.append(f"{profile.materials[0]} medical device")
    fallback.append(f"{profile.mechanism_of_action.value} medical device implant")
    return fallback[:4]


# ---------------------------------------------------------------------------
# Step 2: Patent fetching  (PatentSearch API v1)
# ---------------------------------------------------------------------------

def _search_patentsview(query: str, limit: int = 10) -> list[dict]:
    """
    Query the PatentsView PatentSearch API v1.

    New API (search.patentsview.org/api/v1/patent/) uses:
      - GET requests with URL-encoded q parameter
      - "size" for result count (not "per_page")
      - "_contains" operator for full-text search on string fields
      - Nested objects for assignees and applications

    Filters to patents granted after 2004 to focus on potentially active ones.
    No API key required for the v1 public endpoint.
    """
    query_obj = {
        "_and": [
            {"_contains": {"patent_abstract": query}},
            {"_gte": {"patent_date": "2004-01-01"}},  # Focus on potentially active patents
        ]
    }

    fields = [
        "patent_id",
        "patent_title",
        "patent_abstract",
        "patent_date",
        "assignees.assignee_organization",
        "applications.app_date",
    ]

    params = {
        "q": json.dumps(query_obj),
        "f": json.dumps(fields),
        "o": json.dumps({"size": limit}),
    }

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.get(PATENTSVIEW_API, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("patents") or []
    except httpx.HTTPStatusError as e:
        logger.warning(
            "PatentsView API HTTP error for query '%s': %s — %s",
            query, e.response.status_code, e.response.text[:200],
        )
        return []
    except Exception as e:
        logger.warning("PatentsView API error for query '%s': %s", query, e)
        return []


def _normalize_patentsview_result(raw: dict) -> dict:
    """
    Normalise PatentsView v1 result into our common patent dict format.

    v1 nests assignees and applications as lists of objects, unlike the
    old flat structure. Field name is also patent_id (not patent_number)
    in v1, though we expose it as patent_number for consistency downstream.
    """
    # Assignees is a nested list in v1
    assignees = raw.get("assignees") or []
    assignee = assignees[0].get("assignee_organization", "Unknown") if assignees else "Unknown"

    # Application date is nested
    applications = raw.get("applications") or []
    app_date = applications[0].get("app_date", "") if applications else ""

    patent_date = raw.get("patent_date", "")

    # Rough expiration: 20 years from application date
    expiration = ""
    if app_date and len(app_date) >= 4:
        try:
            expiration = str(int(app_date[:4]) + 20) + app_date[4:]
        except ValueError:
            pass

    return {
        # v1 uses "patent_id" — expose as patent_number for consistency
        "patent_number": raw.get("patent_id", ""),
        "title": raw.get("patent_title", ""),
        "abstract": raw.get("patent_abstract", ""),
        "assignee": assignee,
        "filing_date": app_date,
        "grant_date": patent_date,
        "expiration_date": expiration,
        "source": "patentsview_v1",
    }


def fetch_patents_for_queries(queries: list[str]) -> list[dict]:
    """
    Run all search queries and return a deduplicated list of raw patent dicts.
    """
    all_patents: list[dict] = []
    seen_numbers: set[str] = set()

    for query in queries:
        per_query_limit = max(3, MAX_SEARCH_RESULTS // len(queries) + 2)
        raw_results = _search_patentsview(query, limit=per_query_limit)
        for raw in raw_results:
            normalized = _normalize_patentsview_result(raw)
            num = normalized["patent_number"]
            if num and num not in seen_numbers:
                seen_numbers.add(num)
                all_patents.append(normalized)

    logger.info("Fetched %d unique patents across %d queries", len(all_patents), len(queries))
    return all_patents[:MAX_SEARCH_RESULTS]


# ---------------------------------------------------------------------------
# Step 3: LLM relevance assessment
# ---------------------------------------------------------------------------

RELEVANCE_SYSTEM_PROMPT = """
You are a patent attorney's assistant specializing in medical devices and biotech.

Your job is to assess whether a patent could potentially conflict with a described product.

Given:
1. A product description
2. A patent title and abstract

Assess:
- Whether the patent's claims might "read on" (cover) the described product
- Which specific aspects create overlap
- A relevance rating: "green" (not relevant), "yellow" (possible overlap), or "red" (high overlap risk)

Rating guide:
  green: Patent is expired, clearly different technology, or claims don't read on product
  yellow: Some claim language could apply, or the technology is adjacent — worth legal review
  red: Strong similarity in mechanism, materials, or intended use with apparently active patent

Return JSON:
{
  "relevance": "green" | "yellow" | "red",
  "explanation": "2-3 sentence plain-English explanation of why this patent is or isn't relevant",
  "concerning_claims": ["list of specific claim language or aspects of concern, or empty list if green"],
  "is_likely_active": true/false
}

IMPORTANT: Be conservative. When uncertain, rate yellow not green.
Do NOT provide legal advice. Frame findings as observations, not legal conclusions.
"""


def assess_patent_relevance(patent: dict, product_description: str) -> dict:
    """
    Run LLM relevance assessment for a single patent.
    Returns the relevance dict or a safe default on failure.
    """
    message = (
        f"PRODUCT DESCRIPTION:\n{product_description[:600]}\n\n"
        f"PATENT TITLE: {patent.get('title', 'N/A')}\n\n"
        f"PATENT ABSTRACT:\n{patent.get('abstract', 'N/A')[:800]}\n"
    )

    try:
        data = call_llm_for_json(
            system_prompt=RELEVANCE_SYSTEM_PROMPT,
            user_message=message,
        )
        return data
    except Exception as e:
        logger.warning("Relevance assessment failed for patent %s: %s", patent.get("patent_number"), e)
        return {
            "relevance": "yellow",
            "explanation": "Automated relevance assessment failed. Manual review recommended.",
            "concerning_claims": [],
            "is_likely_active": True,
        }


def _is_patent_active(patent: dict, relevance_data: dict) -> bool:
    """
    Determine if a patent is likely still active.
    Uses LLM's assessment + expiration date estimate.
    """
    if not relevance_data.get("is_likely_active", True):
        return False

    expiration = patent.get("expiration_date", "")
    if expiration and len(expiration) >= 4:
        try:
            exp_year = int(expiration[:4])
            if exp_year < 2025:
                return False
        except ValueError:
            pass

    return True


def _map_relevance(relevance_str: str, is_active: bool) -> PatentRelevance:
    """Map relevance string + active status to PatentRelevance enum."""
    if not is_active:
        return PatentRelevance.GREEN
    mapping = {
        "green": PatentRelevance.GREEN,
        "yellow": PatentRelevance.YELLOW,
        "red": PatentRelevance.RED,
    }
    return mapping.get(relevance_str.lower(), PatentRelevance.YELLOW)


# ---------------------------------------------------------------------------
# Step 4: IP landscape summary
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = """
You are a patent strategist helping an early-stage medtech startup understand
their IP landscape.

Given a product description and a set of patent findings, write a concise
3-4 sentence plain-English summary covering:
1. The overall IP density in this space (crowded vs. open)
2. The most significant risk areas flagged
3. The single most important action item for the team

Be direct and practical. Acknowledge this is not legal advice in one brief phrase.
Do not list the patents — just synthesize the landscape.
"""


def generate_ip_summary(profile: ProductProfile, patents: list[PatentResult]) -> str:
    """Generate a plain-English IP landscape summary."""
    red_count = sum(1 for p in patents if p.relevance == PatentRelevance.RED)
    yellow_count = sum(1 for p in patents if p.relevance == PatentRelevance.YELLOW)

    message = (
        f"Device: {profile.intended_use} for {profile.indication}\n"
        f"Materials: {', '.join(profile.materials) or 'unspecified'}\n\n"
        f"Search returned {len(patents)} potentially relevant patents:\n"
        f"  - High risk (red): {red_count}\n"
        f"  - Moderate concern (yellow): {yellow_count}\n"
        f"  - Low concern (green): {len(patents) - red_count - yellow_count}\n\n"
        "Most concerning patents:\n"
    )
    for p in sorted(patents, key=lambda x: {"red": 0, "yellow": 1, "green": 2}[x.relevance.value])[:3]:
        message += f"  - {p.title} ({p.patent_number}): {p.relevance_explanation[:100]}\n"

    try:
        return call_llm(system_prompt=SUMMARY_SYSTEM_PROMPT, user_message=message)
    except Exception as e:
        logger.warning("IP summary generation failed: %s", e)
        return (
            f"IP search identified {len(patents)} potentially relevant patents "
            f"({red_count} high-risk, {yellow_count} moderate concern). "
            "Consult a patent attorney for a formal FTO analysis."
        )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_ip_radar(profile: ProductProfile) -> IPRadarResult:
    """
    Main entry point for System 3.
    Takes a ProductProfile and returns a full IPRadarResult.
    """
    logger.info("Starting IP radar for: %s", profile.intended_use[:60])

    # Step 1: Generate search queries
    queries = generate_search_queries(profile)
    logger.info("Generated %d search queries: %s", len(queries), queries)

    # Step 2: Fetch patents
    raw_patents = fetch_patents_for_queries(queries)

    if not raw_patents:
        logger.warning("No patents returned from search APIs")
        return IPRadarResult(
            product_profile=profile,
            patents=[],
            search_queries_used=queries,
            summary=(
                "Patent searches returned no results. This may indicate an open IP space, "
                "or it may reflect API limitations. Manual search on Google Patents and "
                "USPTO Full-Text Database is recommended."
            ),
        )

    # Step 3: Assess relevance for top patents
    analyzed_patents: list[PatentResult] = []
    for raw in raw_patents[:MAX_PATENTS_TO_ANALYZE]:
        relevance_data = assess_patent_relevance(raw, profile.raw_description)
        is_active = _is_patent_active(raw, relevance_data)
        relevance_enum = _map_relevance(relevance_data.get("relevance", "yellow"), is_active)

        patent_result = PatentResult(
            patent_number=raw.get("patent_number", "Unknown"),
            title=raw.get("title", "Unknown"),
            abstract=raw.get("abstract", ""),
            assignee=raw.get("assignee", "Unknown"),
            filing_date=raw.get("filing_date", ""),
            expiration_date=raw.get("expiration_date"),
            is_active=is_active,
            relevance=relevance_enum,
            relevance_explanation=relevance_data.get("explanation", ""),
            concerning_claims=relevance_data.get("concerning_claims", []),
        )
        analyzed_patents.append(patent_result)
        logger.info("Patent %s rated: %s", patent_result.patent_number, patent_result.relevance.value)

    # Sort: red → yellow → green
    sort_order = {PatentRelevance.RED: 0, PatentRelevance.YELLOW: 1, PatentRelevance.GREEN: 2}
    analyzed_patents.sort(key=lambda p: sort_order[p.relevance])

    # Step 4: Generate summary
    summary = generate_ip_summary(profile, analyzed_patents)

    return IPRadarResult(
        product_profile=profile,
        patents=analyzed_patents,
        search_queries_used=queries,
        summary=summary,
    )
