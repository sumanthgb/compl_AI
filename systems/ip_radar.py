"""
System 3: IP Radar
===================
Searches for patents that may conflict with the user's device description
and returns plain-language relevance explanations with traffic-light ratings.

Architecture:
  1. Generate diverse search queries from the product profile
  2. Query Google Patents Public Datasets via BigQuery
  3. Deduplicate results
  4. Run each patent through an LLM relevance assessment
  5. Assign traffic-light flags and generate a plain-English IP landscape summary

Data source: Google Patents Public Datasets on BigQuery
  Table: `patents-public-data.google_patents_research.publications`
    Chosen over the raw `patents.publications` table because it exposes plain
    English `title` and `abstract` STRING columns (no localised UNNEST needed)
    and is meaningfully cheaper to scan.
  - Requires a Google Cloud project. Set GOOGLE_CLOUD_PROJECT to your project id.
  - Auth: run `gcloud auth application-default login` OR set
    GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON key path.
  - BigQuery Free Tier gives 1 TiB of query bytes per month. This module runs
    ONE batched query per pipeline invocation (all LLM-generated search
    queries OR'd into a single regex), scanning the abstract column once
    instead of once per query. Measured cost: ~150 GB per pipeline run →
    ~7 runs/month within the free tier. Never introduce SELECT * against
    this table.

THIS IS NOT A FREEDOM-TO-OPERATE (FTO) TOOL.
This tool provides prior art radar to flag patents that should be reviewed
by a registered patent attorney. It does not and cannot provide legal advice.

NEXT STEPS:
  - Add claim-level parsing: the current implementation uses patent abstracts.
    For production, fetch and parse the full claim text (independent claims
    especially) for more accurate relevance scoring.
  - Add continuation/family tracking: flag when a patent has active
    continuations or divisionals, since those may have broader or different claims.
  - Add prosecution history lookup (file wrapper): knowing if a patent
    narrowed its claims during prosecution is critical for FTO analysis.
  - Add a "design around" feature: given a flagged patent, ask the LLM
    to suggest product design modifications that might avoid the patent claims.
  - Add expiration date calculation: most patent APIs return filing/grant dates,
    not expiration. Implement the calculation (20 years from priority date,
    minus any terminal disclaimers, plus any patent term adjustments).
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

from utils.llm_client import call_llm, call_llm_for_json
from utils.models import (
    BigQueryUsage,
    IPRadarResult,
    PatentRelevance,
    PatentResult,
    ProductProfile,
)

logger = logging.getLogger(__name__)

BQ_PATENTS_TABLE = "patents-public-data.google_patents_research.publications"

MAX_PATENTS_TO_ANALYZE = 8    # LLM calls are expensive; cap the deep analysis
MAX_SEARCH_RESULTS = 15       # Raw results to fetch before LLM ranking

# ---------------------------------------------------------------------------
# BigQuery free-tier budget guard
# ---------------------------------------------------------------------------
_FREE_TIER_BYTES = 1024 ** 4   # 1 TiB on-demand query bytes per billing month

# Empirical cost of one batched pipeline run against the scoring SQL below.
# Used to answer "would the next query push me over budget?" without paying
# for a dry-run every time. If the actual scan is higher, we update the
# estimate from the latest total_bytes_billed reading.
_ESTIMATED_QUERY_BYTES = 130 * 10 ** 9   # ~130 GB

# INFORMATION_SCHEMA metadata queries are billed a flat 10 MB each, so we
# cache the reading. 5 minutes is a comfortable middle ground — a stale
# reading can never let us miss the budget cliff by more than ~130 GB.
_USAGE_CACHE_TTL_SEC = 300
_usage_cache: dict[str, float | int] = {}   # {'ts': float, 'used_bytes': int}


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
# Step 2: Patent fetching
# ---------------------------------------------------------------------------

_BQ_SEARCH_SQL = f"""
WITH matched AS (
  SELECT
    publication_number,
    title,
    abstract,
    -- Score = number of DISTINCT keywords found in the abstract. A patent
    -- that hits many of our terms is far more relevant than one that
    -- matches a single generic word like "polymer" or "device".
    ARRAY_LENGTH(ARRAY(
      SELECT DISTINCT x
      FROM UNNEST(REGEXP_EXTRACT_ALL(LOWER(IFNULL(abstract, '')), @pattern)) AS x
    )) AS keyword_hits
  FROM `{BQ_PATENTS_TABLE}`
  WHERE country = 'United States'
    AND REGEXP_CONTAINS(LOWER(IFNULL(abstract, '')), @pattern)
)
SELECT publication_number, title, abstract, keyword_hits
FROM matched
WHERE keyword_hits >= 2                -- suppress single-keyword false positives
ORDER BY keyword_hits DESC,
         publication_number DESC       -- tie-break by recency (pub num starts with year)
LIMIT @lim
"""


_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "are", "was",
    "use", "used", "using", "device", "method", "system", "apparatus",
    "having", "based", "into", "onto", "than", "such", "any", "all",
})


def _keywords_from_query(query: str, max_terms: int = 6) -> list[str]:
    """Extract deduped 3+ letter alphabetic tokens for regex matching.
    Drops English stopwords and generic patent-boilerplate terms so the
    regex stays selective — otherwise ORDER BY would just return newest."""
    seen: set[str] = set()
    words: list[str] = []
    for w in re.findall(r"[A-Za-z]{3,}", query.lower()):
        if w in seen or w in _STOPWORDS:
            continue
        seen.add(w)
        words.append(w)
        if len(words) >= max_terms:
            break
    return words


def _get_bigquery_usage_bytes(client) -> Optional[int]:
    """
    Return bytes billed this month by the current project (cached).
    Returns None when the INFORMATION_SCHEMA query fails (e.g. missing
    bigquery.jobs.listAll permission on a fresh project).
    """
    now = time.time()
    cached_ts = _usage_cache.get("ts", 0)
    if cached_ts and (now - cached_ts) < _USAGE_CACHE_TTL_SEC:
        return int(_usage_cache["used_bytes"])

    sql = """
    SELECT IFNULL(SUM(total_bytes_billed), 0) AS b
    FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
    WHERE creation_time >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP, MONTH)
      AND job_type = 'QUERY'
      AND state = 'DONE'
      AND statement_type != 'SCRIPT'
    """
    try:
        row = next(iter(client.query(sql).result()))
        used = int(row.b or 0)
        _usage_cache["ts"] = now
        _usage_cache["used_bytes"] = used
        return used
    except Exception as e:
        logger.warning("Could not fetch BigQuery usage from INFORMATION_SCHEMA: %s", e)
        return None


def _build_usage_snapshot(used_bytes: Optional[int]) -> Optional[BigQueryUsage]:
    """Build a BigQueryUsage snapshot from a raw byte reading, or None if unknown."""
    if used_bytes is None:
        return None
    remaining = max(0, _FREE_TIER_BYTES - used_bytes)
    return BigQueryUsage(
        used_bytes=used_bytes,
        remaining_bytes=remaining,
        free_tier_bytes=_FREE_TIER_BYTES,
        percent_used=round(used_bytes / _FREE_TIER_BYTES * 100, 2),
        is_exhausted=used_bytes >= _FREE_TIER_BYTES,
        would_incur_charges=(used_bytes + _ESTIMATED_QUERY_BYTES) > _FREE_TIER_BYTES,
    )


def _search_bigquery_batched(
    queries: list[str],
    total_limit: int,
    *,
    allow_paid: bool = False,
) -> tuple[list[dict], Optional[BigQueryUsage], Optional[str]]:
    """
    Run ONE batched BigQuery scan for all queries together.

    BigQuery bills bytes read from columns, not rows returned, so running four
    separate queries would multiply the cost of scanning the abstract column
    four times. This function unions every query's keywords into a single
    regex alternation, scans the abstract column once, and returns up to
    `total_limit` rows for the caller to dedupe and rank.

    Cost controls (must stay in place):
      - Explicit column projection — never SELECT *.
      - country = 'United States' filter.
      - Bounded LIMIT.
      - Parameterised regex, so LLM-generated terms cannot be injected as SQL.

    Requires GOOGLE_CLOUD_PROJECT env var and application-default credentials.

    Returns (rows, usage_snapshot, skip_reason):
      - rows: patent dicts (empty if skipped or errored)
      - usage_snapshot: current BigQuery usage state, or None if we couldn't read it
      - skip_reason: human-readable string if the search was skipped, else None
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    if not project_id:
        logger.warning("GOOGLE_CLOUD_PROJECT not set — skipping Google Patents search")
        return [], None, "GOOGLE_CLOUD_PROJECT is not configured"

    seen: set[str] = set()
    all_keywords: list[str] = []
    for q in queries:
        for kw in _keywords_from_query(q):
            if kw in seen:
                continue
            seen.add(kw)
            all_keywords.append(kw)
    if not all_keywords:
        return [], None, None
    pattern = r"\b(" + "|".join(re.escape(w) for w in all_keywords) + r")\b"

    try:
        from google.cloud import bigquery  # lazy import: keeps startup cheap
    except ImportError:
        logger.warning("google-cloud-bigquery not installed — skipping patent search")
        return [], None, "google-cloud-bigquery client library is not installed"

    try:
        client = bigquery.Client(project=project_id)
    except Exception as e:
        logger.warning("Could not create BigQuery client: %s", e)
        return [], None, f"Could not create BigQuery client: {e}"

    used_bytes = _get_bigquery_usage_bytes(client)
    usage = _build_usage_snapshot(used_bytes)

    # Budget gate — skip cleanly rather than surprise the account owner
    if usage and usage.would_incur_charges and not allow_paid:
        gb_used = usage.used_bytes / 1e9
        pct = usage.percent_used
        reason = (
            f"BigQuery patent search skipped: this project has already used "
            f"{gb_used:.0f} GB of the 1 TiB monthly free tier ({pct:.0f}%), "
            f"and the next scan (~{_ESTIMATED_QUERY_BYTES / 1e9:.0f} GB) would exceed it. "
            f"Re-submit with allow_paid_bigquery=true to run this search anyway "
            f"(it will bill your Google Cloud account at $6.25/TiB scanned)."
        )
        logger.warning(reason)
        return [], usage, reason

    try:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("pattern", "STRING", pattern),
                bigquery.ScalarQueryParameter("lim", "INT64", total_limit),
            ]
        )
        job = client.query(_BQ_SEARCH_SQL, job_config=job_config)
        rows = list(job.result())
        billed = getattr(job, "total_bytes_billed", 0) or 0
        logger.info(
            "BigQuery patent search (%d keywords across %d queries) → %d rows, %.1f GB billed",
            len(all_keywords), len(queries), len(rows), billed / 1e9,
        )
        # Refresh our usage snapshot to reflect this run, so downstream consumers
        # see accurate remaining budget in the response.
        if used_bytes is not None:
            new_used = used_bytes + billed
            _usage_cache["ts"] = time.time()
            _usage_cache["used_bytes"] = new_used
            usage = _build_usage_snapshot(new_used)
        return [dict(row) for row in rows], usage, None
    except Exception as e:
        logger.warning("BigQuery patent search failed: %s", e)
        return [], usage, f"BigQuery query failed: {e}"


# Pre-grant application numbers embed the filing year: US-YYYY######-A#.
# Granted patent numbers (US-#######-B#) do NOT — the digits are just the
# patent number. We can only extract the year for the application format.
_APPLICATION_PUB_NUM_RE = re.compile(r"^[A-Z]{2}-(\d{4})\d{6,}-A")


def _normalize_bq_row(row: dict) -> dict:
    """Normalise a google_patents_research row into our common patent dict format.

    The research table does not carry filing_date or assignee. For pre-grant
    US applications the filing year is embedded in the publication_number and
    we extract it; for granted patents the number carries no year so we leave
    filing_date/expiration empty, and _is_patent_active will treat the patent
    as active (the LLM does the more nuanced assessment downstream).
    """
    pub_num = row.get("publication_number") or ""

    filing_date = ""
    expiration = ""
    m = _APPLICATION_PUB_NUM_RE.match(pub_num)
    if m:
        try:
            year = int(m.group(1))
            filing_date = f"{year}-01-01"
            expiration = f"{year + 20}-01-01"
        except ValueError:
            pass

    return {
        "patent_number": pub_num,
        "title": row.get("title") or "",
        "abstract": row.get("abstract") or "",
        "assignee": "Unknown",  # not exposed by google_patents_research schema
        "filing_date": filing_date,
        "grant_date": "",
        "expiration_date": expiration,
        "source": "google_patents_research",
    }


def fetch_patents_for_queries(
    queries: list[str],
    *,
    allow_paid: bool = False,
) -> tuple[list[dict], Optional[BigQueryUsage], Optional[str]]:
    """
    Run ONE batched BigQuery scan for all queries combined and return a
    deduplicated list of raw patent dicts, alongside the current BigQuery
    usage snapshot and any skip reason. Overfetches by ~2× to give the
    dedup + relevance-ranking layers headroom.
    """
    if not queries:
        return [], None, None

    raw_results, usage, skip_reason = _search_bigquery_batched(
        queries, total_limit=MAX_SEARCH_RESULTS * 2, allow_paid=allow_paid,
    )

    all_patents: list[dict] = []
    seen_numbers: set[str] = set()
    for raw in raw_results:
        normalized = _normalize_bq_row(raw)
        num = normalized["patent_number"]
        if num and num not in seen_numbers:
            seen_numbers.add(num)
            all_patents.append(normalized)

    logger.info("Fetched %d unique patents across %d queries", len(all_patents), len(queries))
    return all_patents[:MAX_SEARCH_RESULTS], usage, skip_reason


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
    # Trust the LLM's assessment first
    if not relevance_data.get("is_likely_active", True):
        return False

    # Check expiration estimate
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

def run_ip_radar(
    profile: ProductProfile,
    *,
    allow_paid_bigquery: bool = False,
) -> IPRadarResult:
    """
    Main entry point for System 3.
    Takes a ProductProfile and returns a full IPRadarResult.

    allow_paid_bigquery: when False (default) the patent search is skipped
      cleanly if it would push BigQuery usage past the 1 TiB monthly free
      tier. The response then carries a skip reason so the frontend can offer
      the user the choice to re-run with paid usage enabled.
    """
    logger.info("Starting IP radar for: %s", profile.intended_use[:60])

    # Step 1: Generate search queries
    queries = generate_search_queries(profile)
    logger.info("Generated %d search queries: %s", len(queries), queries)

    # Step 2: Fetch patents
    raw_patents, usage, skip_reason = fetch_patents_for_queries(
        queries, allow_paid=allow_paid_bigquery,
    )

    if skip_reason:
        # Budget-gated skip — surface the choice to the user with usage stats.
        return IPRadarResult(
            product_profile=profile,
            patents=[],
            search_queries_used=queries,
            summary=(
                skip_reason + " Meanwhile, manual search on Google Patents "
                "(patents.google.com) is a free interim workaround."
            ),
            bigquery_usage=usage,
            bigquery_skipped_reason=skip_reason,
        )

    if not raw_patents:
        has_project = bool(os.getenv("GOOGLE_CLOUD_PROJECT", ""))
        logger.warning("No patents returned from Google Patents BigQuery")
        return IPRadarResult(
            product_profile=profile,
            patents=[],
            search_queries_used=queries,
            summary=(
                "Patent search returned no results. "
                + ("" if has_project else
                   "Note: GOOGLE_CLOUD_PROJECT is not set — configure a Google Cloud "
                   "project and run `gcloud auth application-default login` to enable "
                   "patent database search. ")
                + "Manual search on Google Patents (patents.google.com) and "
                "USPTO Full-Text Database (ppubs.uspto.gov) is recommended for a thorough IP review."
            ),
            bigquery_usage=usage,
        )

    # Step 3 & 4: Assess relevance for top patents
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

    # Step 5: Generate summary
    summary = generate_ip_summary(profile, analyzed_patents)

    return IPRadarResult(
        product_profile=profile,
        patents=analyzed_patents,
        search_queries_used=queries,
        summary=summary,
        bigquery_usage=usage,
    )
