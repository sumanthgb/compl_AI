"""
System 4: Smart Materials Decision Engine
==========================================
The "save months and tens of thousands" system.

Given a baseline roadmap (from System 2), analyzes the materials in the
device and suggests swaps that would:
  1. Reduce testing burden (waive tests with existing biocompatibility data)
  2. Shorten the critical path
  3. Reduce total cost
  4. Not invalidate the 510(k) predicate match (if applicable)

Architecture:
  1. Load the materials knowledge base (defined below — move to DB in prod)
  2. For each material in the device, identify candidate substitutes
  3. For each candidate, regenerate a hypothetical roadmap (rerun System 2)
  4. Diff the baseline roadmap against the hypothetical roadmap
  5. Score recommendations and surface the best one

NEXT STEPS:
  - Build a real materials database (PostgreSQL + full-text search) with:
      * Manufacturer-specific grade data (e.g. Victrex PEEK vs. Solvay PEEK)
      * Published ISO 10993 study citations for each material
      * Regulatory precedent: list of cleared 510(k)s that used this material
      * Sterilization compatibility (gamma, EO, autoclave, e-beam)
      * MRI safety classification (MR Safe / MR Conditional / MR Unsafe)
  - Add a "predicate invalidation check": before recommending a material swap,
    check whether the predicate device for the baseline 510(k) also used that
    material. If not, flag that the swap may weaken the predicate argument.
  - Add combination optimization: instead of single material swaps, explore
    multi-material changes simultaneously using a search algorithm.
  - Integrate with supplier databases (Granta MI, MatWeb) to surface material
    availability, lead times, and cost per unit alongside the testing savings.
  - Add a "material switch regulatory memo" generator: one-click generation of
    the regulatory justification document for a proposed material change.
  - Surface MRI compatibility implications automatically — a metal-to-PEEK swap
    may open a much larger addressable market (MRI-compatible implants).
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

from utils.llm_client import call_llm
from utils.models import (
    ClassificationResult,
    ContactCategory,
    MaterialProfile,
    MaterialSwapRecommendation,
    MaterialsOptimizationResult,
    RoadmapResult,
    TestNode,
)
from systems.roadmap_generator import generate_roadmap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Materials Knowledge Base
# ---------------------------------------------------------------------------
# For each material: which ISO 10993 endpoints have well-established published
# data, which contact categories it's generally safe for, and MRI compatibility.
#
# NEXT STEPS: This should live in a database. Each material record should
# link to specific published studies and cleared 510(k) submissions as evidence.

MATERIALS_KB: dict[str, MaterialProfile] = {

    "peek": MaterialProfile(
        name="PEEK (Polyether Ether Ketone)",
        common_grades=["Victrex PEEK 450G", "Solvay KetaSpire KT-880", "Invibio PEEK-OPTIMA"],
        biocompatibility_endpoints_established=[
            "ISO 10993-5 (cytotoxicity)",
            "ISO 10993-10 (sensitization)",
            "ISO 10993-10 (irritation)",
            "ISO 10993-11 (systemic toxicity)",
            "ISO 10993-3 (genotoxicity)",
            "ISO 10993-6 (implantation)",
        ],
        generally_recognized_safe_for=[
            ContactCategory.SURFACE,
            ContactCategory.EXTERNAL_COMMUNICATING,
            ContactCategory.IMPLANT,
        ],
        mri_compatible=True,
        notes=(
            "Extensive published biocompatibility data. Victrex PEEK 450G and "
            "Invibio PEEK-OPTIMA are the gold standard for spinal implants. "
            "ISO 10993 data packages available from manufacturers. "
            "MRI Safe — significant market advantage over metallic alternatives."
        ),
    ),

    "titanium": MaterialProfile(
        name="Titanium (Ti-6Al-4V ELI / CP Grade 4)",
        common_grades=["Ti-6Al-4V ELI (ASTM F136)", "CP Ti Grade 4 (ASTM F67)"],
        biocompatibility_endpoints_established=[
            "ISO 10993-5 (cytotoxicity)",
            "ISO 10993-10 (sensitization)",
            "ISO 10993-6 (implantation)",
            "ISO 10993-11 (systemic toxicity)",
        ],
        generally_recognized_safe_for=[
            ContactCategory.SURFACE,
            ContactCategory.EXTERNAL_COMMUNICATING,
            ContactCategory.IMPLANT,
        ],
        mri_compatible=None,  # MR Conditional — depends on specific device geometry
        notes=(
            "Industry standard for metallic implants. Ti-6Al-4V ELI per ASTM F136 "
            "has the strongest regulatory precedent. MR Conditional at specific field "
            "strengths — requires labeling. Excellent osseointegration for bone implants."
        ),
    ),

    "316l stainless steel": MaterialProfile(
        name="316L Stainless Steel",
        common_grades=["316L per ASTM F138"],
        biocompatibility_endpoints_established=[
            "ISO 10993-5 (cytotoxicity)",
            "ISO 10993-10 (sensitization)",
            "ISO 10993-6 (implantation)",
        ],
        generally_recognized_safe_for=[
            ContactCategory.SURFACE,
            ContactCategory.EXTERNAL_COMMUNICATING,
            ContactCategory.IMPLANT,
        ],
        mri_compatible=False,  # MR Unsafe for implantable uses
        notes=(
            "Well-established for short-term implant applications and instruments. "
            "Nickel content may cause sensitization in nickel-allergic patients — "
            "flag this in biocompatibility risk assessment. "
            "MR Unsafe — limits market to non-MRI procedures."
        ),
    ),

    "medical grade silicone": MaterialProfile(
        name="Medical Grade Silicone",
        common_grades=["NuSil MED-6382", "Dow SILASTIC MDX4-4210", "Nusil MED-1000"],
        biocompatibility_endpoints_established=[
            "ISO 10993-5 (cytotoxicity)",
            "ISO 10993-10 (sensitization)",
            "ISO 10993-10 (irritation)",
            "ISO 10993-11 (systemic toxicity)",
            "ISO 10993-6 (implantation)",
            "USP Class VI",
        ],
        generally_recognized_safe_for=[
            ContactCategory.SURFACE,
            ContactCategory.EXTERNAL_COMMUNICATING,
            ContactCategory.IMPLANT,
        ],
        mri_compatible=True,
        notes=(
            "Extensive biocompatibility history. USP Class VI data from manufacturers "
            "covers most endpoints. NuSil and Dow provide full ISO 10993 data packages. "
            "Gamma and EO sterilization compatible. MRI Safe."
        ),
    ),

    "uhmwpe": MaterialProfile(
        name="UHMWPE (Ultra-High-Molecular-Weight Polyethylene)",
        common_grades=["GUR 1020", "GUR 1050", "Crosslinked UHMWPE per ASTM F648"],
        biocompatibility_endpoints_established=[
            "ISO 10993-5 (cytotoxicity)",
            "ISO 10993-10 (sensitization)",
            "ISO 10993-6 (implantation)",
            "ISO 10993-4 (hemocompatibility)",
        ],
        generally_recognized_safe_for=[
            ContactCategory.IMPLANT,
        ],
        mri_compatible=True,
        notes=(
            "Gold standard for orthopedic bearing surfaces (hip, knee). "
            "Highly crosslinked UHMWPE (XLPE) has additional data on wear particle biocompatibility. "
            "Must specify the radiation crosslinking level and antioxidant treatment in the submission."
        ),
    ),

    "ptfe": MaterialProfile(
        name="PTFE (Polytetrafluoroethylene / Teflon)",
        common_grades=["ePTFE", "Dense PTFE", "Gore-Tex Medical"],
        biocompatibility_endpoints_established=[
            "ISO 10993-5 (cytotoxicity)",
            "ISO 10993-10 (sensitization)",
            "ISO 10993-6 (implantation)",
            "USP Class VI",
        ],
        generally_recognized_safe_for=[
            ContactCategory.SURFACE,
            ContactCategory.EXTERNAL_COMMUNICATING,
            ContactCategory.IMPLANT,
        ],
        mri_compatible=True,
        notes=(
            "Long regulatory history in vascular grafts and soft tissue augmentation. "
            "ePTFE (expanded PTFE) has extensive cleared device precedent. "
            "MRI Safe. EO sterilization preferred — gamma can affect PTFE properties."
        ),
    ),

    "cobalt chrome": MaterialProfile(
        name="Cobalt-Chromium Alloy",
        common_grades=["CoCrMo ASTM F75", "CoCrMo ASTM F799", "CoCrW ASTM F90"],
        biocompatibility_endpoints_established=[
            "ISO 10993-5 (cytotoxicity)",
            "ISO 10993-10 (sensitization)",
            "ISO 10993-6 (implantation)",
        ],
        generally_recognized_safe_for=[
            ContactCategory.IMPLANT,
        ],
        mri_compatible=None,  # MR Conditional
        notes=(
            "Standard for orthopedic joint replacements and cardiovascular implants. "
            "Metal ion release (Co, Cr, Mo) requires specific chemical characterization "
            "and toxicological risk assessment per ISO 10993-18. "
            "Some patient populations have cobalt sensitivity — document in risk management."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Substitution map
# ---------------------------------------------------------------------------
# Defines which materials are reasonable clinical substitutes for others.
# Keys are normalised lowercase material names.
# Values are lists of candidate substitute material keys.

SUBSTITUTION_MAP: dict[str, list[str]] = {
    "titanium": ["peek", "cobalt chrome"],
    "ti-6al-4v": ["peek", "cobalt chrome"],
    "cobalt chrome": ["peek", "titanium"],
    "316l stainless steel": ["titanium", "peek"],
    "stainless steel": ["titanium", "peek"],
    "peek": ["titanium", "cobalt chrome"],  # Swap direction
    "silicone": ["ptfe", "medical grade silicone"],
    "polyurethane": ["medical grade silicone", "ptfe"],
    "generic polymer": ["peek", "medical grade silicone", "ptfe"],
}


# ---------------------------------------------------------------------------
# Roadmap diffing
# ---------------------------------------------------------------------------

def _diff_roadmaps(
    baseline: RoadmapResult,
    hypothetical: RoadmapResult,
) -> tuple[list[TestNode], list[TestNode], int, int, int, int]:
    """
    Compare two roadmaps and return the delta.
    Returns:
      - tests_eliminated: tests in baseline but not in hypothetical
      - tests_added: tests in hypothetical but not in baseline
      - net_weeks_low, net_weeks_high: positive = saved time
      - net_cost_low, net_cost_high: positive = saved money
    """
    baseline_ids = {t.id for t in baseline.tests}
    hyp_ids = {t.id for t in hypothetical.tests}
    baseline_map = {t.id: t for t in baseline.tests}
    hyp_map = {t.id: t for t in hypothetical.tests}

    eliminated_ids = baseline_ids - hyp_ids
    added_ids = hyp_ids - baseline_ids

    tests_eliminated = [baseline_map[tid] for tid in eliminated_ids]
    tests_added = [hyp_map[tid] for tid in added_ids]

    net_weeks_low = baseline.total_weeks_low - hypothetical.total_weeks_low
    net_weeks_high = baseline.total_weeks_high - hypothetical.total_weeks_high
    net_cost_low = baseline.total_cost_usd_low - hypothetical.total_cost_usd_low
    net_cost_high = baseline.total_cost_usd_high - hypothetical.total_cost_usd_high

    return tests_eliminated, tests_added, net_weeks_low, net_weeks_high, net_cost_low, net_cost_high


# ---------------------------------------------------------------------------
# Predicate impact check
# ---------------------------------------------------------------------------

PREDICATE_IMPACT_SYSTEM_PROMPT = """
You are a 510(k) regulatory specialist.

Given:
1. The original device description and material
2. A proposed material substitution
3. The current predicate device(s) being used for substantial equivalence

Assess whether this material change could affect the 510(k) predicate argument.
A predicate device used materials X — if we switch to Y, does that create
a material difference that FDA reviewers might flag?

Return JSON:
{
  "predicate_impact": "none" | "minor" | "significant",
  "explanation": "1-2 sentence explanation. null if no impact."
}
"""


def check_predicate_impact(
    classification: ClassificationResult,
    original_material: str,
    suggested_material: str,
) -> Optional[str]:
    """
    Check whether a material swap could affect the 510(k) predicate argument.
    Returns a warning string or None.
    """
    if not classification.predicate_devices:
        return None  # No predicates identified — can't assess impact

    predicates_text = ", ".join(
        f"{p.device_name} ({p.k_number})"
        for p in classification.predicate_devices[:3]
    )

    message = (
        f"Device: {classification.product_profile.intended_use}\n"
        f"Original material: {original_material}\n"
        f"Proposed substitute: {suggested_material}\n"
        f"Predicate devices: {predicates_text}\n"
    )

    try:
        data = call_llm_for_json(
            system_prompt=PREDICATE_IMPACT_SYSTEM_PROMPT,
            user_message=message,
        )
        impact = data.get("predicate_impact", "none")
        explanation = data.get("explanation")

        if impact in ("minor", "significant") and explanation:
            return f"[{impact.upper()} PREDICATE IMPACT] {explanation}"
    except Exception as e:
        logger.warning("Predicate impact check failed: %s", e)

    return None


# ---------------------------------------------------------------------------
# Recommendation rationale generation
# ---------------------------------------------------------------------------

RATIONALE_SYSTEM_PROMPT = """
You are a regulatory strategy advisor helping an early-stage medtech team.

Given details about a proposed material substitution and its impact on
their testing roadmap, write a concise 2-3 sentence rationale explaining:
1. Why this material swap makes sense scientifically and regulatorily
2. What the concrete benefit is (time + cost savings)
3. Any important caveats

Be direct, practical, and specific. Don't be overly cautious.
"""


def generate_recommendation_rationale(
    original: str,
    suggested: str,
    eliminated: list[TestNode],
    cost_saved_low: int,
    cost_saved_high: int,
    weeks_saved_low: int,
    weeks_saved_high: int,
) -> str:
    message = (
        f"Original material: {original}\n"
        f"Suggested substitute: {suggested}\n"
        f"Tests eliminated: {', '.join(t.name for t in eliminated) or 'none'}\n"
        f"Estimated savings: ${cost_saved_low:,}–${cost_saved_high:,}, "
        f"{weeks_saved_low}–{weeks_saved_high} weeks\n"
    )

    try:
        return call_llm(system_prompt=RATIONALE_SYSTEM_PROMPT, user_message=message)
    except Exception as e:
        logger.warning("Rationale generation failed: %s", e)
        return (
            f"Switching from {original} to {suggested} leverages existing biocompatibility data "
            f"to potentially eliminate {len(eliminated)} tests, saving an estimated "
            f"${cost_saved_low:,}–${cost_saved_high:,} and {weeks_saved_low}–{weeks_saved_high} weeks."
        )


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

OPTIMIZATION_SUMMARY_PROMPT = """
You are a regulatory strategy advisor.

Given a list of material swap recommendations for a medical device, write
a 2-3 sentence summary that:
1. Identifies the single best recommendation and why
2. States the headline time and cost savings
3. Notes any important caveats or action items

Be direct and specific.
"""


def generate_optimization_summary(
    recommendations: list[MaterialSwapRecommendation],
) -> str:
    if not recommendations:
        return (
            "No material substitutions were identified that would meaningfully "
            "reduce the testing burden for this device. The current material selection "
            "appears well-suited to the regulatory pathway."
        )

    best = recommendations[0]
    recs_text = "\n".join(
        f"- {r.original_material} → {r.suggested_material}: "
        f"save ${r.net_cost_saved_usd_low:,}–${r.net_cost_saved_usd_high:,}, "
        f"{r.net_weeks_saved_low}–{r.net_weeks_saved_high} weeks"
        for r in recommendations
    )

    try:
        return call_llm(
            system_prompt=OPTIMIZATION_SUMMARY_PROMPT,
            user_message=f"Recommendations:\n{recs_text}",
        )
    except Exception as e:
        logger.warning("Optimization summary failed: %s", e)
        return (
            f"Best recommendation: switch {best.original_material} to {best.suggested_material}, "
            f"saving ${best.net_cost_saved_usd_low:,}–${best.net_cost_saved_usd_high:,} "
            f"and {best.net_weeks_saved_low}–{best.net_weeks_saved_high} weeks."
        )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def optimize_materials(baseline_roadmap: RoadmapResult) -> MaterialsOptimizationResult:
    """
    Main entry point for System 4.
    Takes the baseline roadmap and returns a MaterialsOptimizationResult with
    ranked swap recommendations.
    """
    classification = baseline_roadmap.classification
    profile = classification.product_profile
    logger.info("Running materials optimization for %d materials", len(profile.materials))

    recommendations: list[MaterialSwapRecommendation] = []

    for material in profile.materials:
        material_key = material.lower().strip()
        candidates = SUBSTITUTION_MAP.get(material_key, [])

        if not candidates:
            # Fuzzy match against KB keys
            for kb_key in SUBSTITUTION_MAP:
                if kb_key in material_key or material_key in kb_key:
                    candidates = SUBSTITUTION_MAP[kb_key]
                    break

        for candidate_key in candidates:
            if candidate_key not in MATERIALS_KB:
                continue

            candidate_material = MATERIALS_KB[candidate_key]
            logger.info("Evaluating swap: %s → %s", material, candidate_material.name)

            # Create a hypothetical product profile with the material swapped
            hypothetical_profile_materials = [
                candidate_material.name if m.lower().strip() == material_key else m
                for m in profile.materials
            ]

            hypothetical_classification = copy.deepcopy(classification)
            hypothetical_classification.product_profile.materials = hypothetical_profile_materials

            # Regenerate roadmap for the hypothetical configuration
            try:
                hypothetical_roadmap = generate_roadmap(hypothetical_classification)
            except Exception as e:
                logger.warning("Roadmap generation failed for hypothetical: %s", e)
                continue

            # Diff the roadmaps
            (
                tests_eliminated,
                tests_added,
                net_weeks_low,
                net_weeks_high,
                net_cost_low,
                net_cost_high,
            ) = _diff_roadmaps(baseline_roadmap, hypothetical_roadmap)

            # Only recommend if there's a net positive benefit
            if net_cost_low <= 0 and net_weeks_low <= 0:
                continue

            # Check predicate impact
            predicate_impact = check_predicate_impact(
                classification, material, candidate_material.name
            )

            # Generate rationale
            rationale = generate_recommendation_rationale(
                material, candidate_material.name,
                tests_eliminated, net_cost_low, net_cost_high,
                net_weeks_low, net_weeks_high,
            )

            rec = MaterialSwapRecommendation(
                original_material=material,
                suggested_material=candidate_material.name,
                tests_eliminated=tests_eliminated,
                tests_added=tests_added,
                net_weeks_saved_low=net_weeks_low,
                net_weeks_saved_high=net_weeks_high,
                net_cost_saved_usd_low=net_cost_low,
                net_cost_saved_usd_high=net_cost_high,
                predicate_impact=predicate_impact,
                rationale=rationale,
            )
            recommendations.append(rec)

    # Sort by total savings (cost high is the primary sort key)
    recommendations.sort(key=lambda r: r.net_cost_saved_usd_high, reverse=True)

    best_recommendation = recommendations[0] if recommendations else None
    summary = generate_optimization_summary(recommendations)

    return MaterialsOptimizationResult(
        baseline_roadmap=baseline_roadmap,
        recommendations=recommendations,
        best_recommendation=best_recommendation,
        summary=summary,
    )
