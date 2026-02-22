"""
System 1: Classification Engine
================================
Takes a plain-language product description and returns a structured
ClassificationResult including:
  - Device class (I / II / III)
  - Regulatory pathway (510k / De Novo / PMA / etc.)
  - Confidence score
  - Predicate device suggestions
  - Software safety class if applicable

Architecture:
  Step 1 — Extraction: LLM extracts a structured ProductProfile from raw text
  Step 2 — Semantic search: embed the profile and search against FDA product
            code database to find the best matching product code
  Step 3 — Classification logic: apply FDA decision tree using the matched
            product code + profile fields
  Step 4 — Predicate search: find 510(k) cleared predicates if pathway is 510(k)

NEXT STEPS:
  - Replace the stub FDA product code embeddings with a real pre-built vector
    index. FDA publishes the full product classification database at:
    https://api.fda.gov/device/classification.json
    Download all records (~6,500), embed the device_name + medical_specialty_description
    field, and store in a real vector DB (Pinecone, Qdrant, or even a local FAISS index).
  - The predicate search currently queries the FDA 510(k) API. Add pagination
    and date-range filtering so you can bias toward recent predicates (< 5 years).
  - Add a combination product routing layer: if has_drug_component or
    has_biologic_component, call FDA's RCM (Request for Designation) guidance
    logic to determine the lead center (CDER/CBER/CDRH).
  - Store every classification result in a database with the raw description so
    you can build a labeled dataset for fine-tuning.
  - Add IVD (in vitro diagnostic) detection — IVDs follow a separate
    classification framework under 21 CFR 862-866.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import httpx
import numpy as np

from utils.llm_client import call_llm_for_json
from utils.models import (
    ClassificationResult,
    ContactCategory,
    ContactDuration,
    DeviceClass,
    FDALeadCenter,
    MechanismOfAction,
    PredicateDevice,
    ProductCategory,
    ProductProfile,
    RegulatoryPathway,
    SoftwareSafetyClass,
)

logger = logging.getLogger(__name__)

FDA_CLASSIFICATION_API = "https://api.fda.gov/device/classification.json"
FDA_510K_API = "https://api.fda.gov/device/510k.json"
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", "")

# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------
LOW_CONFIDENCE_THRESHOLD = 0.72   # Below this, surface a warning to the user


# ---------------------------------------------------------------------------
# Step 1: Extraction
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """
You are an expert regulatory affairs specialist with deep knowledge of FDA
medical device, biologics, and drug classification.

Your job is to extract a structured profile from a plain-language description.
You must first decide the PRIMARY CATEGORY before anything else — this is the
most important routing decision.

PRIMARY CATEGORY RULES (apply in this order):
1. CELL_GENE_THERAPY if ANY of these are present:
   - Living cells of any kind (autologous, allogeneic, xenogeneic, stem cells, CAR-T, etc.)
   - Organoids
   - Gene editing (CRISPR, viral vectors, siRNA, mRNA, base editing, etc.)
   - Tissue engineering with cellular components
   - Biological grafts derived from living tissue with retained cellular activity
   - Anything that REPAIRS biological function using a biological product
   → Routes to CBER. Requires IND → BLA pathway.

2. DIAGNOSTIC_IVD if:
   - Primary purpose is detecting, measuring, or identifying something biological
   - The test/assay happens OUTSIDE the body (on a sample, in a lab, on a strip)
   - Examples: lateral flow assay, ELISA, PCR test, colorimetric biosensor, sequencing panel
   → Routes to CDRH (or CBER for blood/tissue typing). No biocompatibility required.

3. DIAGNOSTIC_INVIVO if:
   - Primary purpose is detecting, measuring, or identifying something biological
   - The device/probe CONTACTS or ENTERS the body to generate the reading
   - Examples: CGM sensor worn on skin, intravascular IVUS catheter, implanted biosensor
   → Routes to CDRH + biocompatibility testing triggered by contact type.

4. MEDICAL_DEVICE if:
   - Primary purpose is treatment, repair, replacement, or support of biological function
   - Achieved mechanically, electrically, or physically (NOT biologically or chemically)
   - Examples: surgical implant, wearable sensor for therapy, surgical instrument, stent
   → Routes to CDRH. 510(k)/PMA/De Novo depending on risk.

5. DRUG if:
   - Primary mode of action is chemical/pharmacological
   - No significant device or biologic component
   → Routes to CDER.

6. COMBINATION if:
   - Elements from more than one category exist simultaneously
   - Identify which component drives the PRIMARY therapeutic/diagnostic effect
   - The primary mode of action component determines lead center
   - ALL components must be individually cleared (most expensive pathway)

Return ONLY a JSON object with these exact keys:
{
  "product_category": one of ["medical_device","cell_gene_therapy","diagnostic_ivd","diagnostic_invivo","drug","combination","unknown"],
  "is_therapeutic": true/false,
  "is_diagnostic": true/false,
  "diagnostic_location": "in_vitro" | "in_vivo" | null,
  "contains_living_cells": true/false,
  "contains_gene_editing": true/false,
  "contains_tissue_engineering": true/false,
  "is_biological_graft": true/false,
  "primary_mode_of_action": "plain-English description of what drives the primary effect",
  "mechanism_of_action": one of ["mechanical","chemical","biological","electrical","software","combination","unknown"],
  "intended_use": "short description of what the product does",
  "indication": "the clinical condition or patient population it addresses",
  "contact_category": one of ["none","surface","external_communicating","implant"],
  "contact_duration": one of ["limited","prolonged","permanent"],
  "materials": ["list", "of", "materials", "mentioned"],
  "has_drug_component": true/false,
  "has_biologic_component": true/false,
  "has_software_component": true/false,
  "is_implantable": true/false,
  "is_combination_product": true/false,
  "extraction_notes": "any ambiguities or assumptions you made"
}

DEFINITIONS:
- contact_category "surface": intact skin, mucous membrane, or compromised surface
- contact_category "external_communicating": blood path indirect, tissue/bone/dentin, circulating blood
- contact_category "implant": tissue/bone implant, blood implant (fully inside the body)
- contact_duration "limited": < 24 hours; "prolonged": 24h–30 days; "permanent": > 30 days
"""


def extract_product_profile(raw_description: str) -> ProductProfile:
    """
    LLM-powered extraction of structured fields from plain-language description.
    Now captures product category, diagnostic location, and cell/gene therapy signals.
    """
    data = call_llm_for_json(
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_message=f"Extract the product profile from this description:\n\n{raw_description}",
    )

    return ProductProfile(
        raw_description=raw_description,
        # New routing fields
        product_category=data.get("product_category", "unknown"),
        is_therapeutic=data.get("is_therapeutic", False),
        is_diagnostic=data.get("is_diagnostic", False),
        diagnostic_location=data.get("diagnostic_location"),
        contains_living_cells=data.get("contains_living_cells", False),
        contains_gene_editing=data.get("contains_gene_editing", False),
        contains_tissue_engineering=data.get("contains_tissue_engineering", False),
        is_biological_graft=data.get("is_biological_graft", False),
        primary_mode_of_action=data.get("primary_mode_of_action", ""),
        # Existing fields
        mechanism_of_action=data.get("mechanism_of_action", "unknown"),
        intended_use=data.get("intended_use", ""),
        indication=data.get("indication", ""),
        contact_category=data.get("contact_category", "none"),
        contact_duration=data.get("contact_duration", "limited"),
        materials=data.get("materials", []),
        has_drug_component=data.get("has_drug_component", False),
        has_biologic_component=data.get("has_biologic_component", False),
        has_software_component=data.get("has_software_component", False),
        is_implantable=data.get("is_implantable", False),
        is_combination_product=data.get("is_combination_product", False),
        extraction_notes=data.get("extraction_notes", ""),
    )


# ---------------------------------------------------------------------------
# Step 2a: Primary category routing
# ---------------------------------------------------------------------------
# This is the FIRST decision gate — it runs before any FDA product code search.
# It answers: "What KIND of product is this?" before asking "What class is it?"
#
# Routing priority (matches the extraction prompt rules):
#   1. Cell/gene therapy signals → CBER
#   2. Diagnostic (IVD vs. in-vivo) → CDRH ± biocompatibility
#   3. Medical device → CDRH
#   4. Drug → CDER
#   5. Combination → resolve by primary MOA

def route_primary_category(profile: ProductProfile) -> tuple[ProductCategory, FDALeadCenter, RegulatoryPathway, str]:
    """
    Apply the primary category decision tree.

    Returns (product_category, lead_center, pathway, rationale_fragment).
    Pathway may be UNKNOWN here if the category still needs device-class lookup.
    UNKNOWN pathway means "continue to product code search".
    """

    # ---- 1. Cell / Gene Therapy → CBER / IND -------------------------
    cgt_signals = [
        profile.contains_living_cells,
        profile.contains_gene_editing,
        profile.contains_tissue_engineering,
        profile.is_biological_graft,
        profile.product_category == ProductCategory.CELL_GENE_THERAPY,
    ]
    if any(cgt_signals):
        signals_fired = []
        if profile.contains_living_cells:       signals_fired.append("living cells")
        if profile.contains_gene_editing:       signals_fired.append("gene editing / genetic modification")
        if profile.contains_tissue_engineering: signals_fired.append("tissue engineering with cellular components")
        if profile.is_biological_graft:         signals_fired.append("biological graft / tissue-derived product")

        rationale = (
            f"Product is classified as a CELL / GENE THERAPY based on: {', '.join(signals_fired)}. "
            "Primary jurisdiction: CBER (Center for Biologics Evaluation and Research). "
            "Entry pathway: IND (Investigational New Drug Application) required before any clinical use. "
            "Approval pathway: BLA (Biologics License Application). "
            "This is one of the most rigorous pathways — expect extensive CMC (Chemistry, Manufacturing & Controls) "
            "requirements, potency assays, sterility/identity testing specific to biological products, "
            "and a detailed Phase I safety-first clinical development plan."
        )
        return ProductCategory.CELL_GENE_THERAPY, FDALeadCenter.CBER, RegulatoryPathway.IND, rationale

    # ---- 2. Diagnostic ------------------------------------------------
    if profile.is_diagnostic or profile.product_category in (
        ProductCategory.DIAGNOSTIC_IVD, ProductCategory.DIAGNOSTIC_INVIVO
    ):
        loc = profile.diagnostic_location or (
            "in_vitro" if profile.product_category == ProductCategory.DIAGNOSTIC_IVD
            else "in_vivo" if profile.product_category == ProductCategory.DIAGNOSTIC_INVIVO
            else None
        )

        if loc == "in_vitro" or profile.product_category == ProductCategory.DIAGNOSTIC_IVD:
            rationale = (
                "Product is an IN VITRO DIAGNOSTIC (IVD): the test/assay happens on a biological sample "
                "OUTSIDE the body (e.g., blood draw, urine sample, swab). "
                "Primary jurisdiction: CDRH under 21 CFR Parts 862-866. "
                "No biocompatibility testing required (no patient contact). "
                "Pathway depends on risk class — most IVDs are Class II (510(k)) or Class I (exempt). "
                "High-risk IVDs (e.g., HIV confirmatory tests) may be Class III (PMA)."
            )
            return ProductCategory.DIAGNOSTIC_IVD, FDALeadCenter.CDRH, RegulatoryPathway.UNKNOWN, rationale

        elif loc == "in_vivo" or profile.product_category == ProductCategory.DIAGNOSTIC_INVIVO:
            biocompat_note = _get_biocompatibility_flag(profile)
            rationale = (
                "Product is an IN VIVO DIAGNOSTIC: the device contacts or enters the body to generate a reading. "
                f"Patient contact: {profile.contact_category} / {profile.contact_duration}. "
                "Primary jurisdiction: CDRH. "
                f"BIOCOMPATIBILITY FLAGGED: {biocompat_note} "
                "Pathway depends on risk class — wearable non-invasive sensors are typically Class II (510(k)); "
                "implanted diagnostic devices may be Class III (PMA)."
            )
            return ProductCategory.DIAGNOSTIC_INVIVO, FDALeadCenter.CDRH, RegulatoryPathway.UNKNOWN, rationale

        else:
            # Diagnostic but location unclear — flag for LLM fallback
            rationale = (
                "Product appears to be diagnostic but the test location (in vitro vs. in vivo) is ambiguous. "
                "KEY QUESTION: Does this test happen on a sample outside the body, or does the device "
                "contact the patient? This determines whether biocompatibility testing is required."
            )
            return ProductCategory.UNKNOWN, FDALeadCenter.CDRH, RegulatoryPathway.UNKNOWN, rationale

    # ---- 3. Combination product ---------------------------------------
    if profile.is_combination_product or profile.product_category == ProductCategory.COMBINATION:
        return _route_combination_product(profile)

    # ---- 4. Medical device (fallback for physical/electrical/software products) ----
    if profile.product_category == ProductCategory.MEDICAL_DEVICE or profile.mechanism_of_action in (
        MechanismOfAction.MECHANICAL, MechanismOfAction.ELECTRICAL, MechanismOfAction.SOFTWARE
    ):
        rationale = (
            "Product is a MEDICAL DEVICE: achieves its purpose mechanically, electrically, or via software. "
            "Primary jurisdiction: CDRH. "
            "Device class (I/II/III) and pathway (510(k)/PMA/De Novo) to be determined by FDA product code match."
        )
        return ProductCategory.MEDICAL_DEVICE, FDALeadCenter.CDRH, RegulatoryPathway.UNKNOWN, rationale

    # ---- 5. Unknown — return UNKNOWN, proceed to LLM fallback --------
    return ProductCategory.UNKNOWN, FDALeadCenter.UNKNOWN, RegulatoryPathway.UNKNOWN, ""


def _get_biocompatibility_flag(profile: ProductProfile) -> str:
    """
    Generate a plain-English biocompatibility flag for in-vivo diagnostics.
    The roadmap generator will handle the actual test selection — this is the
    human-readable explanation for the classification result.
    """
    contact_map = {
        "surface": "Surface contact (intact skin/mucous membrane)",
        "external_communicating": "External communicating contact (blood path, tissue, or dentin)",
        "implant": "Implant contact (tissue/bone/blood — fully inside body)",
        "none": "No patient contact (review needed — flagged as in-vivo but no contact specified)",
    }
    duration_map = {
        "limited": "< 24 hours (limited duration)",
        "prolonged": "24 hours – 30 days (prolonged)",
        "permanent": "> 30 days (permanent)",
    }

    contact_desc = contact_map.get(profile.contact_category.value, profile.contact_category.value)
    duration_desc = duration_map.get(profile.contact_duration.value, profile.contact_duration.value)

    return (
        f"ISO 10993 biocompatibility testing required. "
        f"Contact type: {contact_desc}. Duration: {duration_desc}. "
        "Required endpoints will be listed in the testing roadmap."
    )


def _route_combination_product(profile: ProductProfile) -> tuple[ProductCategory, FDALeadCenter, RegulatoryPathway, str]:
    """
    Determine lead center for combination products based on primary mode of action (21 CFR 3.4).

    The primary mode of action (PMOA) is the single mode of action that provides
    the most important therapeutic/diagnostic effect. That component's lead center governs.
    All other components must still be cleared independently.
    """
    pmoa = profile.primary_mode_of_action.lower() if profile.primary_mode_of_action else ""

    # Signals that suggest each center's lead
    cber_signals = profile.contains_living_cells or profile.contains_gene_editing or profile.contains_tissue_engineering
    cder_signals = profile.has_drug_component and not cber_signals
    device_leads = profile.mechanism_of_action in (MechanismOfAction.MECHANICAL, MechanismOfAction.ELECTRICAL)

    # Determine lead center by PMOA signals
    if cber_signals:
        lead_center = FDALeadCenter.CBER
        lead_pathway = RegulatoryPathway.IND
        lead_desc = "CBER (biologic/cellular component drives primary effect)"
    elif cder_signals and not device_leads:
        lead_center = FDALeadCenter.CDER
        lead_pathway = RegulatoryPathway.COMBINATION_PRODUCT
        lead_desc = "CDER (drug component drives primary therapeutic effect)"
    else:
        lead_center = FDALeadCenter.CDRH
        lead_pathway = RegulatoryPathway.COMBINATION_PRODUCT
        lead_desc = "CDRH (device component drives primary effect)"

    # Build the component clearance list
    components = []
    if profile.mechanism_of_action in (MechanismOfAction.MECHANICAL, MechanismOfAction.ELECTRICAL):
        components.append("Device component: 510(k) or PMA clearance required from CDRH")
    if profile.has_drug_component:
        components.append("Drug component: NDA/ANDA approval required from CDER")
    if cber_signals:
        components.append("Biologic/cellular component: IND + BLA required from CBER")
    if not components:
        components.append("Component breakdown unclear — Request for Designation (RCM) recommended")

    rationale = (
        f"COMBINATION PRODUCT: Multiple regulatory categories intersect. "
        f"Primary Mode of Action (PMOA): {profile.primary_mode_of_action or 'unclear — see notes'}. "
        f"Lead Center: {lead_desc}. "
        "Each component must be independently cleared — this is the most expensive and time-consuming pathway. "
        f"Required clearances: {'; '.join(components)}. "
        "A Request for Designation (RCM) submission to FDA's Office of Combination Products is strongly recommended "
        "to obtain a binding determination of lead center before investing in the full development program."
    )

    return ProductCategory.COMBINATION, lead_center, lead_pathway, rationale




def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0:
        return 0.0
    return float(np.dot(va, vb) / norm)


def _simple_text_embedding(text: str) -> list[float]:
    """
    Fallback embedding using character n-gram overlap.
    Replace this with a real embedding model in production.

    NEXT STEPS: Use OpenAI text-embedding-3-small or a local sentence-transformer
    (e.g. all-MiniLM-L6-v2 via sentence-transformers library) for real semantic search.
    The FDA product code DB should be pre-embedded and cached — not embedded at query time.
    """
    text = text.lower()
    ngrams: dict[str, int] = {}
    for i in range(len(text) - 2):
        ng = text[i:i+3]
        ngrams[ng] = ngrams.get(ng, 0) + 1
    keys = sorted(ngrams.keys())
    return [ngrams[k] for k in keys[:256]] + [0] * max(0, 256 - len(keys))


def fetch_fda_product_codes(search_term: str, limit: int = 20) -> list[dict]:
    """
    Query the openFDA device classification endpoint.
    Returns a list of raw product code records.
    """
    params = {
        "search": f'device_name:{search_term}',
        "limit": limit,
    }
    if OPENFDA_API_KEY:
        params["api_key"] = OPENFDA_API_KEY
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(FDA_CLASSIFICATION_API, params=params)
            response.raise_for_status()
            return response.json().get("results", [])
    except Exception as e:
        logger.warning("FDA classification API call failed: %s", e)
        return []


def find_best_product_code(profile: ProductProfile) -> tuple[Optional[dict], float]:
    """
    Search the FDA product code database and return the best matching record
    along with a confidence score.

    Returns (record | None, confidence_score).
    """
    # Build a rich search query from the profile
    query_parts = [profile.intended_use, profile.indication]
    if profile.mechanism_of_action not in ("unknown", "combination"):
        query_parts.append(profile.mechanism_of_action.value)
    query = " ".join(p for p in query_parts if p).strip()

    if not query:
        return None, 0.0

    records = fetch_fda_product_codes(query, limit=20)

    if not records:
        # Fallback: try with just the first material
        if profile.materials:
            records = fetch_fda_product_codes(profile.materials[0], limit=10)

    if not records:
        return None, 0.0

    # Score each record by text similarity
    query_vec = _simple_text_embedding(query)
    best_record = None
    best_score = -1.0

    for record in records:
        candidate_text = " ".join(filter(None, [
            record.get("device_name", ""),
            record.get("medical_specialty_description", ""),
            record.get("physical_state", ""),
            record.get("technical_method", ""),
        ]))
        candidate_vec = _simple_text_embedding(candidate_text)
        score = _cosine_similarity(query_vec, candidate_vec)
        if score > best_score:
            best_score = score
            best_record = record

    return best_record, best_score


# ---------------------------------------------------------------------------
# Step 3: Classification decision tree
# ---------------------------------------------------------------------------

def _classify_from_product_code(record: dict, profile: ProductProfile) -> tuple[DeviceClass, RegulatoryPathway, str]:
    """
    Apply FDA classification logic given a matched product code record.
    Returns (device_class, pathway, rationale).
    """
    device_class_raw = record.get("device_class", "").strip()
    exempt = record.get("submission_type_id", "").strip()
    regulation_number = record.get("regulation_number", "")

    # Map FDA's raw class codes
    class_map = {"1": DeviceClass.CLASS_I, "2": DeviceClass.CLASS_II, "3": DeviceClass.CLASS_III}
    device_class = class_map.get(device_class_raw, DeviceClass.UNKNOWN)

    # Combination product overrides everything
    if profile.is_combination_product or profile.has_drug_component or profile.has_biologic_component:
        return DeviceClass.UNKNOWN, RegulatoryPathway.COMBINATION_PRODUCT, (
            "Device contains a drug or biologic component and is likely a combination product. "
            "The lead FDA center (CDER/CBER/CDRH) must be determined via Request for Designation."
        )

    if device_class == DeviceClass.CLASS_I:
        if exempt in ("1", "2"):  # Exempt codes vary; this is illustrative
            pathway = RegulatoryPathway.EXEMPT
            rationale = f"Class I device under 21 CFR {regulation_number}. 510(k) exempt."
        else:
            pathway = RegulatoryPathway.K510
            rationale = f"Class I device under 21 CFR {regulation_number}. Requires 510(k) premarket notification."

    elif device_class == DeviceClass.CLASS_II:
        pathway = RegulatoryPathway.K510
        rationale = f"Class II device under 21 CFR {regulation_number}. Standard 510(k) pathway. "
        # Flag De Novo if device appears novel (no strong predicate expected)
        rationale += "Consider De Novo if no clear predicate device exists."

    elif device_class == DeviceClass.CLASS_III:
        pathway = RegulatoryPathway.PMA
        rationale = (
            f"Class III device under 21 CFR {regulation_number}. "
            "Requires Premarket Approval (PMA). This is the most rigorous pathway — "
            "expect clinical trials, extensive non-clinical testing, and multi-year timeline. "
            "Explore IDE pathway for early clinical investigation."
        )
    else:
        pathway = RegulatoryPathway.UNKNOWN
        rationale = "Could not determine pathway from product code alone. Manual review required."

    return device_class, pathway, rationale


def _classify_without_product_code(profile: ProductProfile) -> tuple[DeviceClass, RegulatoryPathway, str, float]:
    """
    LLM-based fallback classification when no FDA product code match is found.
    Returns (class, pathway, rationale, confidence).
    """
    system_prompt = """
You are an FDA regulatory affairs expert specializing in medical device classification.

Given a product profile, determine:
1. The FDA device class (Class I, Class II, or Class III)
2. The most likely regulatory pathway
3. Your confidence (0.0-1.0)
4. Your rationale

Return JSON with keys:
{
  "device_class": "Class I" | "Class II" | "Class III" | "Unknown",
  "pathway": "510(k) Exempt" | "510(k)" | "De Novo" | "PMA" | "IDE" | "Combination Product" | "HDE" | "Unknown",
  "confidence": 0.0-1.0,
  "rationale": "explanation"
}
"""
    profile_text = (
        f"Mechanism: {profile.mechanism_of_action}\n"
        f"Intended use: {profile.intended_use}\n"
        f"Indication: {profile.indication}\n"
        f"Contact: {profile.contact_category} / {profile.contact_duration}\n"
        f"Implantable: {profile.is_implantable}\n"
        f"Drug component: {profile.has_drug_component}\n"
        f"Biologic component: {profile.has_biologic_component}\n"
        f"Software component: {profile.has_software_component}\n"
        f"Materials: {', '.join(profile.materials) or 'not specified'}\n"
    )

    data = call_llm_for_json(system_prompt=system_prompt, user_message=profile_text)

    class_map = {
        "Class I": DeviceClass.CLASS_I,
        "Class II": DeviceClass.CLASS_II,
        "Class III": DeviceClass.CLASS_III,
    }
    pathway_map = {
        "510(k) Exempt": RegulatoryPathway.EXEMPT,
        "510(k)": RegulatoryPathway.K510,
        "De Novo": RegulatoryPathway.DE_NOVO,
        "PMA": RegulatoryPathway.PMA,
        "IDE": RegulatoryPathway.IDE,
        "Combination Product": RegulatoryPathway.COMBINATION_PRODUCT,
        "HDE": RegulatoryPathway.HDE,
    }

    device_class = class_map.get(data.get("device_class", "Unknown"), DeviceClass.UNKNOWN)
    pathway = pathway_map.get(data.get("pathway", "Unknown"), RegulatoryPathway.UNKNOWN)
    confidence = float(data.get("confidence", 0.5))
    rationale = data.get("rationale", "")

    return device_class, pathway, rationale, confidence


# ---------------------------------------------------------------------------
# Step 4: Software safety classification
# ---------------------------------------------------------------------------

def _classify_software_safety(profile: ProductProfile) -> SoftwareSafetyClass:
    """
    Determine IEC 62304 software safety class.
    Class C: death or serious injury possible from failure
    Class B: non-serious injury possible
    Class A: no injury possible
    """
    if not profile.has_software_component:
        return SoftwareSafetyClass.NOT_APPLICABLE

    system_prompt = """
You are an expert in IEC 62304 medical device software lifecycle standards.
Given a product profile, classify the software safety class:
- Class C: A failure could lead to death or serious injury
- Class B: A failure could lead to non-serious injury
- Class A: A failure cannot lead to injury

Return JSON: {"software_class": "Class A" | "Class B" | "Class C", "rationale": "..."}
"""
    profile_text = (
        f"Device: {profile.intended_use}\n"
        f"Indication: {profile.indication}\n"
        f"Implantable: {profile.is_implantable}\n"
        f"Contact: {profile.contact_category}\n"
    )

    data = call_llm_for_json(system_prompt=system_prompt, user_message=profile_text)
    class_raw = data.get("software_class", "Class B")
    class_map = {
        "Class A": SoftwareSafetyClass.CLASS_A,
        "Class B": SoftwareSafetyClass.CLASS_B,
        "Class C": SoftwareSafetyClass.CLASS_C,
    }
    return class_map.get(class_raw, SoftwareSafetyClass.CLASS_B)


# ---------------------------------------------------------------------------
# Step 5: Predicate device search
# ---------------------------------------------------------------------------

def find_predicate_devices(product_code: str, limit: int = 3) -> list[PredicateDevice]:
    """
    Query the FDA 510(k) database for cleared devices with the same product code.
    Returns a list of potential predicate devices.

    NEXT STEPS:
      - Add semantic re-ranking: embed each 510k device name and return the ones
        closest to the user's product description, not just the most recent.
      - Filter by decision date (prefer predicates < 5 years old — regulators
        prefer recent predicates).
      - Pull the full 510k summary PDF for each predicate and extract the
        testing table using a PDF parsing step. This gives users insight into
        exactly what testing their likely predicate submitted.
    """
    params = {
        "search": f"product_code:{product_code}",
        "limit": limit,
        "sort": "date_received:desc",
    }
    if OPENFDA_API_KEY:
        params["api_key"] = OPENFDA_API_KEY
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(FDA_510K_API, params=params)
            response.raise_for_status()
            results = response.json().get("results", [])
    except Exception as e:
        logger.warning("510(k) API call failed: %s", e)
        return []

    predicates = []
    for r in results:
        predicates.append(PredicateDevice(
            k_number=r.get("k_number", ""),
            device_name=r.get("device_name", ""),
            applicant=r.get("applicant", ""),
            decision_date=r.get("decision_date_as_string", ""),
            similarity_score=0.8,   # Placeholder — replace with real embedding similarity
        ))
    return predicates


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def classify_device(raw_description: str) -> ClassificationResult:
    """
    Main entry point for System 1.
    Takes a plain-language description, returns a fully populated ClassificationResult.

    Routing order:
      1. Extract product profile (LLM)
      2. Primary category router — the critical first gate:
           Cell/gene therapy → CBER/IND (stops here, no product code search needed)
           IVD diagnostic     → CDRH, no biocompat (continues to device class lookup)
           In-vivo diagnostic → CDRH + biocompat flagged (continues to device class lookup)
           Combination        → resolve by PMOA, list all component clearances
           Medical device     → continue to product code search + device class
      3. For CDRH medical devices and diagnostics: product code search → device class → pathway
      4. Software safety classification (if applicable)
      5. Predicate device search (if 510(k) pathway)
    """
    logger.info("Starting classification (length=%d)", len(raw_description))

    # Step 1: Extract structured profile
    profile = extract_product_profile(raw_description)
    logger.info(
        "Extraction: category=%s, MOA=%s, diagnostic=%s, living_cells=%s, gene_editing=%s",
        profile.product_category, profile.mechanism_of_action,
        profile.is_diagnostic, profile.contains_living_cells, profile.contains_gene_editing,
    )

    # Step 2: Primary category routing
    product_category, lead_center, primary_pathway, category_rationale = route_primary_category(profile)
    logger.info("Primary routing: category=%s, lead_center=%s, pathway=%s", product_category, lead_center, primary_pathway)

    # ---- Early exit for CBER/IND (cell/gene therapy) ----
    if lead_center == FDALeadCenter.CBER and primary_pathway == RegulatoryPathway.IND:
        return ClassificationResult(
            product_profile=profile,
            product_category=product_category,
            lead_center=lead_center,
            device_class=DeviceClass.UNKNOWN,  # Not applicable for biologics
            regulatory_pathway=RegulatoryPathway.IND,
            software_safety_class=SoftwareSafetyClass.NOT_APPLICABLE,
            confidence=0.92,  # High confidence — these signals are unambiguous
            combination_product_components=[],
            classification_rationale=category_rationale,
        )

    # ---- Early exit for combination products ----
    if product_category == ProductCategory.COMBINATION:
        components = _build_combination_component_list(profile)
        return ClassificationResult(
            product_profile=profile,
            product_category=ProductCategory.COMBINATION,
            lead_center=lead_center,
            device_class=DeviceClass.UNKNOWN,
            regulatory_pathway=primary_pathway,
            software_safety_class=SoftwareSafetyClass.NOT_APPLICABLE,
            confidence=0.85,
            combination_product_components=components,
            classification_rationale=category_rationale,
        )

    # ---- For CDRH products: continue to device class determination ----
    low_confidence_warning = None
    product_code = None
    regulation_number = None
    predicate_devices = []
    device_class = DeviceClass.UNKNOWN
    pathway = primary_pathway  # May already be set (e.g., UNKNOWN meaning "keep looking")
    detail_rationale = ""

    product_code_record, match_confidence = find_best_product_code(profile)

    if product_code_record and match_confidence >= LOW_CONFIDENCE_THRESHOLD:
        device_class, pathway, detail_rationale = _classify_from_product_code(product_code_record, profile)
        confidence = match_confidence
        product_code = product_code_record.get("product_code")
        regulation_number = product_code_record.get("regulation_number")
    else:
        logger.warning("Low product code match (%.2f). Falling back to LLM.", match_confidence)
        device_class, pathway, detail_rationale, confidence = _classify_without_product_code(profile)

        if confidence < LOW_CONFIDENCE_THRESHOLD or product_code_record is None:
            low_confidence_warning = (
                f"Classification confidence is {confidence:.0%}. "
                "This product description may span multiple product codes or represent a novel type. "
                "Manual review by a regulatory affairs specialist is strongly recommended."
            )

    # Step 4: Software safety class
    software_safety_class = _classify_software_safety(profile)
    if software_safety_class != SoftwareSafetyClass.NOT_APPLICABLE:
        detail_rationale += f" Software safety class: {software_safety_class.value} (IEC 62304)."

    # Step 5: Predicate devices (510(k) only)
    if pathway == RegulatoryPathway.K510 and product_code:
        predicate_devices = find_predicate_devices(product_code)

    # Compose final rationale: category context + device class detail
    full_rationale = (category_rationale + " " + detail_rationale).strip()

    return ClassificationResult(
        product_profile=profile,
        product_category=product_category if product_category != ProductCategory.UNKNOWN else ProductCategory.MEDICAL_DEVICE,
        lead_center=lead_center if lead_center != FDALeadCenter.UNKNOWN else FDALeadCenter.CDRH,
        device_class=device_class,
        regulatory_pathway=pathway,
        product_code=product_code,
        regulation_number=regulation_number,
        software_safety_class=software_safety_class,
        confidence=confidence,
        low_confidence_warning=low_confidence_warning,
        combination_product_components=[],
        predicate_devices=predicate_devices,
        classification_rationale=full_rationale,
    )


def _build_combination_component_list(profile: ProductProfile) -> list[str]:
    """Build the per-component clearance list for combination products."""
    components = []
    if profile.mechanism_of_action in (MechanismOfAction.MECHANICAL, MechanismOfAction.ELECTRICAL):
        components.append("Device component → CDRH: 510(k) or PMA clearance required")
    if profile.has_drug_component:
        components.append("Drug component → CDER: NDA or ANDA approval required")
    if profile.contains_living_cells or profile.contains_gene_editing or profile.contains_tissue_engineering:
        components.append("Biologic/cellular component → CBER: IND (investigational) + BLA (approval) required")
    elif profile.has_biologic_component:
        components.append("Biologic component → CBER: BLA or 351(k) biosimilar pathway required")
    if not components:
        components.append("Component breakdown unclear — Request for Designation (RCM) filing recommended")
    return components
