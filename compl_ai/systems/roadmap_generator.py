"""
System 2: Testing Roadmap Generator  (v2 — ISO 10993-1:2018 / FDA 2023 Guidance)
===================================================================================
Generates a sequenced, DAG-based testing roadmap fully aligned with:
  - ISO 10993-1:2018 (biological evaluation standard)
  - FDA Guidance on ISO 10993-1 (2023 update)
  - FDA Modified Biocompatibility Matrix (Attachment A of 2023 guidance)

KEY DESIGN DECISIONS derived from the document comparison (see uploaded PDF):

1.  CONTACT MATRIX IS THE PRIMARY FILTER
    Every biocompatibility test is driven by (contact_category × contact_duration).
    The document's Table A.1 (FDA modified matrix) is encoded as ENDPOINT_MATRIX.

2.  FDA REQUIRES MULTIPLE CONTACT CATEGORIES FOR COMPLEX DEVICES (FDA Section 4C)
    A pacemaker = subcutaneous implant + intravascular leads → both evaluated independently.

3.  RISK ASSESSMENT IS THE MANDATORY GATEWAY (FDA Section 3D, 4D)
    Must appear at the start of every biocompatibility submission section. Not waivable.

4.  CHEMICAL CHARACTERIZATION IS ALWAYS REQUIRED FOR EC/IMPLANT (FDA Section 4D, 7)
    ISO 10993-18 full chem char cannot be waived for novel materials. Two tiers: screening (surface)
    and full analytical (EC/implant).

5.  NOVEL MATERIAL FLAG REMOVES ALL WAIVERS AND ADDS TESTS (FDA Section 6)
    Novel materials → GPMT mandatory, in vivo genotoxicity required, carcinogenicity SAR needed.

6.  PROLONGED/PERMANENT CONTACT UNLOCKS ADDITIONAL ENDPOINTS (FDA Matrix, Attachment A)
    Subacute/subchronic → prolonged; chronic + carcinogenicity + repro → permanent.

7.  HEMOCOMPATIBILITY HAS TWO TRACKS (FDA Section 6C)
    Direct blood: hemolysis (direct+indirect), complement (SC5b-9 ELISA, direct study),
    thrombogenicity (in vivo). Indirect: hemolysis extract method only.

8.  SENSITIZATION = TWO MANDATED TESTS (FDA Section 6B)
    GPMT + LLNA both evaluated. GPMT mandatory for novel materials. LLNA not for nickel alloys.

9.  GENOTOXICITY = THREE-PART BATTERY (FDA Section 6F)
    Ames + MLA (preferred)/chromosomal aberration + in vivo cytogenetics (novel/extracorporeal).

10. ABSORBABLE MATERIALS → DEGRADATION STUDIES WITH FDA PRE-DISCUSSION (FDA Section 5B, 6I)

11. PYROGENICITY IS SEPARATE FROM STERILITY (FDA Section 6D)

12. IVD DEVICES: ANALYTICAL PERFORMANCE REPLACES BIOCOMPATIBILITY

13. CBER / CELL-GENE THERAPY: ENTIRELY SEPARATE TESTING TRACK

NEXT STEPS:
  - Move MASTER_TEST_LIBRARY and ENDPOINT_MATRIX to PostgreSQL; version them.
  - Add EU MDR / EN ISO 10993 pathway with side-by-side comparison mode.
  - Upgrade critical path to proper CPM with float calculation.
  - Add CBER-specific testing track for combination products.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Optional

from utils.llm_client import call_llm
from utils.models import (
    ClassificationResult,
    ContactCategory,
    ContactDuration,
    FDALeadCenter,
    ProductCategory,
    RegulatoryPathway,
    RoadmapResult,
    SoftwareSafetyClass,
    TestNode,
    TestPhase,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# FDA MODIFIED BIOCOMPATIBILITY ENDPOINT MATRIX  (Attachment A, FDA 2023)
# ===========================================================================
# Structure: ENDPOINT_MATRIX[contact_category_key][duration_key] = set of test IDs

DURATION_KEY = {
    ContactDuration.LIMITED: "limited",
    ContactDuration.PROLONGED: "prolonged",
    ContactDuration.PERMANENT: "permanent",
}

ENDPOINT_MATRIX: dict[str, dict[str, set[str]]] = {

    "surface": {
        "limited": {
            "RISK_ASSESSMENT", "CHEM_CHAR_SCREENING",
            "ISO_10993_5", "ISO_10993_10_SENS", "ISO_10993_10_IRR",
        },
        "prolonged": {
            "RISK_ASSESSMENT", "CHEM_CHAR_SCREENING",
            "ISO_10993_5", "ISO_10993_10_SENS", "ISO_10993_10_IRR",
            "ISO_10993_11_ACUTE",
        },
        "permanent": {
            "RISK_ASSESSMENT", "CHEM_CHAR_SCREENING",
            "ISO_10993_5", "ISO_10993_10_SENS", "ISO_10993_10_IRR",
            "ISO_10993_11_ACUTE", "ISO_10993_11_SUBACUTE",
            "ISO_10993_3_GENO", "ISO_10993_3_CARCINO",
        },
    },

    "external_communicating": {
        "limited": {
            "RISK_ASSESSMENT", "CHEM_CHAR_FULL",
            "ISO_10993_5", "ISO_10993_10_SENS", "ISO_10993_10_IRR",
            "ISO_10993_11_ACUTE", "ISO_10993_3_GENO",
        },
        "prolonged": {
            "RISK_ASSESSMENT", "CHEM_CHAR_FULL",
            "ISO_10993_5", "ISO_10993_10_SENS", "ISO_10993_10_IRR",
            "ISO_10993_11_ACUTE", "ISO_10993_11_SUBACUTE",
            "ISO_10993_3_GENO", "ISO_10993_3_CARCINO",
        },
        "permanent": {
            "RISK_ASSESSMENT", "CHEM_CHAR_FULL",
            "ISO_10993_5", "ISO_10993_10_SENS", "ISO_10993_10_IRR",
            "ISO_10993_11_ACUTE", "ISO_10993_11_SUBACUTE", "ISO_10993_11_CHRONIC",
            "ISO_10993_3_GENO", "ISO_10993_3_CARCINO", "ISO_10993_REPRO",
        },
    },

    "circulating_blood": {
        "limited": {
            "RISK_ASSESSMENT", "CHEM_CHAR_FULL",
            "ISO_10993_5", "ISO_10993_10_SENS", "ISO_10993_10_IRR",
            "ISO_10993_11_ACUTE", "ISO_10993_3_GENO",
            "ISO_10993_4_HEMO_DIRECT", "ISO_10993_PYRO",
        },
        "prolonged": {
            "RISK_ASSESSMENT", "CHEM_CHAR_FULL",
            "ISO_10993_5", "ISO_10993_10_SENS", "ISO_10993_10_IRR",
            "ISO_10993_11_ACUTE", "ISO_10993_11_SUBACUTE",
            "ISO_10993_3_GENO", "ISO_10993_3_CARCINO",
            "ISO_10993_4_HEMO_DIRECT", "ISO_10993_PYRO",
        },
        "permanent": {
            "RISK_ASSESSMENT", "CHEM_CHAR_FULL",
            "ISO_10993_5", "ISO_10993_10_SENS", "ISO_10993_10_IRR",
            "ISO_10993_11_ACUTE", "ISO_10993_11_SUBACUTE", "ISO_10993_11_CHRONIC",
            "ISO_10993_3_GENO", "ISO_10993_3_CARCINO", "ISO_10993_REPRO",
            "ISO_10993_4_HEMO_DIRECT", "ISO_10993_PYRO",
        },
    },

    "blood_path_indirect": {
        "limited": {
            "RISK_ASSESSMENT", "CHEM_CHAR_FULL",
            "ISO_10993_5", "ISO_10993_10_SENS", "ISO_10993_10_IRR",
            "ISO_10993_11_ACUTE", "ISO_10993_3_GENO",
            "ISO_10993_4_HEMO_INDIRECT", "ISO_10993_PYRO",
        },
        "prolonged": {
            "RISK_ASSESSMENT", "CHEM_CHAR_FULL",
            "ISO_10993_5", "ISO_10993_10_SENS", "ISO_10993_10_IRR",
            "ISO_10993_11_ACUTE", "ISO_10993_11_SUBACUTE",
            "ISO_10993_3_GENO",
            "ISO_10993_4_HEMO_INDIRECT", "ISO_10993_PYRO",
        },
        "permanent": {
            "RISK_ASSESSMENT", "CHEM_CHAR_FULL",
            "ISO_10993_5", "ISO_10993_10_SENS", "ISO_10993_10_IRR",
            "ISO_10993_11_ACUTE", "ISO_10993_11_SUBACUTE", "ISO_10993_11_CHRONIC",
            "ISO_10993_3_GENO", "ISO_10993_3_CARCINO", "ISO_10993_REPRO",
            "ISO_10993_4_HEMO_INDIRECT", "ISO_10993_PYRO",
        },
    },

    "implant_tissue": {
        "limited": {
            "RISK_ASSESSMENT", "CHEM_CHAR_FULL",
            "ISO_10993_5", "ISO_10993_10_SENS", "ISO_10993_10_IRR",
            "ISO_10993_11_ACUTE", "ISO_10993_3_GENO",
        },
        "prolonged": {
            "RISK_ASSESSMENT", "CHEM_CHAR_FULL",
            "ISO_10993_5", "ISO_10993_10_SENS", "ISO_10993_10_IRR",
            "ISO_10993_11_ACUTE", "ISO_10993_11_SUBACUTE",
            "ISO_10993_3_GENO", "ISO_10993_6_IMPLANT",
        },
        "permanent": {
            "RISK_ASSESSMENT", "CHEM_CHAR_FULL",
            "ISO_10993_5", "ISO_10993_10_SENS", "ISO_10993_10_IRR",
            "ISO_10993_11_ACUTE", "ISO_10993_11_SUBACUTE", "ISO_10993_11_CHRONIC",
            "ISO_10993_3_GENO", "ISO_10993_3_CARCINO", "ISO_10993_REPRO",
            "ISO_10993_6_IMPLANT",
        },
    },

    "implant_blood": {
        "limited": {
            "RISK_ASSESSMENT", "CHEM_CHAR_FULL",
            "ISO_10993_5", "ISO_10993_10_SENS", "ISO_10993_10_IRR",
            "ISO_10993_11_ACUTE", "ISO_10993_3_GENO",
            "ISO_10993_4_HEMO_DIRECT", "ISO_10993_PYRO",
        },
        "prolonged": {
            "RISK_ASSESSMENT", "CHEM_CHAR_FULL",
            "ISO_10993_5", "ISO_10993_10_SENS", "ISO_10993_10_IRR",
            "ISO_10993_11_ACUTE", "ISO_10993_11_SUBACUTE",
            "ISO_10993_3_GENO", "ISO_10993_6_IMPLANT",
            "ISO_10993_4_HEMO_DIRECT", "ISO_10993_PYRO",
        },
        "permanent": {
            "RISK_ASSESSMENT", "CHEM_CHAR_FULL",
            "ISO_10993_5", "ISO_10993_10_SENS", "ISO_10993_10_IRR",
            "ISO_10993_11_ACUTE", "ISO_10993_11_SUBACUTE", "ISO_10993_11_CHRONIC",
            "ISO_10993_3_GENO", "ISO_10993_3_CARCINO", "ISO_10993_REPRO",
            "ISO_10993_6_IMPLANT",
            "ISO_10993_4_HEMO_DIRECT", "ISO_10993_PYRO",
        },
    },
}


# ===========================================================================
# MASTER TEST LIBRARY  (v2 — FDA 2023 Section 6 compliant)
# ===========================================================================

MASTER_TEST_LIBRARY: dict[str, dict] = {

    "RISK_ASSESSMENT": {
        "name": "Risk Assessment Documentation",
        "standard": "ISO 10993-1:2018 Clause 4 + FDA Guidance Sections 3 & 4D",
        "description": (
            "Mandatory gateway — placed at the START of every biocompatibility submission section (FDA Sec 3D). "
            "Must include: device/material description, manufacturing & sterilization processes, "
            "proposed clinical use and patient population, identification of potential risks "
            "(chemical, physical, surface, particulates), review of available literature/clinical "
            "experience/animal data/prior FDA-reviewed devices, data gap identification, and "
            "testing plan or written waiver justification for each endpoint. "
            "ISO allows no-testing conclusion if history of safe use is adequate and documented."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": [],
        "can_parallelize_with": [],
        "cost_low": 8000, "cost_high": 25000,
        "weeks_low": 4, "weeks_high": 10,
        "waivable_with_existing_data": False,
        "waiver_rationale": None,
        "triggered_by": [],
        "waived_by": [],
        "applicable_pathways": [
            RegulatoryPathway.EXEMPT, RegulatoryPathway.K510,
            RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "FDA Sec 3D: required at submission start. Waiver of the document itself is not permitted.",
    },

    "CHEM_CHAR_SCREENING": {
        "name": "Physical/Chemical Characterization (Screening)",
        "standard": "ISO 10993-18:2020 (screening), FDA Guidance Clause 7",
        "description": (
            "Gather existing physical/chemical information: formulation, processing history, "
            "supplier-provided extractables/leachables data, and published literature. "
            "Used to answer ISO 10993-1 Figure 1 questions and assess whether additional "
            "analytical chemistry is needed. Sufficient for surface-contacting devices with "
            "well-characterized materials. "
            "FDA Clause 7: if all chemicals released at worst-case present no toxicity concern, "
            "no further characterization needed. If concern exists, full ISO 10993-18 required."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["RISK_ASSESSMENT"],
        "can_parallelize_with": ["ISO_10993_5", "ISO_10993_10_SENS"],
        "cost_low": 3000, "cost_high": 10000,
        "weeks_low": 2, "weeks_high": 6,
        "waivable_with_existing_data": True,
        "waiver_rationale": (
            "If all materials have an established safe-use history with identical processing "
            "and the physical form is unchanged, documented rationale may suffice "
            "(ISO 10993-1 Clause 4.4). Must be explicitly stated in submission."
        ),
        "triggered_by": [],
        "waived_by": ["established_biocompatibility_data"],
        "applicable_pathways": [
            RegulatoryPathway.EXEMPT, RegulatoryPathway.K510,
            RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "FDA Clause 7: specifies chemical identity, CAS number, weight %, structure for each chemical.",
    },

    "CHEM_CHAR_FULL": {
        "name": "Full Chemical Characterization (ISO 10993-18)",
        "standard": "ISO 10993-18:2020, FDA Guidance Clause 7",
        "description": (
            "Full analytical chemistry using both polar and nonpolar solvents (ISO 10993-12). "
            "Extraction conditions must reflect clinical use (temperature, duration, contact surface area). "
            "For each identified chemical: CAS number, chemical name, trade name, weight % in "
            "formulation, total amount in device, and toxicological risk assessment. "
            "If toxicity concern exists at worst-case full release: "
            "ADME assessment in clinically relevant animal model may be required (FDA Clause 7). "
            "Sterilization residuals must be included. Required for all EC and implant devices."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["RISK_ASSESSMENT"],
        "can_parallelize_with": ["ISO_10993_5", "ISO_10993_10_SENS", "ISO_10993_10_IRR"],
        "cost_low": 12000, "cost_high": 35000,
        "weeks_low": 6, "weeks_high": 14,
        "waivable_with_existing_data": False,
        "waiver_rationale": (
            "Cannot be waived for EC or implant devices. Manufacturer-provided analytical data "
            "for identical material grade and processing may reduce scope — but must be documented."
        ),
        "triggered_by": [],
        "waived_by": [],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "FDA Clause 7: ADME data required if concern exists at full release concentration.",
    },

    "ISO_10993_5": {
        "name": "Cytotoxicity (ISO 10993-5)",
        "standard": "ISO 10993-5:2009",
        "description": (
            "In vitro cell viability after device/material contact. "
            "FDA preferred method: elution in MEM + 5-10% serum at 37°C for 24-72h (FDA Sec 6A). "
            "Novel materials: BOTH direct contact AND elution methods required. "
            "Inherently cytotoxic materials: serial dilution study to establish threshold. "
            "Coatings/surface modifications with no implantation data: non-standard direct "
            "contact (cells grown on material surface) may be needed."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["CHEM_CHAR_SCREENING"],
        "can_parallelize_with": ["ISO_10993_10_SENS", "ISO_10993_10_IRR", "STERILITY_SAL"],
        "cost_low": 3000, "cost_high": 8000,
        "weeks_low": 3, "weeks_high": 6,
        "waivable_with_existing_data": True,
        "waiver_rationale": (
            "Requires: identical material formulation, processing, sterilization; same contact "
            "category; written justification in risk assessment (FDA Sec 4D). Novel = no waiver."
        ),
        "triggered_by": [],
        "waived_by": ["established_biocompatibility_data"],
        "applicable_pathways": [
            RegulatoryPathway.EXEMPT, RegulatoryPathway.K510,
            RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "FDA 6A: elution in MEM+serum preferred. Novel material = direct + elution both required.",
    },

    "ISO_10993_10_SENS": {
        "name": "Sensitization — GPMT + LLNA (ISO 10993-10)",
        "standard": "ISO 10993-10:2021, FDA Guidance Section 6B",
        "description": (
            "FDA Section 6B requires BOTH GPMT and LLNA to be evaluated. "
            "GPMT: positive controls (same animal source/strain) concurrent or within 3 months; "
            "min 5 guinea pigs. If periodic positive control fails, all subsequent GPMT data "
            "are invalidated (FDA 6B). "
            "LLNA: evaluated case-by-case for chemical mixtures. Acceptable for metals EXCEPT "
            "nickel and nickel-containing alloys. LLNA:BrdU-ELISA and LLNA:DA are alternatives. "
            "Novel materials: GPMT is mandated over LLNA."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["CHEM_CHAR_SCREENING"],
        "can_parallelize_with": ["ISO_10993_5", "ISO_10993_10_IRR", "CHEM_CHAR_FULL"],
        "cost_low": 6000, "cost_high": 14000,
        "weeks_low": 6, "weeks_high": 12,
        "waivable_with_existing_data": True,
        "waiver_rationale": (
            "GPMT and/or LLNA data from identical material lot with same processing and "
            "sterilization. Novel materials cannot use LLNA alone — GPMT required (FDA 6B)."
        ),
        "triggered_by": [],
        "waived_by": ["established_biocompatibility_data"],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "FDA 6B: GPMT mandatory for novel materials. Nickel alloys: GPMT only (not LLNA).",
    },

    "ISO_10993_10_IRR": {
        "name": "Irritation / Intracutaneous Reactivity (ISO 10993-10)",
        "standard": "ISO 10993-10:2021",
        "description": (
            "Estimate irritation potential. Intracutaneous reactivity in rabbits is standard. "
            "Test method must be appropriate for the specific route and duration of clinical exposure."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["CHEM_CHAR_SCREENING"],
        "can_parallelize_with": ["ISO_10993_5", "ISO_10993_10_SENS"],
        "cost_low": 3000, "cost_high": 8000,
        "weeks_low": 4, "weeks_high": 8,
        "waivable_with_existing_data": True,
        "waiver_rationale": "Existing data for identical material, processing, sterilization at equivalent contact.",
        "triggered_by": [],
        "waived_by": ["established_biocompatibility_data"],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "Test method must be appropriate for the route and duration of exposure/contact.",
    },

    "ISO_10993_11_ACUTE": {
        "name": "Acute Systemic Toxicity (ISO 10993-11)",
        "standard": "ISO 10993-11:2017",
        "description": (
            "Single or multiple exposures <24h to estimate potential harm from toxic leachables "
            "and degradation products. Mouse or rat systemic injection study."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["ISO_10993_5"],
        "can_parallelize_with": ["ISO_10993_10_SENS", "ISO_10993_3_GENO"],
        "cost_low": 5000, "cost_high": 12000,
        "weeks_low": 4, "weeks_high": 8,
        "waivable_with_existing_data": True,
        "waiver_rationale": "USP Class VI data or existing systemic toxicity data for identical material/contact.",
        "triggered_by": [],
        "waived_by": ["established_biocompatibility_data", "usp_class_vi"],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "Required when contact allows potential absorption of toxic leachables.",
    },

    "ISO_10993_11_SUBACUTE": {
        "name": "Subacute / Subchronic Toxicity (ISO 10993-11)",
        "standard": "ISO 10993-11:2017",
        "description": (
            "Single or multiple exposures for 24h ≤ period ≤ 10% animal lifespan. "
            "Required for prolonged/permanent EC and implant devices. "
            "May be waived if adequate chronic toxicity data exist — "
            "chronic data cover subacute endpoints (ISO 10993-1 Section 6.3.2)."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["ISO_10993_11_ACUTE"],
        "can_parallelize_with": ["ISO_10993_3_GENO", "ISO_10993_4_HEMO_DIRECT"],
        "cost_low": 15000, "cost_high": 35000,
        "weeks_low": 8, "weeks_high": 16,
        "waivable_with_existing_data": True,
        "waiver_rationale": "Chronic toxicity data covering same materials deemed sufficient per ISO 6.3.2.",
        "triggered_by": [],
        "waived_by": ["established_biocompatibility_data"],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "ISO 6.3.2: waived if chronic data available for the relevant materials.",
    },

    "ISO_10993_11_CHRONIC": {
        "name": "Chronic Toxicity (ISO 10993-11)",
        "standard": "ISO 10993-11:2017",
        "description": (
            "Single or multiple exposures during a major period of animal lifespan. "
            "Required for permanent EC/implant devices (>30 days contact). "
            "Can be combined with implantation study where feasible."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["ISO_10993_11_SUBACUTE"],
        "can_parallelize_with": ["ISO_10993_3_CARCINO", "ISO_10993_REPRO"],
        "cost_low": 30000, "cost_high": 80000,
        "weeks_low": 26, "weeks_high": 52,
        "waivable_with_existing_data": True,
        "waiver_rationale": "Comprehensive chronic toxicity data for all device materials in same contact conditions.",
        "triggered_by": [],
        "waived_by": ["established_biocompatibility_data"],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "May be combined with implantation study if feasible (ISO 10993-11).",
    },

    "ISO_10993_3_GENO": {
        "name": "Genotoxicity Battery (ISO 10993-3)",
        "standard": "ISO 10993-3:2023, FDA Guidance Section 6F",
        "description": (
            "FDA 6F three-part battery — no single test detects all genotoxins: "
            "(1) Bacterial Gene Mutation (Ames, OECD 471) — always required. "
            "(2) In vitro mammalian: MLA [preferred] OR chromosomal aberration OR micronucleus. "
            "(3) In vivo cytogenetics — required for: novel materials; extracorporeal blood-contact "
            "    circuits (any duration, due to high surface area + systemic leaching); devices "
            "    with positive in vitro battery. "
            "Waiver only if chem char + literature adequately address all three components."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["CHEM_CHAR_FULL"],
        "can_parallelize_with": ["ISO_10993_11_ACUTE", "ISO_10993_11_SUBACUTE"],
        "cost_low": 10000, "cost_high": 25000,
        "weeks_low": 8, "weeks_high": 16,
        "waivable_with_existing_data": True,
        "waiver_rationale": (
            "Waiver if: chem char + literature adequately address all three battery components. "
            "Extracorporeal blood contact → in vivo cytogenetics cannot be waived (FDA 6F)."
        ),
        "triggered_by": [],
        "waived_by": ["established_biocompatibility_data"],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "FDA 6F: in vivo component required for novel materials and extracorporeal blood circuits.",
    },

    "ISO_10993_3_CARCINO": {
        "name": "Carcinogenicity Assessment (ISO 10993-3)",
        "standard": "ISO 10993-3:2023, FDA Guidance Section 6G",
        "description": (
            "Required for devices with >30 day contact with breached surfaces, EC, or implants. "
            "Primary method: risk assessment (literature review). "
            "Novel materials: literature review specifically recommended (FDA 6G). "
            "No experimental data available: SAR (structure-activity relationship) modeling required. "
            "IARC monograph chemicals identified: cancer risk assessment with literature evidence."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["ISO_10993_3_GENO", "CHEM_CHAR_FULL"],
        "can_parallelize_with": ["ISO_10993_REPRO", "ISO_10993_11_CHRONIC"],
        "cost_low": 5000, "cost_high": 20000,
        "weeks_low": 4, "weeks_high": 12,
        "waivable_with_existing_data": True,
        "waiver_rationale": (
            "Typically a risk assessment, not animal study. Waiver if: no carcinogenic "
            "constituents in chem char, negative genotoxicity battery, no IARC carcinogens."
        ),
        "triggered_by": [],
        "waived_by": ["established_biocompatibility_data"],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "FDA 6G: SAR modelling required in absence of experimental data for novel materials.",
    },

    "ISO_10993_REPRO": {
        "name": "Reproductive / Developmental Toxicity (ISO 10993-1)",
        "standard": "ISO 10993-1:2018, FDA Guidance Section 6H",
        "description": (
            "Evaluate reproductive function, embryonic development, prenatal/postnatal effects. "
            "FDA 6H: required when biocompatibility evaluation identifies a known or potential "
            "repro/dev toxicity risk AND adequate literature is not available. "
            "Animal testing of reproductive age considered if materials may be systemically "
            "distributed without available reproductive toxicity literature."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["CHEM_CHAR_FULL", "ISO_10993_11_CHRONIC"],
        "can_parallelize_with": ["ISO_10993_3_CARCINO"],
        "cost_low": 25000, "cost_high": 75000,
        "weeks_low": 16, "weeks_high": 52,
        "waivable_with_existing_data": True,
        "waiver_rationale": (
            "Waiver if: no known/potential repro/dev risk identified; adequate literature "
            "available; or materials not systemically distributed."
        ),
        "triggered_by": [],
        "waived_by": ["established_biocompatibility_data"],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "FDA 6H: labeling mitigations likely needed if risk identified and literature inadequate.",
    },

    "ISO_10993_4_HEMO_DIRECT": {
        "name": "Hemocompatibility — Direct Blood Contact (ISO 10993-4)",
        "standard": "ISO 10993-4:2017, FDA Guidance Section 6C",
        "description": (
            "For DIRECT contact with circulating blood. FDA 6C mandates three assessments: "
            "(1) Hemolysis: BOTH direct (ASTM F756) AND indirect (extract) methods required. "
            "(2) Complement activation: direct contact study (NOT extract); SC5b-9 ELISA. "
            "    Physical/chemical properties significantly affect complement — must use actual device. "
            "(3) Thrombogenicity: in vivo animal model in clinically relevant study. "
            "    Geometry, contact conditions, and flow dynamics must be reflected. "
            "Written risk assessment summary required if any testing waived."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["ISO_10993_5", "CHEM_CHAR_FULL"],
        "can_parallelize_with": ["ISO_10993_11_SUBACUTE", "ISO_10993_3_GENO"],
        "cost_low": 20000, "cost_high": 55000,
        "weeks_low": 8, "weeks_high": 20,
        "waivable_with_existing_data": True,
        "waiver_rationale": (
            "Written risk assessment summary required. Data must cover hemolysis, complement "
            "activation, AND thrombogenicity for same material and blood contact configuration."
        ),
        "triggered_by": [],
        "waived_by": ["established_biocompatibility_data"],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "FDA 6C: complement = direct study, SC5b-9 ELISA. Thrombogenicity = in vivo animal.",
    },

    "ISO_10993_4_HEMO_INDIRECT": {
        "name": "Hemocompatibility — Indirect Blood Contact (ISO 10993-4)",
        "standard": "ISO 10993-4:2017, FDA Guidance Section 6C",
        "description": (
            "For INDIRECT contact with circulating blood (fluid passes through device before body). "
            "FDA 6C: only hemolysis testing required for indirect contact. "
            "Method: indirect (extract) method per ASTM F756 only. "
            "Direct contact hemolysis method NOT required. "
            "Complement and thrombogenicity generally not needed unless risk assessment flags concern."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["ISO_10993_5", "CHEM_CHAR_FULL"],
        "can_parallelize_with": ["ISO_10993_11_ACUTE"],
        "cost_low": 5000, "cost_high": 12000,
        "weeks_low": 4, "weeks_high": 8,
        "waivable_with_existing_data": True,
        "waiver_rationale": "Existing hemolysis (extract method) data for same material and indirect contact config.",
        "triggered_by": [],
        "waived_by": ["established_biocompatibility_data"],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "FDA 6C: indirect blood contact → ASTM F756 extract method only.",
    },

    "ISO_10993_PYRO": {
        "name": "Material-Mediated Pyrogenicity (USP <151> / MAT)",
        "standard": "ISO 10993-1:2018, USP <151>, FDA Guidance Section 6D",
        "description": (
            "Tests for material-induced fever response (distinct from endotoxin/sterility). "
            "No single test differentiates material-mediated from endotoxin-mediated pyrogenicity (FDA 6D). "
            "Standard: extraction at 50°C/72h, 70°C/24h, or 121°C/1h (ISO 10993-12:2021), "
            "then USP <151> rabbit bioassay or validated MAT. "
            "Heat-labile materials (drugs, biomolecules, tissue-derived): extract at 37°C instead. "
            "Not needed if chem char + existing information adequately address pyrogenicity."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["CHEM_CHAR_FULL"],
        "can_parallelize_with": ["ISO_10993_4_HEMO_DIRECT", "ISO_10993_11_ACUTE"],
        "cost_low": 3000, "cost_high": 9000,
        "weeks_low": 3, "weeks_high": 6,
        "waivable_with_existing_data": True,
        "waiver_rationale": (
            "Not needed if chem char and prior information adequately address pyrogenicity "
            "for all patient-contacting components (FDA 6D)."
        ),
        "triggered_by": [],
        "waived_by": ["established_biocompatibility_data"],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "FDA 6D: heat-labile materials → 37°C extraction. No test distinguishes material vs. endotoxin.",
    },

    "ISO_10993_6_IMPLANT": {
        "name": "Implantation Study (ISO 10993-6)",
        "standard": "ISO 10993-6:2016, FDA Guidance Section 6E",
        "description": (
            "In vivo local tissue response in rabbit muscle, bone, or clinically relevant site. "
            "FDA 6E: if device geometry confounds interpretation, sub-components/coupons permitted "
            "with justification. High safety-risk implants (brain, vascular): clinically relevant "
            "site preferred. "
            "Absorbable/degradable: interim assessments required at multiple time points "
            "(pre-degradation; during degradation; at steady-state). "
            "Observation: 4-26 weeks depending on contact duration and degradation profile."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["ISO_10993_5", "ISO_10993_3_GENO"],
        "can_parallelize_with": [],
        "cost_low": 25000, "cost_high": 60000,
        "weeks_low": 12, "weeks_high": 32,
        "waivable_with_existing_data": True,
        "waiver_rationale": (
            "Same material, identical processing and sterilization, same or more demanding "
            "contact category. ISO 10993-18 equivalence must be demonstrated. "
            "Novel materials: waiver rarely accepted by FDA."
        ),
        "triggered_by": [],
        "waived_by": ["established_implant_data"],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "FDA 6E: absorbable materials → interim time-point assessments during degradation.",
    },

    "DEGRADATION_ASSESS": {
        "name": "Degradation / Absorption Assessment",
        "standard": "ISO 10993-13/14/15, FDA Guidance Sections 5B & 6I",
        "description": (
            "Required if: device is designed to be absorbed/resorbed; OR toxic degradation products "
            "may be released during contact. "
            "FDA 6I: in vivo degradation in appropriate animal model recommended. "
            "If adverse response: additional in vitro assessments to identify source. "
            "FDA STRONGLY recommends pre-test discussion with FDA before commencing (FDA 6I). "
            "In situ polymerizing materials (FDA 5B): evaluate pre-polymerized, polymerized, "
            "and degrading states separately. "
            "In vitro degradation methods may be used for test article preparation with justification."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["CHEM_CHAR_FULL", "ISO_10993_5"],
        "can_parallelize_with": ["ISO_10993_11_SUBACUTE"],
        "cost_low": 20000, "cost_high": 60000,
        "weeks_low": 12, "weeks_high": 40,
        "waivable_with_existing_data": False,
        "waiver_rationale": (
            "Cannot generally be waived for new absorbable formulations. Prior degradation data "
            "for identical polymer grade, MW, and end-cap chemistry may reduce scope."
        ),
        "triggered_by": [],
        "waived_by": [],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "FDA 6I: discuss with FDA before starting. In situ polymerizing: test all states separately.",
    },

    "NANO_CHAR": {
        "name": "Submicron / Nanotechnology Characterization",
        "standard": "FDA Guidance Section 5D, ISO/TR 10993-22",
        "description": (
            "Required for devices with submicron (<1μm) or nanotechnology components. "
            "Unique properties: aggregation, agglomeration, immunogenicity, unusual toxicity. "
            "FDA 5D: (a) careful nano-scale characterization of test article; "
            "(b) extraction conditions that avoid testing artefacts (nano particles behave "
            "differently in standard extraction solvents); "
            "(c) test article must be representative of clinical device. "
            "Specialized biocompatibility techniques may be warranted."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["RISK_ASSESSMENT"],
        "can_parallelize_with": ["CHEM_CHAR_FULL"],
        "cost_low": 15000, "cost_high": 45000,
        "weeks_low": 8, "weeks_high": 20,
        "waivable_with_existing_data": False,
        "waiver_rationale": None,
        "triggered_by": [],
        "waived_by": [],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "FDA 5D: standard ISO extraction methods may not be appropriate for nanomaterials.",
    },

    "REUSE_VALIDATION": {
        "name": "Reusable Device Processing Cycle Validation",
        "standard": "ISO 10993-1:2018 Section 4.8, AAMI ST79",
        "description": (
            "For reusable devices: biological safety evaluated for maximum validated processing cycles. "
            "Demonstrate repeated cleaning/disinfection/sterilization does not introduce new "
            "biocompatibility risks (surface degradation, cleaning agent residuals). "
            "ISO 10993-1 Section 4.9: re-evaluation required if processing changes occur."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["RISK_ASSESSMENT", "ISO_10993_5"],
        "can_parallelize_with": ["STERILITY_SAL"],
        "cost_low": 8000, "cost_high": 25000,
        "weeks_low": 6, "weeks_high": 16,
        "waivable_with_existing_data": False,
        "waiver_rationale": None,
        "triggered_by": [],
        "waived_by": [],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "ISO 4.8: biological safety for maximum validated processing cycles required.",
    },

    "STERILITY_SAL": {
        "name": "Sterilization Validation (SAL 10⁻⁶)",
        "standard": "ISO 11135 / ISO 11137 / AAMI ST67",
        "description": (
            "SAL 10⁻⁶ sterilization validation. EO (ISO 11135), gamma/e-beam (ISO 11137), "
            "steam (AAMI ST79). EO residuals must be included in ISO 10993-18 chem char."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["RISK_ASSESSMENT"],
        "can_parallelize_with": ["ISO_10993_5", "SHELF_LIFE", "MECHANICAL_PERF"],
        "cost_low": 15000, "cost_high": 40000,
        "weeks_low": 8, "weeks_high": 18,
        "waivable_with_existing_data": False,
        "waiver_rationale": None,
        "triggered_by": [],
        "waived_by": [],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "EO sterilization residuals must be included in ISO 10993-18 chem char (FDA Attachment B).",
    },

    "SHELF_LIFE": {
        "name": "Shelf Life / Accelerated Aging",
        "standard": "ASTM F1980, ISO 11607",
        "description": (
            "Sterile barrier integrity and device performance over claimed shelf life. "
            "Accelerated aging (Q10=2, ASTM F1980) may substitute initially with concurrent real-time study."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["STERILITY_SAL"],
        "can_parallelize_with": ["ISO_10993_5", "MECHANICAL_PERF"],
        "cost_low": 5000, "cost_high": 18000,
        "weeks_low": 8, "weeks_high": 26,
        "waivable_with_existing_data": False,
        "waiver_rationale": None,
        "triggered_by": [],
        "waived_by": [],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "Accelerated aging acceptable with concurrent real-time study.",
    },

    "MECHANICAL_PERF": {
        "name": "Mechanical Performance Testing",
        "standard": "Device-specific ASTM/ISO standards",
        "description": (
            "Tensile, fatigue, wear, compression, torque per device-specific standards. "
            "FDA 5C: mechanical failure risk and biological consequences must be assessed. "
            "If failure alters biological response (debris, surface change), incorporate into biocompat plan."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["RISK_ASSESSMENT"],
        "can_parallelize_with": ["ISO_10993_5", "STERILITY_SAL"],
        "cost_low": 10000, "cost_high": 45000,
        "weeks_low": 6, "weeks_high": 18,
        "waivable_with_existing_data": False,
        "waiver_rationale": None,
        "triggered_by": [],
        "waived_by": [],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "FDA 5C: mechanical failure biological consequences must be included in biocompat evaluation.",
    },

    "SOFTWARE_IEC62304": {
        "name": "Software Lifecycle Documentation (IEC 62304)",
        "standard": "IEC 62304:2006+AMD1:2015",
        "description": "SDLC documentation scaled to software safety class (A/B/C).",
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["RISK_ASSESSMENT"],
        "can_parallelize_with": ["ISO_10993_5", "MECHANICAL_PERF", "ELECTRICAL_SAFETY"],
        "cost_low": 15000, "cost_high": 60000,
        "weeks_low": 12, "weeks_high": 40,
        "waivable_with_existing_data": False,
        "waiver_rationale": None,
        "triggered_by": [],
        "waived_by": [],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": None,
    },

    "SOFTWARE_CYBER": {
        "name": "Cybersecurity Documentation",
        "standard": "FDA Cybersecurity Guidance (2023)",
        "description": "Threat modeling, SBOM, vulnerability disclosure, security testing. Required for networked devices.",
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["SOFTWARE_IEC62304"],
        "can_parallelize_with": [],
        "cost_low": 8000, "cost_high": 30000,
        "weeks_low": 6, "weeks_high": 16,
        "waivable_with_existing_data": False,
        "waiver_rationale": None,
        "triggered_by": [],
        "waived_by": [],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": None,
    },

    "ELECTRICAL_SAFETY": {
        "name": "Electrical Safety & EMC",
        "standard": "IEC 60601-1, IEC 60601-1-2",
        "description": "Leakage current, dielectric strength, grounding, EMC. Required for electrically powered patient-contact devices.",
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["RISK_ASSESSMENT"],
        "can_parallelize_with": ["ISO_10993_5", "SOFTWARE_IEC62304"],
        "cost_low": 20000, "cost_high": 55000,
        "weeks_low": 8, "weeks_high": 18,
        "waivable_with_existing_data": False,
        "waiver_rationale": None,
        "triggered_by": [],
        "waived_by": [],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": None,
    },

    "IVD_ANALYTICAL_PERF": {
        "name": "IVD Analytical Performance Testing",
        "standard": "FDA IVD Guidance, CLSI EP Standards",
        "description": (
            "IVD-specific: analytical sensitivity, specificity, accuracy, precision, "
            "reproducibility, interference, reference range. No biocompatibility required. "
            "Comparison to predicate performance or clinical reference standard."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["RISK_ASSESSMENT"],
        "can_parallelize_with": ["SUBMISSION_510K_PREP"],
        "cost_low": 15000, "cost_high": 60000,
        "weeks_low": 8, "weeks_high": 24,
        "waivable_with_existing_data": False,
        "waiver_rationale": None,
        "triggered_by": [],
        "waived_by": [],
        "applicable_pathways": [
            RegulatoryPathway.K510, RegulatoryPathway.DE_NOVO, RegulatoryPathway.PMA,
        ],
        "fda_notes": "IVD devices: analytical performance replaces biocompatibility as primary test track.",
    },

    # ---- CBER / Cell-Gene Therapy ----
    "CBER_CMC_PACKAGE": {
        "name": "CMC Package — Cell/Gene Therapy (CBER)",
        "standard": "FDA CBER CMC Guidance, ICH Q5A-Q5E",
        "description": (
            "Cell identity, potency (functional), purity, sterility (USP <71>), mycoplasma (USP <63>), "
            "adventitious agents, stability, manufacturing process characterization. "
            "Gene-edited products: off-target edit analysis required."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["RISK_ASSESSMENT"],
        "can_parallelize_with": ["CBER_PRECLINICAL"],
        "cost_low": 200000, "cost_high": 1500000,
        "weeks_low": 26, "weeks_high": 104,
        "waivable_with_existing_data": False,
        "waiver_rationale": None,
        "triggered_by": [],
        "waived_by": [],
        "applicable_pathways": [RegulatoryPathway.IND, RegulatoryPathway.BLA],
        "fda_notes": "CBER: potency must be functional. Off-target edits required for CRISPR products.",
    },

    "CBER_PRECLINICAL": {
        "name": "Preclinical Safety & Efficacy (CBER)",
        "standard": "FDA CBER Guidance, ICH S6, ICH S9",
        "description": (
            "In vitro safety + in vivo toxicology + proof-of-concept efficacy. "
            "Biodistribution (gene/vector products), genotoxicity/insertional mutagenesis (viral vectors), "
            "tumorigenicity (if undifferentiated stem cells)."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["CBER_CMC_PACKAGE"],
        "can_parallelize_with": [],
        "cost_low": 300000, "cost_high": 3000000,
        "weeks_low": 52, "weeks_high": 156,
        "waivable_with_existing_data": False,
        "waiver_rationale": None,
        "triggered_by": [],
        "waived_by": [],
        "applicable_pathways": [RegulatoryPathway.IND, RegulatoryPathway.BLA],
        "fda_notes": "Biodistribution + insertional mutagenesis required for viral vector gene therapy.",
    },

    "CBER_IND_PREP": {
        "name": "IND Application (CBER)",
        "standard": "21 CFR Part 312",
        "description": (
            "IND required before any clinical investigation of cell/gene therapy. "
            "Includes: preclinical data package, CMC, Phase I protocol with safety monitoring."
        ),
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["CBER_PRECLINICAL"],
        "can_parallelize_with": [],
        "cost_low": 150000, "cost_high": 600000,
        "weeks_low": 26, "weeks_high": 78,
        "waivable_with_existing_data": False,
        "waiver_rationale": None,
        "triggered_by": [],
        "waived_by": [],
        "applicable_pathways": [RegulatoryPathway.IND, RegulatoryPathway.BLA],
        "fda_notes": "CBER: IND is mandatory entry point before any clinical investigation.",
    },

    "CLINICAL_STUDY": {
        "name": "Clinical Study / IDE Application",
        "standard": "21 CFR Part 812, ICH E6 GCP",
        "description": "Pivotal clinical study. Significant Risk → IDE required first.",
        "phase": TestPhase.PRE_SUBMISSION,
        "prerequisites": ["ISO_10993_5", "ISO_10993_6_IMPLANT", "MECHANICAL_PERF"],
        "can_parallelize_with": ["STERILITY_SAL"],
        "cost_low": 500000, "cost_high": 10000000,
        "weeks_low": 52, "weeks_high": 312,
        "waivable_with_existing_data": False,
        "waiver_rationale": (
            "PMA: clinical data required. De Novo: may be deferred post-clearance with "
            "post-market clinical follow-up as a special control."
        ),
        "triggered_by": [],
        "waived_by": [],
        "applicable_pathways": [RegulatoryPathway.PMA, RegulatoryPathway.DE_NOVO],
        "fda_notes": None,
    },

    "SUBMISSION_510K_PREP": {
        "name": "510(k) Submission Preparation",
        "standard": "21 CFR 807 Subpart E",
        "description": (
            "Summary/Statement, device description, predicate comparison, performance testing summary, "
            "labeling, biocompatibility section (risk assessment first per FDA Sec 3D)."
        ),
        "phase": TestPhase.SUBMISSION,
        "prerequisites": ["RISK_ASSESSMENT", "ISO_10993_5", "STERILITY_SAL", "MECHANICAL_PERF"],
        "can_parallelize_with": [],
        "cost_low": 20000, "cost_high": 80000,
        "weeks_low": 8, "weeks_high": 20,
        "waivable_with_existing_data": False,
        "waiver_rationale": None,
        "triggered_by": [],
        "waived_by": [],
        "applicable_pathways": [RegulatoryPathway.K510],
        "fda_notes": None,
    },

    "SUBMISSION_PMA_PREP": {
        "name": "PMA Application Preparation",
        "standard": "21 CFR 814",
        "description": "Full PMA: clinical, non-clinical, manufacturing, and labeling modules.",
        "phase": TestPhase.SUBMISSION,
        "prerequisites": ["CLINICAL_STUDY", "ISO_10993_6_IMPLANT", "STERILITY_SAL"],
        "can_parallelize_with": [],
        "cost_low": 100000, "cost_high": 500000,
        "weeks_low": 20, "weeks_high": 52,
        "waivable_with_existing_data": False,
        "waiver_rationale": None,
        "triggered_by": [],
        "waived_by": [],
        "applicable_pathways": [RegulatoryPathway.PMA],
        "fda_notes": None,
    },

    "SUBMISSION_IND_BLA_PREP": {
        "name": "IND → BLA Preparation (CBER)",
        "standard": "21 CFR Part 312 (IND), 21 CFR Part 601 (BLA)",
        "description": "Full biologics development: IND → Phase I/II/III → BLA.",
        "phase": TestPhase.SUBMISSION,
        "prerequisites": ["CBER_IND_PREP", "CBER_PRECLINICAL"],
        "can_parallelize_with": [],
        "cost_low": 500000, "cost_high": 5000000,
        "weeks_low": 52, "weeks_high": 520,
        "waivable_with_existing_data": False,
        "waiver_rationale": None,
        "triggered_by": [],
        "waived_by": [],
        "applicable_pathways": [RegulatoryPathway.BLA, RegulatoryPathway.IND],
        "fda_notes": None,
    },
}


# ===========================================================================
# Materials — waiver eligibility sets
# ===========================================================================

ESTABLISHED_BIOCOMPATIBLE_MATERIALS = {
    "peek", "polyether ether ketone",
    "ptfe", "polytetrafluoroethylene", "teflon",
    "uhmwpe", "ultra-high-molecular-weight polyethylene",
    "titanium", "ti-6al-4v", "ti-6al-4v eli", "cp titanium",
    "316l stainless steel", "stainless steel 316l",
    "cobalt chrome", "cobalt-chromium", "cocr",
    "medical grade silicone", "silicone", "pdms",
    "polyurethane", "polysulfone",
}

ESTABLISHED_IMPLANT_DATA_MATERIALS = {
    "peek", "polyether ether ketone",
    "titanium", "ti-6al-4v eli",
    "uhmwpe", "cobalt chrome",
}

USP_CLASS_VI_MATERIALS = {
    "medical grade silicone", "silicone",
    "ptfe", "polytetrafluoroethylene",
    "polysulfone", "polyurethane",
}

ABSORBABLE_KEYWORDS = {
    "plga", "pla", "pga", "polylactic", "polyglycolide", "polyglycolic",
    "resorbable", "absorbable", "biodegradable", "bioresorbable",
    "polycaprolactone", "pcl", "phb",
}

NANO_KEYWORDS = {
    "nano", "nanoparticle", "nanomaterial", "quantum dot",
    "submicron", "nanocomposite", "nanotube", "nanofiber",
}

NOVEL_KEYWORDS = {
    "novel", "new material", "first-in-class", "proprietary material",
    "custom formulation", "experimental material", "investigational material",
}


# ===========================================================================
# Contact category → matrix key resolution
# ===========================================================================

def _get_matrix_keys(
    contact_category: ContactCategory,
    contact_duration: ContactDuration,
    intended_use: str,
    is_implantable: bool,
) -> list[tuple[str, str]]:
    """
    Map device contact profile to one or more (matrix_key, duration_key) tuples.
    Complex devices may span multiple categories (FDA Section 4C — pacemaker example).
    """
    dur = DURATION_KEY.get(contact_duration, "limited")
    intended_lower = intended_use.lower()

    # Blood-contacting implants get both implant_blood and implant_tissue
    if is_implantable and any(w in intended_lower for w in [
        "blood", "vascular", "cardiac", "heart", "coronary", "intravascular",
        "aortic", "venous", "arterial", "valv", "stent",
    ]):
        return [("implant_blood", dur), ("implant_tissue", dur)]

    if is_implantable or contact_category == ContactCategory.IMPLANT:
        return [("implant_tissue", dur)]

    if any(w in intended_lower for w in [
        "circulating blood", "extracorporeal", "dialysis", "heart-lung", "apheresis",
    ]):
        return [("circulating_blood", dur)]

    if contact_category == ContactCategory.EXTERNAL_COMMUNICATING and any(w in intended_lower for w in [
        "infusion", "iv set", "iv tubing", "blood path", "fluid path",
    ]):
        return [("blood_path_indirect", dur)]

    if contact_category == ContactCategory.EXTERNAL_COMMUNICATING:
        return [("external_communicating", dur)]

    if contact_category == ContactCategory.SURFACE:
        return [("surface", dur)]

    return [("surface", dur)]


# ===========================================================================
# Device flag derivation
# ===========================================================================

def _get_device_flags(classification: ClassificationResult) -> dict[str, bool]:
    profile = classification.product_profile
    intended_lower = profile.intended_use.lower()
    desc_lower = profile.raw_description.lower()
    materials_text = " ".join(profile.materials).lower() + " " + desc_lower

    return {
        "is_sterile": "non-sterile" not in desc_lower,
        "is_reusable": any(w in desc_lower for w in ["reusable", "reuse", "reprocessed", "resterilized"]),
        "has_software": profile.has_software_component,
        "is_electrical": profile.mechanism_of_action.value == "electrical",
        "needs_clinical": classification.regulatory_pathway == RegulatoryPathway.PMA,
        "is_510k": classification.regulatory_pathway in (RegulatoryPathway.K510, RegulatoryPathway.EXEMPT),
        "is_ivd": getattr(classification, "product_category", None) is not None
                  and classification.product_category.value == "diagnostic_ivd",
        "is_cber": classification.lead_center == FDALeadCenter.CBER,
        "has_absorbable": bool(kw for kw in ABSORBABLE_KEYWORDS if kw in materials_text),
        "has_nano": bool(kw for kw in NANO_KEYWORDS if kw in materials_text),
        "has_novel_material": bool(kw for kw in NOVEL_KEYWORDS if kw in materials_text),
        "has_in_situ_polymerizing": any(w in desc_lower for w in [
            "in situ polymerizing", "polymerizes in situ", "cures in vivo",
        ]),
        "is_networked": any(w in desc_lower for w in [
            "bluetooth", "wifi", "wireless", "networked", "connected", "iot", "cloud",
        ]),
        "is_extracorporeal_blood": any(w in intended_lower for w in [
            "extracorporeal", "dialysis", "heart-lung", "apheresis",
        ]),
    }


def _get_active_waivers(classification: ClassificationResult, flags: dict[str, bool]) -> set[str]:
    profile = classification.product_profile
    waivers: set[str] = set()
    normalised = {m.lower().strip() for m in profile.materials}

    if normalised & ESTABLISHED_BIOCOMPATIBLE_MATERIALS:
        waivers.add("established_biocompatibility_data")
    if normalised & ESTABLISHED_IMPLANT_DATA_MATERIALS:
        waivers.add("established_implant_data")
    if normalised & USP_CLASS_VI_MATERIALS:
        waivers.add("usp_class_vi")

    # Novel material voids all waivers (FDA Sec 6B, 6F — new testing required)
    if flags.get("has_novel_material"):
        waivers.clear()

    return waivers


# ===========================================================================
# Test selection engine
# ===========================================================================

def _select_test_ids(
    classification: ClassificationResult,
    flags: dict[str, bool],
) -> list[str]:
    profile = classification.product_profile
    pathway = classification.regulatory_pathway
    required: set[str] = {"RISK_ASSESSMENT"}

    # CBER track is entirely separate
    if flags["is_cber"] or pathway in (RegulatoryPathway.IND, RegulatoryPathway.BLA):
        required.update(["CBER_CMC_PACKAGE", "CBER_PRECLINICAL", "CBER_IND_PREP", "SUBMISSION_IND_BLA_PREP"])
        return sorted(required)

    # IVD: analytical performance replaces biocompatibility
    if flags["is_ivd"]:
        required.add("IVD_ANALYTICAL_PERF")
        if flags["is_510k"]:
            required.add("SUBMISSION_510K_PREP")
        return sorted(required)

    # Contact matrix — may produce multiple slots for complex devices
    contact_slots = _get_matrix_keys(
        contact_category=profile.contact_category,
        contact_duration=profile.contact_duration,
        intended_use=profile.intended_use,
        is_implantable=profile.is_implantable,
    )
    for matrix_key, duration_key in contact_slots:
        required.update(ENDPOINT_MATRIX.get(matrix_key, {}).get(duration_key, set()))

    # Additional flag-driven tests
    if flags["is_sterile"]:
        required.update(["STERILITY_SAL", "SHELF_LIFE"])
    if flags["is_reusable"]:
        required.add("REUSE_VALIDATION")
    if flags["has_software"]:
        required.add("SOFTWARE_IEC62304")
        if flags["is_networked"]:
            required.add("SOFTWARE_CYBER")
    if flags["is_electrical"]:
        required.add("ELECTRICAL_SAFETY")
    if flags["has_absorbable"] or flags["has_in_situ_polymerizing"]:
        required.add("DEGRADATION_ASSESS")
    if flags["has_nano"]:
        required.add("NANO_CHAR")
    if not flags["is_ivd"]:
        required.add("MECHANICAL_PERF")

    # Promote to full chem char if screening is present alongside EC/implant tests
    if "CHEM_CHAR_FULL" in required:
        required.discard("CHEM_CHAR_SCREENING")

    # Submission document
    if pathway == RegulatoryPathway.K510:
        required.add("SUBMISSION_510K_PREP")
    elif pathway == RegulatoryPathway.PMA:
        required.update(["CLINICAL_STUDY", "SUBMISSION_PMA_PREP"])

    # Only return IDs that exist in the library
    valid = {tid for tid in required if tid in MASTER_TEST_LIBRARY}
    if unknown := required - valid:
        logger.warning("Unknown test IDs skipped: %s", unknown)
    return sorted(valid)


def _build_test_nodes(
    test_ids: list[str],
    waivers: set[str],
    flags: dict[str, bool],
) -> list[TestNode]:
    nodes: list[TestNode] = []
    active_ids = set(test_ids)

    for test_id in test_ids:
        if test_id not in MASTER_TEST_LIBRARY:
            continue
        spec = MASTER_TEST_LIBRARY[test_id]
        active_prereqs = [p for p in spec["prerequisites"] if p in active_ids]

        waivable = spec["waivable_with_existing_data"]
        waiver_rationale = spec.get("waiver_rationale")

        if flags.get("has_novel_material") and waivable:
            waivable = False
            waiver_rationale = (
                "WAIVER NOT AVAILABLE: Novel material detected. FDA requires new testing data. "
                "Existing data waivers require identical material grade, processing, and sterilization."
            )
        elif waivable and spec["waived_by"] and (set(spec["waived_by"]) & waivers):
            waiver_rationale = (
                f"POTENTIALLY WAIVABLE: {waiver_rationale} "
                "Provide written justification in the biocompatibility risk assessment section."
            )
        else:
            waivable = False

        nodes.append(TestNode(
            id=test_id,
            name=spec["name"],
            standard=spec["standard"],
            description=spec["description"],
            phase=spec["phase"],
            prerequisites=active_prereqs,
            can_parallelize_with=[p for p in spec["can_parallelize_with"] if p in active_ids],
            estimated_cost_usd_low=spec["cost_low"],
            estimated_cost_usd_high=spec["cost_high"],
            estimated_weeks_low=spec["weeks_low"],
            estimated_weeks_high=spec["weeks_high"],
            waivable_with_existing_data=waivable,
            waiver_rationale=waiver_rationale,
            triggered_by=spec["triggered_by"],
            waived_by=spec["waived_by"],
            applicable_pathways=spec["applicable_pathways"],
            notes=spec.get("fda_notes", "") or "",
        ))

    return nodes


# ===========================================================================
# Critical path, parallelisation, cost rollup
# ===========================================================================

def _compute_critical_path(nodes: list[TestNode]) -> list[str]:
    node_map = {n.id: n for n in nodes}
    active_ids = set(node_map)
    in_edges = {n.id: set(n.prerequisites) & active_ids for n in nodes}
    out_edges: dict[str, set[str]] = defaultdict(set)
    for nid, prereqs in in_edges.items():
        for p in prereqs:
            out_edges[p].add(nid)

    in_degree = {nid: len(prereqs) for nid, prereqs in in_edges.items()}
    queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
    topo_order: list[str] = []
    while queue:
        nid = queue.popleft()
        topo_order.append(nid)
        for s in out_edges[nid]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    eft: dict[str, int] = {}
    for nid in topo_order:
        node = node_map[nid]
        prereq_eft = max((eft.get(p, 0) for p in in_edges[nid]), default=0)
        eft[nid] = prereq_eft + node.estimated_weeks_high

    if not eft:
        return []

    current = max(eft, key=lambda k: eft[k])
    path = [current]
    while True:
        prereqs = [p for p in in_edges.get(current, set()) if p in eft]
        if not prereqs:
            break
        current = max(prereqs, key=lambda k: eft[k])
        path.insert(0, current)
    return path


def _find_parallelization_groups(nodes: list[TestNode]) -> list[list[str]]:
    active_ids = {n.id for n in nodes}
    node_map = {n.id: n for n in nodes}
    groups: list[list[str]] = []
    visited: set[str] = set()

    for node in nodes:
        if node.id in visited:
            continue
        group = {node.id}
        for other_id in node.can_parallelize_with:
            if other_id in active_ids and other_id not in visited:
                other = node_map[other_id]
                if node.id not in other.prerequisites and other_id not in node.prerequisites:
                    group.add(other_id)
        if len(group) > 1:
            groups.append(sorted(group))
            visited.update(group)
    return groups


def _rollup(
    nodes: list[TestNode],
    critical_path: list[str],
    node_map: dict[str, TestNode],
) -> tuple[int, int, int, int]:
    cost_low = sum(n.estimated_cost_usd_low for n in nodes if not n.waivable_with_existing_data)
    cost_high = sum(n.estimated_cost_usd_high for n in nodes if not n.waivable_with_existing_data)
    weeks_low = sum(node_map[nid].estimated_weeks_low for nid in critical_path if nid in node_map)
    weeks_high = sum(node_map[nid].estimated_weeks_high for nid in critical_path if nid in node_map)
    return cost_low, cost_high, weeks_low, weeks_high


# ===========================================================================
# Data gap analysis
# ===========================================================================

DATA_GAP_PROMPT = """
You are a regulatory affairs specialist (ISO 10993-1:2018 + FDA 2023 guidance).
Write a concise 4-6 sentence plain-English analysis covering:
1. Which tests are potentially waivable and what evidence is needed
2. Which tests require new in vitro or in vivo studies
3. The single most time-critical test to initiate first (and why)
4. Any FDA-specific procedure notes the team must know (GPMT vs LLNA, hemocompatibility
   track, pyrogenicity method, genotoxicity battery scope)
5. If absorbable materials: flag the required pre-test FDA discussion (Section 6I)

Reference specific FDA guidance sections (e.g., "FDA Section 6B"). Be direct and specific.
"""


def _generate_data_gap_analysis(nodes: list[TestNode], classification: ClassificationResult) -> str:
    profile = classification.product_profile
    waivable = [n.name for n in nodes if n.waivable_with_existing_data]
    required = [n.name for n in nodes if not n.waivable_with_existing_data]

    text = (
        f"Device: {profile.intended_use} | Contact: {profile.contact_category}/{profile.contact_duration}\n"
        f"Materials: {', '.join(profile.materials) or 'not specified'}\n"
        f"Pathway: {classification.regulatory_pathway} | Center: {classification.lead_center}\n"
        f"Required new studies ({len(required)}): {', '.join(required)}\n"
        f"Potentially waivable ({len(waivable)}): {', '.join(waivable) or 'none'}\n"
    )
    try:
        return call_llm(system_prompt=DATA_GAP_PROMPT, user_message=text)
    except Exception as e:
        logger.warning("Data gap LLM failed: %s", e)
        return (
            f"{len(required)} required new studies; {len(waivable)} potentially waivable. "
            "Initiate risk assessment and chemical characterization first — "
            "they gate all subsequent biocompatibility decisions."
        )


# ===========================================================================
# Public interface
# ===========================================================================

def generate_roadmap(classification: ClassificationResult) -> RoadmapResult:
    logger.info(
        "Generating roadmap: pathway=%s, category=%s, lead_center=%s",
        classification.regulatory_pathway,
        classification.product_category,
        classification.lead_center,
    )

    flags = _get_device_flags(classification)
    waivers = _get_active_waivers(classification, flags)

    logger.info("Flags: %s", {k: v for k, v in flags.items() if v})
    logger.info("Waivers: %s", waivers)

    test_ids = _select_test_ids(classification, flags)
    logger.info("Selected %d tests", len(test_ids))

    nodes = _build_test_nodes(test_ids, waivers, flags)
    critical_path = _compute_critical_path(nodes)
    parallel_groups = _find_parallelization_groups(nodes)

    node_map = {n.id: n for n in nodes}
    cost_low, cost_high, weeks_low, weeks_high = _rollup(nodes, critical_path, node_map)
    data_gap = _generate_data_gap_analysis(nodes, classification)

    return RoadmapResult(
        classification=classification,
        tests=nodes,
        total_cost_usd_low=cost_low,
        total_cost_usd_high=cost_high,
        total_weeks_low=weeks_low,
        total_weeks_high=weeks_high,
        critical_path=critical_path,
        parallelization_opportunities=parallel_groups,
        data_gap_analysis=data_gap,
    )
