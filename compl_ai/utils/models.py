"""
Shared Pydantic models used across all four backend systems.
These form the contracts between systems — don't change a field
without updating every consumer.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MechanismOfAction(str, Enum):
    MECHANICAL = "mechanical"
    CHEMICAL = "chemical"
    BIOLOGICAL = "biological"
    ELECTRICAL = "electrical"
    SOFTWARE = "software"
    COMBINATION = "combination"
    UNKNOWN = "unknown"


class ProductCategory(str, Enum):
    """
    Top-level classification bucket — determined before device class.
    This is the primary routing decision the engine makes.
    """
    MEDICAL_DEVICE = "medical_device"          # Physical object; mechanical/electrical MOA → CDRH
    CELL_GENE_THERAPY = "cell_gene_therapy"    # Living cells, gene editing, tissue engineering → CBER
    DIAGNOSTIC_IVD = "diagnostic_ivd"          # Testing outside the body → CDRH (or CBER for some)
    DIAGNOSTIC_INVIVO = "diagnostic_invivo"    # Testing on/in the body → CDRH + biocompatibility
    DRUG = "drug"                              # Chemical MOA, systemic effect → CDER
    COMBINATION = "combination"                # Multiple categories; primary MOA determines lead center
    UNKNOWN = "unknown"


class FDALeadCenter(str, Enum):
    """
    The FDA center with primary jurisdiction.
    For combination products, this is determined by primary mode of action (21 CFR 3.4).
    """
    CDRH = "CDRH"    # Center for Devices and Radiological Health
    CBER = "CBER"    # Center for Biologics Evaluation and Research
    CDER = "CDER"    # Center for Drug Evaluation and Research
    UNKNOWN = "Unknown"


class ContactCategory(str, Enum):
    NONE = "none"                          # No body contact
    SURFACE = "surface"                    # Intact skin / mucous membrane
    EXTERNAL_COMMUNICATING = "external_communicating"  # Blood path indirect, tissue/bone/dentin, circulating blood
    IMPLANT = "implant"                    # Tissue/bone, blood


class ContactDuration(str, Enum):
    LIMITED = "limited"          # < 24 hours
    PROLONGED = "prolonged"      # 24 hours – 30 days
    PERMANENT = "permanent"      # > 30 days


class DeviceClass(str, Enum):
    CLASS_I = "Class I"
    CLASS_II = "Class II"
    CLASS_III = "Class III"
    UNKNOWN = "Unknown"


class RegulatoryPathway(str, Enum):
    # CDRH pathways
    EXEMPT = "510(k) Exempt"
    K510 = "510(k)"
    DE_NOVO = "De Novo"
    PMA = "PMA"
    IDE = "IDE"
    HDE = "HDE"                              # Humanitarian Device Exemption

    # CBER pathways
    IND = "IND"                              # Investigational New Drug (cell/gene therapy Phase I entry)
    BLA = "BLA"                              # Biologics License Application (CBER approval)

    # Cross-center
    COMBINATION_PRODUCT = "Combination Product"  # Multi-center; primary MOA governs

    UNKNOWN = "Unknown"


class SoftwareSafetyClass(str, Enum):
    CLASS_A = "Class A"   # No injury or damage to health possible
    CLASS_B = "Class B"   # Non-serious injury possible
    CLASS_C = "Class C"   # Death or serious injury possible
    NOT_APPLICABLE = "N/A"


class PatentRelevance(str, Enum):
    GREEN = "green"    # Expired or clearly non-overlapping
    YELLOW = "yellow"  # Possible overlap — review recommended
    RED = "red"        # Active patent, high claim similarity


class TestPhase(str, Enum):
    PRE_SUBMISSION = "pre_submission"
    SUBMISSION = "submission"
    POST_MARKET = "post_market"


# ---------------------------------------------------------------------------
# System 1 — Extraction & Classification
# ---------------------------------------------------------------------------

class ProductProfile(BaseModel):
    """Structured extraction of the user's plain-language description."""
    raw_description: str

    # ------------------------------------------------------------------
    # Primary routing fields (new)
    # ------------------------------------------------------------------
    product_category: ProductCategory = ProductCategory.UNKNOWN
    """
    Top-level bucket before device class is applied.
    Determines lead FDA center and which classification tree to enter.
    """

    is_therapeutic: bool = False
    """True if the product's primary purpose is to treat, repair, or replace a biological function."""

    is_diagnostic: bool = False
    """True if the product's primary purpose is to detect, measure, or identify something biological."""

    diagnostic_location: Optional[str] = None
    """'in_vitro' (sample outside body) or 'in_vivo' (on/in the body). Only set when is_diagnostic=True."""

    # Cell / gene therapy signals
    contains_living_cells: bool = False
    """True if the product contains or IS living cells (autologous, allogeneic, xenogeneic)."""

    contains_gene_editing: bool = False
    """True if product involves CRISPR, viral vectors, siRNA, mRNA, or any genetic modification."""

    contains_tissue_engineering: bool = False
    """True if product involves engineered biological scaffolds with cellular components."""

    is_biological_graft: bool = False
    """True if product is derived from human/animal tissue (decellularized matrix, etc.)."""

    primary_mode_of_action: str = ""
    """
    For combination products: plain-English description of which component
    drives the primary therapeutic/diagnostic effect. Used to assign lead center.
    """

    # ------------------------------------------------------------------
    # Existing fields (unchanged)
    # ------------------------------------------------------------------
    mechanism_of_action: MechanismOfAction = MechanismOfAction.UNKNOWN
    intended_use: str = ""
    indication: str = ""

    contact_category: ContactCategory = ContactCategory.NONE
    contact_duration: ContactDuration = ContactDuration.LIMITED

    materials: list[str] = Field(default_factory=list)

    has_drug_component: bool = False
    has_biologic_component: bool = False
    has_software_component: bool = False
    is_implantable: bool = False
    is_combination_product: bool = False

    extraction_notes: str = ""  # Anything the extractor flagged as ambiguous


class ClassificationResult(BaseModel):
    """Output of the Classification Engine."""
    product_profile: ProductProfile

    # ------------------------------------------------------------------
    # Primary routing result (new)
    # ------------------------------------------------------------------
    product_category: ProductCategory = ProductCategory.UNKNOWN
    lead_center: FDALeadCenter = FDALeadCenter.UNKNOWN
    """The FDA center with primary jurisdiction."""

    combination_product_components: list[str] = Field(default_factory=list)
    """
    For combination products: list each component and its own required clearance.
    e.g. ['Device component: 510(k) clearance required', 'Drug component: NDA/ANDA required']
    This is the 'most expensive pathway' detail — each piece must be cleared independently.
    """

    # ------------------------------------------------------------------
    # Existing fields (unchanged)
    # ------------------------------------------------------------------
    device_class: DeviceClass = DeviceClass.UNKNOWN
    regulatory_pathway: RegulatoryPathway = RegulatoryPathway.UNKNOWN
    product_code: Optional[str] = None
    regulation_number: Optional[str] = None

    software_safety_class: SoftwareSafetyClass = SoftwareSafetyClass.NOT_APPLICABLE

    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    low_confidence_warning: Optional[str] = None

    predicate_devices: list[PredicateDevice] = Field(default_factory=list)
    classification_rationale: str = ""


class PredicateDevice(BaseModel):
    k_number: str          # e.g. K213456
    device_name: str
    applicant: str
    decision_date: str
    similarity_score: float


# ---------------------------------------------------------------------------
# System 2 — Testing Roadmap
# ---------------------------------------------------------------------------

class TestNode(BaseModel):
    """A single test or study in the testing graph."""
    id: str                          # Unique stable ID, e.g. "ISO_10993_5"
    name: str
    standard: str                    # e.g. "ISO 10993-5:2009"
    description: str

    phase: TestPhase
    prerequisites: list[str] = Field(default_factory=list)   # IDs of blocking tests
    can_parallelize_with: list[str] = Field(default_factory=list)

    estimated_cost_usd_low: int
    estimated_cost_usd_high: int
    estimated_weeks_low: int
    estimated_weeks_high: int

    # Whether existing material data may satisfy this requirement
    waivable_with_existing_data: bool = False
    waiver_rationale: Optional[str] = None

    # Which materials or design features trigger or waive this test
    triggered_by: list[str] = Field(default_factory=list)
    waived_by: list[str] = Field(default_factory=list)

    applicable_pathways: list[RegulatoryPathway] = Field(default_factory=list)
    notes: str = ""


class RoadmapResult(BaseModel):
    """Output of the Testing Roadmap Generator."""
    classification: ClassificationResult
    tests: list[TestNode]

    total_cost_usd_low: int
    total_cost_usd_high: int
    total_weeks_low: int
    total_weeks_high: int

    critical_path: list[str]   # Ordered list of test IDs on the critical path
    parallelization_opportunities: list[list[str]]  # Groups that can run simultaneously

    data_gap_analysis: str     # Plain-English summary of what needs new testing vs. existing data


# ---------------------------------------------------------------------------
# System 3 — IP Radar
# ---------------------------------------------------------------------------

class PatentResult(BaseModel):
    patent_number: str
    title: str
    abstract: str
    assignee: str
    filing_date: str
    expiration_date: Optional[str]
    is_active: bool

    relevance: PatentRelevance
    relevance_explanation: str     # Plain-English LLM-generated explanation
    concerning_claims: list[str]   # Specific claim language of concern


class IPRadarResult(BaseModel):
    product_profile: ProductProfile
    patents: list[PatentResult]
    search_queries_used: list[str]
    disclaimer: str = (
        "This output is generated by an AI system and does not constitute "
        "legal advice or a Freedom-to-Operate opinion. Consult a registered "
        "patent attorney before making IP-related business decisions."
    )
    summary: str   # Plain-English paragraph summarizing the IP landscape


# ---------------------------------------------------------------------------
# System 4 — Smart Materials Engine
# ---------------------------------------------------------------------------

class MaterialProfile(BaseModel):
    name: str
    common_grades: list[str] = Field(default_factory=list)
    biocompatibility_endpoints_established: list[str] = Field(default_factory=list)
    generally_recognized_safe_for: list[ContactCategory] = Field(default_factory=list)
    mri_compatible: Optional[bool] = None
    notes: str = ""


class MaterialSwapRecommendation(BaseModel):
    original_material: str
    suggested_material: str
    tests_eliminated: list[TestNode]
    tests_added: list[TestNode]
    net_weeks_saved_low: int
    net_weeks_saved_high: int
    net_cost_saved_usd_low: int
    net_cost_saved_usd_high: int
    predicate_impact: Optional[str]   # If this change affects 510(k) predicate matching
    rationale: str


class MaterialsOptimizationResult(BaseModel):
    baseline_roadmap: RoadmapResult
    recommendations: list[MaterialSwapRecommendation]
    best_recommendation: Optional[MaterialSwapRecommendation]
    summary: str
