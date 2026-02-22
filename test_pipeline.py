"""
Test Suite
===========
Integration tests for all four systems using realistic device descriptions.

Run with: python tests/test_pipeline.py

These are integration tests — they call the real LLM and real APIs.
For unit tests, mock the LLM client and API calls using unittest.mock.

NEXT STEPS:
  - Add pytest fixtures with mocked LLM responses to enable fast offline testing.
  - Add golden-file tests: save known-good classification results and assert
    that future runs match them (catches prompt regressions).
  - Add adversarial test cases: edge cases like combination products, IVDs,
    software-only devices, and Class III implants.
  - Add a load test to verify the parallel execution in the pipeline doesn't
    cause race conditions under concurrent requests.
"""

from __future__ import annotations

import json
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import run_full_pipeline
from systems.classification_engine import classify_device
from systems.roadmap_generator import generate_roadmap
from utils.models import DeviceClass, FDALeadCenter, ProductCategory, RegulatoryPathway


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "name": "Resorbable bone screw (medical device)",
        "description": (
            "A fully resorbable bone screw made from PLGA (poly-lactic-co-glycolic acid) "
            "polymer for fixation of small bone fractures in the hand and wrist. "
            "The screw degrades over 18-24 months, eliminating the need for hardware "
            "removal surgery. Sterilized by gamma irradiation. No drug or biologic components."
        ),
        "expected_class": DeviceClass.CLASS_II,
        "expected_pathway": RegulatoryPathway.K510,
    },
    {
        "name": "AI-powered ECG analysis software (medical device / SaMD)",
        "description": (
            "A cloud-based software platform that uses machine learning to analyze "
            "12-lead ECG waveforms and detect atrial fibrillation in real time. "
            "The software is intended for use in clinical settings and will alert "
            "clinicians to potential AF episodes. No hardware component — software only. "
            "Connects to existing ECG hardware via HL7/FHIR API."
        ),
        "expected_class": DeviceClass.CLASS_II,
        "expected_pathway": RegulatoryPathway.K510,
    },
    {
        "name": "Drug-eluting coronary stent (combination product)",
        "description": (
            "A cobalt-chromium coronary stent with a biodegradable polymer coating "
            "that elutes sirolimus (an immunosuppressant drug) over 90 days to prevent "
            "restenosis. Delivered via catheter to the coronary artery. "
            "Permanent implant. The drug is the primary mode of action for preventing restenosis."
        ),
        "expected_class": DeviceClass.UNKNOWN,
        "expected_pathway": RegulatoryPathway.COMBINATION_PRODUCT,
    },
    {
        "name": "PEEK spinal fusion cage (medical device)",
        "description": (
            "An interbody fusion device made from medical-grade PEEK (Victrex PEEK-OPTIMA) "
            "for lumbar spinal fusion procedures. The cage is packed with autologous bone graft "
            "and inserted between vertebral bodies to restore disc height and promote fusion. "
            "Available in multiple sizes. EO sterilized. No drug or biologic components — "
            "bone graft is surgeon-supplied."
        ),
        "expected_class": DeviceClass.CLASS_II,
        "expected_pathway": RegulatoryPathway.K510,
    },
    # ---- NEW: Cell / gene therapy ----
    {
        "name": "CAR-T cell therapy (cell/gene therapy → CBER/IND)",
        "description": (
            "An autologous CAR-T cell therapy for relapsed/refractory B-cell lymphoma. "
            "Patient T-cells are harvested, genetically engineered via lentiviral vector "
            "to express a CD19-targeting chimeric antigen receptor, expanded ex vivo, "
            "and re-infused. Living cells are the therapeutic product. "
            "No device component — the cells themselves are the therapy."
        ),
        "expected_class": DeviceClass.UNKNOWN,
        "expected_pathway": RegulatoryPathway.IND,
    },
    {
        "name": "CRISPR gene-edited stem cells (cell/gene therapy → CBER/IND)",
        "description": (
            "A therapeutic product consisting of hematopoietic stem cells from a healthy donor, "
            "edited using CRISPR-Cas9 to correct a point mutation in the HBB gene causing "
            "sickle cell disease. Cells are expanded and cryopreserved before infusion. "
            "Gene editing is the primary mechanism of action."
        ),
        "expected_class": DeviceClass.UNKNOWN,
        "expected_pathway": RegulatoryPathway.IND,
    },
    {
        "name": "Lateral flow assay for sepsis biomarker (IVD diagnostic)",
        "description": (
            "A point-of-care lateral flow immunoassay strip that detects procalcitonin (PCT) "
            "in human whole blood. The test is run on a fingerstick blood sample and gives "
            "a visual read result in 15 minutes. Intended for sepsis risk stratification in "
            "the emergency department. No patient-worn component — purely in vitro."
        ),
        "expected_class": DeviceClass.CLASS_II,
        "expected_pathway": RegulatoryPathway.K510,
    },
    {
        "name": "Continuous glucose monitor (in-vivo diagnostic)",
        "description": (
            "A wearable continuous glucose monitoring system consisting of a small sensor "
            "inserted subcutaneously in the upper arm. The sensor measures interstitial glucose "
            "every 5 minutes and transmits readings to a paired smartphone app via Bluetooth. "
            "The sensor is worn continuously for up to 14 days before replacement. "
            "No drug delivery component."
        ),
        "expected_class": DeviceClass.CLASS_II,
        "expected_pathway": RegulatoryPathway.K510,
    },
]


def run_test(test_case: dict, full_pipeline: bool = False) -> bool:
    """
    Run a single test case.
    Returns True if basic assertions pass.
    """
    print(f"\n{'='*60}")
    print(f"TEST: {test_case['name']}")
    print(f"{'='*60}")

    try:
        if full_pipeline:
            result = run_full_pipeline(test_case["description"])
            if not result.success:
                print(f"FAIL: Pipeline failed. Errors: {result.errors}")
                return False
            classification = result.classification
            print(f"Elapsed: {result.elapsed_seconds:.1f}s")
            if result.roadmap:
                print(f"Tests in roadmap: {len(result.roadmap.tests)}")
                print(f"Cost estimate: ${result.roadmap.total_cost_usd_low:,}–${result.roadmap.total_cost_usd_high:,}")
                print(f"Timeline: {result.roadmap.total_weeks_low}–{result.roadmap.total_weeks_high} weeks")
                print(f"Critical path: {' → '.join(result.roadmap.critical_path)}")
            if result.materials_optimization:
                print(f"Material recommendations: {len(result.materials_optimization.recommendations)}")
        else:
            classification = classify_device(test_case["description"])

        print(f"Device class: {classification.device_class}")
        print(f"Pathway: {classification.regulatory_pathway}")
        print(f"Confidence: {classification.confidence:.2%}")
        if classification.low_confidence_warning:
            print(f"Warning: {classification.low_confidence_warning}")
        print(f"Rationale: {classification.classification_rationale[:200]}...")
        if classification.predicate_devices:
            print(f"Predicates found: {[p.k_number for p in classification.predicate_devices]}")

        # Assertions
        expected_class = test_case.get("expected_class")
        expected_pathway = test_case.get("expected_pathway")

        passed = True
        if expected_class and classification.device_class != expected_class:
            print(f"ASSERTION FAIL: Expected class {expected_class}, got {classification.device_class}")
            passed = False
        if expected_pathway and classification.regulatory_pathway != expected_pathway:
            print(f"ASSERTION FAIL: Expected pathway {expected_pathway}, got {classification.regulatory_pathway}")
            passed = False

        if passed:
            print("PASS ✓")
        return passed

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run biotech navigator tests")
    parser.add_argument("--full", action="store_true", help="Run full pipeline (slower, requires API key)")
    parser.add_argument("--case", type=int, help="Run only test case N (0-indexed)")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Tests require a real API key.")
        sys.exit(1)

    cases = TEST_CASES
    if args.case is not None:
        cases = [TEST_CASES[args.case]]

    results = []
    for case in cases:
        passed = run_test(case, full_pipeline=args.full)
        results.append(passed)

    print(f"\n{'='*60}")
    print(f"Results: {sum(results)}/{len(results)} passed")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
