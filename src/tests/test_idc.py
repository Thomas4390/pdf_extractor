#!/usr/bin/env python3
"""
Test script for IDC propositions extraction via VLM.

Usage:
    python -m src.tests.test_idc [pdf_path]

If no PDF path is provided, uses a default test file.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path (src/tests -> src -> project_root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


async def test_extraction(pdf_path: str | None = None):
    """Run extraction test on an IDC PDF."""
    from src.extractors import IDCExtractor
    from src.utils.model_registry import get_model_config
    from src.utils.pdf import get_pdf_hash, pdf_to_images

    # Default test file
    if pdf_path is None:
        pdf_dir = PROJECT_ROOT / "pdf/idc"
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print("ERROR: No PDF files found in pdf/idc/")
            return None
        pdf_path = pdf_files[0]  # Use first PDF
    else:
        pdf_path = Path(pdf_path)

    # Get model configuration
    model_config = get_model_config("IDC")

    print(f"\n{'='*60}")
    print("IDC Propositions VLM Extraction Test")
    print(f"{'='*60}")
    print(f"PDF: {pdf_path.name}")
    print(f"Model: {model_config.model_id}")
    print(f"Mode: {model_config.mode.value.upper()}")
    if model_config.fallback_model_id:
        print(f"Fallback: {model_config.fallback_model_id}")

    # Test PDF utilities
    print(f"\n[1/4] Testing PDF utilities...")
    pdf_hash = get_pdf_hash(pdf_path)
    print(f"  Hash: {pdf_hash[:16]}...")

    images = pdf_to_images(pdf_path)
    print(f"  Pages: {len(images)}")
    print(f"  Image sizes: {[len(img)//1024 for img in images]} KB")

    # Test extractor initialization
    print(f"\n[2/4] Initializing extractor...")
    extractor = IDCExtractor()
    print(f"  Source: {extractor.source_name}")
    print(f"  Pydantic Model: {extractor.model_class.__name__}")

    # Check cache
    print(f"\n[3/4] Checking cache...")
    is_cached = extractor.is_cached(pdf_path)
    print(f"  Cached: {is_cached}")

    # Run extraction
    print(f"\n[4/4] Running VLM extraction...")
    if is_cached:
        print("  → Using cached result (instant)")
    else:
        print("  → Calling VLM API (this may take 10-30 seconds...)")

    start = time.time()

    try:
        report = await extractor.extract(pdf_path)
        elapsed = time.time() - start

        print(f"\n{'='*60}")
        print("EXTRACTION RESULTS")
        print(f"{'='*60}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Titre: {report.titre}")
        print(f"Date rapport: {report.date_rapport or 'N/A'}")
        print(f"Vendeur: {report.vendeur or 'N/A'}")
        print(f"Nombre de propositions: {report.nombre_propositions}")

        # Count by regime type
        types = report.propositions_par_type()
        print(f"\nPar type de régime:")
        for t, count in sorted(types.items()):
            print(f"  {t}: {count}")

        # Show unique insurers
        print(f"\nAssureurs uniques:")
        for assureur in sorted(report.assureurs_uniques):
            print(f"  - {assureur}")

        # Show propositions
        print(f"\n{'='*60}")
        print("PROPOSITIONS")
        print(f"{'='*60}")
        for i, prop in enumerate(report.propositions, 1):
            print(f"\n{i}. Client: {prop.client}")
            print(f"   Assureur: {prop.assureur}")
            print(f"   Type régime: {prop.type_regime}")
            print(f"   Police: {prop.police}")
            print(f"   Statut: {prop.statut}")
            print(f"   Date: {prop.date}")
            print(f"   Nombre: {prop.nombre}")
            print(f"   Taux CPA: {prop.taux_cpa}%")
            print(f"   Couverture: {prop.couverture}")
            print(f"   Prime police: {prop.prime_police}")
            print(f"   Prime commissionnable: {prop.prime_commissionnable}")
            print(f"   Commission: {prop.commission}")

        print(f"\n{'='*60}")
        print("TEST SUCCESSFUL")
        print(f"{'='*60}\n")
        return report

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Entry point."""
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(test_extraction(pdf_path))


if __name__ == "__main__":
    main()
