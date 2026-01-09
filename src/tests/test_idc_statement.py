#!/usr/bin/env python3
"""
Test script for IDC Statement (trailing fees) extraction via VLM.

Usage:
    python -m src.tests.test_idc_statement [pdf_path]

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
    """Run extraction test on an IDC Statement PDF."""
    from src.extractors.idc_statement_extractor import IDCStatementExtractor
    from src.utils.model_registry import get_model_config
    from src.utils.pdf import get_pdf_hash, pdf_to_images

    # Default test file
    if pdf_path is None:
        pdf_dir = PROJECT_ROOT / "pdf/idc_statement"
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print("ERROR: No PDF files found in pdf/idc_statement/")
            return None
        pdf_path = pdf_files[0]  # Use first PDF
    else:
        pdf_path = Path(pdf_path)

    # Get model configuration
    model_config = get_model_config("IDC_STATEMENT")

    print(f"\n{'='*60}")
    print("IDC Statement (Trailing Fees) Extraction Test")
    print(f"{'='*60}")
    print(f"PDF: {pdf_path.name}")
    print(f"Primary: {model_config.model_id} ({model_config.mode.value})")
    if model_config.fallback_model_id:
        fallback_mode = model_config.fallback_mode.value if model_config.fallback_mode else "N/A"
        print(f"Fallback: {model_config.fallback_model_id} ({fallback_mode})")
    if model_config.secondary_fallback_model_id:
        sec_mode = model_config.secondary_fallback_mode.value if model_config.secondary_fallback_mode else "N/A"
        print(f"Secondary: {model_config.secondary_fallback_model_id} ({sec_mode})")

    # Test PDF utilities
    print(f"\n[1/4] Testing PDF utilities...")
    pdf_hash = get_pdf_hash(pdf_path)
    print(f"  Hash: {pdf_hash[:16]}...")

    images = pdf_to_images(pdf_path)
    print(f"  Pages: {len(images)}")
    print(f"  Image sizes: {[len(img)//1024 for img in images]} KB")

    # Test extractor initialization
    print(f"\n[2/4] Initializing extractor...")
    extractor = IDCStatementExtractor()
    print(f"  Source: {extractor.source_name}")
    print(f"  Pydantic Model: {extractor.model_class.__name__}")

    # Check cache
    print(f"\n[3/4] Checking cache...")
    is_cached = extractor.is_cached(pdf_path)
    print(f"  Cached: {is_cached}")

    # Run extraction
    print(f"\n[4/4] Running VLM extraction...")
    if is_cached:
        print("  -> Using cached result (instant)")
    else:
        print("  -> Calling VLM API (this may take 10-30 seconds...)")

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
        print(f"Section conseiller: {report.advisor_section or 'N/A'}")
        print(f"Nombre d'enregistrements: {report.nombre_enregistrements}")

        # Count by company
        companies = report.frais_par_compagnie()
        print(f"\nPar compagnie:")
        for c, count in sorted(companies.items()):
            print(f"  {c}: {count}")

        # Show unique companies
        print(f"\nCompagnies uniques:")
        for company in sorted(report.compagnies_uniques):
            print(f"  - {company}")

        # Show records
        print(f"\n{'='*60}")
        print("FRAIS DE SUIVI (données brutes)")
        print(f"{'='*60}")
        for i, fee in enumerate(report.trailing_fees, 1):
            raw_data = fee.raw_client_data
            # Truncate long raw data for display
            if len(raw_data) > 60:
                raw_display = raw_data[:57] + "..."
            else:
                raw_display = raw_data.replace('\n', ' | ')
            print(f"\n{i}. Raw: {raw_display}")
            print(f"   Compte: {fee.account_number}")
            print(f"   Compagnie: {fee.company}")
            print(f"   Produit: {fee.product}")
            print(f"   Date: {fee.date}")
            print(f"   Frais brut: {fee.gross_trailing_fee}")
            print(f"   Frais net: {fee.net_trailing_fee}")

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
