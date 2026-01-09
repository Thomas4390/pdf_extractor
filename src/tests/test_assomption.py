#!/usr/bin/env python3
"""
Test script for Assomption Vie extraction via VLM.

Usage:
    python -m src.tests.test_assomption [pdf_path]

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
    """Run extraction test on an Assomption Vie PDF."""
    from src.extractors import AssomptionExtractor
    from src.utils.model_registry import get_model_config
    from src.utils.pdf import get_pdf_hash, pdf_to_images

    # Default test file
    if pdf_path is None:
        pdf_dir = PROJECT_ROOT / "pdf/assomption"
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print("ERROR: No PDF files found in pdf/assomption/")
            return None
        pdf_path = pdf_files[0]  # Use first PDF
    else:
        pdf_path = Path(pdf_path)

    # Get model configuration
    model_config = get_model_config("ASSOMPTION")

    print(f"\n{'='*60}")
    print("Assomption Vie VLM Extraction Test")
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
    extractor = AssomptionExtractor()
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
        print(f"Période début: {report.periode_debut}")
        print(f"Période fin: {report.periode_fin}")
        print(f"Date paie: {report.date_paie}")
        print(f"Courtier: {report.nom_courtier}")
        print(f"Numéro courtier: {report.numero_courtier}")
        print(f"Nombre de commissions: {report.nombre_transactions}")

        # Show totals (calculated from records)
        total_commissions = report.calculer_total_commissions()
        total_boni = report.calculer_total_boni()
        print(f"\nTotaux (calculés):")
        print(f"  Total commissions: {total_commissions:,.2f} $")
        print(f"  Total boni: {total_boni:,.2f} $")
        print(f"  Total période: {total_commissions + total_boni:,.2f} $")

        # Show commissions
        print(f"\n{'='*60}")
        print("COMMISSIONS")
        print(f"{'='*60}")
        for i, comm in enumerate(report.commissions, 1):
            print(f"\n{i}. Police: {comm.numero_police}")
            print(f"   Code: {comm.code}")
            print(f"   Assuré: {comm.nom_assure}")
            print(f"   Produit: {comm.produit}")
            print(f"   Date émission: {comm.date_emission}")
            print(f"   Fréquence: {comm.frequence_paiement}")
            print(f"   Facturation: {comm.facturation}")
            print(f"   Prime: {comm.prime:,.2f} $")
            print(f"   Taux commission: {comm.taux_commission}%")
            print(f"   Commission: {comm.commission:,.2f} $")
            if comm.taux_boni or comm.boni:
                print(f"   Taux boni: {comm.taux_boni or 0}%")
                print(f"   Boni: {comm.boni or 0:,.2f} $")

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
