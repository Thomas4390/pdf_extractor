#!/usr/bin/env python3
"""
Test script for UV extraction via VLM.

Usage:
    python -m src.tests.test_uv [pdf_path]

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
    """Run extraction test on a UV PDF."""
    from src.extractors import UVExtractor
    from src.utils.model_registry import get_model_config
    from src.utils.pdf import get_pdf_hash, pdf_to_images

    # Default test file
    if pdf_path is None:
        pdf_dir = PROJECT_ROOT / "pdf/uv"
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print("ERROR: No PDF files found in pdf/uv/")
            return None
        pdf_path = pdf_files[-2]  # Use first PDF
    else:
        pdf_path = Path(pdf_path)

    # Get model configuration
    model_config = get_model_config("UV")

    print(f"\n{'='*60}")
    print("UV VLM Extraction Test")
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
    extractor = UVExtractor()
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
        report = await extractor.extract(pdf_path, force_refresh=True)
        elapsed = time.time() - start

        print(f"\n{'='*60}")
        print("EXTRACTION RESULTS")
        print(f"{'='*60}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Date: {report.date_rapport}")
        print(f"Conseiller: {report.nom_conseiller}")
        print(f"Numéro conseiller: {report.numero_conseiller}")
        print(f"Nombre d'activités: {report.nombre_activites}")
        print(f"Nombre de contrats: {report.nombre_contrats}")
        print(f"Nombre de sous-conseillers: {report.nombre_sous_conseillers}")
        total_calculated = report.calculer_total()
        print(f"Total rémunération: {total_calculated:,.2f} $")

        # Show sous-conseillers
        if report.sous_conseillers_uniques:
            print(f"\nSous-conseillers:")
            for sc in sorted(report.sous_conseillers_uniques):
                print(f"  - {sc}")

        # Show totals per sous-conseiller
        totals_par_sc = report.calculer_total_par_sous_conseiller()
        if totals_par_sc:
            print(f"\nTotaux par sous-conseiller:")
            for sc, total in sorted(totals_par_sc.items()):
                print(f"  {sc}: {total:,.2f} $")

        # Show activities grouped by sous-conseiller
        print(f"\n{'='*60}")
        print("ACTIVITÉS PAR SOUS-CONSEILLER")
        print(f"{'='*60}")
        activites_groupees = report.activites_par_sous_conseiller()
        activity_num = 0
        for sous_conseiller, activites in activites_groupees.items():
            print(f"\n--- {sous_conseiller} ({len(activites)} activités) ---")
            for act in activites:
                activity_num += 1
                print(f"\n{activity_num}. Contrat: {act.contrat}")
                print(f"   Assuré: {act.assure}")
                print(f"   Protection: {act.protection}")
                print(f"   Montant base: {act.montant_base:,.2f} $")
                print(f"   Taux partage: {act.taux_partage}%")
                print(f"   Taux commission: {act.taux_commission}%")
                print(f"   Résultat: {act.resultat:,.2f} $")
                print(f"   Type commission: {act.type_commission}")
                print(f"   Taux boni: {act.taux_boni}%")
                print(f"   Rémunération: {act.remuneration:,.2f} $")

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
