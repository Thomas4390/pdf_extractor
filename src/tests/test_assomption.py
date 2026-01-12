#!/usr/bin/env python3
"""
Test script for Assomption Vie extraction via VLM.

Usage:
    python -m src.tests.test_assomption [pdf_path] [options]

Options:
    --mode <mode>      Extraction mode: vision, text, pdf_native, hybrid (default: from registry)
    --model <model>    Model to use (e.g., gemini-3-flash, qwen3-vl, deepseek)
    --force            Force re-extraction (ignore cache)
    --invalidate       Same as --force

Examples:
    python -m src.tests.test_assomption                          # Default PDF and mode
    python -m src.tests.test_assomption --mode hybrid            # Test hybrid mode
    python -m src.tests.test_assomption --mode vision --force    # Force vision extraction
    python -m src.tests.test_assomption path/to/file.pdf --mode pdf_native
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path (src/tests -> src -> project_root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Model aliases for convenience
MODEL_ALIASES = {
    "gemini-3-flash": "google/gemini-3-flash-preview",
    "qwen3-vl": "qwen/qwen3-vl-235b-a22b-instruct",
    "qwen2.5-vl": "qwen/qwen2.5-vl-72b-instruct",
    "deepseek": "deepseek/deepseek-chat",
}


def parse_args(args: list[str]) -> tuple[str | None, str | None, str | None, bool]:
    """Parse command line arguments.

    Returns:
        Tuple of (pdf_path, mode, model, force_refresh)
    """
    pdf_path = None
    mode = None
    model = None
    force_refresh = False

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--mode":
            if i + 1 < len(args):
                mode = args[i + 1].lower()
                i += 2
            else:
                print("ERROR: --mode requires a value")
                sys.exit(1)
        elif arg == "--model":
            if i + 1 < len(args):
                model_input = args[i + 1]
                model = MODEL_ALIASES.get(model_input, model_input)
                i += 2
            else:
                print("ERROR: --model requires a value")
                sys.exit(1)
        elif arg in ("--force", "--invalidate"):
            force_refresh = True
            i += 1
        elif not arg.startswith("--"):
            pdf_path = arg
            i += 1
        else:
            print(f"WARNING: Unknown option {arg}")
            i += 1

    return pdf_path, mode, model, force_refresh


async def test_extraction(
    pdf_path: str | None = None,
    mode: str | None = None,
    model: str | None = None,
    force_refresh: bool = False,
):
    """Run extraction test on an Assomption Vie PDF."""
    from src.extractors import AssomptionExtractor
    from src.utils.model_registry import (
        get_model_config,
        register_model,
        ModelConfig,
        ExtractionMode,
        OcrEngine,
    )
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

    # Get original model configuration
    original_config = get_model_config("ASSOMPTION")

    # Determine mode and model to use
    effective_mode = mode or original_config.mode.value
    effective_model = model or original_config.model_id

    # Map mode string to ExtractionMode enum
    mode_map = {
        "vision": ExtractionMode.VISION,
        "text": ExtractionMode.TEXT,
        "pdf_native": ExtractionMode.PDF_NATIVE,
        "hybrid": ExtractionMode.HYBRID,
    }

    if effective_mode not in mode_map:
        print(f"ERROR: Invalid mode '{effective_mode}'. Valid modes: {', '.join(mode_map.keys())}")
        sys.exit(1)

    extraction_mode = mode_map[effective_mode]

    # If custom mode or model specified, register temporary config
    if mode or model:
        temp_config = ModelConfig(
            model_id=effective_model,
            mode=extraction_mode,
            fallback_model_id=original_config.fallback_model_id,
            fallback_mode=original_config.fallback_mode,
            page_config=original_config.page_config,
            ocr_engine=OcrEngine.MISTRAL_OCR,
            text_analysis_model="deepseek/deepseek-chat",
        )
        register_model("ASSOMPTION", temp_config)

    print(f"\n{'='*60}")
    print("Assomption Vie VLM Extraction Test")
    print(f"{'='*60}")
    print(f"PDF: {pdf_path.name}")
    print(f"Model: {effective_model}")
    print(f"Mode: {effective_mode.upper()}")
    if force_refresh:
        print(f"Cache: FORCED REFRESH")

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
    print(f"\n[4/4] Running extraction ({effective_mode} mode)...")
    if is_cached and not force_refresh:
        print("  -> Using cached result (instant)")
    else:
        print("  -> Calling API (this may take 10-60 seconds...)")

    start = time.time()

    try:
        report = await extractor.extract(pdf_path, force_refresh=force_refresh)
        elapsed = time.time() - start

        # Get usage info if available
        session = extractor.client.get_session_summary()
        cost = session.get("total_cost", 0)

        print(f"\n{'='*60}")
        print("EXTRACTION RESULTS")
        print(f"{'='*60}")
        print(f"Time: {elapsed:.2f}s")
        if cost > 0:
            print(f"Cost: ${cost:.6f}")
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
    finally:
        # Restore original config
        if mode or model:
            register_model("ASSOMPTION", original_config)


def main():
    """Entry point."""
    pdf_path, mode, model, force_refresh = parse_args(sys.argv[1:])
    asyncio.run(test_extraction(pdf_path, mode, model, force_refresh))


if __name__ == "__main__":
    main()
