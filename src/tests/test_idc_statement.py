#!/usr/bin/env python3
"""
Test script for IDC Statement (trailing fees) extraction via VLM.

Usage:
    python -m src.tests.test_idc_statement [pdf_path] [options]

Options:
    --mode <mode>      Extraction mode: vision, text, pdf_native, hybrid (default: from registry)
    --model <model>    Model to use (e.g., gemini-3-flash, qwen3-vl, deepseek)
    --force            Force re-extraction (ignore cache)
    --invalidate       Same as --force

Examples:
    python -m src.tests.test_idc_statement                          # Default PDF and mode
    python -m src.tests.test_idc_statement --mode hybrid            # Test hybrid mode
    python -m src.tests.test_idc_statement --mode vision --force    # Force vision extraction
    python -m src.tests.test_idc_statement path/to/file.pdf --mode pdf_native
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
    """Run extraction test on an IDC Statement PDF."""
    from src.extractors.idc_statement_extractor import IDCStatementExtractor
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
        pdf_dir = PROJECT_ROOT / "pdf/idc_statement"
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print("ERROR: No PDF files found in pdf/idc_statement/")
            return None
        pdf_path = pdf_files[0]  # Use first PDF
    else:
        pdf_path = Path(pdf_path)

    # Get original model configuration
    original_config = get_model_config("IDC_STATEMENT")

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
        register_model("IDC_STATEMENT", temp_config)

    print(f"\n{'='*60}")
    print("IDC Statement (Trailing Fees) Extraction Test")
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
    extractor = IDCStatementExtractor()
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
        print(f"Titre: {report.titre}")
        print(f"Date rapport: {report.date_rapport or 'N/A'}")
        print(f"Section conseiller: {report.advisor_section or 'N/A'}")
        print(f"Nombre d'enregistrements: {report.nombre_enregistrements}")

        # Show unique advisors (for parsed reports)
        if hasattr(report, 'conseillers_uniques'):
            advisors = report.conseillers_uniques
            if advisors:
                print(f"\nConseillers uniques:")
                for advisor in sorted(advisors):
                    print(f"  - {advisor}")

        # Show records
        print(f"\n{'='*60}")
        print("FRAIS DE SUIVI")
        print(f"{'='*60}")
        for i, fee in enumerate(report.trailing_fees, 1):
            # Check if this is a parsed fee (has advisor_name) or raw fee
            if hasattr(fee, 'advisor_name') and fee.advisor_name:
                # Parsed format
                print(f"\n{i}. Client: {fee.client_first_name or ''} {fee.client_last_name or ''}")
                print(f"   Conseiller: {fee.advisor_name}")
                print(f"   Compagnie code: {fee.company_code or 'N/A'}")
                print(f"   Police: {fee.policy_number or 'N/A'}")
                print(f"   Taux commission: {fee.commission_rate or 'N/A'}%")
            else:
                # Raw format - show truncated raw_client_data
                raw_data = fee.raw_client_data
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
    finally:
        # Restore original config
        if mode or model:
            register_model("IDC_STATEMENT", original_config)


def main():
    """Entry point."""
    pdf_path, mode, model, force_refresh = parse_args(sys.argv[1:])
    asyncio.run(test_extraction(pdf_path, mode, model, force_refresh))


if __name__ == "__main__":
    main()
