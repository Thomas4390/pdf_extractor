#!/usr/bin/env python3
"""
Test script for DataUnifier.

Usage:
    python -m src.tests.test_data_unifier [source] [--raw] [--deepseek]

Sources: UV, IDC, IDC_STATEMENT, ASSOMPTION, ALL (default: ALL)

Options:
    --raw       Use raw extraction mode for IDC_STATEMENT (default is DIRECT mode)
    --deepseek  Use DeepSeek (text mode) instead of vision models for UV extraction

This script tests the DataUnifier by:
1. Loading cached extraction results (or extracting via VLM if not cached)
2. Converting them to standardized DataFrames
3. Displaying the full results

Note: IDC_STATEMENT uses DIRECT extraction by default (parses client_full_name, advisor_name, etc.)
"""

import asyncio
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_full_dataframe(df: pd.DataFrame, title: str):
    """Print a DataFrame with all columns and rows visible."""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")

    # Configure pandas to show all columns and rows
    with pd.option_context(
        'display.max_columns', None,
        'display.max_rows', None,
        'display.width', None,
        'display.max_colwidth', 50,
        'display.expand_frame_repr', False,
    ):
        print(df.to_string(index=False))

    print(f"\n  Total rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"{'=' * 100}\n")


async def test_unifier_with_real_data(
    source: str,
    use_direct: bool = False,
    use_deepseek: bool = False,
):
    """Test DataUnifier with real extracted data from cache or VLM extraction."""
    from src.utils.data_unifier import DataUnifier
    from src.extractors import (
        UVExtractor,
        IDCExtractor,
        IDCStatementExtractor,
        AssomptionExtractor,
    )
    from src.models.idc_statement import IDCStatementReportParsed

    # If using DeepSeek for UV, configure the model registry
    if use_deepseek:
        from src.utils.model_registry import (
            register_model,
            ModelConfig,
            ExtractionMode,
            DEFAULT_TEXT_MODEL,
        )
        # Override UV config to use DeepSeek (text mode)
        register_model("UV", ModelConfig(
            model_id=DEFAULT_TEXT_MODEL,  # deepseek/deepseek-chat
            mode=ExtractionMode.TEXT,
            fallback_model_id=None,
            fallback_mode=None,
        ))
        print("\n  ⚙️  UV configured to use DeepSeek (text extraction mode)")

    # Define test PDFs (using different PDFs from the default ones)
    test_pdfs = {
        "UV": PROJECT_ROOT / "pdf/uv/rappportremun_21622_2025-12-15.pdf",
        "IDC": PROJECT_ROOT / "pdf/idc/Rapport des propositions soumises.20251225_1233.pdf",
        "ASSOMPTION": PROJECT_ROOT / "pdf/assomption/Remuneration - 2025-12-17T071208.298.pdf",
        "IDC_STATEMENT": PROJECT_ROOT / "pdf/idc_statement/Statements (16).pdf",
    }

    extractors = {
        "UV": UVExtractor(),
        "IDC": IDCExtractor(),
        "ASSOMPTION": AssomptionExtractor(),
        "IDC_STATEMENT": IDCStatementExtractor(),
    }

    unifier = DataUnifier()

    sources_to_test = [source] if source != "ALL" else list(test_pdfs.keys())

    # IDC_STATEMENT uses DIRECT mode by default (use_direct=True means use RAW mode)
    use_raw = use_direct  # --raw flag means use RAW mode instead of DIRECT
    mode_str = ""
    if use_raw:
        mode_str += " (RAW mode for IDC_STATEMENT)"
    if use_deepseek:
        mode_str += " (DeepSeek TEXT mode for UV)"
    if not mode_str:
        mode_str = " (DIRECT mode for IDC_STATEMENT)"
    print("\n" + "=" * 100)
    print(f"  DataUnifier Test - Real Data{mode_str}")
    print("=" * 100)

    for src in sources_to_test:
        pdf_path = test_pdfs.get(src)
        if not pdf_path or not pdf_path.exists():
            print(f"\n[{src}] Skipping - PDF not found: {pdf_path}")
            continue

        extractor = extractors[src]

        print(f"\n[{src}] Processing...")
        print(f"  PDF: {pdf_path.name}")

        try:
            # IDC_STATEMENT uses DIRECT mode by default (unless --raw is passed)
            if src == "IDC_STATEMENT" and not use_raw:
                # Check if direct extraction is cached
                is_cached = extractor.is_direct_cached(pdf_path)
                if not is_cached:
                    print(f"  Status: Not cached - running DIRECT VLM extraction (may take 30-60s)...")
                else:
                    print(f"  Status: Using cached DIRECT result")

                # Extract with direct mode (parses client_full_name, advisor_name, etc.)
                result_dict = await extractor.extract_direct(pdf_path)

                # Check for partial extraction
                if result_dict.get("_partial_extraction"):
                    print(f"  ⚠️  PARTIAL EXTRACTION: {result_dict.get('_partial_item_count')} items recovered")
                    print(f"     Reason: {result_dict.get('_partial_reason')}")

                # Create validated model from dict
                report = IDCStatementReportParsed(**result_dict)
                print(f"  Mode: DIRECT (parsed fields: client_full_name, advisor_name, policy_number)")
            else:
                # Standard extraction
                is_cached = extractor.is_cached(pdf_path)
                if not is_cached:
                    print(f"  Status: Not cached - running VLM extraction (may take 10-30s)...")
                else:
                    print(f"  Status: Using cached result")

                report = await extractor.extract(pdf_path)
                if src == "IDC_STATEMENT":
                    print(f"  Mode: RAW (raw_client_data as-is)")
                elif src == "UV" and use_deepseek:
                    print(f"  Mode: TEXT (DeepSeek - deepseek/deepseek-chat)")

            # Convert to DataFrame
            df, board_type = unifier.unify(report, src)

            print(f"  Board Type: {board_type.value}")

            # Print full DataFrame
            print_full_dataframe(df, f"{src} - {board_type.value}")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 100)
    print("  Test completed!")
    print("=" * 100)


async def main():
    """Main entry point."""
    # Parse args
    args = sys.argv[1:]

    # --raw flag to use RAW mode for IDC_STATEMENT (default is DIRECT mode)
    use_raw = "--raw" in args
    args = [a for a in args if a != "--raw"]

    # --deepseek flag to use DeepSeek (text mode) for UV extraction
    use_deepseek = "--deepseek" in args
    args = [a for a in args if a != "--deepseek"]

    source = args[0].upper() if args else "ALL"

    valid_sources = ["UV", "IDC", "ASSOMPTION", "IDC_STATEMENT", "ALL"]
    if source not in valid_sources:
        print(f"Invalid source: {source}")
        print(f"Valid sources: {', '.join(valid_sources)}")
        print(f"Options:")
        print(f"  --raw       Use raw extraction for IDC_STATEMENT instead of direct")
        print(f"  --deepseek  Use DeepSeek (text mode) for UV extraction")
        sys.exit(1)

    await test_unifier_with_real_data(source, use_direct=use_raw, use_deepseek=use_deepseek)


if __name__ == "__main__":
    asyncio.run(main())
