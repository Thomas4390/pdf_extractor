#!/usr/bin/env python3
"""
Test script for parallel batch PDF extraction.

Demonstrates extracting multiple PDFs concurrently using asyncio.

Usage:
    python -m src.tests.test_batch_extraction [pdf_directory]

If no directory is provided, uses pdf/idc_statement/ by default.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


async def test_batch_extraction(pdf_dir: str | None = None, max_concurrent: int = 3):
    """Run batch extraction test on multiple PDFs."""
    from src.extractors.idc_statement_extractor import IDCStatementExtractor
    from src.utils.batch import extract_batch, print_batch_summary
    from src.utils.model_registry import get_model_config

    # Default directory
    if pdf_dir is None:
        pdf_dir = PROJECT_ROOT / "pdf/idc_statement"
    else:
        pdf_dir = Path(pdf_dir)

    if not pdf_dir.exists():
        print(f"ERROR: Directory not found: {pdf_dir}")
        return None

    # Find all PDFs
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"ERROR: No PDF files found in {pdf_dir}")
        return None

    # Get model configuration
    model_config = get_model_config("IDC_STATEMENT")

    print(f"\n{'='*60}")
    print("BATCH PARALLEL EXTRACTION TEST")
    print(f"{'='*60}")
    print(f"Directory: {pdf_dir}")
    print(f"PDF files found: {len(pdf_files)}")
    print(f"Max concurrent: {max_concurrent}")
    print(f"Model: {model_config.model_id}")
    print(f"Mode: {model_config.mode.value.upper()}")

    # List files
    print(f"\nFiles to process:")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf.name}")

    # Create extractor
    extractor = IDCStatementExtractor()

    # Progress callback for real-time updates
    def on_progress(current: int, total: int, pdf_path: Path, success: bool):
        status = "✓" if success else "✗"
        print(f"  [{current}/{total}] {status} {pdf_path.name}")

    # Run batch extraction
    print(f"\n{'='*60}")
    print("STARTING PARALLEL EXTRACTION")
    print(f"{'='*60}\n")

    result = await extract_batch(
        pdf_paths=pdf_files,
        extract_func=extractor.extract_direct,  # Use direct extraction
        extractor=extractor,
        max_concurrent=max_concurrent,
        progress_callback=on_progress,
    )

    # Print summary
    print_batch_summary(result)

    # Show details for each result
    print(f"\n{'='*60}")
    print("EXTRACTION DETAILS")
    print(f"{'='*60}")

    for i, res in enumerate(result.results, 1):
        print(f"\n{i}. {res.pdf_path.name}")
        print(f"   Status: {'SUCCESS' if res.success else 'FAILED'}")
        print(f"   Duration: {res.duration_seconds:.2f}s")
        print(f"   From cache: {res.from_cache}")

        if res.success and res.data:
            fees = res.data.get("trailing_fees", [])
            print(f"   Records: {len(fees)}")

            # Show summary per advisor if available
            advisors = {}
            for fee in fees:
                adv = fee.get("advisor_name", "Unknown")
                if adv:
                    advisors[adv] = advisors.get(adv, 0) + 1

            if advisors:
                print(f"   By advisor:")
                for adv, count in sorted(advisors.items()):
                    print(f"      {adv}: {count}")
        elif res.error:
            print(f"   Error: {res.error}")

    return result


def main():
    """Entry point."""
    pdf_dir = sys.argv[1] if len(sys.argv) > 1 else None
    max_concurrent = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    asyncio.run(test_batch_extraction(pdf_dir, max_concurrent))


if __name__ == "__main__":
    main()
