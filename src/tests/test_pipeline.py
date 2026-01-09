#!/usr/bin/env python3
"""
Test script for Pipeline orchestrator.

Usage:
    python -m src.tests.test_pipeline [command]

Commands:
    mock      - Run tests with mock data only (no API calls)
    single    - Test single PDF extraction (requires PDF files)
    batch     - Test batch PDF extraction (requires PDF files)
    all       - Run all tests

Environment variables:
    OPENROUTER_API_KEY - Required for VLM extraction
    MONDAY_API_KEY     - Required for Monday.com upload tests
"""

import asyncio
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (
    Pipeline,
    SourceType,
    PipelineResult,
    BatchResult,
    ExtractionWarning,
    get_pipeline,
)
from src.utils.data_unifier import BoardType


# =============================================================================
# TEST DATA
# =============================================================================

# Test PDF paths (relative to project root)
TEST_PDFS = {
    SourceType.UV: PROJECT_ROOT / "pdf/uv/rappportremun_21621_2025-10-13.pdf",
    SourceType.IDC: PROJECT_ROOT / "pdf/idc/Rapport des propositions soumises.20251124_1638.pdf",
    SourceType.ASSOMPTION: PROJECT_ROOT / "pdf/assomption/Remuneration (61).pdf",
    SourceType.IDC_STATEMENT: PROJECT_ROOT / "pdf/idc_statement/Détails des frais de suivi.20251105_1113.pdf",
}


# =============================================================================
# MOCK TESTS (No API calls)
# =============================================================================

def test_source_type_enum():
    """Test SourceType enum values."""
    print("\n[1/5] Testing SourceType enum...")

    assert SourceType.UV.value == "UV"
    assert SourceType.IDC.value == "IDC"
    assert SourceType.IDC_STATEMENT.value == "IDC_STATEMENT"
    assert SourceType.ASSOMPTION.value == "ASSOMPTION"
    print("  SourceType enum OK!")
    return True


def test_source_detection():
    """Test auto-detection of source type from paths."""
    print("\n[2/5] Testing source detection...")

    pipeline = Pipeline(use_advisor_matcher=False)

    # Test UV detection
    assert pipeline.detect_source("/path/to/uv/file.pdf") == SourceType.UV
    assert pipeline.detect_source("UV_report.pdf") == SourceType.UV
    print("  ✓ UV detection")

    # Test IDC detection
    assert pipeline.detect_source("/path/to/idc/file.pdf") == SourceType.IDC
    assert pipeline.detect_source("Rapport des propositions.pdf") == SourceType.IDC
    print("  ✓ IDC detection")

    # Test IDC_STATEMENT detection (should match before IDC)
    assert pipeline.detect_source("/path/to/idc_statement/file.pdf") == SourceType.IDC_STATEMENT
    assert pipeline.detect_source("Détails des frais de suivi.pdf") == SourceType.IDC_STATEMENT
    print("  ✓ IDC_STATEMENT detection")

    # Test ASSOMPTION detection
    assert pipeline.detect_source("/path/to/assomption/file.pdf") == SourceType.ASSOMPTION
    assert pipeline.detect_source("Remuneration (61).pdf") == SourceType.ASSOMPTION
    print("  ✓ ASSOMPTION detection")

    # Test detection failure
    try:
        pipeline.detect_source("random_file.pdf")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  ✓ Unknown source raises ValueError")

    print("  Source detection OK!")
    return True


def test_pipeline_result_dataclass():
    """Test PipelineResult dataclass."""
    print("\n[3/5] Testing PipelineResult dataclass...")

    # Test successful result
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    result = PipelineResult(
        pdf_path="/test/file.pdf",
        source=SourceType.UV,
        board_type=BoardType.SALES_PRODUCTION,
        dataframe=df,
        success=True,
        extraction_time_ms=1500
    )

    assert result.success == True
    assert result.row_count == 3
    assert result.source == SourceType.UV
    assert result.error is None
    print("  ✓ Successful result")

    # Test failed result
    result_failed = PipelineResult(
        pdf_path="/test/file.pdf",
        source=SourceType.IDC,
        board_type=BoardType.SALES_PRODUCTION,
        dataframe=pd.DataFrame(),
        success=False,
        error="Extraction failed"
    )

    assert result_failed.success == False
    assert result_failed.row_count == 0
    assert result_failed.error == "Extraction failed"
    print("  ✓ Failed result")

    # Test result with warnings
    result_warning = PipelineResult(
        pdf_path="/test/file.pdf",
        source=SourceType.UV,
        board_type=BoardType.SALES_PRODUCTION,
        dataframe=pd.DataFrame({"a": [1]}),
        success=True,
        warnings=[ExtractionWarning(message="Low confidence", pdf_path="/test/file.pdf")]
    )

    assert len(result_warning.warnings) == 1
    assert result_warning.warnings[0].message == "Low confidence"
    print("  ✓ Result with warnings")

    print("  PipelineResult OK!")
    return True


def test_batch_result_dataclass():
    """Test BatchResult dataclass."""
    print("\n[4/5] Testing BatchResult dataclass...")

    batch = BatchResult()

    # Add successful result
    df1 = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    result1 = PipelineResult(
        pdf_path="/test/file1.pdf",
        source=SourceType.UV,
        board_type=BoardType.SALES_PRODUCTION,
        dataframe=df1,
        success=True
    )
    batch.add(result1)

    assert batch.total_pdfs == 1
    assert batch.successful == 1
    assert batch.failed == 0
    assert batch.total_rows == 2
    print("  ✓ Added successful result")

    # Add failed result
    result2 = PipelineResult(
        pdf_path="/test/file2.pdf",
        source=SourceType.IDC,
        board_type=BoardType.SALES_PRODUCTION,
        dataframe=pd.DataFrame(),
        success=False,
        error="Failed"
    )
    batch.add(result2)

    assert batch.total_pdfs == 2
    assert batch.successful == 1
    assert batch.failed == 1
    assert batch.total_rows == 2
    print("  ✓ Added failed result")

    # Add another successful result
    df3 = pd.DataFrame({"col1": [3, 4, 5], "col2": ["c", "d", "e"]})
    result3 = PipelineResult(
        pdf_path="/test/file3.pdf",
        source=SourceType.UV,
        board_type=BoardType.SALES_PRODUCTION,
        dataframe=df3,
        success=True
    )
    batch.add(result3)

    assert batch.total_pdfs == 3
    assert batch.successful == 2
    assert batch.failed == 1
    assert batch.total_rows == 5
    print("  ✓ Added another successful result")

    # Test combined DataFrame
    combined = batch.get_combined_dataframe()
    assert len(combined) == 5
    assert "_source_file" in combined.columns
    assert "_source_type" in combined.columns
    print("  ✓ Combined DataFrame")

    print("  BatchResult OK!")
    return True


def test_pipeline_initialization():
    """Test Pipeline initialization."""
    print("\n[5/5] Testing Pipeline initialization...")

    import os

    # Save original key
    original_key = os.environ.pop("MONDAY_API_KEY", None)

    try:
        # Without Monday API key
        pipeline = Pipeline(use_advisor_matcher=False)
        assert pipeline.monday_configured == False
        assert len(pipeline.supported_sources) == 4
        print("  ✓ Pipeline without Monday key")

        # With mock Monday API key (won't connect, just tests initialization)
        os.environ["MONDAY_API_KEY"] = "test_key"
        pipeline_with_monday = Pipeline(use_advisor_matcher=False)
        assert pipeline_with_monday.monday_configured == True
        print("  ✓ Pipeline with Monday key from env")

    finally:
        # Restore original key
        if original_key:
            os.environ["MONDAY_API_KEY"] = original_key
        else:
            os.environ.pop("MONDAY_API_KEY", None)

    print("  Pipeline initialization OK!")
    return True


def run_mock_tests():
    """Run all mock tests."""
    print("\n" + "=" * 70)
    print("Pipeline - Mock Tests")
    print("=" * 70)

    tests = [
        test_source_type_enum,
        test_source_detection,
        test_pipeline_result_dataclass,
        test_batch_result_dataclass,
        test_pipeline_initialization,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 70)

    return passed == len(tests)


# =============================================================================
# INTEGRATION TESTS (Require API calls)
# =============================================================================

async def test_single_pdf_extraction():
    """Test single PDF extraction."""
    print("\n[Integration] Testing single PDF extraction...")

    # Find first available test PDF
    test_pdf = None
    source_type = None
    for src, path in TEST_PDFS.items():
        if path.exists():
            test_pdf = path
            source_type = src
            break

    if test_pdf is None:
        print("  Skipping - no test PDFs found")
        return True

    print(f"  Using: {test_pdf.name} ({source_type.value})")

    pipeline = Pipeline(use_advisor_matcher=False)

    # Check if cached
    is_cached = pipeline.is_cached(test_pdf, source_type)
    print(f"  Cached: {is_cached}")

    # Extract
    result = await pipeline.process_pdf(test_pdf, source=source_type)

    print(f"  Success: {result.success}")
    print(f"  Rows: {result.row_count}")
    print(f"  Board type: {result.board_type.value}")
    print(f"  Time: {result.extraction_time_ms}ms")

    if result.warnings:
        print(f"  Warnings: {len(result.warnings)}")
        for w in result.warnings:
            print(f"    - {w.message}")

    if result.error:
        print(f"  Error: {result.error}")

    if result.success and not result.dataframe.empty:
        print(f"\n  DataFrame preview:")
        print(result.dataframe.head(3).to_string())

    return result.success


async def test_batch_pdf_extraction():
    """Test batch PDF extraction."""
    print("\n[Integration] Testing batch PDF extraction...")

    # Get all available test PDFs
    available_pdfs = [(src, path) for src, path in TEST_PDFS.items() if path.exists()]

    if len(available_pdfs) < 2:
        print("  Skipping - need at least 2 test PDFs")
        return True

    print(f"  Found {len(available_pdfs)} test PDFs")

    pipeline = Pipeline(use_advisor_matcher=False, max_parallel=2)

    # Progress callback
    def on_progress(current, total, filename):
        print(f"  [{current}/{total}] Processed: {filename}")

    # Extract batch
    pdf_paths = [path for _, path in available_pdfs]
    results = await pipeline.process_batch(
        pdf_paths=pdf_paths,
        progress_callback=on_progress
    )

    print(f"\n  Batch Summary:")
    print(f"    Total: {results.total_pdfs}")
    print(f"    Successful: {results.successful}")
    print(f"    Failed: {results.failed}")
    print(f"    Total rows: {results.total_rows}")
    print(f"    Time: {results.processing_time_ms}ms")

    if results.all_warnings:
        print(f"    Warnings: {len(results.all_warnings)}")

    # Test combined DataFrame
    combined = results.get_combined_dataframe()
    print(f"\n  Combined DataFrame: {len(combined)} rows")

    if not combined.empty:
        print(f"  Columns: {list(combined.columns)[:10]}...")

    return results.successful > 0


async def run_integration_tests():
    """Run integration tests."""
    print("\n" + "=" * 70)
    print("Pipeline - Integration Tests")
    print("=" * 70)

    import os
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n  Skipping - OPENROUTER_API_KEY not set")
        return True

    tests = [
        test_single_pdf_extraction,
        test_batch_pdf_extraction,
    ]

    passed = 0
    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Integration Results: {passed}/{len(tests)} tests passed")
    print("=" * 70)

    return passed == len(tests)


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Main entry point."""
    command = sys.argv[1].lower() if len(sys.argv) > 1 else "mock"

    if command == "mock":
        run_mock_tests()
    elif command == "single":
        await test_single_pdf_extraction()
    elif command == "batch":
        await test_batch_pdf_extraction()
    elif command == "all":
        run_mock_tests()
        await run_integration_tests()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python -m src.tests.test_pipeline [mock|single|batch|all]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
