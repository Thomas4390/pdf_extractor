# Testing Patterns

**Analysis Date:** 2026-01-11

## Test Framework

**Runner:**
- Standalone Python scripts (no pytest)
- Custom test structure with `asyncio.run()`
- Executable via `python -m src.tests.test_*`

**Assertion Library:**
- Manual assertions with `assert` statements
- Print-based progress feedback
- Exception handling for error verification

**Run Commands:**
```bash
python -m src.tests.test_uv                    # UV extractor tests
python -m src.tests.test_assomption            # Assomption extractor tests
python -m src.tests.test_idc                   # IDC Propositions tests
python -m src.tests.test_idc_statement         # IDC Statements tests
python -m src.tests.test_pipeline              # Pipeline orchestrator tests
python -m src.tests.test_batch_extraction      # Batch processing tests
python -m src.tests.test_monday                # Monday.com integration tests
```

## Test File Organization

**Location:**
- `src/tests/` directory
- All test files in single directory (flat structure)

**Naming:**
- `test_*.py` for all test files
- One test file per major component/extractor

**Structure:**
```
src/tests/
├── __init__.py
├── test_uv.py                     # UV extractor (~140 lines)
├── test_assomption.py             # Assomption extractor
├── test_idc.py                    # IDC Propositions
├── test_idc_statement.py          # IDC Statements
├── test_pipeline.py               # Pipeline orchestrator (448 lines)
├── test_batch_extraction.py       # Parallel extraction (127 lines)
├── test_data_unifier.py           # Data transformation (209 lines)
├── test_monday.py                 # Monday.com integration (406 lines)
├── test_idc_statement_parse.py    # Specialized parsing tests
├── test_idc_statement_compare.py  # Comparison tests
└── test_idc_statement_direct.py   # Direct extraction tests
```

## Test Structure

**Suite Organization:**
```python
#!/usr/bin/env python3
"""
Test script for [Component].

Usage:
    python -m src.tests.test_[component] [optional_args]

Environment variables:
    OPENROUTER_API_KEY - Required for VLM extraction
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.uv_extractor import UVExtractor


async def test_extraction(pdf_path: str | None = None):
    """Main test function."""
    print("\n[1/4] Initializing extractor...")
    # Test logic here

    print("\n[2/4] Running extraction...")
    # More test logic

    print("✓ Test passed!")


def main():
    """Entry point."""
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(test_extraction(pdf_path))


if __name__ == "__main__":
    main()
```

**Patterns:**
- Shebang line: `#!/usr/bin/env python3`
- Module docstring with usage instructions
- Path setup for imports (`PROJECT_ROOT` calculation)
- Async test functions: `async def test_*()`
- Numbered progress sections: `[1/4]`, `[2/4]`, etc.
- Entry point: `main()` with `asyncio.run()`

## Mocking

**Framework:**
- No dedicated mocking library
- Manual mocking via function replacement when needed

**Patterns:**
- Tests generally run against real APIs (integration tests)
- Mock mode available in `test_pipeline.py` via command arg
- Cache checking to avoid redundant API calls

**What to Mock:**
- Not typically mocked - tests are functional/integration style

**What NOT to Mock:**
- Everything - prefer real execution with caching

## Fixtures and Factories

**Test Data:**
```python
# Test PDF paths (relative to project root)
TEST_PDFS = {
    SourceType.UV: PROJECT_ROOT / "pdf/uv/rappportremun_21621_2025-10-13.pdf",
    SourceType.IDC: PROJECT_ROOT / "pdf/idc/Rapport des propositions soumises.20251124_1638.pdf",
    SourceType.ASSOMPTION: PROJECT_ROOT / "pdf/assomption/Remuneration (61).pdf",
    SourceType.IDC_STATEMENT: PROJECT_ROOT / "pdf/idc_statement/Détails des frais de suivi.20251105_1113.pdf",
}
```

**Location:**
- Test data defined inline in test files
- Default fallback to project test PDFs in `pdf/` subdirectories
- Command-line argument for custom PDF path

## Coverage

**Requirements:**
- No coverage target enforced
- Functional/integration testing focus
- Tests verify extraction produces valid output

**Configuration:**
- No coverage tool configured
- Manual verification of results

**View Coverage:**
- Not applicable (no coverage reporting)

## Test Types

**Unit Tests:**
- Minimal - most tests are integration-style
- `test_source_type_enum()` in `test_pipeline.py` (enum validation)
- `test_source_detection()` (path pattern matching)

**Integration Tests:**
- Primary test type
- Test full extraction workflow with real VLM API
- Example: `test_single_pdf_extraction()` in `test_pipeline.py`
- Cache-aware: skip extraction if already cached

**E2E Tests:**
- `test_batch_pdf_extraction()` - Multi-PDF processing
- Monday.com upload tests in `test_monday.py`
- Requires API keys and test PDF files

## Common Patterns

**Async Testing:**
```python
async def test_extraction(pdf_path: str | None = None):
    extractor = UVExtractor()
    result = await extractor.extract(pdf_path)
    assert result is not None
    print("✓ Extraction successful")


def main():
    asyncio.run(test_extraction())
```

**Error Testing:**
```python
try:
    result = await extractor.extract("invalid.pdf")
except FileNotFoundError as e:
    print(f"✓ Expected error: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    raise
```

**Progress Tracking:**
```python
def on_progress(current: int, total: int, pdf_path: Path, success: bool):
    status = "✓" if success else "✗"
    print(f"  [{current}/{total}] {status} {pdf_path.name}")
```

**Cache Checking:**
```python
if extractor.is_cached(pdf_path):
    print("Using cached result")
    result = await extractor.extract(pdf_path)  # Returns cached
else:
    print("Running fresh extraction")
    result = await extractor.extract(pdf_path)  # Calls VLM API
```

**Snapshot Testing:**
- Not used in this codebase
- Prefer explicit assertions and visual output

---

*Testing analysis: 2026-01-11*
*Update when test patterns change*
