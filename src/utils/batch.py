"""
Batch processing utilities for parallel PDF extraction.

Provides async utilities for processing multiple PDFs concurrently
with progress tracking and error handling.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar, Generic
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ExtractionResult(Generic[T]):
    """Result of a single PDF extraction."""
    pdf_path: Path
    success: bool
    data: T | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    from_cache: bool = False


@dataclass
class BatchResult(Generic[T]):
    """Result of a batch extraction."""
    total: int
    successful: int
    failed: int
    cached: int
    results: list[ExtractionResult[T]] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        return (self.successful / self.total * 100) if self.total > 0 else 0.0


async def extract_single(
    pdf_path: Path,
    extract_func: Callable,
    extractor: Any,
    semaphore: asyncio.Semaphore,
    progress_callback: Callable[[int, int, Path, bool], None] | None = None,
    current_index: int = 0,
    total_count: int = 0,
) -> ExtractionResult:
    """
    Extract a single PDF with semaphore-controlled concurrency.

    Args:
        pdf_path: Path to the PDF file
        extract_func: Async extraction function to call
        extractor: Extractor instance (for cache checking)
        semaphore: Semaphore for controlling concurrency
        progress_callback: Optional callback for progress updates
        current_index: Current PDF index (for progress)
        total_count: Total PDF count (for progress)

    Returns:
        ExtractionResult with success/failure info
    """
    async with semaphore:
        start_time = datetime.now()
        from_cache = False

        try:
            # Check if cached
            if hasattr(extractor, 'is_cached'):
                from_cache = extractor.is_cached(pdf_path)

            # Run extraction
            data = await extract_func(pdf_path)
            duration = (datetime.now() - start_time).total_seconds()

            if progress_callback:
                progress_callback(current_index + 1, total_count, pdf_path, True)

            logger.info(f"Extracted {pdf_path.name} in {duration:.2f}s (cached: {from_cache})")

            return ExtractionResult(
                pdf_path=pdf_path,
                success=True,
                data=data,
                duration_seconds=duration,
                from_cache=from_cache,
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)

            if progress_callback:
                progress_callback(current_index + 1, total_count, pdf_path, False)

            logger.error(f"Failed to extract {pdf_path.name}: {error_msg}")

            return ExtractionResult(
                pdf_path=pdf_path,
                success=False,
                error=error_msg,
                duration_seconds=duration,
            )


async def extract_batch(
    pdf_paths: list[Path],
    extract_func: Callable,
    extractor: Any = None,
    max_concurrent: int = 5,
    progress_callback: Callable[[int, int, Path, bool], None] | None = None,
) -> BatchResult:
    """
    Extract multiple PDFs in parallel with controlled concurrency.

    Args:
        pdf_paths: List of PDF file paths to process
        extract_func: Async function to call for each PDF (takes pdf_path as argument)
        extractor: Optional extractor instance (for cache checking)
        max_concurrent: Maximum number of concurrent extractions (default: 5)
        progress_callback: Optional callback(current, total, path, success) for progress updates

    Returns:
        BatchResult with all extraction results

    Example:
        ```python
        extractor = IDCStatementExtractor()
        results = await extract_batch(
            pdf_paths=list(Path("pdf/").glob("*.pdf")),
            extract_func=extractor.extract,
            extractor=extractor,
            max_concurrent=3,
        )
        print(f"Success rate: {results.success_rate:.1f}%")
        ```
    """
    if not pdf_paths:
        return BatchResult(total=0, successful=0, failed=0, cached=0)

    start_time = datetime.now()
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks for all PDFs
    tasks = [
        extract_single(
            pdf_path=pdf_path,
            extract_func=extract_func,
            extractor=extractor,
            semaphore=semaphore,
            progress_callback=progress_callback,
            current_index=i,
            total_count=len(pdf_paths),
        )
        for i, pdf_path in enumerate(pdf_paths)
    ]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Aggregate results
    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    cached = sum(1 for r in results if r.from_cache)
    total_duration = (datetime.now() - start_time).total_seconds()

    return BatchResult(
        total=len(pdf_paths),
        successful=successful,
        failed=failed,
        cached=cached,
        results=list(results),
        total_duration_seconds=total_duration,
    )


def print_progress(current: int, total: int, pdf_path: Path, success: bool) -> None:
    """Default progress callback that prints to console."""
    status = "✓" if success else "✗"
    print(f"  [{current}/{total}] {status} {pdf_path.name}")


def print_batch_summary(result: BatchResult) -> None:
    """Print a summary of batch extraction results."""
    print(f"\n{'='*60}")
    print("BATCH EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total PDFs:     {result.total}")
    print(f"Successful:     {result.successful}")
    print(f"Failed:         {result.failed}")
    print(f"From cache:     {result.cached}")
    print(f"Success rate:   {result.success_rate:.1f}%")
    print(f"Total time:     {result.total_duration_seconds:.2f}s")

    if result.successful > 0:
        avg_time = sum(r.duration_seconds for r in result.results if r.success) / result.successful
        print(f"Avg time/PDF:   {avg_time:.2f}s")

    if result.failed > 0:
        print(f"\nFailed PDFs:")
        for r in result.results:
            if not r.success:
                print(f"  - {r.pdf_path.name}: {r.error}")

    print(f"{'='*60}\n")
