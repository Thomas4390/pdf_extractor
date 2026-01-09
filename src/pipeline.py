"""
Pipeline Orchestrator for PDF Extraction.

Orchestrates the complete workflow:
PDF → VLM Extraction → Data Unification → Monday.com Upload

Features:
- Auto-detection of PDF source type
- Batch parallel processing (configurable limit)
- Error handling with partial data support
- Progress callbacks for UI integration
"""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd

from .extractors import (
    AssomptionExtractor,
    BaseExtractor,
    IDCExtractor,
    IDCStatementExtractor,
    UVExtractor,
)
from .clients.monday import MondayClient, UploadResult
from .utils.data_unifier import BoardType, DataUnifier
from .utils.advisor_matcher import get_advisor_matcher


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MAX_PARALLEL = 3


class SourceType(str, Enum):
    """Supported PDF source types."""
    UV = "UV"
    IDC = "IDC"
    IDC_STATEMENT = "IDC_STATEMENT"
    ASSOMPTION = "ASSOMPTION"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExtractionWarning:
    """Warning generated during extraction."""
    message: str
    pdf_path: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PipelineResult:
    """Result of processing a single PDF."""
    pdf_path: str
    source: SourceType
    board_type: BoardType
    dataframe: pd.DataFrame
    success: bool
    warnings: list[ExtractionWarning] = field(default_factory=list)
    error: Optional[str] = None
    extraction_time_ms: int = 0
    row_count: int = 0

    def __post_init__(self):
        self.row_count = len(self.dataframe) if self.dataframe is not None else 0


@dataclass
class BatchResult:
    """Result of processing multiple PDFs."""
    results: list[PipelineResult] = field(default_factory=list)
    total_pdfs: int = 0
    successful: int = 0
    failed: int = 0
    total_rows: int = 0
    processing_time_ms: int = 0

    def add(self, result: PipelineResult):
        """Add a result to the batch."""
        self.results.append(result)
        self.total_pdfs += 1
        if result.success:
            self.successful += 1
            self.total_rows += result.row_count
        else:
            self.failed += 1

    @property
    def all_warnings(self) -> list[ExtractionWarning]:
        """Get all warnings from all results."""
        warnings = []
        for result in self.results:
            warnings.extend(result.warnings)
        return warnings

    def get_combined_dataframe(self, board_type: Optional[BoardType] = None) -> pd.DataFrame:
        """
        Combine all successful results into a single DataFrame.

        Args:
            board_type: Optional filter by board type

        Returns:
            Combined DataFrame
        """
        dfs = []
        for result in self.results:
            if result.success and not result.dataframe.empty:
                if board_type is None or result.board_type == board_type:
                    # Add source column for tracking
                    df = result.dataframe.copy()
                    df["_source_file"] = Path(result.pdf_path).name
                    df["_source_type"] = result.source.value
                    dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)


# =============================================================================
# PIPELINE
# =============================================================================

class Pipeline:
    """
    Main orchestrator for PDF extraction pipeline.

    Workflow:
    1. Detect PDF source type (or use provided)
    2. Extract data using appropriate VLM extractor
    3. Unify data to standardized DataFrame
    4. Optionally upload to Monday.com

    Example:
        pipeline = Pipeline()

        # Single PDF
        result = await pipeline.process_pdf("path/to/file.pdf")
        print(result.dataframe)

        # Batch processing
        results = await pipeline.process_batch(["file1.pdf", "file2.pdf"])
        combined = results.get_combined_dataframe()
    """

    # Patterns for auto-detecting source type from path/filename
    SOURCE_PATTERNS: dict[str, SourceType] = {
        "uv": SourceType.UV,
        "idc_statement": SourceType.IDC_STATEMENT,
        "idc-statement": SourceType.IDC_STATEMENT,
        "statement": SourceType.IDC_STATEMENT,
        "frais de suivi": SourceType.IDC_STATEMENT,
        "idc": SourceType.IDC,
        "proposition": SourceType.IDC,
        "assomption": SourceType.ASSOMPTION,
        "remuneration": SourceType.ASSOMPTION,
    }

    def __init__(
        self,
        monday_api_key: Optional[str] = None,
        max_parallel: int = DEFAULT_MAX_PARALLEL,
        use_advisor_matcher: bool = True,
    ):
        """
        Initialize the pipeline.

        Args:
            monday_api_key: Optional Monday.com API key for uploads.
                           If not provided, reads from MONDAY_API_KEY env var.
            max_parallel: Maximum PDFs to process in parallel (default: 3)
            use_advisor_matcher: Whether to normalize advisor names (default: True)
        """
        # Initialize extractors
        self._extractors: dict[SourceType, BaseExtractor] = {
            SourceType.UV: UVExtractor(),
            SourceType.IDC: IDCExtractor(),
            SourceType.IDC_STATEMENT: IDCStatementExtractor(),
            SourceType.ASSOMPTION: AssomptionExtractor(),
        }

        # Initialize unifier with optional advisor matcher
        advisor_matcher = get_advisor_matcher() if use_advisor_matcher else None
        self._unifier = DataUnifier(advisor_matcher=advisor_matcher)

        # Initialize Monday client if key provided
        api_key = monday_api_key or os.getenv("MONDAY_API_KEY")
        self._monday_client = MondayClient(api_key=api_key) if api_key else None

        # Semaphore for parallel processing
        self._semaphore = asyncio.Semaphore(max_parallel)
        self._max_parallel = max_parallel

    # -------------------------------------------------------------------------
    # Source detection
    # -------------------------------------------------------------------------

    def detect_source(self, pdf_path: Union[str, Path]) -> SourceType:
        """
        Auto-detect the source type from PDF path.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Detected SourceType

        Raises:
            ValueError: If source cannot be determined
        """
        path_str = str(pdf_path).lower()
        filename = Path(pdf_path).name.lower()

        # Check patterns in order (more specific first)
        for pattern, source in self.SOURCE_PATTERNS.items():
            if pattern in path_str or pattern in filename:
                return source

        raise ValueError(
            f"Cannot detect source type for: {pdf_path}\n"
            f"Please specify source explicitly or ensure the path contains "
            f"one of: {list(self.SOURCE_PATTERNS.keys())}"
        )

    # -------------------------------------------------------------------------
    # Single PDF processing
    # -------------------------------------------------------------------------

    async def process_pdf(
        self,
        pdf_path: Union[str, Path],
        source: Optional[Union[str, SourceType]] = None,
        force_refresh: bool = False,
    ) -> PipelineResult:
        """
        Process a single PDF through the complete pipeline.

        Args:
            pdf_path: Path to the PDF file
            source: Source type (auto-detected if not provided)
            force_refresh: If True, bypass cache and re-extract

        Returns:
            PipelineResult with DataFrame and metadata
        """
        import time
        start_time = time.time()

        pdf_path = Path(pdf_path)
        warnings: list[ExtractionWarning] = []

        # Validate file exists
        if not pdf_path.exists():
            return PipelineResult(
                pdf_path=str(pdf_path),
                source=SourceType.UV,  # Placeholder
                board_type=BoardType.SALES_PRODUCTION,
                dataframe=pd.DataFrame(),
                success=False,
                error=f"File not found: {pdf_path}"
            )

        # Detect or validate source
        try:
            if source is None:
                source_type = self.detect_source(pdf_path)
            elif isinstance(source, str):
                source_type = SourceType(source.upper())
            else:
                source_type = source
        except (ValueError, KeyError) as e:
            return PipelineResult(
                pdf_path=str(pdf_path),
                source=SourceType.UV,
                board_type=BoardType.SALES_PRODUCTION,
                dataframe=pd.DataFrame(),
                success=False,
                error=f"Invalid source: {e}"
            )

        # Process with semaphore for rate limiting
        async with self._semaphore:
            try:
                # Step 1: Extract using VLM
                extractor = self._extractors[source_type]
                report = await extractor.extract(pdf_path, force_refresh=force_refresh)

                # Step 2: Unify to DataFrame
                df, board_type = self._unifier.unify(report, source_type.value)

                # Check for empty results
                if df.empty:
                    warnings.append(ExtractionWarning(
                        message="Extraction returned no data rows",
                        pdf_path=str(pdf_path)
                    ))

                elapsed_ms = int((time.time() - start_time) * 1000)

                return PipelineResult(
                    pdf_path=str(pdf_path),
                    source=source_type,
                    board_type=board_type,
                    dataframe=df,
                    success=True,
                    warnings=warnings,
                    extraction_time_ms=elapsed_ms
                )

            except Exception as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                return PipelineResult(
                    pdf_path=str(pdf_path),
                    source=source_type,
                    board_type=BoardType.SALES_PRODUCTION,
                    dataframe=pd.DataFrame(),
                    success=False,
                    error=str(e),
                    extraction_time_ms=elapsed_ms
                )

    # -------------------------------------------------------------------------
    # Batch processing
    # -------------------------------------------------------------------------

    async def process_batch(
        self,
        pdf_paths: list[Union[str, Path]],
        source: Optional[Union[str, SourceType]] = None,
        force_refresh: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> BatchResult:
        """
        Process multiple PDFs in parallel.

        Args:
            pdf_paths: List of PDF file paths
            source: Common source type (auto-detected per file if not provided)
            force_refresh: If True, bypass cache and re-extract all
            progress_callback: Optional callback(current, total, filename)

        Returns:
            BatchResult with all results and statistics
        """
        import time
        start_time = time.time()

        batch_result = BatchResult()

        # Create tasks for all PDFs
        async def process_with_callback(idx: int, path: Union[str, Path]) -> PipelineResult:
            result = await self.process_pdf(path, source=source, force_refresh=force_refresh)
            if progress_callback:
                progress_callback(idx + 1, len(pdf_paths), Path(path).name)
            return result

        # Execute all tasks with parallel limit enforced by semaphore
        tasks = [
            process_with_callback(i, path)
            for i, path in enumerate(pdf_paths)
        ]

        results = await asyncio.gather(*tasks)

        # Aggregate results
        for result in results:
            batch_result.add(result)

        batch_result.processing_time_ms = int((time.time() - start_time) * 1000)

        return batch_result

    def process_batch_sync(
        self,
        pdf_paths: list[Union[str, Path]],
        source: Optional[Union[str, SourceType]] = None,
        force_refresh: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> BatchResult:
        """Synchronous wrapper for process_batch."""
        return asyncio.run(
            self.process_batch(
                pdf_paths=pdf_paths,
                source=source,
                force_refresh=force_refresh,
                progress_callback=progress_callback
            )
        )

    # -------------------------------------------------------------------------
    # Monday.com upload
    # -------------------------------------------------------------------------

    async def upload_to_monday(
        self,
        result: Union[PipelineResult, BatchResult],
        board_id: int,
        group_id: Optional[str] = None,
        create_missing_columns: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> UploadResult:
        """
        Upload pipeline results to Monday.com.

        Args:
            result: PipelineResult or BatchResult to upload
            board_id: Target board ID
            group_id: Optional group ID
            create_missing_columns: Auto-create missing columns (default: True)
            progress_callback: Optional callback(current, total)

        Returns:
            UploadResult with statistics

        Raises:
            ValueError: If Monday client not configured
        """
        if not self._monday_client:
            raise ValueError(
                "Monday.com client not configured. "
                "Provide monday_api_key or set MONDAY_API_KEY environment variable."
            )

        # Get DataFrame based on result type
        if isinstance(result, PipelineResult):
            if not result.success or result.dataframe.empty:
                return UploadResult(total=0, success=0, failed=0)
            df = result.dataframe
        else:
            # BatchResult
            df = result.get_combined_dataframe()
            if df.empty:
                return UploadResult(total=0, success=0, failed=0)

        # Remove internal columns before upload
        upload_df = df.drop(
            columns=[c for c in df.columns if c.startswith("_")],
            errors="ignore"
        )

        return await self._monday_client.upload_dataframe(
            df=upload_df,
            board_id=board_id,
            group_id=group_id,
            create_missing_columns=create_missing_columns,
            progress_callback=progress_callback
        )

    def upload_to_monday_sync(
        self,
        result: Union[PipelineResult, BatchResult],
        board_id: int,
        group_id: Optional[str] = None,
        create_missing_columns: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> UploadResult:
        """Synchronous wrapper for upload_to_monday."""
        return asyncio.run(
            self.upload_to_monday(
                result=result,
                board_id=board_id,
                group_id=group_id,
                create_missing_columns=create_missing_columns,
                progress_callback=progress_callback
            )
        )

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def is_cached(self, pdf_path: Union[str, Path], source: Optional[SourceType] = None) -> bool:
        """Check if a PDF's extraction is cached."""
        if source is None:
            try:
                source = self.detect_source(pdf_path)
            except ValueError:
                return False

        extractor = self._extractors.get(source)
        if extractor:
            return extractor.is_cached(pdf_path)
        return False

    def invalidate_cache(self, pdf_path: Union[str, Path], source: Optional[SourceType] = None) -> bool:
        """Remove a PDF's cached extraction."""
        if source is None:
            try:
                source = self.detect_source(pdf_path)
            except ValueError:
                return False

        extractor = self._extractors.get(source)
        if extractor:
            return extractor.invalidate_cache(pdf_path)
        return False

    @property
    def supported_sources(self) -> list[SourceType]:
        """List of supported source types."""
        return list(self._extractors.keys())

    @property
    def monday_configured(self) -> bool:
        """Check if Monday.com client is configured."""
        return self._monday_client is not None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_pipeline(
    monday_api_key: Optional[str] = None,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
) -> Pipeline:
    """
    Factory function to get a Pipeline instance.

    Args:
        monday_api_key: Optional Monday.com API key
        max_parallel: Maximum parallel PDF processing

    Returns:
        Configured Pipeline instance
    """
    return Pipeline(
        monday_api_key=monday_api_key,
        max_parallel=max_parallel
    )


async def extract_pdf(
    pdf_path: Union[str, Path],
    source: Optional[str] = None,
) -> pd.DataFrame:
    """
    Quick extraction function for single PDF.

    Args:
        pdf_path: Path to PDF file
        source: Source type (auto-detected if not provided)

    Returns:
        Extracted and unified DataFrame
    """
    pipeline = Pipeline()
    result = await pipeline.process_pdf(pdf_path, source=source)

    if not result.success:
        raise RuntimeError(f"Extraction failed: {result.error}")

    return result.dataframe


def extract_pdf_sync(
    pdf_path: Union[str, Path],
    source: Optional[str] = None,
) -> pd.DataFrame:
    """Synchronous wrapper for extract_pdf."""
    return asyncio.run(extract_pdf(pdf_path, source=source))
