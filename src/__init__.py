"""
VLM-based PDF Extractor for Insurance Commission Data.

This package provides intelligent PDF extraction using Vision Language Models
via OpenRouter API, with local caching and Pydantic validation.
"""

__version__ = "0.1.0"

# Expose key classes at package level for convenience
from .pipeline import (
    Pipeline,
    SourceType,
    PipelineResult,
    BatchResult,
    get_pipeline,
    extract_pdf,
    extract_pdf_sync,
)
from .utils.data_unifier import BoardType, DataUnifier
from .clients.monday import MondayClient, UploadResult

__all__ = [
    # Pipeline
    "Pipeline",
    "SourceType",
    "PipelineResult",
    "BatchResult",
    "get_pipeline",
    "extract_pdf",
    "extract_pdf_sync",
    # Data unification
    "BoardType",
    "DataUnifier",
    # Monday.com
    "MondayClient",
    "UploadResult",
]
