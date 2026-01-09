"""VLM-based extractors for different PDF sources."""

from .assomption_extractor import AssomptionExtractor, extract_assomption_report
from .base import BaseExtractor
from .idc_extractor import IDCExtractor, extract_idc_report
from .idc_statement_extractor import IDCStatementExtractor, extract_idc_statement_report
from .uv_extractor import UVExtractor, extract_uv_report

__all__ = [
    # Base class
    "BaseExtractor",
    # Extractor classes
    "AssomptionExtractor",
    "IDCExtractor",
    "IDCStatementExtractor",
    "UVExtractor",
    # Convenience functions
    "extract_assomption_report",
    "extract_idc_report",
    "extract_idc_statement_report",
    "extract_uv_report",
]
