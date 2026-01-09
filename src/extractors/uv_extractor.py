"""
UV Assurance remuneration report extractor.

Uses Vision LLM to extract structured data from UV Assurance PDF reports.
"""

from pathlib import Path
from typing import Union

from ..models.uv import UVReport
from ..utils.prompt_loader import load_prompts
from .base import BaseExtractor


class UVExtractor(BaseExtractor[UVReport]):
    """
    Extractor for UV Assurance remuneration reports.

    Handles PDF reports containing:
    - Report metadata (date, advisor info)
    - Activity table with contracts, insureds, and commissions
    - Total remuneration
    """

    @property
    def source_name(self) -> str:
        return "uv"

    @property
    def document_type(self) -> str:
        return "UV"

    @property
    def model_class(self) -> type[UVReport]:
        return UVReport

    @property
    def _prompt_config(self):
        """Get prompt configuration from YAML file."""
        return load_prompts(self.source_name)

    @property
    def system_prompt(self) -> str:
        """System prompt for UV extraction (from YAML)."""
        return self._prompt_config.system_prompt

    @property
    def user_prompt(self) -> str:
        """User prompt for UV extraction (from YAML)."""
        return self._prompt_config.user_prompt


async def extract_uv_report(
    pdf_path: Union[str, Path],
    force_refresh: bool = False,
) -> UVReport:
    """
    Convenience function to extract a UV report.

    Args:
        pdf_path: Path to the UV Assurance PDF
        force_refresh: If True, ignore cache

    Returns:
        Validated UVReport instance
    """
    extractor = UVExtractor()
    return await extractor.extract(pdf_path, force_refresh=force_refresh)
