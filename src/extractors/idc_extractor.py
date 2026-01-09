"""
IDC Propositions VLM-based PDF extractor.

Extracts proposition data from IDC "Rapport des propositions soumises"
PDF reports using vision language models.
"""

from pathlib import Path
from typing import Union

from ..models.idc import IDCReport
from ..utils.prompt_loader import load_prompts
from .base import BaseExtractor


class IDCExtractor(BaseExtractor[IDCReport]):
    """
    Extractor for IDC proposition reports.

    These PDFs contain:
    - Report header with vendor info
    - Table of propositions with columns:
      Assureur, Client, Type de régime, Police, Statut, Date,
      Nombre, Taux de CPA, Couverture, Prime de la police,
      Part prime comm., Comm.
    - TOTAUX section at the bottom
    """

    @property
    def source_name(self) -> str:
        return "idc"

    @property
    def document_type(self) -> str:
        return "IDC"

    @property
    def model_class(self) -> type[IDCReport]:
        return IDCReport

    @property
    def _prompt_config(self):
        """Get prompt configuration from YAML file."""
        return load_prompts(self.source_name)

    @property
    def system_prompt(self) -> str:
        """System prompt for IDC extraction (from YAML)."""
        return self._prompt_config.system_prompt

    @property
    def user_prompt(self) -> str:
        """User prompt for IDC extraction (from YAML)."""
        return self._prompt_config.user_prompt


async def extract_idc_report(
    pdf_path: Union[str, Path],
    force_refresh: bool = False,
) -> IDCReport:
    """
    Convenience function to extract an IDC report.

    Args:
        pdf_path: Path to the IDC propositions PDF
        force_refresh: If True, ignore cache

    Returns:
        Validated IDCReport instance
    """
    extractor = IDCExtractor()
    return await extractor.extract(pdf_path, force_refresh=force_refresh)
