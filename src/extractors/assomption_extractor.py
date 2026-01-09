"""
Assomption Vie VLM-based PDF extractor.

Extracts commission and bonus data from Assomption Vie remuneration reports
using vision language models. Merges data from commission pages (page 3)
with bonus pages (page 5) based on policy number.
"""

from pathlib import Path
from typing import Union

from ..models.assomption import AssomptionReport
from ..utils.prompt_loader import load_prompts
from .base import BaseExtractor


class AssomptionExtractor(BaseExtractor[AssomptionReport]):
    """
    Extractor for Assomption Vie remuneration reports.

    These PDFs have a specific structure:
    - Page 1: Summary page with period and broker info
    - Page 2: Balance report
    - Page 3: Commission details table
    - Page 4: Commission summary
    - Page 5: Surcommission (bonus) details table
    - Page 6: Footer/signatures

    The extractor focuses on pages 1, 3, and 5 to extract all relevant data.
    Page configuration is defined in model_registry (pages 0, 2, 4).
    """

    @property
    def source_name(self) -> str:
        return "assomption"

    @property
    def document_type(self) -> str:
        return "ASSOMPTION"

    @property
    def model_class(self) -> type[AssomptionReport]:
        return AssomptionReport

    @property
    def _prompt_config(self):
        """Get prompt configuration from YAML file."""
        return load_prompts(self.source_name)

    @property
    def system_prompt(self) -> str:
        """System prompt for Assomption Vie extraction (from YAML)."""
        return self._prompt_config.system_prompt

    @property
    def user_prompt(self) -> str:
        """User prompt for Assomption Vie extraction (from YAML)."""
        return self._prompt_config.user_prompt

    # Note: Page selection is handled by base class using model_registry.PageConfig
    # Pages 0, 2, 4 (Summary, Commissions, Bonuses) are defined in MODEL_REGISTRY


async def extract_assomption_report(
    pdf_path: Union[str, Path],
    force_refresh: bool = False,
) -> AssomptionReport:
    """
    Convenience function to extract an Assomption report.

    Args:
        pdf_path: Path to the Assomption Vie PDF
        force_refresh: If True, ignore cache

    Returns:
        Validated AssomptionReport instance
    """
    extractor = AssomptionExtractor()
    return await extractor.extract(pdf_path, force_refresh=force_refresh)
