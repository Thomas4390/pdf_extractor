"""
IDC Statement (Trailing Fees) VLM-based PDF extractor.

Extracts trailing fee data from IDC "Détails des frais de suivi"
PDF reports using vision language models.

Supports two extraction modes:
1. Raw extraction: Copies "Nom du client" column as-is for later parsing
2. Direct extraction: VLM parses structured fields directly from PDF
"""

import logging
from pathlib import Path
from typing import Any, Union

from ..models.idc_statement import IDCStatementReport, IDCStatementReportParsed, IDCTrailingFeeRaw
from ..utils.config import settings
from ..utils.model_registry import ExtractionMode, get_model_config
from ..utils.pdf import get_pdf_hash
from ..utils.prompt_loader import load_prompts
from .base import BaseExtractor

logger = logging.getLogger(__name__)


class IDCStatementExtractor(BaseExtractor[IDCStatementReportParsed]):
    """
    Extractor for IDC Statement (trailing fees) reports.

    These PDFs contain:
    - Report header with advisor info
    - "Détails des frais de suivi" table with columns:
      Nom du client, Numéro de compte, Compagnie, Produit, Date,
      Concessionnaire, Frais de suivi brut, Frais de suivi nets
    - Multiple pages with repeating table structure

    DEFAULT BEHAVIOR (direct parsing):
    - extract(): DIRECT parsing with structured fields (company_code, advisor_name,
      client_first_name, client_last_name, etc.)

    Alternative methods:
    - extract_raw(): Use raw prompts (raw_client_data as-is without parsing)
    - extract_direct(): Explicit direct parsing (same as extract())

    Page configuration: Skips first 2 pages (cover/summary) - defined in model_registry.
    """

    @property
    def source_name(self) -> str:
        return "idc_statement"

    @property
    def document_type(self) -> str:
        return "IDC_STATEMENT"

    @property
    def model_class(self) -> type[IDCStatementReportParsed]:
        # Use parsed model by default since we use direct prompts
        return IDCStatementReportParsed

    @property
    def extraction_mode(self) -> ExtractionMode:
        """Get extraction mode from model registry."""
        return get_model_config(self.document_type).mode

    @property
    def _prompt_config(self):
        """Get prompt configuration from YAML file."""
        return load_prompts(self.source_name)

    @property
    def system_prompt(self) -> str:
        """System prompt for IDC Statement extraction (DIRECT mode by default)."""
        # Use direct prompt by default for better field parsing
        prompt = self._prompt_config.system_prompt_direct
        if not prompt:
            # Fallback to raw prompt if direct not available
            return self._prompt_config.system_prompt
        return prompt

    @property
    def user_prompt(self) -> str:
        """User prompt for IDC Statement extraction (DIRECT mode by default)."""
        # Use direct prompt by default for better field parsing
        prompt = self._prompt_config.user_prompt_direct
        if not prompt:
            # Fallback to raw prompt if direct not available
            return self._prompt_config.user_prompt
        return prompt

    @property
    def system_prompt_raw(self) -> str:
        """System prompt for RAW extraction (copies Nom du client as-is)."""
        return self._prompt_config.system_prompt

    @property
    def user_prompt_raw(self) -> str:
        """User prompt for RAW extraction (copies Nom du client as-is)."""
        return self._prompt_config.user_prompt

    @property
    def system_prompt_direct(self) -> str:
        """System prompt for DIRECT extraction with parsed fields (from YAML)."""
        prompt = self._prompt_config.system_prompt_direct
        if not prompt:
            raise ValueError("No system_prompt_direct found in YAML config")
        return prompt

    @property
    def user_prompt_direct(self) -> str:
        """User prompt for DIRECT extraction with parsed fields (from YAML)."""
        prompt = self._prompt_config.user_prompt_direct
        if not prompt:
            raise ValueError("No user_prompt_direct found in YAML config")
        return prompt

    # Note: Page selection (skip first 2 pages) is handled by base class
    # using model_registry.PageConfig(skip_first=2) defined for IDC_STATEMENT

    async def extract_direct(
        self,
        pdf_path: Union[str, Path],
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """
        Extract data with DIRECT parsing of structured fields.

        This method parses the "Nom du client" column directly
        into structured fields (company_code, advisor_name, etc.) in a
        single extraction step.

        Uses text or vision mode based on model registry configuration.

        Args:
            pdf_path: Path to the PDF file
            force_refresh: If True, ignore cache and re-extract

        Returns:
            Extracted data with parsed fields as dictionary
        """
        pdf_path = Path(pdf_path)
        # Use different cache key for direct extraction
        pdf_hash = get_pdf_hash(pdf_path) + "_direct"
        mode = self.extraction_mode

        # Check cache first
        if not force_refresh and settings.cache_enabled:
            cached = self.cache.get(pdf_hash)
            if cached is not None:
                logger.info(f"Cache hit for {pdf_path.name} (direct)")
                return cached

        logger.info(f"Direct extraction from {pdf_path.name} (mode: {mode.value})")

        if mode == ExtractionMode.TEXT:
            # Text mode: extract text and send to LLM
            pdf_text = self.get_text(pdf_path)
            user_prompt_with_text = f"{self.user_prompt_direct}\n\n--- PDF TEXT CONTENT ---\n{pdf_text}"

            logger.info(f"Sending extracted text to LLM for direct extraction ({len(pdf_text)} chars)")

            result = await self.client.extract_with_text(
                system_prompt=self.system_prompt_direct,
                user_prompt=user_prompt_with_text,
            )
        else:
            # Vision mode: send images to VLM
            all_images = self.get_images(pdf_path)

            logger.info(f"Sending {len(all_images)} pages to VLM (direct extraction)")

            result = await self.client.extract_with_vision(
                images=all_images,
                system_prompt=self.system_prompt_direct,
                user_prompt=self.user_prompt_direct,
            )

        # Post-process: normalize advisor names using AdvisorMatcher
        result = self._normalize_advisor_names(result)

        # Cache the result
        if settings.cache_enabled:
            self.cache.set(pdf_hash, result, metadata={
                "source": "idc_statement_direct",
                "filename": pdf_path.name,
                "mode": mode.value,
            })

        return result

    def _normalize_advisor_names(self, result: dict[str, Any]) -> dict[str, Any]:
        """
        Post-process extraction result to normalize advisor names.

        Uses the AdvisorMatcher to map raw advisor names to standardized names.
        """
        from ..utils.advisor_matcher import normalize_advisor_name_or_original

        trailing_fees = result.get("trailing_fees", [])

        for fee in trailing_fees:
            raw_advisor = fee.get("advisor_name")
            if raw_advisor:
                # Normalize using AdvisorMatcher (database-driven)
                normalized = normalize_advisor_name_or_original(raw_advisor)
                fee["advisor_name"] = normalized
                logger.debug(f"Normalized advisor: {raw_advisor} → {normalized}")

        return result

    async def extract_direct_validated(
        self,
        pdf_path: Union[str, Path],
        force_refresh: bool = False,
    ) -> IDCStatementReportParsed:
        """
        Extract with direct parsing and validate against Pydantic model.

        Args:
            pdf_path: Path to the PDF file
            force_refresh: If True, ignore cache and re-extract

        Returns:
            Validated IDCStatementReportParsed instance
        """
        data = await self.extract_direct(pdf_path, force_refresh)
        return IDCStatementReportParsed(**data)

    def is_direct_cached(self, pdf_path: Union[str, Path]) -> bool:
        """Check if a PDF's direct extraction is cached."""
        pdf_hash = get_pdf_hash(pdf_path) + "_direct"
        return self.cache.exists(pdf_hash)

    def invalidate_direct_cache(self, pdf_path: Union[str, Path]) -> bool:
        """Remove a PDF's cached direct extraction."""
        pdf_hash = get_pdf_hash(pdf_path) + "_direct"
        return self.cache.invalidate(pdf_hash)


async def extract_idc_statement_report(
    pdf_path: Union[str, Path],
    force_refresh: bool = False,
    raw: bool = False,
) -> Union[IDCStatementReport, IDCStatementReportParsed]:
    """
    Convenience function to extract an IDC Statement report.

    Args:
        pdf_path: Path to the IDC Statement PDF
        force_refresh: If True, ignore cache
        raw: If True, use RAW extraction (copies raw_client_data without parsing).
             Default is DIRECT mode with parsed fields.

    Returns:
        IDCStatementReportParsed by default (with parsed fields)
        IDCStatementReport if raw=True (with raw_client_data only)
    """
    extractor = IDCStatementExtractor()
    if raw:
        # Use raw prompts - not implemented yet, use extract_direct for now
        # TODO: Add extract_raw_mode() method if needed
        return await extractor.extract(pdf_path, force_refresh=force_refresh)
    return await extractor.extract(pdf_path, force_refresh=force_refresh)
