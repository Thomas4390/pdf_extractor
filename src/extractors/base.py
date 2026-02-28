"""
Base extractor interface for VLM/LLM-based PDF extraction.

Defines the abstract interface that all source-specific extractors must implement.
Supports both vision (images) and text extraction modes.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

from ..clients.cache import ExtractionCache
from ..clients.openrouter import OpenRouterClient
from ..utils.config import get_cache_dir, settings
from ..utils.model_registry import (
    ExtractionMode,
    get_default_text_model,
    get_model_config,
    get_pages_for_extraction,
)
from ..utils.pdf import get_pdf_hash, get_pdf_page_count, pdf_to_images, pdf_to_text

T = TypeVar("T", bound=BaseModel)

MAX_PAGES_PER_CHUNK = 8


class BaseExtractor(ABC, Generic[T]):
    """
    Abstract base class for VLM/LLM-based PDF extractors.

    Provides common functionality for:
    - PDF to image conversion (vision mode)
    - PDF text extraction (text mode)
    - Caching of extraction results
    - OpenRouter API interaction

    Subclasses must implement:
    - system_prompt: The system instructions for the VLM/LLM
    - user_prompt: The user prompt describing what to extract
    - model_class: The Pydantic model for validation
    - source_name: Identifier for this extraction source
    - document_type: Document type for model registry lookup

    Optional overrides:
    - extraction_mode: Override to force a specific mode (vision/text)
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        client: OpenRouterClient | None = None,
    ):
        """
        Initialize the extractor.

        Args:
            cache_dir: Directory for caching results (defaults to settings)
            client: Optional pre-configured OpenRouter client
        """
        self.cache_dir = Path(cache_dir) if cache_dir else get_cache_dir()
        self.cache = ExtractionCache(self.cache_dir)
        self._client = client
        self._configured_client: OpenRouterClient | None = None

    @property
    def client(self) -> OpenRouterClient:
        """Get OpenRouter client configured for this document type."""
        if self._client:
            return self._client

        if self._configured_client is None:
            model_config = get_model_config(self.document_type)
            self._configured_client = OpenRouterClient(
                model=model_config.model_id,
                fallback_model=model_config.fallback_model_id,
                secondary_fallback_model=model_config.secondary_fallback_model_id,
            )
        return self._configured_client

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for the VLM."""
        pass

    @property
    @abstractmethod
    def user_prompt(self) -> str:
        """User prompt describing the extraction task."""
        pass

    @property
    @abstractmethod
    def model_class(self) -> type[T]:
        """Pydantic model class for validation."""
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Source identifier (e.g., 'uv', 'idc')."""
        pass

    @property
    @abstractmethod
    def document_type(self) -> str:
        """Document type for model registry lookup (e.g., 'UV', 'IDC_STATEMENT')."""
        pass

    @property
    def extraction_mode(self) -> ExtractionMode:
        """
        Get extraction mode for this document type.

        Returns mode from model registry by default.
        Override in subclass to force a specific mode.
        """
        return get_model_config(self.document_type).mode

    def get_images(self, pdf_path: str | Path) -> tuple[list[bytes], str]:
        """
        Convert PDF to images for VLM processing.

        Uses page configuration from model_registry to determine which pages to extract.
        Override this method to customize page selection for specific sources.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Tuple of (list of images as bytes, MIME type string)
        """
        pdf_path = Path(pdf_path)
        total_pages = get_pdf_page_count(pdf_path)
        pages = get_pages_for_extraction(self.document_type, total_pages)

        logger.info(
            f"[{self.document_type}] PDF '{pdf_path.name}' has {total_pages} pages, "
            f"extracting pages: {pages}"
        )

        if pages == list(range(total_pages)):
            images, mime_type = pdf_to_images(pdf_path, dpi=settings.pdf_dpi)
        else:
            images, mime_type = pdf_to_images(pdf_path, dpi=settings.pdf_dpi, pages=pages)

        logger.info(f"[{self.document_type}] Converted {len(images)} page(s) to {mime_type} images")
        return images, mime_type

    def get_text(self, pdf_path: str | Path) -> str:
        """
        Extract text from PDF for LLM processing.

        Uses page configuration from model_registry to determine which pages to extract.
        Override this method to customize text extraction for specific sources.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        pdf_path = Path(pdf_path)
        total_pages = get_pdf_page_count(pdf_path)
        pages = get_pages_for_extraction(self.document_type, total_pages)

        logger.info(
            f"[{self.document_type}] PDF '{pdf_path.name}' has {total_pages} pages, "
            f"extracting text from pages: {pages}"
        )

        if pages == list(range(total_pages)):
            # All pages - use default behavior
            text = pdf_to_text(pdf_path)
        else:
            # Specific pages
            text = pdf_to_text(pdf_path, pages=pages)

        logger.info(f"[{self.document_type}] Extracted {len(text)} characters of text")
        return text

    def _merge_chunked_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Merge results from multiple chunked VLM calls.

        Takes the first result as base. For fields that are lists,
        concatenates lists from all chunks. Scalar fields are authoritative
        from the first chunk (subsequent chunks' scalars are ignored).

        Args:
            results: List of raw dict results from each chunk

        Returns:
            Merged dict combining all chunk data
        """
        if not results:
            return {}
        if len(results) == 1:
            return results[0]

        merged = dict(results[0])
        for chunk_idx, subsequent in enumerate(results[1:], start=2):
            for key, value in subsequent.items():
                if isinstance(value, list) and isinstance(merged.get(key), list):
                    before = len(merged[key])
                    merged[key] = merged[key] + value
                    logger.debug(
                        f"[merge] '{key}': chunk {chunk_idx} added {len(value)} items "
                        f"({before} → {len(merged[key])})"
                    )

        # Log summary of merged list fields
        list_summary = {
            k: len(v) for k, v in merged.items() if isinstance(v, list)
        }
        if list_summary:
            logger.info(
                f"[merge] Merged {len(results)} chunks — "
                f"list fields: {list_summary}"
            )

        return merged

    async def extract(
        self,
        pdf_path: str | Path,
        force_refresh: bool = False,
    ) -> T:
        """
        Extract structured data from a PDF.

        Uses vision or text mode based on the document type configuration.

        Args:
            pdf_path: Path to the PDF file
            force_refresh: If True, ignore cache and re-extract

        Returns:
            Validated Pydantic model instance with extracted data

        Raises:
            FileNotFoundError: If PDF doesn't exist
            OpenRouterError: If VLM/LLM extraction fails
            ValidationError: If extracted data doesn't match model
        """
        pdf_path = Path(pdf_path)
        pdf_hash = get_pdf_hash(pdf_path)

        # Check cache unless force refresh
        if not force_refresh and settings.cache_enabled:
            cached = self.cache.get(pdf_hash)
            if cached is not None:
                return self.model_class(**cached)

        mode = self.extraction_mode
        model_config = get_model_config(self.document_type)

        if mode == ExtractionMode.VISION:
            # Vision mode: send PDF pages as images to VLM
            images, mime_type = self.get_images(pdf_path)

            if not images:
                raise ValueError(
                    f"[{self.document_type}] No pages to extract from '{pdf_path.name}'"
                )

            # Adaptive timeout: scale with number of pages, capped at 5x base timeout
            max_timeout = settings.vlm_timeout * 5
            adaptive_timeout = min(
                max(
                    settings.vlm_timeout,
                    len(images) * settings.vlm_timeout_per_page,
                ),
                max_timeout,
            )
            logger.info(
                f"[{self.document_type}] Adaptive timeout: {adaptive_timeout:.0f}s "
                f"for {len(images)} pages (cap: {max_timeout:.0f}s)"
            )

            if len(images) > MAX_PAGES_PER_CHUNK:
                # Chunk pages into batches and merge results
                chunks = [
                    images[i : i + MAX_PAGES_PER_CHUNK]
                    for i in range(0, len(images), MAX_PAGES_PER_CHUNK)
                ]
                logger.info(
                    f"[{self.document_type}] Splitting {len(images)} pages into "
                    f"{len(chunks)} chunks of up to {MAX_PAGES_PER_CHUNK} pages"
                )
                chunk_results: list[dict[str, Any]] = []
                for idx, chunk in enumerate(chunks):
                    logger.info(
                        f"[{self.document_type}] Processing chunk {idx + 1}/{len(chunks)} "
                        f"({len(chunk)} pages)"
                    )
                    raw = await self.client.extract_with_vision(
                        images=chunk,
                        system_prompt=self.system_prompt,
                        user_prompt=self.user_prompt,
                        max_tokens=model_config.max_tokens,
                        mime_type=mime_type,
                        timeout=adaptive_timeout,
                    )
                    chunk_results.append(raw)

                merged = self._merge_chunked_results(chunk_results)
                result = self.model_class(**merged)
            else:
                result = await self.client.validate_and_extract(
                    images=images,
                    system_prompt=self.system_prompt,
                    user_prompt=self.user_prompt,
                    model_class=self.model_class,
                    max_tokens=model_config.max_tokens,
                    mime_type=mime_type,
                    timeout=adaptive_timeout,
                )

        elif mode == ExtractionMode.TEXT:
            # Text mode: extract text via PyMuPDF and send to LLM
            pdf_text = self.get_text(pdf_path)
            user_prompt_with_text = f"{self.user_prompt}\n\n--- PDF TEXT CONTENT ---\n{pdf_text}"
            raw_result = await self.client.extract_with_text(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt_with_text,
                max_tokens=model_config.max_tokens,
            )
            result = self.model_class(**raw_result)

        elif mode == ExtractionMode.PDF_NATIVE:
            # PDF Native mode: send PDF directly via file-parser plugin
            raw_result = await self.client.extract_with_pdf_native(
                pdf_path=str(pdf_path),
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                ocr_engine=model_config.ocr_engine.value,
                max_tokens=model_config.max_tokens,
            )
            result = self.model_class(**raw_result)

        elif mode == ExtractionMode.HYBRID:
            # Hybrid mode: Phase 1 OCR + Phase 2 LLM text analysis
            # Phase 1: Extract text using OCR via file-parser
            ocr_prompt = "Extract ALL text from this PDF document exactly as written. Preserve the structure including tables, lists, and formatting."
            ocr_result = await self.client.extract_with_pdf_native(
                pdf_path=str(pdf_path),
                system_prompt="You are an OCR system. Extract text verbatim.",
                user_prompt=ocr_prompt,
                ocr_engine=model_config.ocr_engine.value,
                model=model_config.text_analysis_model or get_default_text_model(),
            )

            # OCR result might be a dict with content or just text
            if isinstance(ocr_result, dict):
                extracted_text = ocr_result.get("text", str(ocr_result))
            else:
                extracted_text = str(ocr_result)

            # Phase 2: Analyze extracted text with LLM
            analysis_model = model_config.text_analysis_model or get_default_text_model()
            user_prompt_with_text = f"{self.user_prompt}\n\n--- EXTRACTED DOCUMENT TEXT ---\n{extracted_text}"

            raw_result = await self.client.extract_with_text(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt_with_text,
                model=analysis_model,
                max_tokens=model_config.max_tokens,
            )
            result = self.model_class(**raw_result)

        else:
            raise ValueError(f"Unsupported extraction mode: {mode}")

        # Cache the result
        if settings.cache_enabled:
            self.cache.set(
                pdf_hash,
                result.model_dump(mode="json"),
                metadata={
                    "source": self.source_name,
                    "filename": pdf_path.name,
                    "mode": mode.value,
                },
            )

        return result

    async def extract_raw(
        self,
        pdf_path: str | Path,
    ) -> dict[str, Any]:
        """
        Extract data without Pydantic validation.

        Useful for debugging or when schema flexibility is needed.
        Uses vision, text, pdf_native, or hybrid mode based on the document type configuration.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Raw dictionary from VLM/LLM response
        """
        pdf_path = Path(pdf_path)
        mode = self.extraction_mode
        model_config = get_model_config(self.document_type)

        if mode == ExtractionMode.VISION:
            images, mime_type = self.get_images(pdf_path)

            if not images:
                raise ValueError(
                    f"[{self.document_type}] No pages to extract from '{pdf_path.name}'"
                )

            # Adaptive timeout: scale with number of pages, capped at 5x base timeout
            max_timeout = settings.vlm_timeout * 5
            adaptive_timeout = min(
                max(
                    settings.vlm_timeout,
                    len(images) * settings.vlm_timeout_per_page,
                ),
                max_timeout,
            )
            logger.info(
                f"[{self.document_type}] extract_raw adaptive timeout: {adaptive_timeout:.0f}s "
                f"for {len(images)} pages"
            )

            if len(images) > MAX_PAGES_PER_CHUNK:
                chunks = [
                    images[i : i + MAX_PAGES_PER_CHUNK]
                    for i in range(0, len(images), MAX_PAGES_PER_CHUNK)
                ]
                logger.info(
                    f"[{self.document_type}] extract_raw splitting {len(images)} pages into "
                    f"{len(chunks)} chunks"
                )
                chunk_results: list[dict[str, Any]] = []
                for idx, chunk in enumerate(chunks):
                    logger.info(
                        f"[{self.document_type}] extract_raw chunk {idx + 1}/{len(chunks)} "
                        f"({len(chunk)} pages)"
                    )
                    raw = await self.client.extract_with_vision(
                        images=chunk,
                        system_prompt=self.system_prompt,
                        user_prompt=self.user_prompt,
                        max_tokens=model_config.max_tokens,
                        mime_type=mime_type,
                        timeout=adaptive_timeout,
                    )
                    chunk_results.append(raw)
                return self._merge_chunked_results(chunk_results)

            return await self.client.extract_with_vision(
                images=images,
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                max_tokens=model_config.max_tokens,
                mime_type=mime_type,
                timeout=adaptive_timeout,
            )

        elif mode == ExtractionMode.TEXT:
            pdf_text = self.get_text(pdf_path)
            user_prompt_with_text = f"{self.user_prompt}\n\n--- PDF TEXT CONTENT ---\n{pdf_text}"
            return await self.client.extract_with_text(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt_with_text,
                max_tokens=model_config.max_tokens,
            )

        elif mode == ExtractionMode.PDF_NATIVE:
            return await self.client.extract_with_pdf_native(
                pdf_path=str(pdf_path),
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                ocr_engine=model_config.ocr_engine.value,
                max_tokens=model_config.max_tokens,
            )

        elif mode == ExtractionMode.HYBRID:
            # Phase 1: OCR extraction
            ocr_prompt = "Extract ALL text from this PDF document exactly as written. Preserve the structure."
            ocr_result = await self.client.extract_with_pdf_native(
                pdf_path=str(pdf_path),
                system_prompt="You are an OCR system. Extract text verbatim.",
                user_prompt=ocr_prompt,
                ocr_engine=model_config.ocr_engine.value,
                model=model_config.text_analysis_model or get_default_text_model(),
            )

            if isinstance(ocr_result, dict):
                extracted_text = ocr_result.get("text", str(ocr_result))
            else:
                extracted_text = str(ocr_result)

            # Phase 2: LLM analysis
            analysis_model = model_config.text_analysis_model or get_default_text_model()
            user_prompt_with_text = f"{self.user_prompt}\n\n--- EXTRACTED DOCUMENT TEXT ---\n{extracted_text}"

            return await self.client.extract_with_text(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt_with_text,
                model=analysis_model,
                max_tokens=model_config.max_tokens,
            )

        else:
            raise ValueError(f"Unsupported extraction mode: {mode}")

    def is_cached(self, pdf_path: str | Path) -> bool:
        """Check if a PDF's extraction is cached."""
        pdf_hash = get_pdf_hash(pdf_path)
        return self.cache.exists(pdf_hash)

    def invalidate_cache(self, pdf_path: str | Path) -> bool:
        """Remove a PDF's cached extraction."""
        pdf_hash = get_pdf_hash(pdf_path)
        return self.cache.invalidate(pdf_hash)
