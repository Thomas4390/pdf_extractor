"""
Model registry for document type to model configuration mapping.

Defines which VLM/LLM model to use for each document type,
and whether to use vision (images) or text extraction mode.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ExtractionMode(Enum):
    """Mode of extraction from PDF."""
    VISION = "vision"  # Send PDF pages as images
    TEXT = "text"  # Extract text from PDF and send as text


@dataclass
class PageConfig:
    """Configuration for page selection in PDF extraction."""
    # Specific pages to extract (0-indexed). If empty, extracts all pages.
    pages: list[int] = field(default_factory=list)
    # Number of pages to skip from the start (alternative to listing specific pages)
    skip_first: int = 0
    # Number of pages to skip from the end
    skip_last: int = 0


@dataclass
class ModelConfig:
    """Configuration for a model and document extraction."""
    model_id: str
    mode: ExtractionMode
    fallback_model_id: Optional[str] = None
    fallback_mode: Optional[ExtractionMode] = None
    secondary_fallback_model_id: Optional[str] = None
    secondary_fallback_mode: Optional[ExtractionMode] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    # Page configuration for this document type
    page_config: Optional[PageConfig] = None


# Default models
DEFAULT_VISION_MODEL = "qwen/qwen2.5-vl-72b-instruct"  # Primary: fast, cost-effective
ADVANCED_VISION_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"  # Secondary: more capable
DEFAULT_TEXT_MODEL = "deepseek/deepseek-chat"  # Tertiary: text fallback (V3 stable)

# Document type to model configuration mapping
MODEL_REGISTRY: dict[str, ModelConfig] = {
    # UV Assurance - Qwen 2.5 → Qwen3 → DeepSeek
    # All pages are relevant
    "UV": ModelConfig(
        model_id=DEFAULT_VISION_MODEL,
        mode=ExtractionMode.VISION,
        fallback_model_id=ADVANCED_VISION_MODEL,
        fallback_mode=ExtractionMode.VISION,
        secondary_fallback_model_id=DEFAULT_TEXT_MODEL,
        secondary_fallback_mode=ExtractionMode.TEXT,
        page_config=None,  # Use all pages
    ),

    # Assomption Vie - Qwen 2.5 → Qwen3 → DeepSeek
    # Pages: 1 (summary), 3 (commissions), 5 (bonuses) - 0-indexed: 0, 2, 4
    "ASSOMPTION": ModelConfig(
        model_id=DEFAULT_VISION_MODEL,
        mode=ExtractionMode.VISION,
        fallback_model_id=ADVANCED_VISION_MODEL,
        fallback_mode=ExtractionMode.VISION,
        secondary_fallback_model_id=DEFAULT_TEXT_MODEL,
        secondary_fallback_mode=ExtractionMode.TEXT,
        page_config=PageConfig(pages=[0, 2, 4]),  # Summary, Commissions, Bonuses
    ),

    # IDC Propositions - Qwen 2.5 → Qwen3 → DeepSeek
    # All pages are relevant
    "IDC": ModelConfig(
        model_id=DEFAULT_VISION_MODEL,
        mode=ExtractionMode.VISION,
        fallback_model_id=ADVANCED_VISION_MODEL,
        fallback_mode=ExtractionMode.VISION,
        secondary_fallback_model_id=DEFAULT_TEXT_MODEL,
        secondary_fallback_mode=ExtractionMode.TEXT,
        page_config=None,  # Use all pages
    ),

    # IDC Statements (trailing fees) - Complex documents
    # Using Qwen3 VL as primary (most capable), Qwen 2.5 as fallback
    # Skip first 2 pages (cover and summary)
    "IDC_STATEMENT": ModelConfig(
        model_id=ADVANCED_VISION_MODEL,
        mode=ExtractionMode.VISION,
        fallback_model_id=DEFAULT_VISION_MODEL,
        fallback_mode=ExtractionMode.VISION,
        secondary_fallback_model_id=DEFAULT_TEXT_MODEL,
        secondary_fallback_mode=ExtractionMode.TEXT,
        page_config=PageConfig(skip_first=2),  # Skip cover and summary pages
    ),
}


def get_model_config(document_type: str) -> ModelConfig:
    """
    Get model configuration for a document type.

    Args:
        document_type: Document type identifier (e.g., "UV", "IDC_STATEMENT")

    Returns:
        ModelConfig for the document type

    Raises:
        KeyError: If document type is not registered
    """
    doc_type_upper = document_type.upper()
    if doc_type_upper not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown document type: {document_type}. "
            f"Available types: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[doc_type_upper]


def register_model(document_type: str, config: ModelConfig) -> None:
    """
    Register or update model configuration for a document type.

    Args:
        document_type: Document type identifier
        config: Model configuration
    """
    MODEL_REGISTRY[document_type.upper()] = config


def list_document_types() -> list[str]:
    """Get list of all registered document types."""
    return list(MODEL_REGISTRY.keys())


def get_default_vision_model() -> str:
    """Get the default vision model ID."""
    return DEFAULT_VISION_MODEL


def get_default_text_model() -> str:
    """Get the default text-only model ID."""
    return DEFAULT_TEXT_MODEL


def get_pages_for_extraction(
    document_type: str,
    total_pages: int,
) -> list[int]:
    """
    Get the list of page indices to extract for a document type.

    Args:
        document_type: Document type identifier
        total_pages: Total number of pages in the PDF

    Returns:
        List of 0-indexed page numbers to extract
    """
    config = get_model_config(document_type)
    page_config = config.page_config

    if page_config is None:
        # No page config means extract all pages
        return list(range(total_pages))

    if page_config.pages:
        # Specific pages are defined, filter to valid ones
        return [p for p in page_config.pages if 0 <= p < total_pages]

    # Use skip_first and skip_last
    start = page_config.skip_first
    end = total_pages - page_config.skip_last
    return list(range(start, max(start, end)))
