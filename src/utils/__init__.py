"""Utility modules for PDF processing and configuration."""

from .advisor_matcher import (
    Advisor,
    AdvisorMatcher,
    get_advisor_matcher,
    normalize_advisor_name,
    normalize_advisor_name_full,
    normalize_advisor_name_full_or_original,
    normalize_advisor_name_or_original,
)
from .batch import (
    BatchResult,
    ExtractionResult,
    extract_batch,
    extract_single,
    print_batch_summary,
    print_progress,
)
from .config import get_cache_dir, get_openrouter_api_key, settings
from .data_unifier import (
    BoardType,
    DataUnifier,
)
from .model_registry import (
    ExtractionMode,
    ModelConfig,
    PageConfig,
    get_model_config,
    get_pages_for_extraction,
    list_document_types,
    register_model,
)
from .pdf import get_pdf_hash, get_pdf_page_count, pdf_to_images, pdf_to_text
from .prompt_loader import (
    PromptConfig,
    clear_prompt_cache,
    get_prompt,
    load_prompts,
)
from .raw_data_parser import (
    ParsedClientData,
    parse_raw_client_data,
    parse_raw_entries_batch,
)

__all__ = [
    # Config
    "get_openrouter_api_key",
    "get_cache_dir",
    "settings",
    # PDF utilities
    "pdf_to_images",
    "pdf_to_text",
    "get_pdf_hash",
    "get_pdf_page_count",
    # Model registry
    "ExtractionMode",
    "ModelConfig",
    "PageConfig",
    "get_model_config",
    "get_pages_for_extraction",
    "register_model",
    "list_document_types",
    # Raw data parsing
    "ParsedClientData",
    "parse_raw_client_data",
    "parse_raw_entries_batch",
    # Batch processing
    "ExtractionResult",
    "BatchResult",
    "extract_single",
    "extract_batch",
    "print_progress",
    "print_batch_summary",
    # Advisor matching
    "Advisor",
    "AdvisorMatcher",
    "get_advisor_matcher",
    "normalize_advisor_name",
    "normalize_advisor_name_full",
    "normalize_advisor_name_full_or_original",
    "normalize_advisor_name_or_original",
    # Data unification
    "BoardType",
    "DataUnifier",
    # Prompt loading
    "PromptConfig",
    "load_prompts",
    "get_prompt",
    "clear_prompt_cache",
]
