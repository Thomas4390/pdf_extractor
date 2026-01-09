"""Utility modules for PDF processing and configuration."""

from .config import get_openrouter_api_key, get_cache_dir, settings
from .pdf import pdf_to_images, pdf_to_text, get_pdf_hash, get_pdf_page_count
from .model_registry import (
    ExtractionMode,
    ModelConfig,
    PageConfig,
    get_model_config,
    get_pages_for_extraction,
    register_model,
    list_document_types,
)
from .raw_data_parser import (
    ParsedClientData,
    parse_raw_client_data,
    parse_raw_entries_batch,
)
from .batch import (
    ExtractionResult,
    BatchResult,
    extract_single,
    extract_batch,
    print_progress,
    print_batch_summary,
)
from .advisor_matcher import (
    Advisor,
    AdvisorMatcher,
    get_advisor_matcher,
    normalize_advisor_name,
    normalize_advisor_name_or_original,
)
from .data_unifier import (
    BoardType,
    DataUnifier,
)
from .prompt_loader import (
    PromptConfig,
    load_prompts,
    get_prompt,
    clear_prompt_cache,
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
