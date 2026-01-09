"""
Prompt management for VLM/LLM extractors.

Provides functions to load prompts from YAML files.
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Prompts directory
PROMPTS_DIR = Path(__file__).parent

# Cache for loaded prompts
_prompts_cache: dict[str, dict] = {}


def load_prompts(document_type: str) -> dict:
    """
    Load prompts for a document type from YAML file.

    Args:
        document_type: Document type identifier (e.g., "UV", "IDC_STATEMENT")

    Returns:
        Dictionary with prompt fields (system_prompt, user_prompt, etc.)

    Raises:
        FileNotFoundError: If prompts file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    doc_type_lower = document_type.lower()

    # Check cache first
    if doc_type_lower in _prompts_cache:
        return _prompts_cache[doc_type_lower]

    # Find the YAML file
    yaml_path = PROMPTS_DIR / f"{doc_type_lower}.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Prompts file not found: {yaml_path}. "
            f"Available prompts: {list_available_prompts()}"
        )

    # Load and cache
    with open(yaml_path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    _prompts_cache[doc_type_lower] = prompts
    logger.debug(f"Loaded prompts for {document_type} from {yaml_path}")

    return prompts


def get_system_prompt(document_type: str, variant: Optional[str] = None) -> str:
    """
    Get system prompt for a document type.

    Args:
        document_type: Document type identifier
        variant: Optional variant (e.g., "direct" for system_prompt_direct)

    Returns:
        System prompt string
    """
    prompts = load_prompts(document_type)
    key = f"system_prompt_{variant}" if variant else "system_prompt"

    if key not in prompts:
        raise KeyError(
            f"System prompt '{key}' not found for {document_type}. "
            f"Available keys: {list(prompts.keys())}"
        )

    return prompts[key]


def get_user_prompt(document_type: str, variant: Optional[str] = None) -> str:
    """
    Get user prompt for a document type.

    Args:
        document_type: Document type identifier
        variant: Optional variant (e.g., "direct" for user_prompt_direct)

    Returns:
        User prompt string
    """
    prompts = load_prompts(document_type)
    key = f"user_prompt_{variant}" if variant else "user_prompt"

    if key not in prompts:
        raise KeyError(
            f"User prompt '{key}' not found for {document_type}. "
            f"Available keys: {list(prompts.keys())}"
        )

    return prompts[key]


def list_available_prompts() -> list[str]:
    """Get list of available prompt files (document types)."""
    return [
        f.stem
        for f in PROMPTS_DIR.glob("*.yaml")
        if not f.name.startswith("_")
    ]


def clear_cache() -> None:
    """Clear the prompts cache."""
    _prompts_cache.clear()


__all__ = [
    "load_prompts",
    "get_system_prompt",
    "get_user_prompt",
    "list_available_prompts",
    "clear_cache",
    "PROMPTS_DIR",
]
