"""
Prompt loader utility for loading extraction prompts from YAML files.

Provides a centralized way to load and access prompts for different document types.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Directory containing prompt YAML files
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class PromptConfig:
    """Configuration for a document type's extraction prompts."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @property
    def document_type(self) -> str:
        """Document type identifier."""
        return self._data.get("document_type", "")

    @property
    def system_prompt(self) -> str:
        """System prompt for extraction."""
        return self._data.get("system_prompt", "")

    @property
    def user_prompt(self) -> str:
        """User prompt for extraction."""
        return self._data.get("user_prompt", "")

    @property
    def system_prompt_direct(self) -> str | None:
        """System prompt for direct extraction mode (if available)."""
        return self._data.get("system_prompt_direct")

    @property
    def user_prompt_direct(self) -> str | None:
        """User prompt for direct extraction mode (if available)."""
        return self._data.get("user_prompt_direct")

    def has_direct_prompts(self) -> bool:
        """Check if direct extraction prompts are available."""
        return bool(self.system_prompt_direct and self.user_prompt_direct)


@lru_cache(maxsize=32)
def load_prompts(source_name: str) -> PromptConfig:
    """
    Load prompts for a given source from YAML file.

    Args:
        source_name: Name of the source (e.g., 'uv', 'idc_statement')

    Returns:
        PromptConfig with loaded prompts

    Raises:
        FileNotFoundError: If prompt file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    yaml_path = PROMPTS_DIR / f"{source_name}.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {yaml_path}")

    logger.debug(f"Loading prompts from {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return PromptConfig(data)


def get_prompt(source_name: str, prompt_type: str) -> str:
    """
    Get a specific prompt for a source.

    Args:
        source_name: Name of the source (e.g., 'uv', 'idc_statement')
        prompt_type: Type of prompt ('system_prompt', 'user_prompt',
                     'system_prompt_direct', 'user_prompt_direct')

    Returns:
        The prompt string

    Raises:
        FileNotFoundError: If prompt file doesn't exist
        ValueError: If prompt type is not found
    """
    config = load_prompts(source_name)
    prompt = getattr(config, prompt_type, None)

    if prompt is None:
        raise ValueError(f"Prompt type '{prompt_type}' not found for source '{source_name}'")

    return prompt


def clear_prompt_cache() -> None:
    """Clear the prompt loading cache (useful for testing)."""
    load_prompts.cache_clear()
