"""
Configuration management for the VLM PDF Extractor.

Handles environment variables and application settings using Pydantic.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load .env from project root
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Keys
    openrouter_api_key: Optional[str] = Field(
        default=None,
        alias="OPENROUTER_API_KEY",
        description="OpenRouter API key for VLM access",
    )

    # VLM Configuration
    vlm_model: str = Field(
        default="qwen/qwen3-vl-235b-a22b-instruct",
        description="Primary Vision LLM model to use",
    )
    vlm_fallback_model: str = Field(
        default="qwen/qwen2.5-vl-72b-instruct",
        description="Fallback Vision LLM model used when primary fails",
    )
    vlm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for VLM responses (lower = more deterministic)",
    )
    vlm_max_retries: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum retry attempts for VLM calls",
    )
    vlm_timeout: float = Field(
        default=400.0,
        description="Timeout in seconds for VLM API calls",
    )
    vlm_max_tokens: int = Field(
        default=16384,
        description="Maximum tokens for VLM response (high to prevent truncation)",
    )

    # Hybrid Mode Configuration
    default_ocr_engine: str = Field(
        default="mistral-ocr",
        description="OCR engine for PDF_NATIVE/HYBRID modes: pdf-text (free) or mistral-ocr (paid)",
    )
    hybrid_analysis_model: str = Field(
        default="deepseek/deepseek-chat",
        description="LLM model for HYBRID mode Phase 2 text analysis",
    )

    # PDF Processing
    pdf_dpi: int = Field(
        default=300,
        ge=72,
        le=600,
        description="DPI for PDF to image conversion",
    )

    # Cache Configuration
    cache_enabled: bool = Field(
        default=True,
        description="Enable local result caching",
    )
    cache_dir: Path = Field(
        default=PROJECT_ROOT / "cache",
        description="Directory for cached extraction results",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def get_settings() -> Settings:
    """Get application settings (fresh instance each time for dev)."""
    return Settings()


# Convenience alias - creates fresh instance
settings = get_settings()


def get_openrouter_api_key() -> str:
    """
    Retrieve OpenRouter API key from environment.

    Returns:
        str: The API key

    Raises:
        ValueError: If API key is not configured
    """
    key = settings.openrouter_api_key
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY not configured. "
            "Add it to your .env file or set it as an environment variable."
        )
    return key


def get_cache_dir() -> Path:
    """
    Get the cache directory path, creating it if necessary.

    Returns:
        Path: The cache directory path
    """
    cache_dir = settings.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
