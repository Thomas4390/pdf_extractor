"""
Configuration management for the VLM PDF Extractor.

Handles environment variables and application settings using Pydantic.
Supports both local .env files and Streamlit Cloud secrets.
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


def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get secret from Streamlit secrets or environment variables.

    Priority: Streamlit secrets > Environment variables > Default
    """
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    # Fall back to environment variables
    return os.environ.get(key, default)


_BOARD_IDS = {
    "prod": {
        "paiement_historique": 8553813876,
        "vente_production": 9423464449,
        "ae_tracker": 9142978904,
        "data": 9142121714,
    },
    "test": {
        "paiement_historique": 18402704159,
        "vente_production": 18402704199,
        "ae_tracker": 9142978904,
        "data": 18402704401,
    },
}


_resolved_boards: Optional[dict[str, int]] = None
_resolved_env: Optional[str] = None


def _resolve_board_ids() -> dict[str, int]:
    """Resolve board IDs based on ENVIRONMENT setting (test/prod).

    Priority for environment: env var > .env file > Streamlit secrets > default.
    Priority for board IDs: Streamlit secrets boards.{env} > hardcoded _BOARD_IDS.

    Results are cached for the process lifetime.
    """
    global _resolved_boards, _resolved_env

    if _resolved_boards is not None:
        return _resolved_boards

    # Snapshot env var BEFORE any Streamlit import (Streamlit may overwrite
    # os.environ with values from secrets.toml).
    env = os.environ.get("ENVIRONMENT", "").strip().lower() or None

    # Fallback to Streamlit secrets / default
    if not env:
        env = (_get_secret("ENVIRONMENT") or "prod").strip().lower()

    if env not in _BOARD_IDS:
        env = "prod"

    _resolved_env = env

    # Try Streamlit secrets boards table first
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "boards" in st.secrets:
            boards_section = st.secrets["boards"]
            if env in boards_section:
                tbl = boards_section[env]
                _resolved_boards = {
                    "paiement_historique": int(tbl.get("paiement_historique", _BOARD_IDS[env]["paiement_historique"])),
                    "vente_production": int(tbl.get("vente_production", _BOARD_IDS[env]["vente_production"])),
                    "ae_tracker": int(tbl.get("ae_tracker", _BOARD_IDS[env]["ae_tracker"])),
                    "data": int(tbl.get("data", _BOARD_IDS[env]["data"])),
                }
                return _resolved_boards
    except Exception:
        pass

    _resolved_boards = _BOARD_IDS[env]
    return _resolved_boards


def get_environment() -> str:
    """Return the resolved environment name ('test' or 'prod')."""
    _resolve_board_ids()  # ensure resolution happened
    return _resolved_env or "prod"


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Environment (test / prod)
    environment: str = Field(
        default="prod",
        alias="ENVIRONMENT",
        description="Environment: 'test' or 'prod' — selects Monday.com board IDs",
    )

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
        default=200,
        ge=72,
        le=600,
        description="DPI for PDF to image conversion",
    )
    vlm_timeout_per_page: float = Field(
        default=60.0,
        description="Timeout in seconds per page for adaptive VLM timeout calculation",
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

    # Monday.com Board IDs — resolved dynamically from ENVIRONMENT
    @property
    def monday_board_paiement_historique(self) -> int:
        return _resolve_board_ids()["paiement_historique"]

    @property
    def monday_board_vente_production(self) -> int:
        return _resolve_board_ids()["vente_production"]

    @property
    def monday_board_ae_tracker(self) -> int:
        return _resolve_board_ids()["ae_tracker"]

    @property
    def monday_board_data(self) -> int:
        return _resolve_board_ids()["data"]

    # Monday.com Client
    monday_batch_size: int = Field(
        default=50, ge=1, le=200,
        description="Number of items per batch upload to Monday.com",
    )
    monday_max_concurrent: int = Field(
        default=5, ge=1, le=20,
        description="Maximum concurrent requests to Monday.com API",
    )
    monday_rate_limit_delay: float = Field(
        default=0.5, ge=0.0, le=5.0,
        description="Delay in seconds between Monday.com API requests",
    )
    monday_max_retries: int = Field(
        default=3, ge=1, le=10,
        description="Maximum retries for transient Monday.com errors",
    )
    monday_retry_base_delay: float = Field(
        default=2.0, ge=0.5, le=30.0,
        description="Base delay in seconds for retry backoff",
    )

    # Verification
    verification_tolerance_pct: float = Field(
        default=10.0, ge=1.0, le=50.0,
        description="Tolerance percentage for Reçu vs PA verification",
    )

    # Advisor matching
    advisor_fuzzy_threshold: float = Field(
        default=0.85, ge=0.5, le=1.0,
        description="Minimum similarity ratio for fuzzy advisor name matching",
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
    Retrieve OpenRouter API key from Streamlit secrets or environment.

    Priority: Streamlit secrets > Environment variables > Settings

    Returns:
        str: The API key

    Raises:
        ValueError: If API key is not configured
    """
    # Try Streamlit secrets first, then environment, then settings
    key = _get_secret("OPENROUTER_API_KEY") or settings.openrouter_api_key
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY not configured. "
            "Add it to your .env file, environment variables, or Streamlit secrets."
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
