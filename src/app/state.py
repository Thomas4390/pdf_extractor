"""
Session state management for the Streamlit application.

Centralizes all session state initialization, access, and reset logic.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

from src.pipeline import Pipeline
from src.utils.aggregator import SOURCE_BOARDS
from src.utils.data_unifier import BoardType


def get_default_selected_sources() -> dict[str, int]:
    """Get default source board selections from SOURCE_BOARDS config.

    Returns:
        Dict mapping source_key to board_id for sources that have a default board_id
    """
    result = {}
    for source_key, config in SOURCE_BOARDS.items():
        # Use getattr to handle cached versions without board_id attribute
        board_id = getattr(config, 'board_id', None)
        if board_id is not None:
            result[source_key] = board_id
    return result


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get secret from Streamlit secrets or environment."""
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.

    Removes directory components and special characters that could
    be used to escape the intended directory.

    Args:
        filename: Original filename from user upload

    Returns:
        Safe filename with only the base name and allowed characters
    """
    # Extract only the base filename (remove any path components)
    safe_name = Path(filename).name

    # Remove any remaining path separators and null bytes
    safe_name = safe_name.replace('/', '_').replace('\\', '_').replace('\x00', '')

    # Remove leading dots to prevent hidden files
    safe_name = safe_name.lstrip('.')

    # If empty after sanitization, use a default name
    if not safe_name:
        safe_name = "uploaded_file.pdf"

    return safe_name


def cleanup_temp_files() -> None:
    """Clean up temporary files from previous sessions."""
    temp_paths = st.session_state.get('temp_pdf_paths', [])
    if temp_paths:
        for path in temp_paths:
            try:
                if isinstance(path, Path) and path.exists():
                    # Get parent temp directory
                    temp_dir = path.parent
                    if temp_dir.exists() and str(temp_dir).startswith(tempfile.gettempdir()):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        break  # All files are in same temp dir
            except Exception:
                pass  # Ignore cleanup errors
        st.session_state.temp_pdf_paths = []


def init_session_state() -> None:
    """Initialize all session state variables."""
    defaults = {
        # Stage (Phase 1: Multi-stage wizard)
        "stage": 1,  # 1=Configuration, 2=Preview, 3=Upload

        # Pipeline
        "pipeline": None,

        # File upload
        "uploaded_files": [],
        "temp_pdf_paths": [],

        # Extraction
        "extraction_results": {},
        "batch_result": None,
        "combined_data": None,
        "extraction_usage": None,  # UsageStats for cost/model tracking
        "selected_model": None,  # Custom model override for extraction
        "file_group_overrides": {},  # Per-file group overrides {filename: group_name}

        # Processing state
        "is_processing": False,
        "processing_progress": 0.0,
        "current_file": "",

        # Monday.com
        "monday_api_key": get_secret("MONDAY_API_KEY"),
        "monday_boards": None,
        "selected_board_id": None,
        "selected_group_id": None,
        "selected_board_type": BoardType.HISTORICAL_PAYMENTS,
        "monday_groups": None,
        "upload_result": None,
        "is_uploading": False,
        "_current_board_name": "",
        "boards_loading": False,
        "boards_error": None,

        # Options
        "selected_source": None,
        "force_refresh": False,
        "data_modified": False,
        "upload_key_counter": 0,  # Counter to reset file uploader widget

        # Advisor management (Phase 2)
        "advisor_matcher": None,

        # Verification (Phase 3)
        "verification_tolerance": 10.0,

        # UI state
        "show_columns": False,

        # Aggregation mode (Phase 5)
        "app_mode": "extraction",  # "extraction", "aggregation", or "column_conversion"

        # Column conversion mode
        "conv_board_id": None,  # Selected board ID for conversion
        "conv_column_id": None,  # Selected column ID to convert
        "conv_column_title": "Conseiller",  # Default column name to convert
        "conv_result": None,  # Migration result dict
        "conv_is_executing": False,  # True during migration
        "conv_use_mapping": True,  # Whether to use advisor name mapping
        "agg_step": 1,  # 1-4 for aggregation wizard
        "agg_selected_sources": get_default_selected_sources(),  # {source_key: board_id}
        "agg_period": None,  # DatePeriod enum
        "agg_target_board_id": None,
        "agg_source_data": {},  # {source_key: DataFrame} - raw data from Monday.com
        "agg_data_loaded": False,  # True when raw data has been loaded
        "agg_filtered_data": {},  # {source_key: DataFrame}
        "agg_aggregated_data": {},  # {source_key: DataFrame}
        "agg_combined_data": None,  # Combined DataFrame
        "agg_edited_data": None,  # Edited DataFrame (may differ from combined)
        "agg_upsert_result": None,
        "agg_is_loading": False,
        "agg_is_executing": False,
        "agg_use_custom_group": False,  # Toggle for manual group name
        "agg_custom_group_name": "",  # Custom group name input
        "agg_unknown_advisors": [],  # List of unknown advisor names filtered out
        "agg_detail_advisor": "Tous",  # Selected advisor in detail tab
        "agg_detail_source": "Toutes",  # Selected source in detail tab
        "agg_flexible_period": None,  # FlexiblePeriod object for flexible date selection
        "agg_metrics_loaded": False,  # True when metrics have been imported
        "agg_metrics_group": "",  # Group name from which metrics were loaded
        "agg_validation_passed": True,  # Validation status for pre-upload check
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get_pipeline() -> Pipeline:
    """Get or create Pipeline instance."""
    if st.session_state.pipeline is None:
        st.session_state.pipeline = Pipeline(
            monday_api_key=st.session_state.monday_api_key,
            max_parallel=3,
            use_advisor_matcher=True
        )
    return st.session_state.pipeline


def reset_pipeline() -> None:
    """Reset pipeline state to start over."""
    keys_to_reset = [
        'stage', 'uploaded_files', 'temp_pdf_paths', 'extraction_results',
        'batch_result', 'combined_data', 'extraction_usage', 'is_processing', 'processing_progress',
        'current_file', 'selected_board_id', 'selected_group_id', 'monday_groups',
        'upload_result', 'is_uploading', 'selected_source', 'data_modified',
        'show_columns', '_current_board_name', 'selected_model', 'file_group_overrides',
        'extraction_error', 'extraction_traceback'
    ]
    for key in keys_to_reset:
        if key == 'stage':
            st.session_state[key] = 1
        elif key in ['uploaded_files', 'temp_pdf_paths']:
            st.session_state[key] = []
        elif key in ['extraction_results', 'file_group_overrides']:
            st.session_state[key] = {}
        elif key in ['is_processing', 'is_uploading', 'data_modified', 'show_columns']:
            st.session_state[key] = False
        elif key == 'processing_progress':
            st.session_state[key] = 0.0
        else:
            st.session_state[key] = None

    # Increment upload key counter to reset file uploader widget
    st.session_state.upload_key_counter = st.session_state.get('upload_key_counter', 0) + 1


def reset_aggregation_state() -> None:
    """Reset aggregation mode state."""
    st.session_state.agg_step = 1
    st.session_state.agg_selected_sources = get_default_selected_sources()
    st.session_state.agg_period = None
    st.session_state.agg_target_board_id = None
    st.session_state.agg_source_data = {}
    st.session_state.agg_data_loaded = False
    st.session_state.agg_filtered_data = {}
    st.session_state.agg_aggregated_data = {}
    st.session_state.agg_combined_data = None
    st.session_state.agg_edited_data = None
    st.session_state.agg_upsert_result = None
    st.session_state.agg_is_loading = False
    st.session_state.agg_is_executing = False
    st.session_state.agg_use_custom_group = False
    st.session_state.agg_custom_group_name = ""
    st.session_state.agg_unknown_advisors = []
    st.session_state.agg_detail_advisor = "Tous"
    st.session_state.agg_detail_source = "Toutes"
    # FlexiblePeriod and metrics state
    st.session_state.agg_flexible_period = None
    st.session_state.agg_metrics_loaded = False
    st.session_state.agg_metrics_group = ""
    st.session_state.agg_validation_passed = True
