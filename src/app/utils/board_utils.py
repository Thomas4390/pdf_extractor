"""
Board utility functions for Monday.com integration.

Provides helpers for sorting, filtering, and loading boards.
"""

import re
import threading
from typing import Optional

import pandas as pd
import streamlit as st

from src.app.utils.async_helpers import run_async


# Global storage for background loading results (thread-safe)
_background_agg_data: dict = {}
_background_agg_loading: bool = False
_background_agg_error: Optional[str] = None
_background_agg_progress: dict = {}


def sort_and_filter_boards(boards: list, search_query: str = "") -> list:
    """Sort boards with priority keywords first and filter by search query."""
    if not boards:
        return []

    filtered_boards = boards
    if search_query and search_query.strip():
        search_lower = search_query.lower().strip()
        filtered_boards = [b for b in boards if search_lower in b['name'].lower()]

    priority_1_keywords = ['paiement', 'historique']
    priority_2_keywords = ['vente', 'production']

    def get_priority(board_name: str) -> tuple:
        name_lower = board_name.lower()
        if any(kw in name_lower for kw in priority_1_keywords):
            return (0, name_lower)
        if any(kw in name_lower for kw in priority_2_keywords):
            return (1, name_lower)
        return (2, name_lower)

    return sorted(filtered_boards, key=lambda b: get_priority(b['name']))


def detect_board_type_from_name(board_name: str) -> str:
    """Detect the board type based on regex patterns in the board name."""
    if not board_name:
        return "Paiements Historiques"

    name_lower = board_name.lower()

    sales_patterns = [
        r'vente[s]?', r'production[s]?', r'sales?', r'prod\b',
        r'commercial', r'soumis', r'proposition[s]?',
    ]

    payment_patterns = [
        r'paiement[s]?', r'historique[s]?', r'payment[s]?', r'history',
        r'hist\b', r'reçu[s]?', r'commission[s]?', r'statement[s]?',
    ]

    for pattern in sales_patterns:
        if re.search(pattern, name_lower):
            return "Ventes et Production"

    for pattern in payment_patterns:
        if re.search(pattern, name_lower):
            return "Paiements Historiques"

    return "Paiements Historiques"


def load_boards_async(force_rerun: bool = False) -> None:
    """Load Monday.com boards automatically when API key is set."""
    # Import here to avoid circular imports
    from src.app.state import get_pipeline

    if (st.session_state.monday_api_key and
        st.session_state.monday_boards is None and
        not st.session_state.boards_loading):
        try:
            st.session_state.boards_loading = True
            st.session_state.boards_error = None
            pipeline = get_pipeline()
            if pipeline.monday_configured:
                boards = run_async(pipeline.monday.list_boards())
                st.session_state.monday_boards = boards
            st.session_state.boards_loading = False
            if force_rerun:
                st.rerun()
        except Exception as e:
            st.session_state.boards_loading = False
            st.session_state.boards_error = str(e)


def _load_aggregation_data_thread(api_key: str, selected_sources: dict) -> None:
    """
    Background thread function to load aggregation source data.

    Args:
        api_key: Monday.com API key
        selected_sources: Dict of {source_key: board_id}
    """
    global _background_agg_data, _background_agg_loading, _background_agg_error, _background_agg_progress

    from src.clients.monday import MondayClient
    from src.utils.aggregator import SOURCE_BOARDS

    _background_agg_loading = True
    _background_agg_error = None
    _background_agg_data = {}
    _background_agg_progress = {"current": 0, "total": len(selected_sources), "current_source": ""}

    try:
        client = MondayClient(api_key=api_key)

        for idx, (source_key, board_id) in enumerate(selected_sources.items()):
            config = SOURCE_BOARDS.get(source_key)
            if not config:
                continue

            _background_agg_progress = {
                "current": idx,
                "total": len(selected_sources),
                "current_source": config.display_name,
            }

            try:
                items = client.extract_board_data_sync(board_id)
                df = client.board_items_to_dataframe(items)
                _background_agg_data[source_key] = df
            except Exception:
                _background_agg_data[source_key] = pd.DataFrame()

        _background_agg_progress = {
            "current": len(selected_sources),
            "total": len(selected_sources),
            "current_source": "Terminé",
        }

    except Exception as e:
        _background_agg_error = str(e)
    finally:
        _background_agg_loading = False


def start_background_aggregation_load() -> bool:
    """
    Start loading aggregation data in the background.

    Returns:
        True if loading was started, False if already loading or no API key
    """
    global _background_agg_loading, _background_agg_data

    if _background_agg_loading:
        return False

    if _background_agg_data and not _background_agg_error:
        return False

    api_key = st.session_state.get("monday_api_key")
    if not api_key:
        return False

    selected_sources = st.session_state.get("agg_selected_sources", {})
    if not selected_sources:
        return False

    thread = threading.Thread(
        target=_load_aggregation_data_thread,
        args=(api_key, selected_sources),
        daemon=True,
    )
    thread.start()

    return True


def get_background_aggregation_status() -> dict:
    """Get the current status of background aggregation loading."""
    return {
        "loading": _background_agg_loading,
        "error": _background_agg_error,
        "data": _background_agg_data,
        "progress": _background_agg_progress,
    }


def apply_background_aggregation_data() -> bool:
    """Apply the background-loaded data to session state."""
    global _background_agg_data

    if _background_agg_loading:
        return False

    if not _background_agg_data:
        return False

    st.session_state.agg_source_data = _background_agg_data.copy()
    st.session_state.agg_data_loaded = True

    return True


def reset_background_aggregation_data() -> None:
    """Reset the background aggregation data."""
    global _background_agg_data, _background_agg_loading, _background_agg_error, _background_agg_progress

    _background_agg_data = {}
    _background_agg_loading = False
    _background_agg_error = None
    _background_agg_progress = {}
