"""
Utility modules for the Streamlit application.

Contains helper functions for async operations, date handling,
board operations, and UI navigation.
"""

from src.app.utils.async_helpers import run_async
from src.app.utils.date_utils import (
    get_months_fr,
    date_to_group,
    detect_date_from_filename,
    detect_groups_from_data,
    analyze_groups_in_data,
)
from src.app.utils.board_utils import (
    sort_and_filter_boards,
    detect_board_type_from_name,
    load_boards_async,
)
from src.app.utils.navigation import (
    render_breadcrumb,
    render_stepper,
)

__all__ = [
    # Async helpers
    "run_async",
    # Date utilities
    "get_months_fr",
    "date_to_group",
    "detect_date_from_filename",
    "detect_groups_from_data",
    "analyze_groups_in_data",
    # Board utilities
    "sort_and_filter_boards",
    "detect_board_type_from_name",
    "load_boards_async",
    # Navigation
    "render_breadcrumb",
    "render_stepper",
]
