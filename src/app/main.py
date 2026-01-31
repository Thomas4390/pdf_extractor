"""
PDF Extractor - Streamlit Application

Multi-stage wizard application for extracting commission data from PDFs
and uploading to Monday.com.

Features:
- Multi-stage wizard (Configuration -> Preview -> Upload)
- Batch PDF processing with progress tracking
- Advisor management tab with CRUD operations
- Verification of Reçu vs calculated Commission
- Automatic date/group detection from data
- Multi-month file handling
- Excel/CSV file replacement
- Aggregation mode for combining data from multiple boards

Run with: streamlit run src/app/main.py
"""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import application modules
from src.app.styles import apply_custom_styles
from src.app.state import init_session_state
from src.app.sidebar import render_sidebar
from src.app.utils.board_utils import load_boards_async, start_background_aggregation_load

# Import extraction stages
from src.app.extraction import render_stage_1, render_stage_2, render_stage_3

# Import aggregation mode
from src.app.aggregation import render_aggregation_mode

# Import column conversion mode
from src.app.column_conversion import render_column_conversion_mode


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Commission Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

apply_custom_styles()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main() -> None:
    """Main application entry point."""
    # Clear prompt cache once per session to ensure fresh prompts are loaded
    if '_prompt_cache_cleared' not in st.session_state:
        import sys
        # Find and clear the prompt_loader cache if loaded
        for module_name, module in list(sys.modules.items()):
            if 'prompt_loader' in module_name and hasattr(module, 'load_prompts'):
                if hasattr(module.load_prompts, 'cache_clear'):
                    module.load_prompts.cache_clear()
        st.session_state._prompt_cache_cleared = True

    init_session_state()

    # Auto-load boards at startup if API key is available
    if (st.session_state.monday_api_key and
        st.session_state.monday_boards is None and
        not st.session_state.boards_loading):
        load_boards_async()

    # Start background aggregation data loading once boards are available
    if (st.session_state.monday_api_key and
        st.session_state.monday_boards is not None and
        not st.session_state.get("agg_data_loaded", False)):
        start_background_aggregation_load()

    render_sidebar()

    # Route based on mode
    if st.session_state.app_mode == "aggregation":
        render_aggregation_mode()
    elif st.session_state.app_mode == "column_conversion":
        render_column_conversion_mode()
    else:
        # Route to appropriate extraction stage
        if st.session_state.stage == 1:
            render_stage_1()
        elif st.session_state.stage == 2:
            render_stage_2()
        else:
            render_stage_3()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
