"""
Streamlit application for Commission Pipeline.

This package contains:
- main.py: Main application logic and stage rendering
- styles.py: CSS styles and theming
- components.py: Reusable UI components

Run with: streamlit run src/app/main.py
"""

from .styles import apply_custom_styles, get_css
from .components import (
    render_breadcrumb,
    render_stepper,
    render_metrics_dashboard,
    render_upload_dashboard,
    render_success_box,
    verify_recu_vs_com,
    get_verification_stats,
    reorder_columns_for_display,
)

__all__ = [
    "main",
    # Styles
    "apply_custom_styles",
    "get_css",
    # Components
    "render_breadcrumb",
    "render_stepper",
    "render_metrics_dashboard",
    "render_upload_dashboard",
    "render_success_box",
    "verify_recu_vs_com",
    "get_verification_stats",
    "reorder_columns_for_display",
]
