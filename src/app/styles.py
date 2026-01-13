"""
Styles module for the Commission Pipeline Streamlit application.

Contains all CSS styles and theming for the modern teal/gray design system.
"""

import streamlit as st


def apply_custom_styles() -> None:
    """Apply custom CSS styles to the Streamlit application."""
    st.markdown(get_css(), unsafe_allow_html=True)


def get_css() -> str:
    """Return the complete CSS stylesheet."""
    return """
<style>
    /* ===========================================
       Modern Design System
       Palette: Teal/Gray Professional Theme
       =========================================== */

    /* CSS Variables */
    :root {
        --primary-500: #3d5a80;
        --primary-600: #2c4a6e;
        --accent-400: #4fd1c5;
        --accent-500: #38b2ac;
        --accent-600: #319795;
        --success: #48bb78;
        --warning: #ed8936;
        --error: #f56565;
        --surface: #ffffff;
        --surface-hover: #f7fafc;
        --border: #e2e8f0;
        --border-focus: #38b2ac;
        --text-primary: #1a202c;
        --text-secondary: #718096;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.07);
        --shadow-lg: 0 10px 15px rgba(0,0,0,0.1);
    }

    /* Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    /* Main container */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    /* ===========================================
       BUTTONS - Modern Gradient Style
       =========================================== */

    /* Primary button */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, var(--accent-500) 0%, var(--accent-600) 100%);
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-sm);
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(56, 178, 172, 0.35);
    }

    /* Secondary/default button */
    .stButton > button {
        border-radius: 12px;
        font-weight: 500;
        padding: 0.55rem 1.25rem;
        border: 1.5px solid var(--border);
        background: var(--surface);
        color: var(--text-primary);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: var(--accent-500);
        background: var(--surface-hover);
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }

    /* ===========================================
       METRICS - Large & Bold
       =========================================== */

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
    }

    /* ===========================================
       CARDS & EXPANDERS
       =========================================== */

    [data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 16px;
        background: var(--surface);
        box-shadow: var(--shadow-sm);
        transition: box-shadow 0.2s ease;
    }
    [data-testid="stExpander"]:hover {
        box-shadow: var(--shadow-md);
    }
    [data-testid="stExpander"] summary {
        font-weight: 600;
        color: var(--text-primary);
    }

    /* ===========================================
       TABS - Modern Pill Style
       =========================================== */

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--surface-hover);
        padding: 6px;
        border-radius: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--surface);
        box-shadow: var(--shadow-sm);
    }

    /* ===========================================
       FORMS & INPUTS
       =========================================== */

    [data-testid="stForm"] {
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        background: var(--surface);
    }

    /* Text inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div {
        border-radius: 10px;
        border: 1.5px solid var(--border);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus-within {
        border-color: var(--accent-500);
        box-shadow: 0 0 0 3px rgba(56, 178, 172, 0.15);
    }

    /* ===========================================
       DATAFRAME - Clean Table Style
       =========================================== */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border);
    }
    [data-testid="stDataFrame"] th {
        background: var(--primary-500) !important;
        color: white !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.03em;
    }
    [data-testid="stDataFrame"] tr:nth-child(even) {
        background: var(--surface-hover);
    }
    [data-testid="stDataFrame"] tr:hover {
        background: rgba(56, 178, 172, 0.08);
    }

    /* ===========================================
       FILE UPLOADER - Drag & Drop Zone
       =========================================== */

    [data-testid="stFileUploader"] {
        border: 2px dashed var(--border);
        border-radius: 16px;
        padding: 1rem;
        background: var(--surface-hover);
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-500);
        background: rgba(56, 178, 172, 0.05);
    }

    /* ===========================================
       ALERTS & STATUS BOXES
       =========================================== */

    .success-box {
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.1) 0%, rgba(72, 187, 120, 0.05) 100%);
        border: 1px solid var(--success);
        border-radius: 12px;
        padding: 1rem;
    }
    .warning-box {
        background: linear-gradient(135deg, rgba(237, 137, 54, 0.1) 0%, rgba(237, 137, 54, 0.05) 100%);
        border: 1px solid var(--warning);
        border-radius: 12px;
        padding: 1rem;
    }

    /* ===========================================
       STEPPER STYLES
       =========================================== */

    .stepper-step {
        text-align: center;
        padding: 16px 12px;
        border-radius: 14px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    .stepper-step.current {
        background: linear-gradient(135deg, var(--accent-500) 0%, var(--accent-600) 100%);
        color: white;
        box-shadow: 0 8px 20px rgba(56, 178, 172, 0.3);
    }
    .stepper-step.completed {
        background: linear-gradient(135deg, var(--success) 0%, #38a169 100%);
        color: white;
        cursor: pointer;
    }
    .stepper-step.completed:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    .stepper-step.future {
        background: var(--surface-hover);
        color: var(--text-secondary);
        border: 1px solid var(--border);
    }
    .stepper-step .step-icon {
        font-size: 1.75rem;
        margin-bottom: 4px;
    }
    .stepper-step .step-label {
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* ===========================================
       DASHBOARD HEADER
       =========================================== */

    .metrics-dashboard {
        background: linear-gradient(135deg, var(--primary-500) 0%, var(--primary-600) 100%);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-lg);
    }
    .metrics-dashboard .metric-item {
        text-align: center;
        color: white;
    }
    .metrics-dashboard .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
    }
    .metrics-dashboard .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.85;
    }

    /* ===========================================
       ANIMATIONS
       =========================================== */

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .animate-fade-in {
        animation: fadeInUp 0.4s ease-out;
    }

    /* Reduce spacing */
    .element-container {
        margin-bottom: 0.5rem;
    }

    /* ===========================================
       BREADCRUMB
       =========================================== */

    .breadcrumb {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-bottom: 1rem;
    }
    .breadcrumb-item {
        color: var(--text-secondary);
    }
    .breadcrumb-item.active {
        color: var(--text-primary);
        font-weight: 500;
    }
    .breadcrumb-separator {
        color: var(--border);
    }

    /* ===========================================
       SIDEBAR STYLES
       =========================================== */

    /* Sidebar header */
    .sidebar-header {
        background: linear-gradient(135deg, var(--primary-500) 0%, var(--primary-600) 100%);
        color: white;
        padding: 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sidebar-header h2 {
        color: white;
        margin: 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .sidebar-header .version {
        font-size: 0.75rem;
        opacity: 0.8;
        margin-top: 4px;
    }

    /* Sidebar section */
    .sidebar-section {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .sidebar-section-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Status indicator */
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .status-indicator.connected {
        background: rgba(72, 187, 120, 0.15);
        color: var(--success);
        border: 1px solid rgba(72, 187, 120, 0.3);
    }
    .status-indicator.disconnected {
        background: rgba(245, 101, 101, 0.15);
        color: var(--error);
        border: 1px solid rgba(245, 101, 101, 0.3);
    }
    .status-indicator.loading {
        background: rgba(237, 137, 54, 0.15);
        color: var(--warning);
        border: 1px solid rgba(237, 137, 54, 0.3);
    }

    /* Info card in sidebar */
    .sidebar-info-card {
        background: linear-gradient(135deg, rgba(56, 178, 172, 0.1) 0%, rgba(56, 178, 172, 0.05) 100%);
        border: 1px solid rgba(56, 178, 172, 0.2);
        border-radius: 10px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .sidebar-info-card .label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        margin-bottom: 2px;
    }
    .sidebar-info-card .value {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    /* Help section styling */
    .help-section {
        font-size: 0.85rem;
        line-height: 1.6;
    }
    .help-section h4 {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--accent-600);
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid var(--border);
    }
    .help-section h4:first-child {
        margin-top: 0;
    }
    .help-section ul {
        margin: 0.5rem 0;
        padding-left: 1.25rem;
    }
    .help-section li {
        margin-bottom: 0.35rem;
    }
    .help-section code {
        background: var(--surface-hover);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    .help-tip {
        background: rgba(56, 178, 172, 0.1);
        border-left: 3px solid var(--accent-500);
        padding: 0.5rem 0.75rem;
        border-radius: 0 8px 8px 0;
        margin: 0.75rem 0;
        font-size: 0.8rem;
    }

    /* Sidebar stats */
    .sidebar-stats {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin-top: 0.75rem;
    }
    .sidebar-stat {
        background: var(--surface-hover);
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
    }
    .sidebar-stat .number {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--accent-600);
    }
    .sidebar-stat .label {
        font-size: 0.65rem;
        text-transform: uppercase;
        color: var(--text-secondary);
    }
</style>
"""
