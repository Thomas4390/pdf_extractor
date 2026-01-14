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

Run with: streamlit run src/app/main.py
"""

import asyncio
import io
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Coroutine, Optional, TypeVar

T = TypeVar('T')


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine safely, handling existing event loops.

    This function handles the common issue where asyncio.run() fails
    when called from within an existing event loop (e.g., in Streamlit
    or Jupyter environments).

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(coro)

    # Event loop already running - use nest_asyncio if available,
    # otherwise run in a separate thread
    try:
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except ImportError:
        # nest_asyncio not available - run in thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()

import pandas as pd
import streamlit as st

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import Pipeline, SourceType, BatchResult, UsageStats
from src.utils.data_unifier import BoardType
from src.utils.model_registry import get_available_models, get_model_config

# Import UI modules
from src.app.styles import apply_custom_styles
from src.app.components import (
    render_metrics_dashboard,
    render_upload_dashboard,
    render_success_box,
    verify_recu_vs_com,
    get_verification_stats,
    reorder_columns_for_display,
)


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
# CUSTOM CSS - Applied from styles module
# =============================================================================

apply_custom_styles()


# =============================================================================
# SESSION STATE
# =============================================================================

def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get secret from Streamlit secrets or environment."""
    import os
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
    import shutil

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


# =============================================================================
# PHASE 4: DATE/GROUP DETECTION UTILITIES
# =============================================================================

def get_months_fr() -> dict:
    """Retourne le dictionnaire des mois en français."""
    return {
        1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
        5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
        9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
    }


def date_to_group(date_val, fallback_group: str = None) -> str:
    """
    Convertit une date en nom de groupe "Mois YYYY".

    Args:
        date_val: Date (string YYYY-MM-DD, YYYY/MM/DD, datetime, ou Timestamp)
        fallback_group: Groupe à utiliser si la date n'est pas parsable

    Returns:
        str: Nom du groupe (ex: "Octobre 2025")
    """
    months_fr = get_months_fr()

    # Si None ou NaN, utiliser fallback ou date du jour
    if date_val is None or pd.isna(date_val):
        if fallback_group:
            return fallback_group
        now = datetime.now()
        return f"{months_fr[now.month]} {now.year}"

    # Gérer les Timestamp pandas directement
    if isinstance(date_val, pd.Timestamp):
        return f"{months_fr[date_val.month]} {date_val.year}"

    # Gérer les datetime
    if isinstance(date_val, datetime):
        return f"{months_fr[date_val.month]} {date_val.year}"

    date_str = str(date_val).strip()

    # Pattern YYYY-MM-DD ou YYYY/MM/DD
    match = re.match(r'(\d{4})[-/](\d{2})[-/](\d{2})', date_str)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        if 1 <= month <= 12:
            return f"{months_fr[month]} {year}"

    # Pattern DD/MM/YYYY ou DD-MM-YYYY
    match = re.match(r'(\d{2})[-/](\d{2})[-/](\d{4})', date_str)
    if match:
        month = int(match.group(2))
        year = int(match.group(3))
        if 1 <= month <= 12:
            return f"{months_fr[month]} {year}"

    # Essayer de parser avec pandas
    try:
        parsed = pd.to_datetime(date_str)
        if pd.notna(parsed):
            return f"{months_fr[parsed.month]} {parsed.year}"
    except Exception:
        pass

    # Fallback
    if fallback_group:
        return fallback_group
    now = datetime.now()
    return f"{months_fr[now.month]} {now.year}"


def detect_date_from_filename(filename: str) -> Optional[str]:
    """
    Détecte la date/mois à partir du nom de fichier PDF.

    Patterns supportés:
    - rappportremun_21622_2025-10-20.pdf -> "Octobre 2025"
    - Rapport des propositions soumises.20251017_1517.pdf -> "Octobre 2025"
    - 20251017_report.pdf -> "Octobre 2025"

    Returns:
        str: Nom du groupe (ex: "Octobre 2025") ou None si non détecté
    """
    months_fr = get_months_fr()

    # Patterns de date dans le nom de fichier
    patterns = [
        (r'(\d{4})-(\d{2})-(\d{2})', 1, 2),      # 2025-10-20
        (r'\.(\d{4})(\d{2})(\d{2})_', 1, 2),     # .20251017_
        (r'_(\d{4})(\d{2})(\d{2})', 1, 2),       # _20251017
        (r'^(\d{4})(\d{2})(\d{2})', 1, 2),       # 20251017 at start
    ]

    for pattern, year_pos, month_pos in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                year = int(match.group(year_pos))
                month = int(match.group(month_pos))
                if 1 <= month <= 12 and 2020 <= year <= 2030:
                    return f"{months_fr[month]} {year}"
            except (ValueError, IndexError):
                continue

    return None


def detect_groups_from_data(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Analyse le DataFrame extrait et assigne un groupe à chaque ligne basé sur la date.

    Args:
        df: DataFrame avec les données extraites
        source: Type de source (UV, IDC, IDC_STATEMENT, ASSOMPTION)

    Returns:
        DataFrame avec colonne '_target_group' ajoutée par ligne
    """
    df = df.copy()

    # Trouver la colonne de date
    date_column = None
    for col in ['Date', 'date', 'Émission', 'effective_date', 'report_date']:
        if col in df.columns:
            non_null_count = df[col].notna().sum()
            if non_null_count > 0:
                date_column = col
                break

    if date_column is None:
        # Pas de colonne de date - utiliser date du jour
        months_fr = get_months_fr()
        now = datetime.now()
        default_group = f"{months_fr[now.month]} {now.year}"
        df['_target_group'] = default_group
        return df

    # Assigner un groupe à chaque ligne basé sur sa date
    df['_target_group'] = df[date_column].apply(date_to_group)

    return df


def analyze_groups_in_data(df: pd.DataFrame) -> dict:
    """
    Analyse les groupes présents dans un DataFrame.

    Returns:
        {
            'unique_groups': ['Octobre 2025', 'Novembre 2025', ...],
            'spans_multiple_months': True/False,
            'group_counts': {'Octobre 2025': 15, 'Novembre 2025': 3}
        }
    """
    if '_target_group' not in df.columns:
        return {
            'unique_groups': [],
            'spans_multiple_months': False,
            'group_counts': {}
        }

    unique_groups = df['_target_group'].unique().tolist()
    group_counts = df['_target_group'].value_counts().to_dict()

    return {
        'unique_groups': unique_groups,
        'spans_multiple_months': len(unique_groups) > 1,
        'group_counts': group_counts
    }


# =============================================================================
# BOARD HELPERS
# =============================================================================

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


# =============================================================================
# PHASE 1: STEPPER COMPONENT (Enhanced with Click Navigation)
# =============================================================================

def render_breadcrumb() -> None:
    """Render breadcrumb navigation showing current context."""
    parts = ["Accueil"]

    if st.session_state.selected_source:
        parts.append(st.session_state.selected_source)

    if st.session_state.uploaded_files:
        file_count = len(st.session_state.uploaded_files)
        parts.append(f"{file_count} fichier{'s' if file_count > 1 else ''}")

    if st.session_state._current_board_name:
        board_name = st.session_state._current_board_name
        if len(board_name) > 25:
            board_name = board_name[:22] + "..."
        parts.append(f'Board "{board_name}"')

    breadcrumb_html = '<div class="breadcrumb">'
    for i, part in enumerate(parts):
        is_active = i == len(parts) - 1
        breadcrumb_html += f'<span class="breadcrumb-item{"" if not is_active else " active"}">{part}</span>'
        if i < len(parts) - 1:
            breadcrumb_html += '<span class="breadcrumb-separator">›</span>'
    breadcrumb_html += '</div>'

    st.markdown(breadcrumb_html, unsafe_allow_html=True)


def render_stepper() -> None:
    """Render the clickable progress stepper in main content area."""
    stages = [
        ("1", "Configuration", "📁"),
        ("2", "Prévisualisation", "🔍"),
        ("3", "Upload", "☁️")
    ]

    cols = st.columns(3)
    for i, (num, name, icon) in enumerate(stages):
        stage_num = i + 1
        with cols[i]:
            is_current = stage_num == st.session_state.stage
            is_completed = stage_num < st.session_state.stage
            is_future = stage_num > st.session_state.stage

            # Determine CSS class
            if is_current:
                css_class = "current"
            elif is_completed:
                css_class = "completed"
            else:
                css_class = "future"

            # Render step visual
            display_icon = "✅" if is_completed else icon
            st.markdown(f"""
            <div class="stepper-step {css_class}">
                <div class="step-icon">{display_icon}</div>
                <div class="step-label">{name}</div>
            </div>
            """, unsafe_allow_html=True)

            # Add clickable button for completed stages
            if is_completed:
                if st.button(f"← Retour", key=f"stepper_nav_{stage_num}", width="stretch"):
                    # Reset extraction state when going back to stage 1
                    if stage_num == 1:
                        st.session_state.combined_data = None
                        st.session_state.extraction_results = {}
                        st.session_state.batch_result = None
                        st.session_state.extraction_usage = None
                        st.session_state.upload_result = None
                        st.session_state.file_group_overrides = {}
                    st.session_state.stage = stage_num
                    st.rerun()


# =============================================================================
# PHASE 7: SIDEBAR
# =============================================================================

def render_sidebar() -> None:
    """Render enhanced sidebar with better design and comprehensive help."""
    with st.sidebar:
        # Header with app branding
        st.markdown("""
        <div class="sidebar-header">
            <h2>📊 Commission Pipeline</h2>
            <div class="version">v1.0 • Extraction & Upload</div>
        </div>
        """, unsafe_allow_html=True)

        # Connection status section
        st.markdown('<div class="sidebar-section-title">🔗 Connexions</div>', unsafe_allow_html=True)

        api_from_secrets = get_secret('MONDAY_API_KEY') is not None

        # Monday.com API status
        if st.session_state.monday_api_key:
            status_text = "API Secrets" if api_from_secrets else "API Connectée"
            st.markdown(f"""
            <div class="status-indicator connected">
                <span>✓</span> <span>Monday.com: {status_text}</span>
            </div>
            """, unsafe_allow_html=True)

            if not api_from_secrets:
                if st.button("Déconnecter", key="disconnect_api", width="stretch"):
                    st.session_state.monday_api_key = None
                    st.session_state.monday_boards = None
                    st.rerun()
        else:
            st.markdown("""
            <div class="status-indicator disconnected">
                <span>✗</span> <span>Monday.com: Non connecté</span>
            </div>
            """, unsafe_allow_html=True)

            api_key = st.text_input(
                "Clé API Monday.com",
                type="password",
                placeholder="Entrez votre clé API...",
                key="sidebar_api_key",
                label_visibility="collapsed"
            )
            if api_key:
                if st.button("🔌 Connecter", type="primary", width="stretch"):
                    st.session_state.monday_api_key = api_key
                    st.rerun()

            st.caption("💡 Ou configurez `MONDAY_API_KEY` dans secrets.toml")

        # Board status with stats
        if st.session_state.monday_api_key:
            st.markdown("---")
            if st.session_state.boards_loading:
                st.markdown("""
                <div class="status-indicator loading">
                    <span>⏳</span> <span>Chargement des boards...</span>
                </div>
                """, unsafe_allow_html=True)
            elif st.session_state.get('boards_error'):
                st.error(f"Erreur: {st.session_state.boards_error}")
                if st.button("🔄 Réessayer", width="stretch", type="primary"):
                    st.session_state.boards_error = None
                    st.session_state.monday_boards = None
                    load_boards_async(force_rerun=True)
            elif st.session_state.monday_boards:
                board_count = len(st.session_state.monday_boards)
                st.markdown(f"""
                <div class="sidebar-info-card">
                    <div class="label">Boards disponibles</div>
                    <div class="value">📋 {board_count} boards</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("🔄 Rafraîchir", width="stretch"):
                    st.session_state.monday_boards = None
                    load_boards_async(force_rerun=True)
            else:
                if st.button("📥 Charger les boards", width="stretch", type="primary"):
                    load_boards_async(force_rerun=True)

        st.markdown("---")

        # Session info
        st.markdown('<div class="sidebar-section-title">📈 Session actuelle</div>', unsafe_allow_html=True)

        # Show current stage
        stage_names = {1: "Configuration", 2: "Prévisualisation", 3: "Upload"}
        current_stage_name = stage_names.get(st.session_state.stage, "Inconnu")

        files_count = len(st.session_state.uploaded_files) if st.session_state.uploaded_files else 0
        rows_count = len(st.session_state.combined_data) if st.session_state.combined_data is not None else 0

        st.markdown(f"""
        <div class="sidebar-stats">
            <div class="sidebar-stat">
                <div class="number">{st.session_state.stage}/3</div>
                <div class="label">Étape</div>
            </div>
            <div class="sidebar-stat">
                <div class="number">{files_count}</div>
                <div class="label">Fichiers</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if rows_count > 0:
            st.markdown(f"""
            <div class="sidebar-info-card" style="margin-top: 0.5rem;">
                <div class="label">Lignes extraites</div>
                <div class="value">{rows_count} lignes</div>
            </div>
            """, unsafe_allow_html=True)

        # Quick actions
        if st.session_state.stage > 1:
            if st.button("⬅️ Recommencer", width="stretch"):
                reset_pipeline()
                st.rerun()

        st.markdown("---")

        # Comprehensive help section
        with st.expander("📖 Guide d'utilisation", expanded=False):
            st.markdown("""
            <div class="help-section">

            <h4>📄 Sources PDF supportées</h4>
            <ul>
                <li><strong>UV Assurance</strong> - Relevés de commissions UV</li>
                <li><strong>IDC</strong> - Relevés Industrial Alliance</li>
                <li><strong>IDC Statement</strong> - Statements détaillés IDC</li>
                <li><strong>Assomption Vie</strong> - Relevés Assomption</li>
            </ul>

            <h4>🔄 Workflow en 3 étapes</h4>
            <ul>
                <li><strong>Étape 1:</strong> Sélectionner la source et uploader les PDFs</li>
                <li><strong>Étape 2:</strong> Vérifier et modifier les données extraites</li>
                <li><strong>Étape 3:</strong> Exporter vers Monday.com</li>
            </ul>

            <h4>✨ Fonctionnalités</h4>
            <ul>
                <li>Extraction automatique via IA (VLM)</li>
                <li>Vérification des commissions calculées</li>
                <li>Normalisation des noms de conseillers</li>
                <li>Support multi-fichiers et multi-mois</li>
                <li>Cache intelligent pour éviter les re-extractions</li>
            </ul>

            <div class="help-tip">
                <strong>💡 Astuce:</strong> Les fichiers déjà extraits sont mis en cache.
                Réuploadez le même PDF pour utiliser le cache et économiser du temps.
            </div>

            <h4>⚙️ Configuration requise</h4>
            <ul>
                <li><code>MONDAY_API_KEY</code> - Clé API Monday.com</li>
                <li><code>OPENROUTER_API_KEY</code> - Pour l'extraction IA</li>
                <li><code>GOOGLE_SHEETS_*</code> - Pour la base conseillers</li>
            </ul>

            <h4>🔍 Vérification des données</h4>
            <p>Le système vérifie automatiquement que:</p>
            <ul>
                <li><strong>✓ OK</strong> - Commission dans la tolérance (±10%)</li>
                <li><strong>✅ Bonus</strong> - Commission supérieure au calcul</li>
                <li><strong>⚠️ Écart</strong> - Commission inférieure au calcul</li>
            </ul>

            <h4>❓ Support</h4>
            <p>En cas de problème, vérifiez:</p>
            <ul>
                <li>La qualité du PDF (scan lisible)</li>
                <li>La connexion API Monday.com</li>
                <li>Les logs dans la console</li>
            </ul>

            </div>
            """, unsafe_allow_html=True)

        # Footer
        st.markdown("---")
        st.caption("🛠️ Commission Pipeline v1.0")
        st.caption("Powered by OpenRouter & Monday.com")


# =============================================================================
# PHASE 2: ADVISOR MANAGEMENT TAB
# =============================================================================

def render_advisor_management_tab() -> None:
    """Render advisor management interface."""
    st.markdown("### 👥 Gestion des Conseillers")

    try:
        from src.utils.advisor_matcher import get_advisor_matcher, Advisor
    except ImportError:
        st.error("Module advisor_matcher non disponible")
        return

    # Initialize matcher
    if st.session_state.advisor_matcher is None:
        st.session_state.advisor_matcher = get_advisor_matcher()

    matcher = st.session_state.advisor_matcher

    # Check if Google Sheets is configured
    if not matcher.is_configured:
        st.warning(f"""
        ⚠️ **Google Sheets non configuré**

        La gestion des conseillers nécessite une connexion à Google Sheets.

        **Pour configurer:**
        1. Créez un projet Google Cloud et activez l'API Sheets
        2. Créez un compte de service et téléchargez le fichier JSON
        3. Configurez les variables d'environnement:
           - `GOOGLE_SHEETS_SPREADSHEET_ID` - ID de votre spreadsheet
           - `GOOGLE_SHEETS_CREDENTIALS_FILE` - Chemin vers le fichier JSON

        **Ou dans Streamlit secrets:**
        ```toml
        [gcp_service_account]
        type = "service_account"
        project_id = "..."
        # ... autres champs du service account
        ```

        *La normalisation des noms de conseillers sera désactivée.*
        """)
        return

    st.info("""
    **Gestion des noms de conseillers**

    Cette section permet de gérer les conseillers et leurs variations de noms.
    Le système utilise ces données pour normaliser automatiquement les noms
    lors de l'extraction des données PDF.

    **Format de sortie:** Prénom, Initiale (ex: "Thomas, L")
    """)

    st.divider()

    # Statistics
    advisors = matcher.get_all_advisors()
    total_variations = sum(len(a.variations) for a in advisors)

    cols = st.columns(3)
    cols[0].metric("Conseillers", len(advisors))
    cols[1].metric("Variations totales", total_variations)
    cols[2].metric("Stockage", f"{'☁️ Cloud' if matcher.storage_backend == 'google_sheets' else '💾 Local'}")

    st.divider()

    # Add new advisor
    st.markdown("#### ➕ Ajouter un conseiller")

    with st.form("add_advisor_form", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            new_first_name = st.text_input(
                "Prénom",
                placeholder="Ex: Thomas",
                key="new_advisor_first_name"
            )

        with col2:
            new_last_name = st.text_input(
                "Nom de famille",
                placeholder="Ex: Lussier",
                key="new_advisor_last_name"
            )

        new_variations = st.text_input(
            "Variations (séparées par des virgules)",
            placeholder="Ex: Tom, T. Lussier, Tommy",
            help="Entrez les différentes façons dont ce nom peut apparaître",
            key="new_advisor_variations"
        )

        submitted = st.form_submit_button("➕ Ajouter le conseiller", type="primary")

        if submitted:
            if new_first_name and new_last_name:
                variations = []
                if new_variations:
                    variations = [v.strip() for v in new_variations.split(',') if v.strip()]

                # Check if exists
                existing = matcher.find_advisor(new_first_name, new_last_name)
                if existing:
                    st.error(f"❌ Ce conseiller existe déjà")
                else:
                    try:
                        advisor = matcher.add_advisor(new_first_name, new_last_name, variations)
                        st.success(f"✅ Conseiller ajouté: {advisor.display_name_compact}")
                        st.session_state.advisor_matcher = get_advisor_matcher()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Erreur: {e}")
            else:
                st.error("❌ Veuillez entrer le prénom et le nom de famille")

    st.divider()

    # List existing advisors
    st.markdown("#### 📋 Conseillers existants")

    if not advisors:
        st.info("Aucun conseiller enregistré. Ajoutez-en un ci-dessus.")
    else:
        for idx, advisor in enumerate(advisors):
            with st.expander(f"**{advisor.display_name_compact}** ({advisor.full_name})", expanded=False):
                st.markdown(f"**Prénom:** {advisor.first_name}")
                st.markdown(f"**Nom:** {advisor.last_name}")
                st.markdown(f"**Format compact:** {advisor.display_name_compact}")

                st.markdown("**Variations:**")
                if advisor.variations:
                    for var in advisor.variations:
                        st.text(f"  • {var}")
                else:
                    st.caption("Aucune variation définie")

    st.divider()

    # Test matching
    st.markdown("#### 🔍 Tester la correspondance")

    test_name = st.text_input(
        "Entrez un nom à tester",
        placeholder="Ex: Thomas Lussier, Lussier Thomas, T. Lussier...",
        key="test_name_input"
    )

    if test_name:
        result = matcher.match_compact(test_name)
        if result:
            st.success(f"✅ Correspondance trouvée: **{result}**")
        else:
            st.warning(f"⚠️ Aucune correspondance pour: \"{test_name}\"")


# =============================================================================
# STAGE 1: CONFIGURATION (PDF EXTRACTION TAB)
# =============================================================================

def render_pdf_extraction_tab() -> None:
    """Render PDF extraction tab with batch processing support."""

    # File upload
    st.markdown("### 📤 Upload des fichiers PDF")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Use counter in key to reset widget when pipeline is reset
        upload_key = f"pdf_upload_main_{st.session_state.get('upload_key_counter', 0)}"
        uploaded_files = st.file_uploader(
            "Déposez vos fichiers PDF ici",
            type=['pdf'],
            accept_multiple_files=True,
            help="Sélectionnez un ou plusieurs fichiers PDF du même type",
            key=upload_key
        )

        if uploaded_files:
            is_batch = len(uploaded_files) > 1
            if is_batch:
                st.success(f"✅ {len(uploaded_files)} fichiers chargés")
            else:
                st.success(f"✅ Fichier chargé: {uploaded_files[0].name}")

    with col2:
        source_options = [s.value for s in SourceType]
        source = st.selectbox(
            "Source",
            options=source_options,
            help="Type de document PDF",
            key="source_select"
        )

    if not uploaded_files:
        st.info("👆 Commencez par uploader un ou plusieurs fichiers PDF pour continuer.")
        return

    st.session_state.uploaded_files = uploaded_files
    st.session_state.selected_source = source

    # Show file details with detected dates for batch mode
    is_batch = len(uploaded_files) > 1
    if is_batch:
        with st.expander(f"📁 Détail des {len(uploaded_files)} fichiers", expanded=True):
            has_undetected = False
            for i, f in enumerate(uploaded_files):
                detected_group = detect_date_from_filename(f.name)
                col_file, col_date = st.columns([3, 1])
                with col_file:
                    st.text(f"{i+1}. {f.name}")
                with col_date:
                    if detected_group:
                        st.caption(f"→ {detected_group}")
                    else:
                        st.caption("→ 📅 À détecter")
                        has_undetected = True
            if has_undetected:
                st.info("💡 Certains groupes seront détectés après extraction")

    st.divider()

    # Board selection
    st.markdown("### 📋 Destination Monday.com")

    if st.session_state.monday_boards is None and st.session_state.monday_api_key:
        load_boards_async()

    if st.session_state.boards_loading:
        st.info("⏳ Chargement des boards...")
        return

    if st.session_state.monday_boards:
        search = st.text_input(
            "🔍 Rechercher un board",
            placeholder="Filtrer par nom...",
            key="pdf_search_board"
        )

        sorted_boards = sort_and_filter_boards(st.session_state.monday_boards, search)

        if sorted_boards:
            board_options = {b['name']: b['id'] for b in sorted_boards}
            selected_name = st.selectbox(
                "Board de destination",
                options=list(board_options.keys()),
                key="pdf_board_select"
            )
            st.session_state.selected_board_id = board_options[selected_name]
            st.session_state._current_board_name = selected_name

            # Auto-detect board type
            detected_type = detect_board_type_from_name(selected_name)
            st.caption(f"🔍 Type détecté: **{detected_type}**")
        else:
            st.warning("Aucun board trouvé")
    else:
        st.warning("⚠️ Configurez la clé API Monday.com dans la sidebar")

    st.divider()

    # Configuration options
    st.markdown("### ⚙️ Configuration")

    type_options = ["Paiements Historiques", "Ventes et Production"]
    target_type = st.selectbox(
        "Type de table",
        options=type_options,
        key="pdf_target_type"
    )

    st.session_state.selected_board_type = (
        BoardType.SALES_PRODUCTION if target_type == "Ventes et Production"
        else BoardType.HISTORICAL_PAYMENTS
    )

    # Info about data processing based on board type
    if target_type == "Ventes et Production":
        st.caption(
            "ℹ️ **Traitement des données:** Les lignes sont regroupées par numéro de police. "
            "Plusieurs entrées avec le même numéro seront agrégées. "
            "Colonnes: Date, Police, Client, Compagnie, Statut, Conseiller, PA, Com, Boni, etc."
        )
    else:
        st.caption(
            "ℹ️ **Traitement des données:** Chaque ligne représente un paiement individuel. "
            "Colonnes: Police, Client, Compagnie, Statut, Conseiller, PA, Com, Boni, Sur-Com, Reçu, Date."
        )

    force_refresh = st.checkbox(
        "Forcer la ré-extraction (ignorer le cache)",
        value=False,
        key="pdf_force_refresh"
    )
    st.session_state.force_refresh = force_refresh

    st.divider()

    # Submit button
    button_text = f"🚀 Extraire {len(uploaded_files)} fichier{'s' if is_batch else ''}"

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        can_proceed = st.session_state.selected_board_id is not None
        if st.button(button_text, type="primary", width="stretch", disabled=not can_proceed):
            # Reset extraction state for new extraction
            st.session_state.combined_data = None
            st.session_state.extraction_results = {}
            st.session_state.batch_result = None
            st.session_state.extraction_usage = None
            st.session_state.upload_result = None
            st.session_state.file_group_overrides = {}
            st.session_state.stage = 2
            st.rerun()

    if not can_proceed:
        st.caption("⚠️ Sélectionnez un board de destination pour continuer")


def render_stage_1() -> None:
    """Render configuration stage with tabs."""
    st.markdown("## Pipeline de Commissions")
    render_stepper()
    render_breadcrumb()

    # Check API key
    if not st.session_state.monday_api_key:
        st.warning("Veuillez d'abord configurer votre clé API Monday.com dans la barre latérale.")
        return

    # Tabs for different workflows
    tab1, tab2 = st.tabs(["Extraction PDF", "Gestion Conseillers"])

    with tab1:
        render_pdf_extraction_tab()

    with tab2:
        render_advisor_management_tab()


# =============================================================================
# STAGE 2: PREVIEW
# =============================================================================

def run_extraction() -> None:
    """Run the extraction process with detailed progress display."""
    uploaded_files = st.session_state.uploaded_files
    if not uploaded_files:
        return

    st.session_state.is_processing = True
    st.session_state.extraction_results = {}
    st.session_state.combined_data = None

    # Get model configuration
    source = st.session_state.selected_source
    model_config = get_model_config(source) if source else None

    # Check if a custom model was selected
    selected_model = st.session_state.get('selected_model')
    if selected_model:
        # Temporarily update the model registry for this extraction
        from src.utils.model_registry import register_model, ModelConfig, ExtractionMode
        if source:
            original_config = model_config
            new_config = ModelConfig(
                model_id=selected_model,
                mode=original_config.mode,
                fallback_model_id=original_config.fallback_model_id,
                fallback_mode=original_config.fallback_mode,
                secondary_fallback_model_id=original_config.secondary_fallback_model_id,
                secondary_fallback_mode=original_config.secondary_fallback_mode,
                temperature=original_config.temperature,
                max_tokens=original_config.max_tokens,
                page_config=original_config.page_config,
                ocr_engine=original_config.ocr_engine,
                text_analysis_model=original_config.text_analysis_model,
            )
            register_model(source, new_config)
            model_config = new_config

    pipeline = get_pipeline()

    # Reset extractor clients if a custom model was selected
    # This ensures new clients are created with the updated model config
    if selected_model:
        pipeline.reset_extractor_clients()

    # ===== ENHANCED PROGRESS UI =====
    st.markdown("---")

    # Model info - simple elegant display
    if model_config:
        primary_model = model_config.model_id.split("/")[-1] if "/" in model_config.model_id else model_config.model_id
        st.markdown(f"""
        <div style="display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 20px; margin-bottom: 16px;">
            <span style="font-size: 16px;">🤖</span>
            <span style="color: white; font-weight: 500; font-size: 14px;">Modèle: {primary_model}</span>
        </div>
        """, unsafe_allow_html=True)

    # File list with status
    st.markdown("### 📁 Fichiers en cours de traitement")
    file_status_container = st.container()

    # Create placeholders for each file
    file_placeholders = {}
    with file_status_container:
        for i, file in enumerate(uploaded_files):
            file_placeholders[file.name] = {
                'container': st.empty(),
                'status': 'pending',
                'index': i
            }
            file_placeholders[file.name]['container'].markdown(
                f"⏳ **{i+1}/{len(uploaded_files)}** - `{file.name}` - *En attente...*"
            )

    # Overall progress
    progress_bar = st.progress(0, text="Démarrage de l'extraction...")
    status_text = st.empty()
    fallback_alert = st.empty()

    # Clean up previous temp files before creating new ones
    cleanup_temp_files()

    # Save files to temp directory with sanitized names
    temp_dir = tempfile.mkdtemp()
    temp_paths = []

    for file in uploaded_files:
        # Sanitize filename to prevent path traversal attacks
        safe_filename = sanitize_filename(file.name)
        temp_path = Path(temp_dir) / safe_filename
        temp_path.write_bytes(file.read())
        temp_paths.append(temp_path)
        file.seek(0)

    st.session_state.temp_pdf_paths = temp_paths

    # Track current file and results
    file_results = {}

    # Progress callback with enhanced display
    def on_progress(current: int, total: int, filename: str) -> None:
        progress = current / total if total > 0 else 0
        progress_bar.progress(progress, text=f"Traitement: {filename} ({current}/{total})")

        # Update file status
        for fname, placeholder in file_placeholders.items():
            idx = placeholder['index']
            if idx < current - 1:
                # Completed
                if fname in file_results:
                    result = file_results[fname]
                    if result.success:
                        rows = result.row_count
                        placeholder['container'].markdown(
                            f"✅ **{idx+1}/{total}** - `{fname}` - **{rows} lignes extraites**"
                        )
                    else:
                        placeholder['container'].markdown(
                            f"❌ **{idx+1}/{total}** - `{fname}` - *Erreur: {result.error[:50]}...*" if result.error and len(result.error) > 50 else f"❌ **{idx+1}/{total}** - `{fname}` - *Erreur: {result.error}*"
                        )
            elif idx == current - 1:
                # Currently processing (just finished)
                placeholder['container'].markdown(
                    f"🔄 **{idx+1}/{total}** - `{fname}` - *Finalisation...*"
                )
            elif fname == filename:
                # Currently being processed
                placeholder['container'].markdown(
                    f"🔄 **{idx+1}/{total}** - `{fname}` - *Extraction en cours...*"
                )

    try:
        # Run extraction
        batch_result = run_async(
            pipeline.process_batch(
                pdf_paths=temp_paths,
                source=st.session_state.selected_source,
                force_refresh=st.session_state.force_refresh,
                progress_callback=on_progress
            )
        )

        st.session_state.batch_result = batch_result

        # Store results and update file status
        results = {}
        fallback_used_files = []

        for result in batch_result.results:
            filename = Path(result.pdf_path).name
            results[filename] = result
            file_results[filename] = result

            # Update final status for each file
            if filename in file_placeholders:
                idx = file_placeholders[filename]['index']
                total = len(uploaded_files)
                if result.success:
                    rows = result.row_count
                    model_used = ""
                    fallback_indicator = ""

                    if result.usage and result.usage.model:
                        actual_model = result.usage.model.split("/")[-1] if "/" in result.usage.model else result.usage.model
                        model_used = f" ({actual_model})"

                        # Check if fallback was used
                        if model_config:
                            primary = model_config.model_id.split("/")[-1] if "/" in model_config.model_id else model_config.model_id
                            if actual_model != primary:
                                fallback_indicator = " 🔄"
                                fallback_used_files.append((filename, actual_model))

                    file_placeholders[filename]['container'].markdown(
                        f"✅ **{idx+1}/{total}** - `{filename}` - **{rows} lignes**{model_used}{fallback_indicator}"
                    )
                else:
                    error_msg = result.error[:60] + "..." if result.error and len(result.error) > 60 else (result.error or "Erreur inconnue")
                    file_placeholders[filename]['container'].markdown(
                        f"❌ **{idx+1}/{total}** - `{filename}` - *{error_msg}*"
                    )

        # Show fallback alerts if any
        if fallback_used_files:
            with fallback_alert.container():
                st.warning(f"⚠️ **Modèle de secours utilisé** pour {len(fallback_used_files)} fichier(s)")
                for fname, model in fallback_used_files:
                    st.caption(f"  • `{fname}` → {model}")

        st.session_state.extraction_results = results

        # Store usage stats
        st.session_state.extraction_usage = batch_result.total_usage

        # Get combined data and detect groups
        combined_df = batch_result.get_combined_dataframe()
        if combined_df is not None and not combined_df.empty:
            combined_df = detect_groups_from_data(combined_df, st.session_state.selected_source)
        st.session_state.combined_data = combined_df

        # Show success with model info
        progress_bar.progress(1.0, text="Extraction terminée!")
        if batch_result.failed == 0:
            status_text.success(f"✅ **{batch_result.total_rows} lignes** extraites de **{batch_result.successful} fichier(s)**")
        else:
            status_text.warning(f"⚠️ **{batch_result.total_rows} lignes** extraites. **{batch_result.failed} fichier(s)** en erreur.")

        # Show cost summary if available
        if batch_result.total_usage and batch_result.total_usage.cost > 0:
            st.info(f"💰 Coût total: **${batch_result.total_usage.cost:.4f}** | Tokens: {batch_result.total_usage.total_tokens:,}")

    except Exception as e:
        status_text.error(f"Échec de l'extraction: {e}")
        # Store the error for display in render_stage_2
        st.session_state.extraction_error = str(e)
        import traceback
        st.session_state.extraction_traceback = traceback.format_exc()

    finally:
        st.session_state.is_processing = False
        st.session_state.selected_model = None  # Reset custom model after extraction
        st.session_state.force_refresh = False  # Reset force refresh flag


def render_stage_2() -> None:
    """Render data preview stage with modern tabs layout."""
    st.markdown("## 📊 Pipeline de Commissions")
    render_stepper()
    render_breadcrumb()

    # Extract data if not done
    if st.session_state.combined_data is None:
        if not st.session_state.uploaded_files:
            st.error("Aucun fichier à traiter")
            if st.button("Recommencer"):
                reset_pipeline()
                st.rerun()
            return

        with st.spinner("Extraction en cours..."):
            run_extraction()
            st.rerun()
        return

    df = st.session_state.combined_data

    if df is None or df.empty:
        st.error("Aucune donnée extraite")

        # Show detailed error information if available
        batch_result = st.session_state.get('batch_result')
        if batch_result:
            if batch_result.failed > 0:
                st.warning(f"⚠️ {batch_result.failed} fichier(s) en erreur sur {batch_result.total_pdfs}")
                for result in batch_result.results:
                    if not result.success and result.error:
                        st.error(f"**{Path(result.pdf_path).name}**: {result.error}")
            elif batch_result.total_pdfs == 0:
                st.info("Aucun fichier n'a été traité. Vérifiez que les fichiers sont bien des PDFs valides.")
            else:
                st.info("L'extraction a réussi mais n'a retourné aucune donnée. Le PDF ne contient peut-être pas les informations attendues.")
        else:
            # Check for stored extraction error
            extraction_error = st.session_state.get('extraction_error')
            if extraction_error:
                st.error(f"**Erreur d'extraction**: {extraction_error}")
                with st.expander("Voir le traceback complet"):
                    st.code(st.session_state.get('extraction_traceback', 'No traceback available'))
            else:
                # Check if API key is configured
                from src.utils.config import _get_secret, settings
                api_key = _get_secret("OPENROUTER_API_KEY") or settings.openrouter_api_key
                if not api_key:
                    st.error("⚠️ **OPENROUTER_API_KEY** n'est pas configurée. Ajoutez-la dans les secrets Streamlit ou le fichier .env")

        if st.button("Recommencer"):
            reset_pipeline()
            st.rerun()
        return

    # ===========================================
    # METRICS DASHBOARD HEADER (Always Visible)
    # ===========================================
    usage = st.session_state.get('extraction_usage')
    model_name = ""
    cost_display = "Cache"
    if usage:
        model_name = usage.model.split("/")[-1] if usage.model and "/" in usage.model else (usage.model or "N/A")
        cost_display = f"${usage.cost:.4f}" if usage.cost > 0 else "Cache"

    # Determine status
    has_verification_cols = 'Reçu' in df.columns and 'PA' in df.columns
    if has_verification_cols:
        df_verified = verify_recu_vs_com(df, tolerance_pct=st.session_state.get('verification_tolerance', 10.0))
        stats = get_verification_stats(df_verified)
        status_icon = "OK" if stats['ecart'] == 0 else f"{stats['ecart']} Ecarts"
    else:
        df_verified = df
        stats = None
        status_icon = "OK"

    # Dashboard metrics
    render_metrics_dashboard(
        row_count=len(df),
        cost=cost_display,
        model=model_name,
        status=status_icon
    )

    # ===========================================
    # TABBED INTERFACE (Replaces Expanders)
    # ===========================================
    tab_data, tab_verify, tab_config, tab_actions = st.tabs([
        "Données",
        "Vérification",
        "Configuration",
        "Actions"
    ])

    # ----- TAB 1: DONNÉES -----
    with tab_data:
        # Quick stats
        cols = st.columns(4)
        cols[0].metric("Lignes", len(df))
        cols[1].metric("Colonnes", len(df.columns))
        if '# de Police' in df.columns:
            cols[2].metric("Contrats", df['# de Police'].notna().sum())
        elif 'Contrat' in df.columns:
            cols[2].metric("Contrats", df['Contrat'].notna().sum())
        else:
            cols[2].metric("Contrats", "-")
        cols[3].metric("Doublons", df.duplicated().sum())

        st.markdown("---")

        # Display the dataframe
        if has_verification_cols:
            st.dataframe(reorder_columns_for_display(df_verified), width="stretch", height=400)
        else:
            st.dataframe(reorder_columns_for_display(df), width="stretch", height=400)

        # Extraction details
        results = st.session_state.extraction_results
        if results:
            with st.expander("Détails de l'extraction", expanded=False):
                for filename, result in results.items():
                    if result.success:
                        st.markdown(f"**{filename}**: {result.row_count} lignes ({result.extraction_time_ms}ms)")
                    else:
                        st.markdown(f"**{filename}**: {result.error}")

    # ----- TAB 2: VÉRIFICATION -----
    with tab_verify:
        if has_verification_cols:
            st.markdown("### Vérification Reçu vs Commission")
            st.caption("Formule: `Com Calculée = ROUND((PA x 0.4) x 0.5, 2)`")

            tolerance = st.slider(
                "Tolérance (%)",
                min_value=1.0,
                max_value=50.0,
                value=st.session_state.get('verification_tolerance', 10.0),
                step=1.0,
                key="verification_tolerance_slider"
            )
            st.session_state.verification_tolerance = tolerance

            df_verified = verify_recu_vs_com(df, tolerance_pct=tolerance)
            stats = get_verification_stats(df_verified)

            # Stats display
            stat_cols = st.columns(4)
            stat_cols[0].metric("OK", stats['ok'])
            stat_cols[1].metric("Bonus", stats['bonus'])
            stat_cols[2].metric("Ecart", stats['ecart'])
            stat_cols[3].metric("N/A", stats['na'])

            if stats['ecart'] > 0:
                st.warning(f"**{stats['ecart']} ligne(s)** ont un écart négatif")
            if stats['bonus'] > 0:
                st.success(f"**{stats['bonus']} ligne(s)** ont un bonus")

            st.markdown("---")

            # Find verification column
            verif_col = [col for col in df_verified.columns if col.startswith('Vérification')]
            if verif_col:
                verif_col = verif_col[0]

                # Show table with ecarts (negative differences)
                if stats['ecart'] > 0:
                    st.markdown("#### ⚠️ Lignes avec écarts négatifs")
                    ecart_df = df_verified[df_verified[verif_col].astype(str).str.contains('⚠️', na=False)]

                    # Select relevant columns for display
                    display_cols = ['# de Police', 'Nom Client', 'Conseiller', 'PA', 'Reçu', 'Com Calculée', verif_col]
                    display_cols = [c for c in display_cols if c in ecart_df.columns]

                    st.dataframe(
                        ecart_df[display_cols],
                        width="stretch",
                        hide_index=True
                    )

                # Show table with bonus (positive differences)
                if stats['bonus'] > 0:
                    st.markdown("#### ✅ Lignes avec bonus")
                    bonus_df = df_verified[df_verified[verif_col].astype(str).str.contains('✅', na=False)]

                    display_cols = ['# de Police', 'Nom Client', 'Conseiller', 'PA', 'Reçu', 'Com Calculée', verif_col]
                    display_cols = [c for c in display_cols if c in bonus_df.columns]

                    st.dataframe(
                        bonus_df[display_cols],
                        width="stretch",
                        hide_index=True
                    )

                # Show all data with verification
                with st.expander("📊 Voir toutes les données avec vérification", expanded=False):
                    st.dataframe(
                        df_verified,
                        width="stretch",
                        hide_index=True
                    )
        else:
            st.info("La vérification n'est pas disponible pour ce type de données (colonnes 'Reçu' et 'PA' requises).")

    # ----- TAB 3: CONFIGURATION -----
    with tab_config:
        st.markdown("### Résumé de la Configuration")

        config_cols = st.columns(2)
        with config_cols[0]:
            st.markdown("**Source**")
            st.info(st.session_state.selected_source or "N/A")

            st.markdown("**Fichiers**")
            st.info(f"{len(st.session_state.uploaded_files)} fichier(s)")

        with config_cols[1]:
            board_name = st.session_state._current_board_name or "N/A"
            st.markdown("**Board de destination**")
            st.info(board_name[:40] + "..." if len(board_name) > 40 else board_name)

            st.markdown("**Type de table**")
            st.info("Ventes" if st.session_state.selected_board_type == BoardType.SALES_PRODUCTION else "Paiements")

        st.markdown("---")

        # ===========================================
        # MODEL SELECTION & RE-EXTRACTION
        # ===========================================
        st.markdown("### 🤖 Modèle d'extraction")

        # Show current model info
        current_model = model_name if model_name else "Modèle par défaut"
        st.markdown(f"**Modèle actuel:** `{current_model}`")
        st.markdown(f"**Coût de l'extraction:** {cost_display}")

        st.caption("Sélectionnez un autre modèle pour ré-extraire les données.")

        available_models = get_available_models()
        model_options = list(available_models.keys())
        model_labels = [f"{v}" for v in available_models.values()]

        # Create a mapping for display
        model_display_map = dict(zip(model_labels, model_options))

        col_model, col_btn = st.columns([3, 1])
        with col_model:
            selected_label = st.selectbox(
                "Choisir un modèle",
                options=model_labels,
                index=0,
                key="model_selector",
                help="Sélectionnez un modèle VLM pour l'extraction"
            )
            selected_model_id = model_display_map[selected_label]

        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Ré-extraire", type="primary", width="stretch"):
                st.session_state.selected_model = selected_model_id
                st.session_state.combined_data = None
                st.session_state.extraction_results = {}
                st.session_state.extraction_usage = None
                st.session_state.force_refresh = True
                # IMPORTANT: Reset pipeline to force new extractors with new model
                st.session_state.pipeline = None
                st.rerun()

        st.markdown("---")

        # ===========================================
        # PER-FILE GROUP ASSIGNMENT
        # ===========================================
        st.markdown("### 📁 Groupes de destination par fichier")

        if '_source_file' in df.columns:
            unique_files = df['_source_file'].unique().tolist()

            if len(unique_files) > 1:
                st.info(f"🎯 **{len(unique_files)} fichiers** - Chaque fichier peut être envoyé vers un groupe Monday.com différent.")

                # Show current group summary
                with st.expander("📊 Résumé actuel des groupes", expanded=True):
                    group_summary = {}
                    for filename in unique_files:
                        file_df = df[df['_source_file'] == filename]
                        group = file_df['_target_group'].iloc[0] if '_target_group' in file_df.columns else "Auto"
                        if group not in group_summary:
                            group_summary[group] = []
                        group_summary[group].append((filename, len(file_df)))

                    for group, files in group_summary.items():
                        st.markdown(f"**📁 {group}**")
                        for fname, count in files:
                            st.caption(f"  └─ `{fname}` ({count} lignes)")

            # Generate group options
            months_fr = get_months_fr()
            now = datetime.now()
            group_options_list = []
            for offset in range(-6, 7):
                month = now.month + offset
                year = now.year
                if month < 1:
                    month += 12
                    year -= 1
                elif month > 12:
                    month -= 12
                    year += 1
                group_options_list.append(f"{months_fr[month]} {year}")

            # Initialize file_group_overrides if not exists
            if 'file_group_overrides' not in st.session_state:
                st.session_state.file_group_overrides = {}

            st.markdown("#### ✏️ Modifier les groupes")
            st.caption("Sélectionnez un groupe prédéfini ou entrez un nom personnalisé pour chaque fichier.")

            for filename in unique_files:
                file_df = df[df['_source_file'] == filename]
                current_group = file_df['_target_group'].iloc[0] if '_target_group' in file_df.columns else "Auto"
                row_count = len(file_df)

                # Use container with border for better visibility
                with st.container():
                    col_file, col_group, col_manual = st.columns([2, 2, 2])

                    with col_file:
                        # Truncate long filenames
                        display_name = filename[:25] + "..." if len(filename) > 25 else filename
                        st.markdown(f"**📄 {display_name}**")
                        st.caption(f"{row_count} lignes • Actuel: `{current_group}`")

                    with col_group:
                        # Get current override or detected group
                        current_override = st.session_state.file_group_overrides.get(filename, current_group)

                        # Find index of current group in options
                        try:
                            default_idx = group_options_list.index(current_override)
                        except ValueError:
                            default_idx = 6  # Middle of the list (current month)

                        selected_group = st.selectbox(
                            f"Groupe pour {filename}",
                            options=group_options_list,
                            index=default_idx,
                            key=f"group_select_{filename}",
                            label_visibility="collapsed"
                        )

                        # Always track the selection
                        if selected_group != current_group:
                            st.session_state.file_group_overrides[filename] = selected_group

                    with col_manual:
                        manual_input = st.text_input(
                            f"Groupe personnalisé pour {filename}",
                            placeholder="Ex: Janvier 2026",
                            key=f"manual_group_{filename}",
                            label_visibility="collapsed"
                        )
                        if manual_input and manual_input.strip():
                            st.session_state.file_group_overrides[filename] = manual_input.strip()

                    st.markdown("---")

            # Always show apply button when multiple files
            col_apply, col_reset = st.columns([3, 1])
            with col_apply:
                if st.button("✅ Appliquer les modifications de groupes", type="primary", width="stretch"):
                    # Update the dataframe with new groups
                    changes_made = False
                    for filename in unique_files:
                        # Check selectbox value
                        select_key = f"group_select_{filename}"
                        manual_key = f"manual_group_{filename}"

                        new_group = None
                        # Manual input takes priority
                        if manual_key in st.session_state and st.session_state[manual_key]:
                            new_group = st.session_state[manual_key].strip()
                        elif filename in st.session_state.file_group_overrides:
                            new_group = st.session_state.file_group_overrides[filename]
                        elif select_key in st.session_state:
                            new_group = st.session_state[select_key]

                        if new_group:
                            df.loc[df['_source_file'] == filename, '_target_group'] = new_group
                            changes_made = True

                    if changes_made:
                        st.session_state.combined_data = df
                        st.session_state.file_group_overrides = {}  # Reset overrides
                        st.success("✅ Groupes mis à jour! Les fichiers seront uploadés dans leurs groupes respectifs.")
                        st.rerun()
                    else:
                        st.info("Aucune modification détectée.")

            with col_reset:
                if st.button("🔄 Reset", width="stretch"):
                    st.session_state.file_group_overrides = {}
                    st.rerun()

        elif '_target_group' in df.columns:
            # Single file mode - show simple group override
            groups_info = analyze_groups_in_data(df)

            st.markdown("### Groupes Détectés")
            if groups_info['spans_multiple_months']:
                st.warning(f"Les données couvrent **{len(groups_info['unique_groups'])} mois différents**.")

            for group, count in groups_info['group_counts'].items():
                st.markdown(f"- **{group}**: {count} lignes")

            st.markdown("---")

            # Manual override with text input option
            st.markdown("### Modifier le groupe")
            st.caption("Sélectionnez ou entrez manuellement un groupe.")

            months_fr = get_months_fr()
            now = datetime.now()
            group_options = ["(Garder auto-détection)"]

            for offset in range(-3, 4):
                month = now.month + offset
                year = now.year
                if month < 1:
                    month += 12
                    year -= 1
                elif month > 12:
                    month -= 12
                    year += 1
                group_options.append(f"{months_fr[month]} {year}")

            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                manual_group = st.selectbox("Groupe prédéfini", group_options, key="manual_group_override")
            with col2:
                custom_group = st.text_input(
                    "Ou entrez un groupe personnalisé",
                    placeholder="Ex: Janvier 2026",
                    key="custom_group_input"
                )
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                final_group = custom_group.strip() if custom_group and custom_group.strip() else (
                    manual_group if manual_group != "(Garder auto-détection)" else None
                )
                if final_group:
                    if st.button("Appliquer", width="stretch", type="primary"):
                        df['_target_group'] = final_group
                        st.session_state.combined_data = df
                        st.success(f"Groupe modifié: {final_group}")
                        st.rerun()

        # Column info
        st.markdown("---")
        st.markdown("### Informations Colonnes")
        col_info = pd.DataFrame({
            'Colonne': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.notna().sum().values,
            'Null': df.isna().sum().values
        })
        st.dataframe(col_info, width="stretch", height=200)

    # ----- TAB 4: ACTIONS -----
    with tab_actions:
        st.markdown("### Exporter les données")

        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Télécharger CSV",
                data=csv,
                file_name=f"commissions_{st.session_state.selected_source}.csv",
                mime="text/csv",
                width="stretch"
            )
        with col2:
            # Excel export
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Commissions')
            excel_data = output.getvalue()
            st.download_button(
                "Télécharger Excel",
                data=excel_data,
                file_name=f"commissions_{st.session_state.selected_source}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch"
            )

        st.markdown("---")
        st.markdown("### Remplacer les données")
        st.caption("Uploader un fichier Excel/CSV modifié pour remplacer les données extraites.")

        excel_file = st.file_uploader(
            "Fichier Excel/CSV modifié",
            type=['xlsx', 'xls', 'csv'],
            key="excel_upload"
        )

        if excel_file:
            try:
                if excel_file.name.endswith('.csv'):
                    uploaded_df = pd.read_csv(excel_file)
                else:
                    uploaded_df = pd.read_excel(excel_file)

                st.success(f"{excel_file.name} chargé ({len(uploaded_df)} lignes)")

                if st.button("Utiliser ce fichier", type="primary"):
                    st.session_state.combined_data = uploaded_df
                    st.session_state.data_modified = True
                    st.rerun()

            except Exception as e:
                st.error(f"Erreur: {e}")

    # ===========================================
    # STICKY ACTION FOOTER
    # ===========================================
    st.markdown("---")

    footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

    with footer_col1:
        if st.button("Retour", width="stretch"):
            reset_pipeline()
            st.rerun()

    with footer_col3:
        if st.button("Uploader vers Monday.com", type="primary", width="stretch"):
            st.session_state.stage = 3
            st.rerun()




# =============================================================================
# STAGE 3: UPLOAD
# =============================================================================

def render_stage_3() -> None:
    """Render upload stage with modern styling."""
    st.markdown("## Pipeline de Commissions")
    render_stepper()
    render_breadcrumb()

    df = st.session_state.combined_data

    if df is None or df.empty:
        st.error("Aucune donnée à uploader")
        if st.button("Recommencer"):
            reset_pipeline()
            st.rerun()
        return

    if st.session_state.data_modified:
        st.warning("Upload de données modifiées")

    # Summary Dashboard (similar to Stage 2)
    unique_groups = df['_target_group'].unique() if '_target_group' in df.columns else []
    board_name = st.session_state._current_board_name or "N/A"

    render_upload_dashboard(
        item_count=len(df),
        board_name=board_name,
        group_count=len(unique_groups) if len(unique_groups) > 0 else 1,
        file_count=len(st.session_state.extraction_results)
    )

    # Groups breakdown
    if '_target_group' in df.columns and len(unique_groups) > 1:
        with st.expander("Détail par groupe", expanded=False):
            for group in unique_groups:
                group_count = len(df[df['_target_group'] == group])
                st.markdown(f"**{group}**: {group_count} items")

    st.markdown("---")

    # Upload process
    if st.session_state.upload_result is None:
        # Check if upload is in progress
        if st.session_state.is_uploading:
            # Execute the upload (after rerun from button click)
            execute_upload(df)
        else:
            st.info("Les données vont être uploadées vers Monday.com.")

            footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

            with footer_col1:
                if st.button("Retour", width="stretch"):
                    st.session_state.stage = 2
                    st.rerun()

            with footer_col3:
                if st.button(
                    "🚀 Confirmer l'upload",
                    type="primary",
                    width="stretch"
                ):
                    # Set uploading state and rerun to hide the button
                    st.session_state.is_uploading = True
                    st.rerun()
    else:
        render_upload_result(st.session_state.upload_result)

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("Nouveau pipeline", width="stretch"):
                reset_pipeline()
                st.rerun()
        with col3:
            board_id = st.session_state.selected_board_id
            st.link_button("Ouvrir Monday.com", f"https://monday.com/boards/{board_id}", width="stretch")


def execute_upload(df: pd.DataFrame) -> None:
    """Execute the upload to Monday.com."""
    st.session_state.upload_result = None

    pipeline = get_pipeline()
    board_id = st.session_state.selected_board_id

    progress_bar = st.progress(0, text="Démarrage de l'upload...")
    status_text = st.empty()

    try:
        # Get unique groups
        if '_target_group' in df.columns:
            unique_groups = df['_target_group'].unique().tolist()
        else:
            unique_groups = [f"{get_months_fr()[datetime.now().month]} {datetime.now().year}"]

        total_items = len(df)
        items_uploaded = 0
        items_failed = 0
        all_errors = []

        # Track items processed so far for progress display
        items_processed_before_group = 0

        for group_idx, group_name in enumerate(unique_groups):
            status_text.markdown(f"📁 **Groupe {group_idx + 1}/{len(unique_groups)}:** {group_name}")

            # Filter data for this group
            if '_target_group' in df.columns:
                group_df = df[df['_target_group'] == group_name].copy()
            else:
                group_df = df.copy()

            # Remove internal columns
            export_df = group_df.drop(columns=[c for c in group_df.columns if c.startswith('_')], errors='ignore')

            try:
                # Create or get group
                group_result = run_async(
                    pipeline.monday.get_or_create_group(board_id, str(group_name))
                )
                group_id = group_result.id if group_result.success else None

                if not group_id:
                    raise Exception(f"Impossible de créer le groupe: {group_result.error}")

                # Progress callback - show overall progress
                def on_progress(current: int, total: int) -> None:
                    # Calculate overall progress across all groups
                    overall_current = items_processed_before_group + current
                    overall_progress = overall_current / total_items
                    progress_bar.progress(
                        min(overall_progress, 0.99),
                        text=f"Upload: {group_name} ({overall_current}/{total_items})"
                    )

                # Upload
                result = run_async(
                    pipeline.monday.upload_dataframe(
                        df=export_df,
                        board_id=board_id,
                        group_id=group_id,
                        progress_callback=on_progress
                    )
                )

                items_uploaded += result.success
                items_failed += result.failed
                all_errors.extend(result.errors)

                # Update items processed before next group
                items_processed_before_group += len(export_df)

            except Exception as e:
                items_failed += len(export_df)
                items_processed_before_group += len(export_df)
                all_errors.append(f"Groupe {group_name}: {str(e)}")

        progress_bar.progress(1.0, text=f"Upload terminé! ({total_items}/{total_items})")
        status_text.empty()

        # Store result
        st.session_state.upload_result = {
            "total": total_items,
            "success": items_uploaded,
            "failed": items_failed,
            "errors": all_errors,
            "groups": len(unique_groups),
        }

        if items_failed == 0:
            st.success(f"{items_uploaded} éléments uploadés dans {len(unique_groups)} groupe(s)!")
        else:
            st.warning(f"{items_uploaded}/{total_items} uploadés. {items_failed} en erreur.")

    except Exception as e:
        st.error(f"Échec de l'upload: {e}")
        st.session_state.upload_result = {
            "total": len(df),
            "success": 0,
            "failed": len(df),
            "errors": [str(e)],
            "groups": 0,
        }

    finally:
        st.session_state.is_uploading = False
        progress_bar.empty()
        st.rerun()


def render_upload_result(result: dict) -> None:
    """Render upload result summary with modern styling."""
    total = result.get("total", 0)
    success = result.get("success", 0)
    failed = result.get("failed", 0)
    groups = result.get("groups", 1)

    if failed == 0:
        render_success_box(
            title="Upload réussi!",
            message=f"{success}/{total} éléments uploadés dans {groups} groupe(s)."
        )
    else:
        st.warning(f"{success}/{total} éléments uploadés. {failed} en erreur.")

    errors = result.get("errors", [])
    if errors:
        with st.expander("Voir les erreurs"):
            for error in errors[:10]:
                st.error(error)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main() -> None:
    """Main application entry point."""
    # Clear prompt cache once per session to ensure fresh prompts are loaded
    if '_prompt_cache_cleared' not in st.session_state:
        # Use functools to clear the LRU cache directly
        from functools import lru_cache
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

    render_sidebar()

    # Route to appropriate stage
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
