"""
PDF Extractor - Streamlit Application

Single-page application for extracting commission data from PDFs
and uploading to Monday.com.

Run with: streamlit run src/app/main.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import Pipeline, SourceType, BatchResult
from src.utils.data_unifier import BoardType


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

st.markdown("""
<style>
    /* Modern button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Main container */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        max-width: 1200px;
    }

    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    /* Card styling */
    [data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
    }

    /* Reduce spacing */
    .element-container {
        margin-bottom: 0.5rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }

    /* Success/warning boxes */
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


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


def init_session_state() -> None:
    """Initialize all session state variables."""
    defaults = {
        # Pipeline
        "pipeline": None,

        # File upload
        "uploaded_files": [],
        "temp_pdf_paths": [],

        # Extraction
        "extraction_results": {},
        "batch_result": None,
        "combined_data": None,

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

        # Options
        "selected_source": None,
        "force_refresh": False,
        "data_modified": False,
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


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar() -> None:
    """Render the sidebar."""
    with st.sidebar:
        st.title("📊 Pipeline Commissions")
        st.caption("Extraire • Prévisualiser • Exporter")

        st.divider()

        # API Configuration
        st.subheader("🔑 Configuration")

        # Monday.com API Key
        api_key = st.session_state.monday_api_key
        with st.expander("Monday.com API", expanded=not api_key):
            new_key = st.text_input(
                "Clé API",
                value=api_key or "",
                type="password",
                key="sidebar_monday_key"
            )
            if st.button("Sauvegarder", key="save_monday_key"):
                if new_key:
                    st.session_state.monday_api_key = new_key
                    st.session_state.pipeline = None
                    st.session_state.monday_boards = None
                    st.success("Sauvegardé!")
                    st.rerun()

        # Status indicators
        st.divider()
        st.subheader("📊 Statut")

        # Files status
        files = st.session_state.uploaded_files
        if files:
            st.success(f"📁 {len(files)} fichier(s) chargé(s)")
        else:
            st.info("📁 Aucun fichier")

        # Source type
        source = st.session_state.get("selected_source")
        if source:
            st.info(f"📄 Source: {source}")

        # Extraction status
        results = st.session_state.extraction_results
        if results:
            success = sum(1 for r in results.values() if r.success)
            st.success(f"✅ {success}/{len(results)} extraits")
        else:
            st.info("📝 Aucune extraction")

        # Monday.com status
        if api_key:
            st.success("🔗 Monday.com connecté")
            boards = st.session_state.monday_boards
            if boards:
                st.caption(f"📋 {len(boards)} boards disponibles")
        else:
            st.warning("🔗 Monday.com non configuré")

        st.divider()

        # Quick actions
        if st.button("🔄 Tout réinitialiser", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ["monday_api_key"]:
                    del st.session_state[key]
            st.rerun()

        st.divider()

        # Supported sources
        with st.expander("ℹ️ Sources supportées"):
            st.markdown("""
            - **UV Assurance**
            - **IDC Propositions**
            - **IDC Statement**
            - **Assomption Vie**
            """)


# =============================================================================
# SECTION 1: FILE UPLOAD
# =============================================================================

def render_upload_section() -> None:
    """Render the file upload section."""
    st.header("📤 1. Upload des fichiers PDF")

    col_upload, col_source = st.columns([2, 1])

    with col_upload:
        # File uploader
        uploaded_files = st.file_uploader(
            "Déposez vos fichiers PDF ici ou cliquez pour parcourir",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
            help="Supportés: UV, IDC, IDC Statement, Assomption"
        )

        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files

    with col_source:
        # Source type selection - REQUIRED (no auto-detect)
        source_options = [s.value for s in SourceType]
        selected_source = st.selectbox(
            "Type de document *",
            source_options,
            key="source_selector",
            help="Sélectionnez le type de document PDF"
        )
        st.session_state.selected_source = selected_source

    if uploaded_files:
        # Show file list
        st.subheader(f"📁 {len(uploaded_files)} fichier(s) sélectionné(s)")

        for file in uploaded_files:
            col1, col2 = st.columns([3, 1])

            with col1:
                st.text(f"📄 {file.name}")

            with col2:
                st.caption(f"{file.size / 1024:.1f} KB")

    # Extraction options
    with st.expander("⚙️ Options d'extraction", expanded=False):
        st.session_state.force_refresh = st.checkbox(
            "Forcer la ré-extraction",
            value=st.session_state.force_refresh,
            help="Ignorer le cache"
        )

    # Extract button
    col1, col2 = st.columns(2)

    with col1:
        extract_disabled = (
            len(st.session_state.uploaded_files) == 0 or
            st.session_state.is_processing or
            not st.session_state.selected_source
        )
        if st.button(
            "🚀 Extraire les données",
            type="primary",
            disabled=extract_disabled,
            use_container_width=True
        ):
            run_extraction()

    with col2:
        if st.button("🗑️ Effacer les fichiers", use_container_width=True):
            st.session_state.uploaded_files = []
            st.session_state.extraction_results = {}
            st.session_state.batch_result = None
            st.session_state.combined_data = None
            st.rerun()


def run_extraction() -> None:
    """Run the extraction process."""
    uploaded_files = st.session_state.uploaded_files
    if not uploaded_files:
        return

    st.session_state.is_processing = True
    st.session_state.extraction_results = {}
    st.session_state.combined_data = None

    pipeline = get_pipeline()

    # Progress containers
    progress_bar = st.progress(0, text="Démarrage de l'extraction...")
    status_text = st.empty()

    # Save files to temp directory
    temp_dir = tempfile.mkdtemp()
    temp_paths = []

    for file in uploaded_files:
        temp_path = Path(temp_dir) / file.name
        temp_path.write_bytes(file.read())
        temp_paths.append(temp_path)
        file.seek(0)

    st.session_state.temp_pdf_paths = temp_paths

    # Progress callback
    def on_progress(current: int, total: int, filename: str) -> None:
        progress = current / total if total > 0 else 0
        progress_bar.progress(progress, text=f"Traitement: {filename}")

    try:
        # Run extraction
        batch_result = asyncio.run(
            pipeline.process_batch(
                pdf_paths=temp_paths,
                source=st.session_state.selected_source,
                force_refresh=st.session_state.force_refresh,
                progress_callback=on_progress
            )
        )

        st.session_state.batch_result = batch_result

        # Store results
        results = {}
        for result in batch_result.results:
            filename = Path(result.pdf_path).name
            results[filename] = result

        st.session_state.extraction_results = results
        st.session_state.combined_data = batch_result.get_combined_dataframe()

        # Show success
        if batch_result.failed == 0:
            status_text.success(f"✅ {batch_result.total_rows} lignes extraites de {batch_result.successful} fichier(s)!")
        else:
            status_text.warning(f"⚠️ {batch_result.total_rows} lignes extraites. {batch_result.failed} fichier(s) en erreur.")

    except Exception as e:
        status_text.error(f"Échec de l'extraction: {e}")

    finally:
        st.session_state.is_processing = False
        progress_bar.empty()


# =============================================================================
# SECTION 2: DATA PREVIEW & EDIT
# =============================================================================

def render_preview_section() -> None:
    """Render the data preview and edit section."""
    st.header("📋 2. Prévisualisation & Édition")

    combined_data = st.session_state.combined_data

    if combined_data is None or combined_data.empty:
        st.info("Aucune donnée à prévisualiser. Uploadez et extrayez d'abord les fichiers PDF.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Lignes", len(combined_data))

    with col2:
        st.metric("Colonnes", len(combined_data.columns))

    with col3:
        if "Total" in combined_data.columns:
            total = pd.to_numeric(combined_data["Total"], errors="coerce").sum()
            st.metric("Total Commissions", f"{total:,.2f} $")
        elif "Com" in combined_data.columns:
            total = pd.to_numeric(combined_data["Com"], errors="coerce").sum()
            st.metric("Total Com", f"{total:,.2f} $")

    with col4:
        if "Compagnie" in combined_data.columns:
            st.metric("Compagnies", combined_data["Compagnie"].nunique())

    # Extraction results details
    results = st.session_state.extraction_results
    if results:
        with st.expander("📊 Détails de l'extraction", expanded=False):
            for filename, result in results.items():
                if result.success:
                    st.markdown(f"✅ **{filename}**: {result.row_count} lignes ({result.extraction_time_ms}ms)")
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(f"  ⚠️ {w.message}")
                else:
                    st.markdown(f"❌ **{filename}**: {result.error}")

    # Data editor tabs
    tab1, tab2 = st.tabs(["📝 Éditer les données", "📥 Télécharger"])

    with tab1:
        render_data_editor(combined_data)

    with tab2:
        render_download_options(combined_data)


def render_data_editor(df: pd.DataFrame) -> None:
    """Render the editable data table."""
    # Filter controls
    with st.expander("🔍 Filtres", expanded=False):
        col1, col2, col3 = st.columns(3)

        filtered_df = df.copy()

        with col1:
            if "Compagnie" in df.columns:
                companies = ["Tous"] + sorted(df["Compagnie"].dropna().unique().tolist())
                selected = st.selectbox("Compagnie", companies, key="filter_company")
                if selected != "Tous":
                    filtered_df = filtered_df[filtered_df["Compagnie"] == selected]

        with col2:
            if "Statut" in df.columns:
                statuses = ["Tous"] + sorted(df["Statut"].dropna().unique().tolist())
                selected = st.selectbox("Statut", statuses, key="filter_status")
                if selected != "Tous":
                    filtered_df = filtered_df[filtered_df["Statut"] == selected]

        with col3:
            if "_source_file" in df.columns:
                sources = ["Tous"] + sorted(df["_source_file"].dropna().unique().tolist())
                selected = st.selectbox("Fichier source", sources, key="filter_source")
                if selected != "Tous":
                    filtered_df = filtered_df[filtered_df["_source_file"] == selected]

        if len(filtered_df) != len(df):
            st.caption(f"Affichage de {len(filtered_df)} sur {len(df)} lignes")

    # Hide internal columns for display
    display_cols = [c for c in filtered_df.columns if not c.startswith("_")]
    display_df = filtered_df[display_cols]

    # Column config
    column_config = {}
    for col in display_df.columns:
        if col in ["PA", "Com", "Boni", "Sur-Com", "Reçu", "Total"]:
            column_config[col] = st.column_config.NumberColumn(col, format="%.2f")
        elif col in ["Date", "Paie"]:
            column_config[col] = st.column_config.DateColumn(col, format="YYYY-MM-DD")
        elif col in ["Verifié", "Complet"]:
            column_config[col] = st.column_config.CheckboxColumn(col)
        elif col == "Statut":
            column_config[col] = st.column_config.SelectboxColumn(
                col,
                options=["Approved", "Pending", "Rejected", "Received", ""]
            )

    # Editable table
    edited_df = st.data_editor(
        display_df,
        key="data_editor",
        num_rows="dynamic",
        column_config=column_config,
        height=400,
        use_container_width=True
    )

    # Update state if edited
    if not edited_df.equals(display_df):
        st.session_state.data_modified = True
        # Restore hidden columns
        for col in filtered_df.columns:
            if col.startswith("_") and col in filtered_df.columns:
                edited_df[col] = filtered_df[col].values[:len(edited_df)]
        st.session_state.combined_data = edited_df

    if st.session_state.data_modified:
        st.info("📝 Les données ont été modifiées")

    # Reset button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Annuler les modifications"):
            batch_result = st.session_state.batch_result
            if batch_result:
                st.session_state.combined_data = batch_result.get_combined_dataframe()
                st.session_state.data_modified = False
                st.rerun()


def render_download_options(df: pd.DataFrame) -> None:
    """Render download options."""
    # Filter internal columns
    export_df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")

    st.markdown(f"**{len(export_df)} lignes** prêtes à télécharger")

    col1, col2 = st.columns(2)

    with col1:
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger CSV",
            data=csv,
            file_name="commission_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        try:
            import io
            buffer = io.BytesIO()
            export_df.to_excel(buffer, index=False, engine="openpyxl")
            buffer.seek(0)
            st.download_button(
                "📥 Télécharger Excel",
                data=buffer,
                file_name="commission_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except ImportError:
            st.caption("Installez openpyxl pour l'export Excel")


# =============================================================================
# BOARD HELPERS
# =============================================================================

def sort_and_filter_boards(boards: list, search_query: str = "") -> list:
    """Sort boards with priority keywords first and filter by search query."""
    import re

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
    """
    Detect the board type based on regex patterns in the board name.

    Returns:
        "Ventes et Production" or "Paiements Historiques"
    """
    import re

    if not board_name:
        return "Paiements Historiques"

    name_lower = board_name.lower()

    # Regex patterns for Sales/Production
    sales_patterns = [
        r'vente[s]?',
        r'production[s]?',
        r'sales?',
        r'prod\b',
        r'commercial',
        r'soumis',
        r'proposition[s]?',
    ]

    # Regex patterns for Historical Payments
    payment_patterns = [
        r'paiement[s]?',
        r'historique[s]?',
        r'payment[s]?',
        r'history',
        r'hist\b',
        r'reçu[s]?',
        r'commission[s]?',
        r'statement[s]?',
    ]

    # Check for sales/production patterns first
    for pattern in sales_patterns:
        if re.search(pattern, name_lower):
            return "Ventes et Production"

    # Check for payment patterns
    for pattern in payment_patterns:
        if re.search(pattern, name_lower):
            return "Paiements Historiques"

    # Default to Historical Payments
    return "Paiements Historiques"


def on_board_select_change():
    """Callback when board selection changes - auto-detect board type."""
    if 'board_selector' in st.session_state:
        board_name = st.session_state.board_selector
        detected_type = detect_board_type_from_name(board_name)
        st.session_state._detected_board_type = detected_type


# =============================================================================
# SECTION 3: MONDAY.COM EXPORT
# =============================================================================

def render_export_section() -> None:
    """Render the Monday.com export section."""
    st.header("📤 3. Export to Monday.com")

    combined_data = st.session_state.combined_data
    api_key = st.session_state.monday_api_key

    if combined_data is None or combined_data.empty:
        st.info("Aucune donnée à exporter. Complétez d'abord l'extraction.")
        return

    if not api_key:
        st.warning("⚠️ Clé API Monday.com non configurée. Ajoutez-la dans la barre latérale.")
        return

    # Fetch boards if not loaded
    if st.session_state.monday_boards is None:
        fetch_boards()

    # Board selection with search
    st.subheader("📋 Sélection du Board")

    col1, col2 = st.columns([3, 1])

    with col1:
        # Search box
        search_query = st.text_input(
            "🔍 Rechercher un board",
            placeholder="Filtrer par nom...",
            key="board_search"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Rafraîchir", use_container_width=True):
            fetch_boards()

    # Board dropdown with filtered/sorted results
    render_board_selection(search_query)

    # Board type detection and configuration
    board_id = st.session_state.selected_board_id
    if board_id:
        # Show detected board type
        board_name = st.session_state.get("_current_board_name", "")
        detected_type = detect_board_type_from_name(board_name)

        st.caption(f"🔍 Type détecté: **{detected_type}**")

        # Board type override
        col1, col2 = st.columns(2)
        with col1:
            type_options = ["Paiements Historiques", "Ventes et Production"]
            default_idx = type_options.index(detected_type) if detected_type in type_options else 0
            selected_type = st.selectbox(
                "Type de board",
                type_options,
                index=default_idx,
                key="board_type_selector",
                help="Le type détermine les colonnes exportées"
            )
            st.session_state.selected_board_type = (
                BoardType.SALES_PRODUCTION if selected_type == "Ventes et Production"
                else BoardType.HISTORICAL_PAYMENTS
            )

        st.divider()

        # Group selection
        render_group_selection(board_id)

    st.divider()

    # Upload section
    render_upload_controls(combined_data)


def fetch_boards() -> None:
    """Fetch boards from Monday.com."""
    pipeline = get_pipeline()

    if not pipeline.monday_configured:
        st.error("Monday.com not configured")
        return

    with st.spinner("Loading boards..."):
        try:
            boards = asyncio.run(pipeline.monday.list_boards())
            st.session_state.monday_boards = boards
        except Exception as e:
            st.error(f"Failed to fetch boards: {e}")
            st.session_state.monday_boards = []


def render_board_selection(search_query: str = "") -> None:
    """Render board selection dropdown with search and filtering."""
    boards = st.session_state.monday_boards

    if boards is None:
        st.info("Chargement des boards...")
        return

    if not boards:
        st.warning("Aucun board trouvé")
        return

    # Filter and sort boards
    sorted_boards = sort_and_filter_boards(boards, search_query)

    if not sorted_boards:
        st.warning(f"Aucun board ne correspond à '{search_query}'")
        return

    board_options = {b["name"]: b["id"] for b in sorted_boards}
    selected_name = st.selectbox(
        "Board de destination",
        list(board_options.keys()),
        key="board_selector"
    )

    if selected_name:
        new_board_id = board_options[selected_name]
        st.session_state._current_board_name = selected_name
        if st.session_state.selected_board_id != new_board_id:
            st.session_state.selected_board_id = new_board_id
            st.session_state.monday_groups = None
            st.session_state.selected_group_id = None
        st.caption(f"ID: {new_board_id}")


def render_group_selection(board_id: int) -> None:
    """Render group selection."""
    st.subheader("📁 Sélection du Groupe")

    # Fetch groups if needed
    if st.session_state.monday_groups is None:
        fetch_groups(board_id)

    groups = st.session_state.monday_groups or []

    # Group selection
    col1, col2 = st.columns([2, 2])

    with col1:
        group_options = {"(Créer un nouveau groupe)": None}
        group_options.update({g["title"]: g["id"] for g in groups})

        selected_title = st.selectbox(
            "Groupe de destination",
            list(group_options.keys()),
            key="group_selector"
        )

        if selected_title == "(Créer un nouveau groupe)":
            st.session_state.selected_group_id = None
        else:
            st.session_state.selected_group_id = group_options[selected_title]

    with col2:
        if selected_title == "(Créer un nouveau groupe)":
            new_name = st.text_input(
                "Nom du nouveau groupe",
                placeholder="Ex: Janvier 2025",
                key="new_group_name"
            )
            st.session_state.new_group_name = new_name
        else:
            st.session_state.new_group_name = None


def fetch_groups(board_id: int) -> None:
    """Fetch groups for a board."""
    pipeline = get_pipeline()

    try:
        groups = asyncio.run(pipeline.monday.list_groups(board_id))
        st.session_state.monday_groups = groups
    except Exception as e:
        st.error(f"Failed to fetch groups: {e}")
        st.session_state.monday_groups = []


def render_upload_controls(df: pd.DataFrame) -> None:
    """Render upload controls and execute upload."""
    st.subheader("🚀 Upload")

    board_id = st.session_state.selected_board_id
    group_id = st.session_state.selected_group_id
    new_group_name = getattr(st.session_state, "new_group_name", None)
    board_type = st.session_state.get("selected_board_type", BoardType.HISTORICAL_PAYMENTS)

    # Validation
    can_upload = board_id and (group_id or new_group_name)

    # Export data preview
    export_df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Lignes", len(export_df))

    with col2:
        st.metric("Colonnes", len(export_df.columns))

    with col3:
        board_status = "✓" if board_id else "Non sélectionné"
        st.metric("Board", board_status)

    with col4:
        type_label = "Ventes" if board_type == BoardType.SALES_PRODUCTION else "Paiements"
        st.metric("Type", type_label)

    # Upload button
    col1, col2 = st.columns([2, 1])

    with col1:
        is_uploading = st.session_state.is_uploading
        if st.button(
            "🚀 Uploader vers Monday.com",
            type="primary",
            disabled=not can_upload or is_uploading,
            use_container_width=True
        ):
            run_upload(export_df, board_id, group_id, new_group_name)

    with col2:
        if st.button("🔄 Réinitialiser", use_container_width=True):
            st.session_state.upload_result = None
            st.rerun()

    # Show results
    upload_result = st.session_state.upload_result
    if upload_result:
        st.divider()
        render_upload_result(upload_result)


def run_upload(
    df: pd.DataFrame,
    board_id: int,
    group_id: Optional[str],
    new_group_name: Optional[str]
) -> None:
    """Execute the upload to Monday.com."""
    st.session_state.is_uploading = True
    st.session_state.upload_result = None

    pipeline = get_pipeline()

    progress_bar = st.progress(0, text="Démarrage de l'upload...")
    status_text = st.empty()

    try:
        # Create group if needed
        actual_group_id = group_id
        if new_group_name and not group_id:
            status_text.info(f"Création du groupe: {new_group_name}")
            actual_group_id = asyncio.run(
                pipeline.monday.get_or_create_group(board_id, new_group_name)
            )

        # Progress callback
        def on_progress(current: int, total: int, item_name: str) -> None:
            progress = current / total if total > 0 else 0
            progress_bar.progress(progress, text=f"Upload: {current}/{total}")

        # Upload
        status_text.info("Upload des données...")
        result = asyncio.run(
            pipeline.monday.upload_dataframe(
                df=df,
                board_id=board_id,
                group_id=actual_group_id,
                progress_callback=on_progress
            )
        )

        # Store result
        st.session_state.upload_result = {
            "total": result.total,
            "success": result.success,
            "failed": result.failed,
            "errors": result.errors,
        }

        if result.failed == 0:
            status_text.success(f"✅ {result.success} éléments uploadés!")
        else:
            status_text.warning(f"⚠️ {result.success}/{result.total} uploadés. {result.failed} en erreur.")

    except Exception as e:
        status_text.error(f"Échec de l'upload: {e}")
        st.session_state.upload_result = {
            "total": len(df),
            "success": 0,
            "failed": len(df),
            "errors": [str(e)],
        }

    finally:
        st.session_state.is_uploading = False
        progress_bar.empty()


def render_upload_result(result: dict) -> None:
    """Render upload result summary."""
    total = result.get("total", 0)
    success = result.get("success", 0)
    failed = result.get("failed", 0)

    if failed == 0:
        st.success(f"✅ {success}/{total} éléments uploadés avec succès vers Monday.com!")
    else:
        st.warning(f"⚠️ {success}/{total} éléments uploadés. {failed} en erreur.")

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
    init_session_state()
    render_sidebar()

    # Title
    st.title("📊 Pipeline de Commissions")
    st.caption("Extraire les données de commissions des PDFs et les exporter vers Monday.com")

    st.divider()

    # All sections on single page
    render_upload_section()

    st.divider()

    render_preview_section()

    st.divider()

    render_export_section()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
