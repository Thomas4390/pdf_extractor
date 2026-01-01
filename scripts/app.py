"""
Streamlit Application - Insurance Commission Data Pipeline
===========================================================

Application web pour extraire, visualiser et uploader les données
de commissions d'assurance vers Monday.com.

Author: Thomas
Date: 2025-10-30
Version: 2.0.0 - UI/UX Refactored
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Project root directory (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent

# Import pipeline components
from main import (
    InsuranceCommissionPipeline,
    PipelineConfig,
    InsuranceSource,
)
from unify_notation import BoardType

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
# CUSTOM CSS FOR MODERN LOOK
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

    /* Card-like containers */
    .css-1r6slb0 {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        padding: 1rem;
    }

    /* Stepper styling */
    .step-active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
    }
    .step-completed {
        color: #28a745;
        font-weight: 500;
    }
    .step-pending {
        color: #6c757d;
    }

    /* Reduce spacing */
    .element-container {
        margin-bottom: 0.5rem;
    }

    /* Form styling */
    [data-testid="stForm"] {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        background: #fafafa;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 500;
        font-size: 0.95rem;
    }

    /* Hide anchor links */
    .css-15zrgzn {display: none}
    .css-zt5igj {display: none}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'stage': 1,
        'pdf_file': None,
        'pdf_path': None,
        'extracted_data': None,
        'pipeline': None,
        'config': None,
        'upload_results': None,
        'data_modified': False,
        'monday_boards': None,
        'selected_board_id': None,
        'monday_api_key': None,
        'boards_loading': False,
        'verification_tolerance': 10.0,  # Default tolerance percentage for Reçu vs Com verification
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def verify_recu_vs_com(df: pd.DataFrame, tolerance_pct: float = 10.0) -> pd.DataFrame:
    """
    Verify that Reçu is within tolerance range of calculated Com for each row.

    The comparison uses a CALCULATED commission value based on the formula:
        Com_calculée = ROUND((PA * 0.4) * 0.5, 2)

    This is different from the 'Com' column which may contain other calculated values.
    The original 'Com', 'Boni', 'Sur-Com' columns are preserved unchanged.

    Args:
        df: DataFrame with 'Reçu' and 'PA' columns
        tolerance_pct: Tolerance percentage (default 10%)

    Returns:
        DataFrame with added columns:
        - 'Com Calculée': The calculated commission for comparison
        - 'Vérification': Status flag
            - '✅ Bonus' if Reçu > Com_calculée * (1 + tolerance) - positive flag (good)
            - '⚠️ Écart' if Reçu < Com_calculée * (1 - tolerance) - negative flag (problem)
            - '✓ OK' if within tolerance range
            - '-' if data is missing
    """
    result_df = df.copy()

    # Check if required columns exist (need PA and Reçu for calculation)
    if 'Reçu' not in result_df.columns or 'PA' not in result_df.columns:
        return result_df

    # Convert to numeric
    recu = pd.to_numeric(result_df['Reçu'], errors='coerce')
    pa = pd.to_numeric(result_df['PA'], errors='coerce')

    # Calculate expected commission using formula: ROUND((PA * 0.4) * 0.5, 2)
    # This represents: PA * sharing_rate(40%) * commission_rate(50%)
    com_calculee = (pa * 0.4 * 0.5).round(2)

    # Add calculated commission column for transparency
    result_df['Com Calculée'] = com_calculee

    # Calculate tolerance bounds based on calculated commission
    tolerance = tolerance_pct / 100.0
    lower_bound = com_calculee * (1 - tolerance)
    upper_bound = com_calculee * (1 + tolerance)

    # Calculate percentage difference for display
    pct_diff = ((recu - com_calculee) / com_calculee * 100).round(1)

    # Create verification column
    verification = []
    for i in range(len(result_df)):
        r = recu.iloc[i]
        c = com_calculee.iloc[i]
        diff = pct_diff.iloc[i]

        if pd.isna(r) or pd.isna(c) or c == 0:
            verification.append('-')
        elif r > upper_bound.iloc[i]:
            verification.append(f'✅ +{diff}%')  # Positive flag (bonus/good)
        elif r < lower_bound.iloc[i]:
            verification.append(f'⚠️ {diff}%')  # Negative flag (problem)
        else:
            verification.append('✓ OK')

    # Column name includes tolerance to show user which tolerance is applied
    result_df[f'Vérification (±{tolerance_pct:.0f}%)'] = verification

    return result_df


def get_verification_stats(df: pd.DataFrame) -> dict:
    """
    Get statistics about verification results.

    Returns:
        Dictionary with counts of each verification status
    """
    # Find verification column (name includes tolerance percentage)
    verif_cols = [col for col in df.columns if col.startswith('Vérification')]
    if not verif_cols:
        return {'ok': 0, 'bonus': 0, 'ecart': 0, 'na': 0}

    verif = df[verif_cols[0]].astype(str)

    return {
        'ok': verif.str.contains('OK', na=False).sum(),
        'bonus': verif.str.contains('✅', na=False).sum(),
        'ecart': verif.str.contains('⚠️', na=False).sum(),
        'na': (verif == '-').sum()
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_boards_async():
    """Load Monday.com boards automatically when API key is set."""
    if (st.session_state.monday_api_key and
        st.session_state.monday_boards is None and
        not st.session_state.boards_loading):
        try:
            st.session_state.boards_loading = True
            from monday_automation import MondayClient
            client = MondayClient(api_key=st.session_state.monday_api_key)
            st.session_state.monday_boards = client.list_boards()
            st.session_state.boards_loading = False
        except Exception:
            st.session_state.boards_loading = False


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


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to a temporary location and return the path."""
    temp_dir = Path("./temp")
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)


def cleanup_temp_file(file_path: str = None):
    """Clean up temporary file."""
    if file_path is None:
        file_path = st.session_state.get('pdf_path')
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass


def reset_pipeline():
    """Reset pipeline state to start over."""
    cleanup_temp_file()
    keys_to_reset = ['stage', 'pdf_file', 'pdf_path', 'extracted_data',
                     'pipeline', 'config', 'upload_results', 'data_modified',
                     'monday_boards', 'selected_board_id']
    for key in keys_to_reset:
        if key == 'stage':
            st.session_state[key] = 1
        else:
            st.session_state[key] = None
    st.session_state.data_modified = False


def render_stepper():
    """Render the progress stepper in main content area."""
    stages = [
        ("1", "Configuration", "📁"),
        ("2", "Prévisualisation", "🔍"),
        ("3", "Upload", "☁️")
    ]

    cols = st.columns(3)
    for i, (num, name, icon) in enumerate(stages):
        stage_num = i + 1
        with cols[i]:
            if stage_num == st.session_state.stage:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px; color: white;">
                    <div style="font-size: 1.5rem;">{icon}</div>
                    <div style="font-weight: 600;">{name}</div>
                </div>
                """, unsafe_allow_html=True)
            elif stage_num < st.session_state.stage:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: #d4edda;
                border-radius: 10px; color: #155724;">
                    <div style="font-size: 1.5rem;">✅</div>
                    <div style="font-weight: 500;">{name}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: #f8f9fa;
                border-radius: 10px; color: #6c757d;">
                    <div style="font-size: 1.5rem;">{icon}</div>
                    <div>{name}</div>
                </div>
                """, unsafe_allow_html=True)


# =============================================================================
# SIDEBAR - SIMPLIFIED
# =============================================================================

def render_sidebar():
    """Render simplified sidebar."""
    with st.sidebar:
        st.markdown("## 🔑 Configuration")

        # API Key section - compact
        if st.session_state.monday_api_key:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success("API connectée", icon="✅")
            with col2:
                if st.button("✏️", help="Modifier la clé API"):
                    st.session_state.monday_api_key = None
                    st.session_state.monday_boards = None
                    st.rerun()

            # Show boards count
            if st.session_state.monday_boards:
                st.caption(f"📋 {len(st.session_state.monday_boards)} boards disponibles")
        else:
            api_key = st.text_input(
                "Clé API Monday.com",
                type="password",
                placeholder="Entrez votre clé API...",
                key="sidebar_api_key"
            )
            if api_key:
                if st.button("Connecter", type="primary", use_container_width=True):
                    st.session_state.monday_api_key = api_key
                    st.rerun()

        st.divider()

        # Quick actions
        st.markdown("### ⚡ Actions rapides")

        if st.session_state.stage > 1:
            if st.button("⬅️ Retour au début", use_container_width=True):
                reset_pipeline()
                st.rerun()

        if st.session_state.monday_api_key and st.session_state.monday_boards is None:
            if st.button("📥 Charger les boards", use_container_width=True, type="primary"):
                load_boards_async()
                st.rerun()

        if st.session_state.monday_boards:
            if st.button("🔄 Rafraîchir boards", use_container_width=True):
                st.session_state.monday_boards = None
                load_boards_async()
                st.rerun()

        st.divider()

        # Help section - collapsible
        with st.expander("ℹ️ Aide", expanded=False):
            st.markdown("""
            **Sources supportées:**
            - UV Assurance
            - IDC / IDC Statement
            - Assomption Vie
            - Monday.com Legacy

            **Besoin d'aide?**
            Contactez le support technique.
            """)


# =============================================================================
# STAGE 1: CONFIGURATION - REORGANIZED
# =============================================================================

def render_stage_1():
    """Render configuration stage with improved UX."""

    # Header with stepper
    st.markdown("## 📊 Pipeline de Commissions")
    render_stepper()
    st.write("")

    # Check API key first
    if not st.session_state.monday_api_key:
        st.warning("👈 Veuillez d'abord configurer votre clé API Monday.com dans la barre latérale.")
        return

    # Auto-load boards
    if st.session_state.monday_boards is None:
        load_boards_async()
        if st.session_state.boards_loading:
            st.info("⏳ Chargement des boards en cours...")
            return

    # Tabs for different workflows
    tab1, tab2 = st.tabs(["📄 Extraction PDF", "🔄 Migration Monday.com"])

    # =========================================================================
    # TAB 1: PDF EXTRACTION
    # =========================================================================
    with tab1:
        render_pdf_extraction_tab()

    # =========================================================================
    # TAB 2: MONDAY.COM MIGRATION
    # =========================================================================
    with tab2:
        render_monday_migration_tab()


def detect_board_type_from_name(board_name: str) -> str:
    """
    Detect the board type based on regex patterns in the board name.

    Uses regex to match common variations of keywords for each board type.

    Args:
        board_name: Name of the board

    Returns:
        "Ventes et Production" or "Paiements Historiques"
    """
    import re

    if not board_name:
        return "Paiements Historiques"

    name_lower = board_name.lower()

    # Regex patterns for Sales/Production (more flexible matching)
    sales_patterns = [
        r'vente[s]?',           # vente, ventes
        r'production[s]?',      # production, productions
        r'sales?',              # sale, sales
        r'prod\b',              # prod (abbreviation)
        r'commercial',          # commercial
        r'soumis',              # soumissions
        r'proposition[s]?',     # proposition, propositions
    ]

    # Regex patterns for Historical Payments
    payment_patterns = [
        r'paiement[s]?',        # paiement, paiements
        r'historique[s]?',      # historique, historiques
        r'payment[s]?',         # payment, payments
        r'history',             # history
        r'hist\b',              # hist (abbreviation)
        r'reçu[s]?',            # reçu, reçus
        r'commission[s]?',      # commission, commissions (often payment related)
        r'statement[s]?',       # statement, statements
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
    """Callback when board selection changes - auto-detect and update target type."""
    if 'pdf_board_select' in st.session_state:
        board_name = st.session_state.pdf_board_select
        detected_type = detect_board_type_from_name(board_name)
        # Store detected type - will be used to set the selectbox index
        st.session_state._detected_board_type = detected_type
        # Also update aggregate checkbox based on detected type
        st.session_state.pdf_aggregate = detected_type == "Ventes et Production"
        # Force the target type widget to update by deleting its key
        # This allows the index parameter to take effect on next render
        if 'pdf_target_type' in st.session_state:
            del st.session_state.pdf_target_type


def on_target_type_change():
    """Callback when target type changes - update aggregate checkbox accordingly."""
    if 'pdf_target_type' in st.session_state:
        # Coché pour Ventes et Production, décoché pour Paiements Historiques
        st.session_state.pdf_aggregate = st.session_state.pdf_target_type == "Ventes et Production"


def render_pdf_extraction_tab():
    """Render PDF extraction tab with simplified flow."""

    # Step 1: Upload PDF first (most important action)
    st.markdown("### 📤 Upload du fichier PDF")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Déposez votre fichier PDF ici",
            type=['pdf'],
            help="Fichier PDF contenant les données de commissions",
            key="pdf_upload_main"
        )

        if uploaded_file:
            st.success(f"✅ Fichier chargé: {uploaded_file.name}")

    with col2:
        source = st.selectbox(
            "Source",
            options=["UV", "IDC", "IDC Statement", "ASSOMPTION"],
            help="Type de document PDF"
        )

    if not uploaded_file:
        st.info("👆 Commencez par uploader un fichier PDF pour continuer.")
        return

    st.divider()

    # Step 2: Board selection
    st.markdown("### 📋 Destination Monday.com")

    board_mode = st.radio(
        "Board de destination",
        options=["Utiliser un board existant", "Créer un nouveau board"],
        horizontal=True,
        key="pdf_board_mode_radio"
    )

    selected_board_id = None
    board_name = None
    reuse_board = True
    reuse_group = True

    if board_mode == "Utiliser un board existant":
        if st.session_state.monday_boards:
            # Search
            search = st.text_input(
                "🔍 Rechercher",
                placeholder="Filtrer par nom...",
                key="pdf_search_board"
            )

            sorted_boards = sort_and_filter_boards(
                st.session_state.monday_boards,
                search_query=search
            )

            if sorted_boards:
                board_options = {f"{b['name']}": b['id'] for b in sorted_boards}
                selected_name = st.selectbox(
                    "Board",
                    options=list(board_options.keys()),
                    key="pdf_board_select",
                    on_change=on_board_select_change
                )
                selected_board_id = board_options[selected_name]
                board_name = selected_name

                # Auto-detect board type on first load (when no callback has been triggered yet)
                if 'pdf_target_type' not in st.session_state:
                    detected_type = detect_board_type_from_name(selected_name)
                    st.session_state.pdf_target_type = detected_type
                    st.session_state.pdf_aggregate = detected_type == "Ventes et Production"

                st.caption(f"ID: {selected_board_id}")
            else:
                st.warning("Aucun board trouvé.")
                board_name = None  # No board selected
        else:
            st.warning("Chargez d'abord vos boards via la barre latérale.")
            board_name = None  # No boards loaded
    else:
        board_name = st.text_input(
            "Nom du nouveau board",
            placeholder=f"Commissions {source}",
            key="pdf_new_board_name"
        )
        if not board_name:
            board_name = f"Commissions {source}"

        col1, col2 = st.columns(2)
        with col1:
            reuse_board = st.checkbox("Réutiliser si existe", value=True, key="pdf_reuse_board")
        with col2:
            reuse_group = st.checkbox("Réutiliser groupe", value=True, key="pdf_reuse_group")

    # Step 3: Configuration options
    st.markdown("### ⚙️ Configuration")

    # Determine current type - prioritize detected type from callback, then widget value, then default
    type_options = ["Paiements Historiques", "Ventes et Production"]

    # Use detected type if available (from board selection callback), otherwise use widget state
    if '_detected_board_type' in st.session_state:
        current_type = st.session_state._detected_board_type
    elif 'pdf_target_type' in st.session_state:
        current_type = st.session_state.pdf_target_type
    else:
        current_type = 'Paiements Historiques'

    current_index = type_options.index(current_type) if current_type in type_options else 0

    # Groupe (optionnel)
    month_group = st.text_input(
        "Groupe (optionnel)",
        placeholder="Ex: Novembre 2025",
        key="pdf_month_group"
    )

    # Type de table avec détection automatique
    if board_name:
        detected = detect_board_type_from_name(board_name)
        st.caption(f"🔍 Type détecté automatiquement: **{detected}**")

    target_type = st.selectbox(
        "Type de table",
        options=type_options,
        index=current_index,
        key="pdf_target_type",
        on_change=on_target_type_change
    )

    # Sync detected type with actual widget value after render
    st.session_state._detected_board_type = target_type

    # Agrégation
    aggregate = st.checkbox(
        "Agréger par contrat",
        value=st.session_state.get('pdf_aggregate', False),
        help="Combine les lignes avec le même numéro de contrat",
        key="pdf_aggregate"
    )

    st.divider()

    # Submit button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Extraire les données", type="primary", use_container_width=True):
            # Validation
            errors = []
            if board_mode == "Utiliser un board existant" and not selected_board_id:
                errors.append("Veuillez sélectionner un board")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                # Save file and create config
                pdf_path = save_uploaded_file(uploaded_file)

                target_board_type = (BoardType.SALES_PRODUCTION
                                    if target_type == "Ventes et Production"
                                    else BoardType.HISTORICAL_PAYMENTS)

                config = PipelineConfig(
                    source=InsuranceSource(source.replace(" ", "_").upper()),
                    pdf_path=pdf_path,
                    month_group=month_group if month_group else None,
                    board_name=board_name,
                    monday_api_key=st.session_state.monday_api_key,
                    output_dir=str(PROJECT_ROOT / "results"),
                    reuse_board=reuse_board,
                    reuse_group=reuse_group,
                    aggregate_by_contract=aggregate,
                    source_board_id=None,
                    source_group_id=None,
                    target_board_type=target_board_type
                )

                st.session_state.pdf_file = uploaded_file
                st.session_state.pdf_path = pdf_path
                st.session_state.config = config
                st.session_state.stage = 2
                st.rerun()


def render_monday_migration_tab():
    """Render Monday.com migration tab."""

    st.info("""
    **🔄 Migration de Board**

    Convertit un ancien board Monday.com vers le nouveau format standardisé.
    Cette fonctionnalité est conçue pour une migration unique.
    """)

    if not st.session_state.monday_boards:
        st.warning("Chargez d'abord vos boards via la barre latérale.")
        return

    # Source board selection
    st.markdown("### 📥 Board source")

    search_source = st.text_input(
        "🔍 Rechercher",
        placeholder="Filtrer par nom...",
        key="legacy_search"
    )

    sorted_boards = sort_and_filter_boards(
        st.session_state.monday_boards,
        search_query=search_source
    )

    if not sorted_boards:
        st.warning("Aucun board trouvé.")
        return

    source_options = {f"{b['name']}": b['id'] for b in sorted_boards}
    source_name = st.selectbox(
        "Board à convertir",
        options=list(source_options.keys()),
        key="legacy_source_select"
    )
    source_board_id = source_options[source_name]

    st.divider()

    # Target board
    st.markdown("### 📤 Board destination")

    target_name = st.text_input(
        "Nom du nouveau board",
        placeholder="Commissions - Nouveau Format",
        key="legacy_target_name"
    )

    col1, col2 = st.columns(2)
    with col1:
        reuse_board = st.checkbox("Réutiliser si existe", value=True, key="legacy_reuse_board")
    with col2:
        reuse_group = st.checkbox("Réutiliser groupes", value=True, key="legacy_reuse_group")

    # Advanced options
    with st.expander("⚙️ Options avancées", expanded=False):
        aggregate = st.checkbox(
            "Agréger par contrat",
            value=False,
            help="Normalement désactivé pour préserver la structure",
            key="legacy_aggregate"
        )

        st.caption("""
        **Constantes appliquées:**
        - sharing_rate = 40%
        - commission_rate = 50%
        - bonus_rate = 175%
        - on_commission_rate = 75%
        """)

    st.divider()

    # Submit
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Convertir le board", type="primary", use_container_width=True):
            if not target_name or not target_name.strip():
                st.error("Veuillez entrer un nom pour le nouveau board")
            else:
                config = PipelineConfig(
                    source=InsuranceSource.MONDAY_LEGACY,
                    pdf_path=None,
                    month_group=None,
                    board_name=target_name.strip(),
                    monday_api_key=st.session_state.monday_api_key,
                    output_dir=str(PROJECT_ROOT / "results/monday_legacy"),
                    reuse_board=reuse_board,
                    reuse_group=reuse_group,
                    aggregate_by_contract=aggregate,
                    source_board_id=int(source_board_id),
                    source_group_id=None,
                    target_board_type=None
                )

                st.session_state.config = config
                st.session_state.stage = 2
                st.rerun()


# =============================================================================
# STAGE 2: PREVIEW - CLEANER LAYOUT
# =============================================================================

def render_stage_2():
    """Render data preview stage."""

    st.markdown("## 📊 Pipeline de Commissions")
    render_stepper()
    st.write("")

    config = st.session_state.config

    # Config summary - compact
    with st.expander("📋 Configuration", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Source", config.source.value)
        cols[1].metric("Board", config.board_name[:20] + "..." if len(config.board_name) > 20 else config.board_name)
        cols[2].metric("Groupe", config.month_group or "Défaut")
        cols[3].metric("Agrégation", "Oui" if config.aggregate_by_contract else "Non")

    # Extract data if not done
    if st.session_state.extracted_data is None:
        source_type = "Monday.com" if config.source == InsuranceSource.MONDAY_LEGACY else "PDF"

        with st.spinner(f"🔄 Extraction depuis {source_type}..."):
            try:
                pipeline = InsuranceCommissionPipeline(config)

                if not pipeline._step1_extract_data():
                    st.error("❌ Échec de l'extraction")
                    if st.button("🔄 Recommencer"):
                        reset_pipeline()
                        st.rerun()
                    return

                if not pipeline._step2_process_data():
                    st.error("❌ Échec du traitement")
                    if st.button("🔄 Recommencer"):
                        reset_pipeline()
                        st.rerun()
                    return

                st.session_state.extracted_data = pipeline.final_data
                st.session_state.pipeline = pipeline
                st.rerun()

            except Exception as e:
                st.error(f"❌ Erreur: {e}")
                with st.expander("Détails"):
                    st.exception(e)
                if st.button("🔄 Recommencer"):
                    reset_pipeline()
                    st.rerun()
                return

    df = st.session_state.extracted_data

    if df is None or df.empty:
        st.error("❌ Aucune donnée extraite")
        if st.button("🔄 Recommencer"):
            reset_pipeline()
            st.rerun()
        return

    # Modified data notice
    if st.session_state.data_modified:
        st.info("📝 Données modifiées (fichier uploadé)")

    # Statistics - compact cards
    st.markdown("### 📊 Aperçu")

    cols = st.columns(4)
    cols[0].metric("Lignes", len(df))
    cols[1].metric("Colonnes", len(df.columns))
    if '# de Police' in df.columns:
        cols[2].metric("Contrats", df['# de Police'].notna().sum())
    cols[3].metric("Doublons", df.duplicated().sum())

    # Verification section - only if Reçu and PA columns exist (PA needed for formula)
    has_verification_cols = 'Reçu' in df.columns and 'PA' in df.columns

    if has_verification_cols:
        st.markdown("### 🔍 Vérification Reçu vs Commission")
        st.caption("Formule: `Com Calculée = ROUND((PA × 0.4) × 0.5, 2)`")

        # Tolerance slider
        col1, col2 = st.columns([2, 3])
        with col1:
            # Use on_change to ensure state is updated before verification
            if 'verification_tolerance' not in st.session_state:
                st.session_state.verification_tolerance = 10.0

            tolerance = st.slider(
                "Tolérance (%)",
                min_value=1.0,
                max_value=50.0,
                value=st.session_state.verification_tolerance,
                step=1.0,
                help="Écart acceptable entre Reçu et Com. Le pourcentage affiché est l'écart réel entre Reçu et Com Calculée.",
                key="tolerance_slider",
                on_change=lambda: setattr(st.session_state, 'verification_tolerance', st.session_state.tolerance_slider)
            )
            # Sync state
            st.session_state.verification_tolerance = tolerance

        # Apply verification
        df_verified = verify_recu_vs_com(df, tolerance_pct=tolerance)
        stats = get_verification_stats(df_verified)

        with col2:
            # Display verification stats
            stat_cols = st.columns(4)
            stat_cols[0].metric("✓ OK", stats['ok'], help="Reçu dans la tolérance de Com Calculée")
            stat_cols[1].metric("✅ Bonus", stats['bonus'], help=f"Reçu > Com Calculée + {tolerance}%")
            stat_cols[2].metric("⚠️ Écart", stats['ecart'], help=f"Reçu < Com Calculée - {tolerance}%")
            stat_cols[3].metric("- N/A", stats['na'], help="PA ou Reçu manquant")

        # Show warnings if there are issues
        if stats['ecart'] > 0:
            st.warning(f"⚠️ **{stats['ecart']} ligne(s)** ont un écart négatif (Reçu inférieur à la commission attendue)")

        if stats['bonus'] > 0:
            st.success(f"✅ **{stats['bonus']} ligne(s)** ont un bonus (Reçu supérieur à la commission attendue)")

        # Explanation of verification column
        st.caption(f"📌 **Colonne Vérification (±{tolerance:.0f}%)**: Le pourcentage affiché est l'écart réel entre Reçu et Com Calculée. "
                   f"Les valeurs hors tolérance (±{tolerance:.0f}%) sont marquées ✅ (bonus) ou ⚠️ (écart).")

        # Data preview with verification column
        st.dataframe(df_verified, use_container_width=True, height=350)

        # Store verified data for potential use
        df_display = df_verified
    else:
        # Data preview without verification
        st.dataframe(df, use_container_width=True, height=350)
        df_display = df

    # Actions row
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Télécharger CSV",
            data=csv,
            file_name=f"commissions_{config.source.value}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        # Column info in expander
        if st.button("ℹ️ Colonnes", use_container_width=True):
            st.session_state.show_columns = not st.session_state.get('show_columns', False)

    with col3:
        if st.button("⬅️ Retour", use_container_width=True):
            reset_pipeline()
            st.rerun()

    with col4:
        if st.button("➡️ Uploader", type="primary", use_container_width=True):
            st.session_state.stage = 3
            st.rerun()

    # Show column info if toggled
    if st.session_state.get('show_columns', False):
        st.markdown("#### Informations colonnes")
        col_info = pd.DataFrame({
            'Colonne': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.notna().sum().values,
            'Null': df.isna().sum().values
        })
        st.dataframe(col_info, use_container_width=True, height=200)

    # Excel upload option
    with st.expander("📤 Remplacer par un fichier modifié", expanded=False):
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

                st.success(f"✅ {excel_file.name} chargé ({len(uploaded_df)} lignes)")

                if st.button("✅ Utiliser ce fichier", type="primary"):
                    st.session_state.extracted_data = uploaded_df
                    st.session_state.pipeline.final_data = uploaded_df
                    st.session_state.data_modified = True
                    st.rerun()

            except Exception as e:
                st.error(f"Erreur: {e}")

    # Groups display for Monday Legacy
    if config.source == InsuranceSource.MONDAY_LEGACY:
        with st.expander("📁 Groupes du board source", expanded=False):
            try:
                from monday_automation import MondayClient
                client = MondayClient(api_key=config.monday_api_key)
                groups = client.list_groups(board_id=config.source_board_id)
                groups = [g for g in groups if g['title'] != 'Group Title']

                if groups:
                    st.success(f"{len(groups)} groupes seront recréés")
                    for g in groups:
                        st.caption(f"• {g['title']}")
                else:
                    st.info("Aucun groupe personnalisé trouvé")
            except Exception as e:
                st.error(f"Erreur: {e}")


# =============================================================================
# STAGE 3: UPLOAD - STREAMLINED
# =============================================================================

def render_stage_3():
    """Render upload stage."""

    st.markdown("## 📊 Pipeline de Commissions")
    render_stepper()
    st.write("")

    df = st.session_state.extracted_data
    config = st.session_state.config

    if st.session_state.data_modified:
        st.warning("⚠️ Upload de données modifiées")

    # Summary
    st.markdown("### 📋 Résumé de l'upload")

    cols = st.columns(3)
    cols[0].metric("Lignes", len(df))
    cols[1].metric("Board", config.board_name[:25] + "..." if len(config.board_name) > 25 else config.board_name)
    cols[2].metric("Groupe", config.month_group or "Défaut")

    st.divider()

    # Upload process
    if st.session_state.upload_results is None:
        st.info("Les données vont être uploadées vers Monday.com.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.stage = 2
                st.rerun()

        with col2:
            if st.button("🚀 Confirmer l'upload", type="primary", use_container_width=True):
                execute_upload()
    else:
        render_upload_results()


def execute_upload():
    """Execute the upload to Monday.com."""
    config = st.session_state.config
    df = st.session_state.extracted_data

    progress = st.progress(0)
    status = st.empty()

    try:
        pipeline = st.session_state.pipeline

        # Step 3: Setup board
        status.text("Configuration du board...")
        progress.progress(25)

        if not pipeline._step3_setup_monday_board():
            st.error("❌ Échec de la configuration du board")
            return

        progress.progress(50)

        # Step 4: Upload
        status.text("Upload des données...")

        is_monday_legacy = config.source == InsuranceSource.MONDAY_LEGACY
        has_groups = hasattr(pipeline, 'groups_to_create') and pipeline.groups_to_create

        if is_monday_legacy and has_groups:
            success = pipeline._step4_upload_to_monday()
            results = pipeline.upload_results if hasattr(pipeline, 'upload_results') else []
        else:
            items = pipeline._prepare_monday_items(df)
            results = []
            batch_size = 10

            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                status.text(f"Upload... ({min(i + batch_size, len(items))}/{len(items)})")

                batch_results = pipeline.monday_client.create_items_batch(
                    board_id=pipeline.board_id,
                    items=batch,
                    group_id=pipeline.group_id
                )
                results.extend(batch_results)

                pct = 50 + int(50 * (i + len(batch)) / len(items))
                progress.progress(min(pct, 100))

        progress.progress(100)
        status.empty()

        # Analyze results
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        if successful > 0:
            st.session_state.upload_results = {
                'success': True,
                'board_id': pipeline.board_id,
                'group_id': pipeline.group_id,
                'items_uploaded': successful,
                'items_failed': failed
            }
            st.rerun()
        else:
            st.error("❌ Aucun item uploadé")

    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        with st.expander("Détails"):
            st.exception(e)


def render_upload_results():
    """Render upload results."""
    results = st.session_state.upload_results
    config = st.session_state.config

    if results['success']:
        st.balloons()
        st.success("✅ Upload terminé avec succès!")

        cols = st.columns(4)
        cols[0].metric("Items créés", results['items_uploaded'])
        cols[1].metric("Échecs", results['items_failed'])
        cols[2].metric("Board ID", results['board_id'])
        cols[3].metric("Group ID", results['group_id'] or "Défaut")

        st.divider()

        if results['items_failed'] == 0:
            st.info(f"""
            🎉 **Upload réussi!**

            **{results['items_uploaded']}** items créés dans le board **{config.board_name}**
            """)
        else:
            st.warning(f"""
            ⚠️ **Upload partiel**

            {results['items_uploaded']} items créés, {results['items_failed']} échecs
            """)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Nouveau pipeline", type="primary", use_container_width=True):
                reset_pipeline()
                st.rerun()

        with col2:
            if results['board_id']:
                url = f"https://monday.com/boards/{results['board_id']}"
                st.link_button("🔗 Ouvrir Monday.com", url, use_container_width=True)
    else:
        st.error("❌ L'upload a échoué")
        if st.button("🔄 Recommencer"):
            reset_pipeline()
            st.rerun()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()

    if st.session_state.stage == 1:
        render_stage_1()
    elif st.session_state.stage == 2:
        render_stage_2()
    elif st.session_state.stage == 3:
        render_stage_3()


if __name__ == "__main__":
    main()
