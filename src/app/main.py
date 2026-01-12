"""
PDF Extractor - Streamlit Application

Multi-stage wizard application for extracting commission data from PDFs
and uploading to Monday.com.

Features:
- Multi-stage wizard (Configuration → Preview → Upload)
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
from datetime import datetime
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

    /* Form styling */
    [data-testid="stForm"] {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        background: #fafafa;
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
        'batch_result', 'combined_data', 'is_processing', 'processing_progress',
        'current_file', 'selected_board_id', 'selected_group_id', 'monday_groups',
        'upload_result', 'is_uploading', 'selected_source', 'data_modified',
        'show_columns', '_current_board_name'
    ]
    for key in keys_to_reset:
        if key == 'stage':
            st.session_state[key] = 1
        elif key in ['uploaded_files', 'temp_pdf_paths']:
            st.session_state[key] = []
        elif key == 'extraction_results':
            st.session_state[key] = {}
        elif key in ['is_processing', 'is_uploading', 'data_modified', 'show_columns']:
            st.session_state[key] = False
        elif key == 'processing_progress':
            st.session_state[key] = 0.0
        else:
            st.session_state[key] = None


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


def reorder_columns_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Réordonne les colonnes pour l'affichage:
    1. Colonnes normales (sans underscore)
    2. Colonnes de calcul/vérification
    3. Colonnes avec underscore (_source_file, _target_group, etc.)
    """
    cols = df.columns.tolist()

    # Séparer les colonnes
    underscore_cols = [c for c in cols if c.startswith('_')]
    calc_verify_cols = [c for c in cols if 'Vérification' in c or c == 'Com Calculée']
    normal_cols = [c for c in cols if c not in underscore_cols and c not in calc_verify_cols]

    # Nouvel ordre
    new_order = normal_cols + calc_verify_cols + underscore_cols

    return df[new_order]


# =============================================================================
# PHASE 3: VERIFICATION FUNCTIONS
# =============================================================================

def verify_recu_vs_com(df: pd.DataFrame, tolerance_pct: float = 10.0) -> pd.DataFrame:
    """
    Verify that Reçu is within tolerance range of calculated Com for each row.

    The comparison uses a CALCULATED commission value based on the formula:
        Com_calculée = ROUND((PA * 0.4) * 0.5, 2)

    Args:
        df: DataFrame with 'Reçu' and 'PA' columns
        tolerance_pct: Tolerance percentage (default 10%)

    Returns:
        DataFrame with added columns:
        - 'Com Calculée': The calculated commission for comparison
        - 'Vérification (±X%)': Status flag
    """
    result_df = df.copy()

    # Check if required columns exist
    if 'Reçu' not in result_df.columns or 'PA' not in result_df.columns:
        return result_df

    # Convert to numeric
    recu = pd.to_numeric(result_df['Reçu'], errors='coerce')
    pa = pd.to_numeric(result_df['PA'], errors='coerce')

    # Calculate expected commission: ROUND((PA * 0.4) * 0.5, 2)
    com_calculee = (pa * 0.4 * 0.5).round(2)

    # Add calculated commission column
    result_df['Com Calculée'] = com_calculee

    # Calculate tolerance bounds
    tolerance = tolerance_pct / 100.0
    lower_bound = com_calculee * (1 - tolerance)
    upper_bound = com_calculee * (1 + tolerance)

    # Calculate percentage difference
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
            verification.append(f'✅ +{diff}%')  # Bonus
        elif r < lower_bound.iloc[i]:
            verification.append(f'⚠️ {diff}%')  # Problem
        else:
            verification.append('✓ OK')

    result_df[f'Vérification (±{tolerance_pct:.0f}%)'] = verification

    return result_df


def get_verification_stats(df: pd.DataFrame) -> dict:
    """Get statistics about verification results."""
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
                boards = asyncio.run(pipeline.monday.list_boards())
                st.session_state.monday_boards = boards
            st.session_state.boards_loading = False
            if force_rerun:
                st.rerun()
        except Exception as e:
            st.session_state.boards_loading = False
            st.session_state.boards_error = str(e)


# =============================================================================
# PHASE 1: STEPPER COMPONENT
# =============================================================================

def render_stepper() -> None:
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
# PHASE 7: SIDEBAR
# =============================================================================

def render_sidebar() -> None:
    """Render simplified sidebar."""
    with st.sidebar:
        st.markdown("## 🔑 Configuration")

        api_from_secrets = get_secret('MONDAY_API_KEY') is not None

        if st.session_state.monday_api_key:
            col1, col2 = st.columns([3, 1])
            with col1:
                if api_from_secrets:
                    st.success("API (secrets)", icon="🔐")
                else:
                    st.success("API connectée", icon="✅")
            with col2:
                if not api_from_secrets:
                    if st.button("✏️", help="Modifier la clé API"):
                        st.session_state.monday_api_key = None
                        st.session_state.monday_boards = None
                        st.rerun()

            if st.session_state.monday_boards:
                st.caption(f"📋 {len(st.session_state.monday_boards)} boards disponibles")
        else:
            api_key = st.text_input(
                "Clé API Monday.com",
                type="password",
                placeholder="Entrez votre clé API...",
                key="sidebar_api_key",
                help="Ou configurez MONDAY_API_KEY dans .streamlit/secrets.toml"
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

        # Board loading status
        if st.session_state.boards_loading:
            st.info("⏳ Chargement des boards...")
        elif st.session_state.get('boards_error'):
            st.error(f"❌ {st.session_state.boards_error}")
            if st.button("🔄 Réessayer", use_container_width=True, type="primary"):
                st.session_state.boards_error = None
                st.session_state.monday_boards = None
                load_boards_async(force_rerun=True)
        elif st.session_state.monday_boards:
            st.success(f"✅ {len(st.session_state.monday_boards)} boards chargés")
            if st.button("🔄 Rafraîchir boards", use_container_width=True):
                st.session_state.monday_boards = None
                load_boards_async(force_rerun=True)
        elif st.session_state.monday_api_key:
            if st.button("📥 Charger les boards", use_container_width=True, type="primary"):
                load_boards_async(force_rerun=True)

        st.divider()

        # Help section
        with st.expander("ℹ️ Aide", expanded=False):
            st.markdown("""
            **Sources supportées:**
            - UV Assurance
            - IDC / IDC Statement
            - Assomption Vie

            **Stages:**
            1. Configuration & Upload PDF
            2. Prévisualisation & Édition
            3. Export vers Monday.com

            **Besoin d'aide?**
            Contactez le support technique.
            """)


# =============================================================================
# PHASE 2: ADVISOR MANAGEMENT TAB
# =============================================================================

def render_advisor_management_tab() -> None:
    """Render advisor management interface."""
    st.markdown("### 👥 Gestion des Conseillers")

    st.info("""
    **Gestion des noms de conseillers**

    Cette section permet de gérer les conseillers et leurs variations de noms.
    Le système utilise ces données pour normaliser automatiquement les noms
    lors de l'extraction des données PDF.

    **Format de sortie:** Prénom, Initiale (ex: "Thomas, L")
    """)

    try:
        from src.utils.advisor_matcher import get_advisor_matcher, Advisor
    except ImportError:
        st.error("Module advisor_matcher non disponible")
        return

    # Initialize matcher
    if st.session_state.advisor_matcher is None:
        st.session_state.advisor_matcher = get_advisor_matcher()

    matcher = st.session_state.advisor_matcher

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
        uploaded_files = st.file_uploader(
            "Déposez vos fichiers PDF ici",
            type=['pdf'],
            accept_multiple_files=True,
            help="Sélectionnez un ou plusieurs fichiers PDF du même type",
            key="pdf_upload_main"
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
        if st.button(button_text, type="primary", use_container_width=True, disabled=not can_proceed):
            st.session_state.stage = 2
            st.rerun()

    if not can_proceed:
        st.caption("⚠️ Sélectionnez un board de destination pour continuer")


def render_stage_1() -> None:
    """Render configuration stage with tabs."""
    st.markdown("## 📊 Pipeline de Commissions")
    render_stepper()
    st.write("")

    # Check API key
    if not st.session_state.monday_api_key:
        st.warning("👈 Veuillez d'abord configurer votre clé API Monday.com dans la barre latérale.")
        return

    # Tabs for different workflows
    tab1, tab2 = st.tabs(["📄 Extraction PDF", "👥 Gestion Conseillers"])

    with tab1:
        render_pdf_extraction_tab()

    with tab2:
        render_advisor_management_tab()


# =============================================================================
# STAGE 2: PREVIEW
# =============================================================================

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

        # Get combined data and detect groups
        combined_df = batch_result.get_combined_dataframe()
        if combined_df is not None and not combined_df.empty:
            combined_df = detect_groups_from_data(combined_df, st.session_state.selected_source)
        st.session_state.combined_data = combined_df

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


def render_stage_2() -> None:
    """Render data preview stage."""
    st.markdown("## 📊 Pipeline de Commissions")
    render_stepper()
    st.write("")

    # Extract data if not done
    if st.session_state.combined_data is None:
        if not st.session_state.uploaded_files:
            st.error("❌ Aucun fichier à traiter")
            if st.button("🔄 Recommencer"):
                reset_pipeline()
                st.rerun()
            return

        with st.spinner("🔄 Extraction en cours..."):
            run_extraction()
            st.rerun()
        return

    df = st.session_state.combined_data

    if df is None or df.empty:
        st.error("❌ Aucune donnée extraite")
        if st.button("🔄 Recommencer"):
            reset_pipeline()
            st.rerun()
        return

    # Config summary
    with st.expander("📋 Configuration", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Source", st.session_state.selected_source)
        cols[1].metric("Fichiers", len(st.session_state.uploaded_files))
        cols[2].metric("Board", st.session_state._current_board_name[:20] + "..." if len(st.session_state._current_board_name) > 20 else st.session_state._current_board_name)
        cols[3].metric("Type", "Ventes" if st.session_state.selected_board_type == BoardType.SALES_PRODUCTION else "Paiements")

    # Multi-month warning (Phase 5)
    if '_target_group' in df.columns:
        groups_info = analyze_groups_in_data(df)
        if groups_info['spans_multiple_months']:
            st.warning(f"⚠️ Les données couvrent **{len(groups_info['unique_groups'])} mois différents**. "
                       f"Les lignes seront automatiquement assignées à leur groupe respectif.")
            with st.expander("📅 Détail par groupe", expanded=False):
                for group, count in groups_info['group_counts'].items():
                    st.markdown(f"**{group}**: {count} lignes")

    # Manual group override (Phase 5)
    if '_target_group' in df.columns:
        with st.expander("📅 Modifier le groupe de destination", expanded=False):
            st.caption("Si la détection automatique de date n'est pas correcte, vous pouvez assigner manuellement un groupe.")

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

            col1, col2 = st.columns([3, 1])
            with col1:
                manual_group = st.selectbox("Groupe manuel", group_options, key="manual_group_override")
            with col2:
                if manual_group != "(Garder auto-détection)":
                    if st.button("✅ Appliquer", use_container_width=True):
                        df['_target_group'] = manual_group
                        st.session_state.combined_data = df
                        st.success(f"Groupe modifié: {manual_group}")
                        st.rerun()

    # Statistics
    st.markdown("### 📊 Aperçu")

    cols = st.columns(4)
    cols[0].metric("Lignes", len(df))
    cols[1].metric("Colonnes", len(df.columns))
    if '# de Police' in df.columns:
        cols[2].metric("Contrats", df['# de Police'].notna().sum())
    elif 'Contrat' in df.columns:
        cols[2].metric("Contrats", df['Contrat'].notna().sum())
    else:
        cols[2].metric("", "")
    cols[3].metric("Doublons", df.duplicated().sum())

    # Phase 3: Verification section
    has_verification_cols = 'Reçu' in df.columns and 'PA' in df.columns

    if has_verification_cols:
        st.markdown("### 🔍 Vérification Reçu vs Commission")
        st.caption("Formule: `Com Calculée = ROUND((PA × 0.4) × 0.5, 2)`")

        col1, col2 = st.columns([2, 3])
        with col1:
            tolerance = st.slider(
                "Tolérance (%)",
                min_value=1.0,
                max_value=50.0,
                value=10.0,
                step=1.0,
                key="verification_tolerance_slider"
            )

        df_verified = verify_recu_vs_com(df, tolerance_pct=tolerance)
        stats = get_verification_stats(df_verified)

        with col2:
            stat_cols = st.columns(4)
            stat_cols[0].metric("✓ OK", stats['ok'])
            stat_cols[1].metric("✅ Bonus", stats['bonus'])
            stat_cols[2].metric("⚠️ Écart", stats['ecart'])
            stat_cols[3].metric("- N/A", stats['na'])

        if stats['ecart'] > 0:
            st.warning(f"⚠️ **{stats['ecart']} ligne(s)** ont un écart négatif")

        if stats['bonus'] > 0:
            st.success(f"✅ **{stats['bonus']} ligne(s)** ont un bonus")

        st.dataframe(reorder_columns_for_display(df_verified), use_container_width=True, height=350)
        df_display = df_verified
    else:
        st.dataframe(reorder_columns_for_display(df), use_container_width=True, height=350)
        df_display = df

    # Extraction details
    results = st.session_state.extraction_results
    if results:
        with st.expander("📊 Détails de l'extraction", expanded=False):
            for filename, result in results.items():
                if result.success:
                    st.markdown(f"✅ **{filename}**: {result.row_count} lignes ({result.extraction_time_ms}ms)")
                else:
                    st.markdown(f"❌ **{filename}**: {result.error}")

    # Actions row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Télécharger CSV",
            data=csv,
            file_name=f"commissions_{st.session_state.selected_source}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        if st.button("ℹ️ Colonnes", use_container_width=True):
            st.session_state.show_columns = not st.session_state.get('show_columns', False)
            st.rerun()

    with col3:
        if st.button("⬅️ Retour", use_container_width=True):
            reset_pipeline()
            st.rerun()

    with col4:
        if st.button("➡️ Uploader", type="primary", use_container_width=True):
            st.session_state.stage = 3
            st.rerun()

    # Column info
    if st.session_state.get('show_columns', False):
        st.markdown("#### Informations colonnes")
        col_info = pd.DataFrame({
            'Colonne': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.notna().sum().values,
            'Null': df.isna().sum().values
        })
        st.dataframe(col_info, use_container_width=True, height=200)

    # Phase 6: Excel upload replacement
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
                    st.session_state.combined_data = uploaded_df
                    st.session_state.data_modified = True
                    st.rerun()

            except Exception as e:
                st.error(f"Erreur: {e}")


# =============================================================================
# STAGE 3: UPLOAD
# =============================================================================

def render_stage_3() -> None:
    """Render upload stage."""
    st.markdown("## 📊 Pipeline de Commissions")
    render_stepper()
    st.write("")

    df = st.session_state.combined_data

    if df is None or df.empty:
        st.error("❌ Aucune donnée à uploader")
        if st.button("🔄 Recommencer"):
            reset_pipeline()
            st.rerun()
        return

    if st.session_state.data_modified:
        st.warning("⚠️ Upload de données modifiées")

    # Summary
    st.markdown("### 📋 Résumé de l'upload")

    unique_groups = df['_target_group'].unique() if '_target_group' in df.columns else []

    cols = st.columns(4)
    cols[0].metric("Items total", len(df))
    cols[1].metric("Board", st.session_state._current_board_name[:20] + "..." if len(st.session_state._current_board_name) > 20 else st.session_state._current_board_name)
    cols[2].metric("Groupes", len(unique_groups) if len(unique_groups) > 0 else 1)
    cols[3].metric("Fichiers", len(st.session_state.extraction_results))

    # Groups breakdown
    if '_target_group' in df.columns and len(unique_groups) > 1:
        with st.expander("📁 Détail par groupe", expanded=False):
            for group in unique_groups:
                group_count = len(df[df['_target_group'] == group])
                st.markdown(f"**{group}**: {group_count} items")

    st.divider()

    # Upload process
    if st.session_state.upload_result is None:
        st.info(f"Les données vont être uploadées vers Monday.com.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.stage = 2
                st.rerun()

        with col2:
            if st.button("🚀 Confirmer l'upload", type="primary", use_container_width=True):
                execute_upload(df)
    else:
        render_upload_result(st.session_state.upload_result)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Nouveau pipeline", use_container_width=True):
                reset_pipeline()
                st.rerun()
        with col2:
            if st.button("📋 Voir le board", use_container_width=True):
                board_id = st.session_state.selected_board_id
                st.markdown(f"[Ouvrir Monday.com](https://monday.com/boards/{board_id})")


def execute_upload(df: pd.DataFrame) -> None:
    """Execute the upload to Monday.com."""
    st.session_state.is_uploading = True
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
                group_result = asyncio.run(
                    pipeline.monday.get_or_create_group(board_id, str(group_name))
                )
                group_id = group_result.id if group_result.success else None

                if not group_id:
                    raise Exception(f"Impossible de créer le groupe: {group_result.error}")

                # Progress callback
                def on_progress(current: int, total: int) -> None:
                    nonlocal items_uploaded
                    overall_progress = (items_uploaded + current) / total_items
                    progress_bar.progress(min(overall_progress, 0.99), text=f"Upload: {group_name} ({current}/{total})")

                # Upload
                result = asyncio.run(
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

            except Exception as e:
                items_failed += len(export_df)
                all_errors.append(f"Groupe {group_name}: {str(e)}")

        progress_bar.progress(1.0)
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
            st.success(f"✅ {items_uploaded} éléments uploadés dans {len(unique_groups)} groupe(s)!")
        else:
            st.warning(f"⚠️ {items_uploaded}/{total_items} uploadés. {items_failed} en erreur.")

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
    """Render upload result summary."""
    total = result.get("total", 0)
    success = result.get("success", 0)
    failed = result.get("failed", 0)
    groups = result.get("groups", 1)

    if failed == 0:
        st.success(f"✅ {success}/{total} éléments uploadés avec succès dans {groups} groupe(s)!")
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
