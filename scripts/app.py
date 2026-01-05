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
from advisor_matcher import AdvisorMatcher, Advisor

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

def get_secret(key: str, default: str = None) -> str:
    """
    Get a secret value from multiple sources (priority order):
    1. Streamlit secrets
    2. Environment variables
    3. Default value
    """
    # Try Streamlit secrets first
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    # Fallback to environment variable
    value = os.environ.get(key)
    if value:
        return value

    return default


def init_session_state():
    """Initialize session state variables."""
    # Try to load Monday API key from secrets
    monday_api_key = get_secret('MONDAY_API_KEY')

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
        'monday_api_key': monday_api_key,  # Auto-load from secrets if available
        'boards_loading': False,
        'boards_error': None,
        # Batch processing state
        'batch_mode': False,
        'uploaded_files': [],
        'extraction_results': {},  # {filename: {'success': bool, 'data': df, 'error': str, 'group': str}}
        'combined_data': None,
        'processing_progress': 0,
        'current_processing_file': '',
        'batch_configs': [],  # List of configs for each PDF
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

def load_boards_async(force_rerun: bool = False):
    """Load Monday.com boards automatically when API key is set."""
    if (st.session_state.monday_api_key and
        st.session_state.monday_boards is None and
        not st.session_state.boards_loading):
        try:
            st.session_state.boards_loading = True
            st.session_state.boards_error = None
            from monday_automation import MondayClient
            client = MondayClient(api_key=st.session_state.monday_api_key)
            st.session_state.monday_boards = client.list_boards()
            st.session_state.boards_loading = False
            if force_rerun:
                st.rerun()
        except Exception as e:
            st.session_state.boards_loading = False
            st.session_state.boards_error = str(e)


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


def get_months_fr():
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
        fallback_group: Groupe à utiliser si la date n'est pas parsable (optionnel)

    Returns:
        str: Nom du groupe (ex: "Octobre 2025")
    """
    import re
    from datetime import datetime

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
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{months_fr[month]} {year}"

    # Essayer de parser avec pandas
    try:
        parsed = pd.to_datetime(date_str)
        if pd.notna(parsed):
            return f"{months_fr[parsed.month]} {parsed.year}"
    except:
        pass

    # Fallback
    if fallback_group:
        return fallback_group
    now = datetime.now()
    return f"{months_fr[now.month]} {now.year}"


def detect_date_from_filename(filename: str) -> str:
    """
    Détecte la date/mois à partir du nom de fichier PDF.

    Patterns supportés:
    - rappportremun_21622_2025-10-20.pdf -> "Octobre 2025"
    - Rapport des propositions soumises.20251017_1517.pdf -> "Octobre 2025"
    - 20251017_report.pdf -> "Octobre 2025"

    Args:
        filename: Nom du fichier PDF

    Returns:
        str: Nom du groupe (ex: "Octobre 2025") ou None si non détecté
    """
    import re
    from datetime import datetime

    months_fr = get_months_fr()

    # Patterns de date dans le nom de fichier
    patterns = [
        (r'(\d{4})-(\d{2})-(\d{2})', 1, 2),      # 2025-10-20 (year, month at pos 1, 2)
        (r'\.(\d{4})(\d{2})(\d{2})_', 1, 2),     # .20251017_ (year, month at pos 1, 2)
        (r'_(\d{4})(\d{2})(\d{2})', 1, 2),       # _20251017 (year, month at pos 1, 2)
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

    # Retourner None si pas de date détectée dans le nom de fichier
    # La vraie détection se fera après extraction des données
    return None


def detect_groups_from_data(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Analyse le DataFrame extrait et assigne un groupe à chaque ligne basé sur la date.

    Stratégie par source:
    - UV: Utilise la colonne 'Date' (date unique du rapport journalier)
    - IDC/IDC_STATEMENT: Utilise la colonne 'Date' de chaque ligne
    - ASSOMPTION: Utilise la colonne 'Date' (date d'émission) de chaque ligne

    Args:
        df: DataFrame avec les données extraites (après standardisation)
        source: Type de source (UV, IDC, IDC_STATEMENT, ASSOMPTION)

    Returns:
        DataFrame avec colonne '_target_group' ajoutée par ligne
    """
    df = df.copy()

    # Trouver la colonne de date
    date_column = None
    for col in ['Date', 'date', 'Émission', 'effective_date', 'report_date']:
        if col in df.columns:
            # Vérifier que la colonne contient des valeurs non-null
            non_null_count = df[col].notna().sum()
            if non_null_count > 0:
                date_column = col
                print(f"   📅 Colonne date trouvée: '{col}' ({non_null_count}/{len(df)} valeurs non-null)")
                # Afficher un exemple de valeur
                first_valid = df[col].dropna().iloc[0] if non_null_count > 0 else None
                print(f"   📅 Exemple de valeur: {first_valid} (type: {type(first_valid).__name__})")
                break
            else:
                print(f"   ⚠️ Colonne '{col}' trouvée mais toutes les valeurs sont null")

    if date_column is None:
        # Pas de colonne de date trouvée - utiliser date du jour pour tout
        print(f"   ⚠️ Aucune colonne de date valide trouvée. Colonnes disponibles: {list(df.columns)}")
        from datetime import datetime
        months_fr = get_months_fr()
        now = datetime.now()
        default_group = f"{months_fr[now.month]} {now.year}"
        df['_target_group'] = default_group
        return df

    # Assigner un groupe à chaque ligne basé sur sa date
    df['_target_group'] = df[date_column].apply(date_to_group)

    return df


def reorder_columns_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Réordonne les colonnes pour l'affichage:
    1. Colonnes normales (sans underscore)
    2. Com Calculées et Vérifications
    3. Colonnes avec underscore (_source_file, _target_group, etc.)
    """
    cols = df.columns.tolist()

    # Séparer les colonnes
    underscore_cols = [c for c in cols if c.startswith('_')]
    calc_verify_cols = [c for c in cols if c in ['Com Calculées', 'Vérifications', 'Com Calculée']]
    normal_cols = [c for c in cols if c not in underscore_cols and c not in calc_verify_cols]

    # Nouvel ordre: normales + calc/verify + underscore
    new_order = normal_cols + calc_verify_cols + underscore_cols

    return df[new_order]


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


def extract_with_details(pdf_path: str, source: str, aggregate: bool, target_board_type) -> dict:
    """
    Extrait les données d'un PDF avec des statistiques détaillées sur chaque étape.

    Permet de fournir des messages d'erreur précis sur ce qui a échoué.

    Args:
        pdf_path: Chemin vers le fichier PDF
        source: Type de source (UV, IDC, etc.)
        aggregate: Agréger par contrat
        target_board_type: Type de board cible

    Returns:
        {
            'success': bool,
            'final_data': DataFrame ou None,
            'error_type': str (None si succès),
            'error_message': str (message détaillé),
            'stats': {
                'rows_extracted': int,
                'rows_after_standardization': int,
                'rows_after_filter': int,
                'rows_final': int,
                'sharing_rates_found': list (taux uniques trouvés)
            }
        }
    """
    from unify_notation import CommissionDataUnifier

    result = {
        'success': False,
        'final_data': None,
        'error_type': None,
        'error_message': '',
        'stats': {
            'rows_extracted': 0,
            'rows_after_standardization': 0,
            'rows_after_filter': 0,
            'rows_final': 0,
            'sharing_rates_found': []
        }
    }

    try:
        # Étape 1: Extraction brute
        raw_df = None
        metadata = None

        if source == 'UV':
            from uv_extractor import RemunerationReportExtractor
            extractor = RemunerationReportExtractor(pdf_path)
            data = extractor.extract_all()
            if data['activites'] is None or data['activites'].empty:
                result['error_type'] = 'extraction_failed'
                result['error_message'] = "Aucune donnée trouvée dans le PDF"
                return result
            raw_df = data['activites']
            metadata = {
                'date': data.get('date'),
                'nom_conseiller': data.get('nom_conseiller'),
                'numero_conseiller': data.get('numero_conseiller')
            }

        elif source == 'IDC':
            from idc_extractor import PDFPropositionParser
            parser = PDFPropositionParser(pdf_path)
            raw_df = parser.parse()
            if raw_df.empty:
                result['error_type'] = 'extraction_failed'
                result['error_message'] = "Aucune donnée trouvée dans le PDF"
                return result

        elif source == 'IDC_STATEMENT':
            from idc_statements_extractor import PDFStatementParser
            parser = PDFStatementParser(pdf_path)
            raw_df = parser.parse_trailing_fees()
            if raw_df.empty:
                result['error_type'] = 'extraction_failed'
                result['error_message'] = "Aucune donnée trouvée dans le PDF"
                return result

        elif source == 'ASSOMPTION':
            from assomption_extractor import extract_pdf_data
            raw_df = extract_pdf_data(pdf_path)
            if raw_df.empty:
                result['error_type'] = 'extraction_failed'
                result['error_message'] = "Aucune donnée trouvée dans le PDF"
                return result
        else:
            result['error_type'] = 'unknown_source'
            result['error_message'] = f"Source non reconnue: {source}"
            return result

        result['stats']['rows_extracted'] = len(raw_df)

        # Étape 2: Standardisation
        unifier = CommissionDataUnifier(output_dir=str(PROJECT_ROOT / "results"))

        if source == 'UV':
            standardized = unifier.convert_uv_to_standard(raw_df, metadata)
        elif source == 'IDC':
            standardized = unifier.convert_idc_to_standard(raw_df)
        elif source == 'IDC_STATEMENT':
            standardized = unifier.convert_idc_statement_to_standard(raw_df)
        elif source == 'ASSOMPTION':
            standardized = unifier.convert_assomption_to_standard(raw_df)

        result['stats']['rows_after_standardization'] = len(standardized)

        # Collecter les taux de partage uniques
        if 'sharing_rate' in standardized.columns:
            unique_rates = standardized['sharing_rate'].dropna().unique().tolist()
            result['stats']['sharing_rates_found'] = [float(r) for r in unique_rates]

        # Étape 3: Filtrage par sharing_rate (sauf IDC_STATEMENT)
        if source == 'IDC_STATEMENT':
            filtered = standardized
        else:
            filtered = unifier.filter_by_sharing_rate(standardized, target_rate=0.4)

        result['stats']['rows_after_filter'] = len(filtered)

        # Vérifier si le filtre a tout éliminé
        if len(filtered) == 0 and len(standardized) > 0:
            rates_str = ", ".join([f"{r*100:.0f}%" for r in result['stats']['sharing_rates_found']])
            result['error_type'] = 'filtered_out'
            result['error_message'] = (
                f"Extraction réussie ({len(standardized)} lignes), mais aucune ligne "
                f"avec taux de partage = 40%. Taux trouvés: {rates_str}"
            )
            return result

        # Étape 4: Agrégation (optionnelle)
        if aggregate and source not in ['IDC_STATEMENT']:
            final = unifier.aggregate_by_contract_number(filtered)
        else:
            final = filtered

        # Étape 5: Filtrage des colonnes finales
        board_type_enum = target_board_type if hasattr(target_board_type, 'value') else BoardType.HISTORICAL_PAYMENTS
        final = unifier.filter_final_columns(final, board_type=board_type_enum)

        result['stats']['rows_final'] = len(final)

        if len(final) == 0:
            result['error_type'] = 'no_data_after_processing'
            result['error_message'] = "Aucune donnée après le traitement complet"
            return result

        result['success'] = True
        result['final_data'] = final
        return result

    except ImportError as e:
        result['error_type'] = 'import_error'
        result['error_message'] = f"Module d'extraction non disponible: {e}"
        return result
    except Exception as e:
        result['error_type'] = 'exception'
        result['error_message'] = str(e)
        return result


def process_batch_pdfs(uploaded_files, source: str, board_name: str, api_key: str,
                       target_board_type, aggregate: bool, reuse_board: bool,
                       reuse_group: bool, progress_callback=None) -> dict:
    """
    Traite plusieurs PDFs séquentiellement avec gestion d'erreurs résiliente.

    La détection de groupe est basée sur les dates DANS les données extraites:
    - Chaque ligne est assignée à un groupe basé sur sa date
    - Un fichier peut contenir des lignes pour plusieurs mois différents

    Args:
        uploaded_files: Liste des fichiers uploadés
        source: Type de source (UV, IDC, etc.)
        board_name: Nom du board de destination
        api_key: Clé API Monday.com
        target_board_type: Type de board (HISTORICAL_PAYMENTS ou SALES_PRODUCTION)
        aggregate: Agréger par contrat
        reuse_board: Réutiliser le board si existe
        reuse_group: Réutiliser le groupe si existe
        progress_callback: Fonction de callback pour la progression

    Returns:
        {
            'successful': [(filename, dataframe, groups_info), ...],
            'failed': [(filename, error_message), ...],
            'combined_df': DataFrame ou None,
            'files_with_multiple_months': [filename, ...]  # Fichiers couvrant plusieurs mois
        }
    """
    results = {
        'successful': [],
        'failed': [],
        'combined_df': None,
        'files_with_multiple_months': []
    }

    for i, pdf_file in enumerate(uploaded_files):
        pdf_path = None
        try:
            # Callback de progression
            if progress_callback:
                progress_callback(i, len(uploaded_files), pdf_file.name, "Extraction...")

            # Sauvegarder temporairement
            pdf_path = save_uploaded_file(pdf_file)

            # Normaliser le nom de source
            source_normalized = source.replace(" ", "_").upper()

            # Extraire avec détails (permet des messages d'erreur précis)
            extraction_result = extract_with_details(
                pdf_path=pdf_path,
                source=source_normalized,
                aggregate=aggregate,
                target_board_type=target_board_type
            )

            if not extraction_result['success']:
                # Message d'erreur détaillé
                raise Exception(extraction_result['error_message'])

            # Ajouter colonne source
            df = extraction_result['final_data'].copy()
            df['_source_file'] = pdf_file.name

            # Détecter les groupes DEPUIS LES DONNÉES (pas le nom de fichier)
            df = detect_groups_from_data(df, source)

            # Analyser les groupes présents
            groups_info = analyze_groups_in_data(df)

            # Marquer si le fichier couvre plusieurs mois
            if groups_info['spans_multiple_months']:
                results['files_with_multiple_months'].append(pdf_file.name)

            results['successful'].append((pdf_file.name, df, groups_info))

        except Exception as e:
            results['failed'].append((pdf_file.name, str(e)))

        finally:
            if pdf_path:
                cleanup_temp_file(pdf_path)

    # Combiner les DataFrames
    if results['successful']:
        all_dfs = [item[1] for item in results['successful']]
        results['combined_df'] = pd.concat(all_dfs, ignore_index=True)

        # Ajouter un ordre d'extraction pour le tri
        results['combined_df']['_extraction_order'] = range(len(results['combined_df']))

    return results


def reset_pipeline():
    """Reset pipeline state to start over."""
    cleanup_temp_file()
    keys_to_reset = ['stage', 'pdf_file', 'pdf_path', 'extracted_data',
                     'pipeline', 'config', 'upload_results', 'data_modified',
                     'monday_boards', 'selected_board_id',
                     # Batch processing state
                     'batch_mode', 'uploaded_files', 'extraction_results',
                     'combined_data', 'processing_progress', 'current_processing_file',
                     'batch_configs', 'batch_config_params']
    for key in keys_to_reset:
        if key == 'stage':
            st.session_state[key] = 1
        elif key == 'batch_mode':
            st.session_state[key] = False
        elif key in ['uploaded_files', 'batch_configs']:
            st.session_state[key] = []
        elif key == 'extraction_results':
            st.session_state[key] = {}
        elif key == 'processing_progress':
            st.session_state[key] = 0
        elif key == 'current_processing_file':
            st.session_state[key] = ''
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

        # Check if API key comes from secrets
        api_from_secrets = get_secret('MONDAY_API_KEY') is not None

        # API Key section - compact
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

            # Show boards count
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

        # Show loading status, error, or refresh button
        if st.session_state.boards_loading:
            st.info("⏳ Chargement des boards...")
        elif st.session_state.get('boards_error'):
            st.error(f"❌ Erreur: {st.session_state.boards_error}")
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
            # API key present but boards not loaded - try loading
            if st.button("📥 Charger les boards", use_container_width=True, type="primary"):
                load_boards_async(force_rerun=True)

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

    # Show loading message if boards are still loading
    if st.session_state.boards_loading:
        st.info("⏳ Chargement des boards en cours...")
        return

    # Tabs for different workflows (Migration Monday.com removed - no longer used)
    tab1, tab2 = st.tabs(["📄 Extraction PDF", "👥 Gestion Conseillers"])

    # =========================================================================
    # TAB 1: PDF EXTRACTION
    # =========================================================================
    with tab1:
        render_pdf_extraction_tab()

    # =========================================================================
    # TAB 2: ADVISOR MANAGEMENT
    # =========================================================================
    with tab2:
        render_advisor_management_tab()


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
    """Render PDF extraction tab with batch processing support."""

    # Step 1: Upload PDF(s) - supports multiple files
    st.markdown("### 📤 Upload des fichiers PDF")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Déposez vos fichiers PDF ici",
            type=['pdf'],
            accept_multiple_files=True,  # Enable batch upload
            help="Sélectionnez un ou plusieurs fichiers PDF du même type",
            key="pdf_upload_main"
        )

        # Display upload summary
        if uploaded_files:
            is_batch = len(uploaded_files) > 1
            if is_batch:
                st.success(f"✅ {len(uploaded_files)} fichiers chargés")
            else:
                st.success(f"✅ Fichier chargé: {uploaded_files[0].name}")

    with col2:
        source = st.selectbox(
            "Source",
            options=["UV", "IDC", "IDC Statement", "ASSOMPTION"],
            help="Type de document PDF (tous les fichiers doivent être du même type)"
        )

    if not uploaded_files:
        st.info("👆 Commencez par uploader un ou plusieurs fichiers PDF pour continuer.")
        return

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
                st.info("💡 Certains groupes seront détectés après extraction (basé sur les dates dans le contenu du PDF)")
            else:
                st.info("💡 Les groupes sont détectés automatiquement à partir des noms de fichiers")

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
            st.warning("⏳ Les boards sont en cours de chargement...")
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

    # Groupe (only for single file mode - batch uses auto-detection)
    if not is_batch:
        month_group = st.text_input(
            "Groupe (optionnel)",
            placeholder="Ex: Novembre 2025",
            key="pdf_month_group",
            help="Laissez vide pour détecter automatiquement à partir du nom du fichier"
        )
    else:
        month_group = None  # Batch mode uses auto-detection
        st.info("📅 En mode batch, les groupes sont détectés automatiquement depuis les noms de fichiers")

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

    # Submit button - adapted text for batch mode
    button_text = f"🚀 Extraire {len(uploaded_files)} fichier{'s' if is_batch else ''}"

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(button_text, type="primary", use_container_width=True):
            # Validation
            errors = []
            if board_mode == "Utiliser un board existant" and not selected_board_id:
                errors.append("Veuillez sélectionner un board")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                target_board_type = (BoardType.SALES_PRODUCTION
                                    if target_type == "Ventes et Production"
                                    else BoardType.HISTORICAL_PAYMENTS)

                if is_batch:
                    # BATCH MODE: Process multiple PDFs
                    st.session_state.batch_mode = True
                    st.session_state.uploaded_files = uploaded_files

                    # Store config parameters for batch processing (don't create PipelineConfig yet)
                    # PipelineConfig will be created per-file during processing
                    st.session_state.batch_config_params = {
                        'source': source,
                        'board_name': board_name,
                        'api_key': st.session_state.monday_api_key,
                        'reuse_board': reuse_board,
                        'reuse_group': reuse_group,
                        'aggregate': aggregate,
                        'target_board_type': target_board_type
                    }

                    st.session_state.stage = 2
                    st.rerun()
                else:
                    # SINGLE FILE MODE: Original behavior
                    st.session_state.batch_mode = False
                    uploaded_file = uploaded_files[0]

                    # Use manual group if provided, otherwise auto-detect
                    if month_group:
                        final_group = month_group
                    else:
                        final_group = detect_date_from_filename(uploaded_file.name)
                        # Si pas de date dans le nom, utiliser un placeholder
                        # La vraie détection se fera après extraction via detect_groups_from_data()
                        if not final_group:
                            from datetime import datetime
                            months_fr = get_months_fr()
                            now = datetime.now()
                            final_group = f"{months_fr[now.month]} {now.year}"

                    pdf_path = save_uploaded_file(uploaded_file)

                    config = PipelineConfig(
                        source=InsuranceSource(source.replace(" ", "_").upper()),
                        pdf_path=pdf_path,
                        month_group=final_group,
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
        st.warning("⏳ Les boards sont en cours de chargement ou non disponibles. Vérifiez votre connexion API.")
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
# ADVISOR MANAGEMENT TAB
# =============================================================================

def render_advisor_management_tab():
    """Render advisor management interface."""
    st.markdown("### 👥 Gestion des Conseillers")

    st.info("""
    **Gestion des noms de conseillers**

    Cette section permet de gérer les conseillers et leurs variations de noms.
    Le système utilise ces données pour normaliser automatiquement les noms
    lors de l'extraction des données PDF.

    **Format de sortie:** Prénom + Première lettre du nom (ex: "Thomas L.")

    **Stockage cloud (optionnel):** Configurez les variables d'environnement
    `GOOGLE_SHEETS_CREDENTIALS_FILE` et `GOOGLE_SHEETS_SPREADSHEET_ID` pour
    synchroniser les données avec Google Sheets.
    """)

    # Initialize session state for advisor management
    if 'advisor_matcher' not in st.session_state:
        st.session_state.advisor_matcher = AdvisorMatcher()

    if 'editing_advisor_idx' not in st.session_state:
        st.session_state.editing_advisor_idx = None

    if 'adding_advisor' not in st.session_state:
        st.session_state.adding_advisor = False

    matcher = st.session_state.advisor_matcher

    st.divider()

    # Statistics
    stats = matcher.export_statistics()
    cols = st.columns(4)
    cols[0].metric("Conseillers", stats['total_advisors'])
    cols[1].metric("Variations totales", stats['total_variations'])

    # Storage backend indicator
    backend = stats.get('storage_backend', 'local')
    if backend == 'google_sheets':
        cols[2].metric("Stockage", "☁️ Google Sheets")
    else:
        cols[2].metric("Stockage", "💾 Local (JSON)")

    # Sync option - only "Vers cloud"
    with cols[3]:
        if backend == 'google_sheets':
            if st.button("☁️ Synchroniser", help="Envoyer les données vers Google Sheets", use_container_width=True):
                try:
                    synced, errors = matcher.sync_to_gsheets()
                    if errors == 0:
                        st.success(f"✅ {synced} conseillers synchronisés")
                        st.session_state.advisor_matcher = AdvisorMatcher()
                        st.rerun()
                    else:
                        st.error("❌ Erreur de synchronisation")
                except Exception as e:
                    st.error(f"❌ {e}")
        else:
            st.caption("Mode hors-ligne")
            st.info("Configurer GOOGLE_SHEETS_* pour le cloud", icon="ℹ️")

    st.divider()

    # Add new advisor section
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
            help="Entrez les différentes façons dont ce nom peut apparaître dans les rapports",
            key="new_advisor_variations"
        )

        submitted = st.form_submit_button("➕ Ajouter le conseiller", type="primary")

        if submitted:
            if new_first_name and new_last_name:
                # Parse variations
                variations = []
                if new_variations:
                    variations = [v.strip() for v in new_variations.split(',') if v.strip()]

                # Check if advisor already exists
                existing = matcher.find_advisor_by_name(new_first_name, new_last_name)
                if existing:
                    st.error(f"❌ Ce conseiller existe déjà: {existing[1].display_name}")
                else:
                    advisor = matcher.add_advisor(new_first_name, new_last_name, variations)
                    st.success(f"✅ Conseiller ajouté: {advisor.display_name}")
                    # Refresh matcher in session state
                    st.session_state.advisor_matcher = AdvisorMatcher()
                    st.rerun()
            else:
                st.error("❌ Veuillez entrer le prénom et le nom de famille")

    st.divider()

    # List of existing advisors
    st.markdown("#### 📋 Conseillers existants")

    if not matcher.advisors:
        st.info("Aucun conseiller enregistré. Ajoutez-en un ci-dessus.")
    else:
        for idx, advisor in enumerate(matcher.advisors):
            with st.expander(f"**{advisor.display_name}** ({advisor.full_name})", expanded=False):
                # Show current info
                st.markdown(f"**Prénom:** {advisor.first_name}")
                st.markdown(f"**Nom:** {advisor.last_name}")
                st.markdown(f"**Nom affiché:** {advisor.display_name}")

                # Variations section
                st.markdown("**Variations:**")
                if advisor.variations:
                    for var_idx, variation in enumerate(advisor.variations):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.text(f"  • {variation}")
                        with col2:
                            if st.button("🗑️", key=f"del_var_{idx}_{var_idx}",
                                       help="Supprimer cette variation"):
                                matcher.remove_variation(idx, var_idx)
                                st.session_state.advisor_matcher = AdvisorMatcher()
                                st.rerun()
                else:
                    st.caption("Aucune variation définie")

                # Add variation
                col1, col2 = st.columns([3, 1])
                with col1:
                    new_var = st.text_input(
                        "Nouvelle variation",
                        placeholder="Ex: Tommy",
                        key=f"new_var_{idx}",
                        label_visibility="collapsed"
                    )
                with col2:
                    if st.button("➕", key=f"add_var_{idx}", help="Ajouter variation"):
                        if new_var:
                            matcher.add_variation(idx, new_var)
                            st.session_state.advisor_matcher = AdvisorMatcher()
                            st.rerun()

                st.divider()

                # Edit advisor
                st.markdown("**Modifier le conseiller:**")
                col1, col2 = st.columns(2)
                with col1:
                    edit_first = st.text_input(
                        "Prénom",
                        value=advisor.first_name,
                        key=f"edit_first_{idx}"
                    )
                with col2:
                    edit_last = st.text_input(
                        "Nom",
                        value=advisor.last_name,
                        key=f"edit_last_{idx}"
                    )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Sauvegarder", key=f"save_{idx}", type="primary"):
                        if edit_first and edit_last:
                            matcher.update_advisor(idx, edit_first, edit_last)
                            st.session_state.advisor_matcher = AdvisorMatcher()
                            st.success("✅ Conseiller mis à jour")
                            st.rerun()

                with col2:
                    if st.button("🗑️ Supprimer", key=f"delete_{idx}", type="secondary"):
                        matcher.delete_advisor(idx)
                        st.session_state.advisor_matcher = AdvisorMatcher()
                        st.warning(f"Conseiller supprimé")
                        st.rerun()

    st.divider()

    # Test matching section
    st.markdown("#### 🔍 Tester la correspondance")

    test_name = st.text_input(
        "Entrez un nom à tester",
        placeholder="Ex: Thomas Lussier, Lussier Thomas, T. Lussier...",
        key="test_name_input"
    )

    if test_name:
        result = matcher.match(test_name)
        if result:
            st.success(f"✅ Correspondance trouvée: **{result}**")
        else:
            st.warning(f"⚠️ Aucune correspondance pour: \"{test_name}\"")
            st.caption("Le nom original sera conservé tel quel.")


# =============================================================================
# BATCH PROCESSING HELPERS
# =============================================================================

def _process_batch_extraction(batch_params: dict):
    """
    Process batch PDF extraction with visual progress feedback.
    Called from render_stage_2 when in batch mode.

    Args:
        batch_params: Dict with source, board_name, api_key, reuse_board, reuse_group,
                      aggregate, target_board_type
    """
    uploaded_files = st.session_state.uploaded_files
    total_files = len(uploaded_files)

    st.markdown("### 🔄 Extraction en cours...")

    # Progress container
    progress_bar = st.progress(0)
    status_container = st.empty()
    details_container = st.container()

    results = {
        'successful': [],
        'failed': [],
        'files_with_multiple_months': []
    }

    # Extract parameters
    source = batch_params.get('source', 'UV')
    board_name = batch_params.get('board_name', '')
    api_key = batch_params.get('api_key', '')
    reuse_board = batch_params.get('reuse_board', True)
    reuse_group = batch_params.get('reuse_group', True)
    aggregate = batch_params.get('aggregate', False)
    target_board_type = batch_params.get('target_board_type', BoardType.HISTORICAL_PAYMENTS)

    first_pipeline = None

    for i, pdf_file in enumerate(uploaded_files):
        pdf_path = None
        try:
            # Update progress
            progress = (i / total_files)
            progress_bar.progress(progress)

            status_container.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <strong>📄 Traitement du fichier {i+1}/{total_files}</strong><br>
                <span style="color: #6c757d;">{pdf_file.name}</span>
            </div>
            """, unsafe_allow_html=True)

            # Save file temporarily
            pdf_path = save_uploaded_file(pdf_file)

            # Normaliser le nom de source
            source_normalized = source.replace(" ", "_").upper()

            # Extraire avec détails (permet des messages d'erreur précis)
            extraction_result = extract_with_details(
                pdf_path=pdf_path,
                source=source_normalized,
                aggregate=aggregate,
                target_board_type=target_board_type
            )

            if not extraction_result['success']:
                # Message d'erreur détaillé basé sur le type d'erreur
                error_msg = extraction_result['error_message']
                raise Exception(error_msg)

            # Create config for reference (for upload later)
            pdf_config = PipelineConfig(
                source=InsuranceSource(source_normalized),
                pdf_path=pdf_path,
                month_group=None,  # Will be detected per-row from data
                board_name=board_name,
                monday_api_key=api_key,
                output_dir=str(PROJECT_ROOT / "results"),
                reuse_board=reuse_board,
                reuse_group=reuse_group,
                aggregate_by_contract=aggregate,
                target_board_type=target_board_type
            )

            # Create pipeline for reference (for upload later)
            pipeline = InsuranceCommissionPipeline(pdf_config)
            pipeline.final_data = extraction_result['final_data']

            # Keep first pipeline for upload reference
            if first_pipeline is None:
                first_pipeline = pipeline

            # Add source metadata
            df = extraction_result['final_data'].copy()
            df['_source_file'] = pdf_file.name

            # Detect groups FROM DATA (not filename)
            df = detect_groups_from_data(df, source)

            # Analyze groups
            groups_info = analyze_groups_in_data(df)

            # Mark if file spans multiple months
            if groups_info['spans_multiple_months']:
                results['files_with_multiple_months'].append(pdf_file.name)

            results['successful'].append((pdf_file.name, df, groups_info))

            # Show success in details
            if groups_info['spans_multiple_months']:
                groups_str = ", ".join(groups_info['unique_groups'])
                with details_container:
                    st.warning(f"⚠️ {pdf_file.name} → Multi-mois: {groups_str} ({len(df)} items)")
            else:
                group_name = groups_info['unique_groups'][0] if groups_info['unique_groups'] else "N/A"
                with details_container:
                    st.success(f"✅ {pdf_file.name} → {group_name} ({len(df)} items)")

        except Exception as e:
            results['failed'].append((pdf_file.name, str(e)))
            with details_container:
                st.error(f"❌ {pdf_file.name}: {e}")

        finally:
            if pdf_path:
                cleanup_temp_file(pdf_path)

    # Complete progress
    progress_bar.progress(1.0)
    status_container.empty()

    # Combine successful dataframes
    if results['successful']:
        all_dfs = [item[1] for item in results['successful']]  # item = (filename, df, groups_info)
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Add extraction order for sorting
        combined_df['_extraction_order'] = range(len(combined_df))

        # Store results
        st.session_state.extracted_data = combined_df
        st.session_state.extraction_results = results

        # Use the first successful pipeline as reference (for upload later)
        st.session_state.pipeline = first_pipeline
        st.session_state.batch_config_params = batch_params

        st.rerun()
    else:
        st.error("❌ Aucun fichier n'a pu être traité avec succès")
        if st.button("🔄 Recommencer"):
            reset_pipeline()
            st.rerun()


def _render_batch_summary(results: dict):
    """Render batch extraction results summary."""
    successful = results.get('successful', [])
    failed = results.get('failed', [])
    files_with_multiple_months = results.get('files_with_multiple_months', [])

    total = len(successful) + len(failed)
    success_count = len(successful)

    # Summary metrics
    st.markdown("### 📋 Résumé de l'extraction batch")

    # Count total items and unique groups
    total_items = 0
    all_groups = set()
    for filename, df, groups_info in successful:
        total_items += len(df)
        all_groups.update(groups_info.get('unique_groups', []))

    cols = st.columns(4)
    cols[0].metric("Fichiers traités", f"{success_count}/{total}")
    cols[1].metric("Items extraits", total_items)
    cols[2].metric("Groupes détectés", len(all_groups))
    cols[3].metric("Échecs", len(failed), delta_color="inverse" if failed else "off")

    # Show files spanning multiple months (warning)
    if files_with_multiple_months:
        st.warning(f"⚠️ **{len(files_with_multiple_months)} fichier(s) couvrent plusieurs mois.** "
                   f"Les lignes seront automatiquement assignées à leur groupe respectif.")
        with st.expander("📅 Fichiers multi-mois", expanded=True):
            for filename, df, groups_info in successful:
                if filename in files_with_multiple_months:
                    st.markdown(f"**{filename}**")
                    for group, count in groups_info.get('group_counts', {}).items():
                        st.caption(f"  • {group}: {count} lignes")

    # Show failed files if any
    if failed:
        with st.expander("❌ Fichiers en erreur", expanded=True):
            for filename, error in failed:
                st.error(f"**{filename}**: {error}")

    # Show successful files breakdown
    if successful:
        with st.expander(f"✅ {success_count} fichiers traités avec succès", expanded=False):
            for filename, df, groups_info in successful:
                unique_groups = groups_info.get('unique_groups', [])
                group_counts = groups_info.get('group_counts', {})

                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.text(filename)
                with col2:
                    if len(unique_groups) == 1:
                        st.caption(f"→ {unique_groups[0]}")
                    else:
                        st.caption(f"→ {len(unique_groups)} groupes")
                with col3:
                    st.caption(f"{len(df)} items")

    st.divider()


# =============================================================================
# STAGE 2: PREVIEW - CLEANER LAYOUT (with batch support)
# =============================================================================

def render_stage_2():
    """Render data preview stage with batch processing support."""

    st.markdown("## 📊 Pipeline de Commissions")
    render_stepper()
    st.write("")

    is_batch = st.session_state.get('batch_mode', False)

    # Get config info based on mode
    if is_batch:
        batch_params = st.session_state.get('batch_config_params', {})
        source_name = batch_params.get('source', 'N/A')
        board_name = batch_params.get('board_name', 'N/A')
        aggregate = batch_params.get('aggregate', False)
    else:
        config = st.session_state.config
        source_name = config.source.value if config else 'N/A'
        board_name = config.board_name if config else 'N/A'
        aggregate = config.aggregate_by_contract if config else False

    # Config summary - compact
    with st.expander("📋 Configuration", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Source", source_name)
        cols[1].metric("Board", board_name[:20] + "..." if len(board_name) > 20 else board_name)
        if is_batch:
            cols[2].metric("Mode", f"Batch ({len(st.session_state.uploaded_files)} fichiers)")
        else:
            cols[2].metric("Groupe", config.month_group or "Auto-détecté" if config else "N/A")
        cols[3].metric("Agrégation", "Oui" if aggregate else "Non")

    # Extract data if not done
    if st.session_state.extracted_data is None:
        if is_batch:
            # BATCH MODE: Process multiple PDFs with progress
            batch_params = st.session_state.get('batch_config_params', {})
            _process_batch_extraction(batch_params)
            return
        else:
            # SINGLE FILE MODE
            if not config:
                st.error("❌ Configuration non trouvée")
                if st.button("🔄 Recommencer"):
                    reset_pipeline()
                    st.rerun()
                return

            is_pdf_source = config.source != InsuranceSource.MONDAY_LEGACY
            source_type = "PDF" if is_pdf_source else "Monday.com"

            with st.spinner(f"🔄 Extraction depuis {source_type}..."):
                try:
                    if is_pdf_source and config.pdf_path:
                        # Utiliser extract_with_details() pour des messages d'erreur précis
                        extraction_result = extract_with_details(
                            pdf_path=config.pdf_path,
                            source=config.source.value,
                            aggregate=config.aggregate_by_contract,
                            target_board_type=config.target_board_type
                        )

                        if not extraction_result['success']:
                            # Message d'erreur détaillé
                            error_msg = extraction_result['error_message']
                            error_type = extraction_result['error_type']

                            if error_type == 'filtered_out':
                                st.warning(f"⚠️ {error_msg}")
                            else:
                                st.error(f"❌ {error_msg}")

                            if st.button("🔄 Recommencer"):
                                reset_pipeline()
                                st.rerun()
                            return

                        # Create pipeline for reference
                        pipeline = InsuranceCommissionPipeline(config)
                        pipeline.final_data = extraction_result['final_data']
                    else:
                        # MONDAY_LEGACY: utiliser le pipeline standard
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

    # Batch mode: show extraction results summary
    if is_batch and st.session_state.get('extraction_results'):
        results = st.session_state.extraction_results
        _render_batch_summary(results)

        # Manual group override option for batch mode
        if '_target_group' in df.columns:
            unique_groups = df['_target_group'].unique().tolist()

            with st.expander("📅 Modifier le groupe de destination", expanded=False):
                st.caption("Si la détection automatique de date n'est pas correcte, vous pouvez assigner manuellement un groupe à toutes les lignes.")

                col1, col2 = st.columns([3, 1])

                with col1:
                    # Generate group options (current month ± 3 months)
                    from datetime import datetime
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

                    # Also add any detected groups not in the list
                    for g in unique_groups:
                        if g not in group_options:
                            group_options.insert(1, g)

                    manual_group = st.selectbox(
                        "Groupe manuel",
                        options=group_options,
                        index=0,
                        key="manual_group_override"
                    )

                with col2:
                    if manual_group != "(Garder auto-détection)":
                        if st.button("✅ Appliquer", use_container_width=True, key="apply_manual_group"):
                            # Apply manual group to all rows
                            df['_target_group'] = manual_group
                            st.session_state.extracted_data = df
                            st.success(f"Groupe modifié: {manual_group}")
                            st.rerun()

                # Show current groups
                st.markdown("**Groupes actuels:**")
                for group in unique_groups:
                    count = len(df[df['_target_group'] == group])
                    st.caption(f"• {group}: {count} lignes")

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

        # Tolerance slider - simple direct usage without complex callbacks
        col1, col2 = st.columns([2, 3])
        with col1:
            tolerance = st.slider(
                "Tolérance (%)",
                min_value=1.0,
                max_value=50.0,
                value=10.0,
                step=1.0,
                help="Écart acceptable entre Reçu et Com. Le pourcentage affiché est l'écart réel entre Reçu et Com Calculée.",
                key="verification_tolerance_slider"
            )

        # Apply verification with current slider value
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

        # Data preview with verification column (reorder columns for display)
        st.dataframe(reorder_columns_for_display(df_verified), use_container_width=True, height=350)

        # Store verified data for potential use
        df_display = df_verified
    else:
        # Data preview without verification (reorder columns for display)
        st.dataframe(reorder_columns_for_display(df), use_container_width=True, height=350)
        df_display = df

    # Actions row
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Télécharger CSV",
            data=csv,
            file_name=f"commissions_{source_name}.csv",
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

    # Groups display for Monday Legacy (not applicable in batch mode)
    if not is_batch and config and config.source == InsuranceSource.MONDAY_LEGACY:
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
# STAGE 3: UPLOAD - STREAMLINED (with batch support)
# =============================================================================

def render_stage_3():
    """Render upload stage with batch support."""

    st.markdown("## 📊 Pipeline de Commissions")
    render_stepper()
    st.write("")

    df = st.session_state.extracted_data
    config = st.session_state.config
    is_batch = st.session_state.get('batch_mode', False)
    batch_params = st.session_state.get('batch_config_params', {})

    # Get board_name from config or batch_params
    board_name = batch_params.get('board_name', '') if is_batch else (config.board_name if config else '')

    if st.session_state.data_modified:
        st.warning("⚠️ Upload de données modifiées")

    # Summary
    st.markdown("### 📋 Résumé de l'upload")

    if is_batch:
        # Batch mode summary
        unique_groups = df['_target_group'].unique() if '_target_group' in df.columns else []
        cols = st.columns(4)
        cols[0].metric("Items total", len(df))
        cols[1].metric("Board", board_name[:20] + "..." if len(board_name) > 20 else board_name)
        cols[2].metric("Groupes", len(unique_groups))
        cols[3].metric("Fichiers", len(st.session_state.get('extraction_results', {}).get('successful', [])))

        # Show groups breakdown
        if '_target_group' in df.columns:
            with st.expander("📁 Détail par groupe", expanded=False):
                for group in unique_groups:
                    group_count = len(df[df['_target_group'] == group])
                    st.markdown(f"**{group}**: {group_count} items")
    else:
        # Single file mode summary
        cols = st.columns(3)
        cols[0].metric("Lignes", len(df))
        cols[1].metric("Board", board_name[:25] + "..." if len(board_name) > 25 else board_name)
        cols[2].metric("Groupe", config.month_group if config else "Auto-détecté")

    st.divider()

    # Upload process
    if st.session_state.upload_results is None:
        if is_batch:
            st.info(f"Les données vont être uploadées vers Monday.com dans {len(df['_target_group'].unique()) if '_target_group' in df.columns else 1} groupe(s).")
        else:
            st.info("Les données vont être uploadées vers Monday.com.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.stage = 2
                st.rerun()

        with col2:
            if st.button("🚀 Confirmer l'upload", type="primary", use_container_width=True):
                if is_batch:
                    execute_batch_upload()
                else:
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


def execute_batch_upload():
    """Execute batch upload to Monday.com with multiple groups."""
    batch_params = st.session_state.get('batch_config_params', {})
    df = st.session_state.extracted_data

    # Get reuse_group from batch_params
    reuse_group = batch_params.get('reuse_group', True)

    # Get unique groups from the combined DataFrame
    if '_target_group' not in df.columns:
        st.error("❌ Colonne '_target_group' manquante dans les données")
        return

    unique_groups = df['_target_group'].unique().tolist()
    total_items = len(df)

    # Progress tracking
    progress_bar = st.progress(0)
    status_container = st.empty()
    details_container = st.container()

    try:
        pipeline = st.session_state.pipeline

        # Step 1: Setup board (without creating groups yet)
        status_container.markdown("⚙️ **Configuration du board...**")
        progress_bar.progress(5)

        # Setup the board using the first pipeline
        if not pipeline._step3_setup_monday_board():
            st.error("❌ Échec de la configuration du board")
            return

        progress_bar.progress(10)

        # Step 2: Process each group
        results_by_group = {}
        all_results = []
        items_uploaded = 0
        items_failed = 0
        groups_processed = 0

        for group_idx, group_name in enumerate(unique_groups):
            group_items = df[df['_target_group'] == group_name]
            group_count = len(group_items)

            status_container.markdown(f"📁 **Groupe {group_idx + 1}/{len(unique_groups)}:** {group_name} ({group_count} items)")

            try:
                # Create group
                from monday_automation import MondayClient

                group_result = pipeline.monday_client.create_group(
                    board_id=pipeline.board_id,
                    group_name=str(group_name),
                    group_color="#0086c0",
                    reuse_existing=reuse_group
                )

                if not group_result.success:
                    with details_container:
                        st.warning(f"⚠️ Impossible de créer le groupe '{group_name}': {group_result.error}")
                    items_failed += group_count
                    continue

                group_id = group_result.group_id

                # Prepare items for this group (exclude metadata columns)
                group_df = group_items.drop(columns=['_source_file', '_target_group', '_extraction_order'], errors='ignore')
                items = pipeline._prepare_monday_items(group_df)

                # Upload items in batches
                batch_size = 10
                group_results = []

                for i in range(0, len(items), batch_size):
                    batch = items[i:i + batch_size]

                    batch_results = pipeline.monday_client.create_items_batch(
                        board_id=pipeline.board_id,
                        items=batch,
                        group_id=group_id
                    )
                    group_results.extend(batch_results)

                    # Update progress
                    items_done = items_uploaded + len(group_results)
                    pct = 10 + int(85 * items_done / total_items)
                    progress_bar.progress(min(pct, 95))

                # Count results for this group
                group_success = sum(1 for r in group_results if r.success)
                group_fail = len(group_results) - group_success

                items_uploaded += group_success
                items_failed += group_fail
                all_results.extend(group_results)

                results_by_group[group_name] = {
                    'success': group_success,
                    'failed': group_fail,
                    'group_id': group_id
                }

                groups_processed += 1

                with details_container:
                    if group_fail == 0:
                        st.success(f"✅ {group_name}: {group_success} items uploadés")
                    else:
                        st.warning(f"⚠️ {group_name}: {group_success} uploadés, {group_fail} échecs")

            except Exception as e:
                items_failed += group_count
                with details_container:
                    st.error(f"❌ {group_name}: Erreur - {str(e)}")

        progress_bar.progress(100)
        status_container.empty()

        # Save results
        if items_uploaded > 0:
            st.session_state.upload_results = {
                'success': True,
                'board_id': pipeline.board_id,
                'group_id': None,  # Multiple groups
                'items_uploaded': items_uploaded,
                'items_failed': items_failed,
                'is_batch': True,
                'groups_processed': groups_processed,
                'total_groups': len(unique_groups),
                'results_by_group': results_by_group
            }
            st.rerun()
        else:
            st.error("❌ Aucun item uploadé")

    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        with st.expander("Détails"):
            st.exception(e)


def render_upload_results():
    """Render upload results with batch mode support."""
    results = st.session_state.upload_results
    config = st.session_state.config
    is_batch = results.get('is_batch', False)
    batch_params = st.session_state.get('batch_config_params', {})

    # Get board_name from config or batch_params
    board_name = batch_params.get('board_name', '') if is_batch else (config.board_name if config else '')

    if results['success']:
        st.balloons()
        st.success("✅ Upload terminé avec succès!")

        if is_batch:
            # Batch mode metrics
            cols = st.columns(4)
            cols[0].metric("Items créés", results['items_uploaded'])
            cols[1].metric("Échecs", results['items_failed'])
            cols[2].metric("Groupes", f"{results['groups_processed']}/{results['total_groups']}")
            cols[3].metric("Board ID", results['board_id'])

            st.divider()

            # Show results by group
            if results.get('results_by_group'):
                with st.expander("📁 Détail par groupe", expanded=True):
                    for group_name, group_data in results['results_by_group'].items():
                        if group_data['failed'] == 0:
                            st.success(f"✅ **{group_name}**: {group_data['success']} items")
                        else:
                            st.warning(f"⚠️ **{group_name}**: {group_data['success']} créés, {group_data['failed']} échecs")

            if results['items_failed'] == 0:
                st.info(f"""
                🎉 **Upload batch réussi!**

                **{results['items_uploaded']}** items créés dans **{results['groups_processed']}** groupe(s)
                dans le board **{board_name}**
                """)
            else:
                st.warning(f"""
                ⚠️ **Upload batch partiel**

                {results['items_uploaded']} items créés, {results['items_failed']} échecs
                dans {results['groups_processed']} groupe(s)
                """)
        else:
            # Single file mode metrics
            cols = st.columns(4)
            cols[0].metric("Items créés", results['items_uploaded'])
            cols[1].metric("Échecs", results['items_failed'])
            cols[2].metric("Board ID", results['board_id'])
            cols[3].metric("Group ID", results['group_id'] or "Défaut")

            st.divider()

            if results['items_failed'] == 0:
                st.info(f"""
                🎉 **Upload réussi!**

                **{results['items_uploaded']}** items créés dans le board **{board_name}**
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

    # Auto-load boards at startup if API key is available
    load_boards_async()

    render_sidebar()

    if st.session_state.stage == 1:
        render_stage_1()
    elif st.session_state.stage == 2:
        render_stage_2()
    elif st.session_state.stage == 3:
        render_stage_3()


if __name__ == "__main__":
    main()
