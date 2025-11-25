"""
Streamlit Application - Insurance Commission Data Pipeline
===========================================================

Application web pour extraire, visualiser et uploader les données
de commissions d'assurance vers Monday.com.

Author: Thomas
Date: 2025-10-30
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path
from io import StringIO
import sys

# Import pipeline components
from main import (
    InsuranceCommissionPipeline,
    PipelineConfig,
    InsuranceSource,
    ColorPrint
)
from unify_notation import BoardType

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Insurance Commission Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'stage' not in st.session_state:
        st.session_state.stage = 1

    if 'pdf_file' not in st.session_state:
        st.session_state.pdf_file = None

    if 'pdf_path' not in st.session_state:
        st.session_state.pdf_path = None

    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None

    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None

    if 'config' not in st.session_state:
        st.session_state.config = None

    if 'upload_results' not in st.session_state:
        st.session_state.upload_results = None

    if 'data_modified' not in st.session_state:
        st.session_state.data_modified = False

    # For Monday.com board selection
    if 'monday_boards' not in st.session_state:
        st.session_state.monday_boards = None

    if 'selected_board_id' not in st.session_state:
        st.session_state.selected_board_id = None

    # Global Monday.com API Key
    if 'monday_api_key' not in st.session_state:
        st.session_state.monday_api_key = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sort_and_filter_boards(boards: list, search_query: str = "") -> list:
    """
    Sort boards with priority keywords first and filter by search query.

    Priority order:
    1. Boards containing "paiement" or "historique" (case-insensitive)
    2. Boards containing "vente" or "production" (case-insensitive)
    3. All other boards (alphabetically)

    Args:
        boards: List of board dictionaries with 'name' and 'id' keys
        search_query: Optional search string to filter boards by name

    Returns:
        Sorted and filtered list of boards
    """
    if not boards:
        return []

    # Filter by search query if provided
    filtered_boards = boards
    if search_query and search_query.strip():
        search_lower = search_query.lower().strip()
        filtered_boards = [
            b for b in boards
            if search_lower in b['name'].lower()
        ]

    # Define priority keywords
    priority_1_keywords = ['paiement', 'historique']
    priority_2_keywords = ['vente', 'production']

    def get_priority(board_name: str) -> tuple:
        """
        Return a tuple for sorting: (priority_level, board_name_lower)
        Lower priority_level = higher priority (appears first)
        """
        name_lower = board_name.lower()

        # Priority 1: paiement/historique
        if any(kw in name_lower for kw in priority_1_keywords):
            return (0, name_lower)

        # Priority 2: vente/production
        if any(kw in name_lower for kw in priority_2_keywords):
            return (1, name_lower)

        # Priority 3: all others (alphabetically)
        return (2, name_lower)

    # Sort boards by priority then alphabetically
    sorted_boards = sorted(filtered_boards, key=lambda b: get_priority(b['name']))

    return sorted_boards


def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to a temporary location and return the path.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Path to the saved file
    """
    # Create temp directory if it doesn't exist
    temp_dir = Path("./temp")
    temp_dir.mkdir(exist_ok=True)

    # Save file
    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(file_path)


def cleanup_temp_file(file_path: str = None):
    """
    Clean up temporary file.

    Args:
        file_path: Path to file to delete. If None, uses session state pdf_path
    """
    if file_path is None:
        file_path = st.session_state.get('pdf_path')

    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            # Silent fail - don't interrupt user experience
            pass


def reset_pipeline():
    """Reset pipeline state to start over."""
    # Clean up temporary file before reset
    cleanup_temp_file()

    st.session_state.stage = 1
    st.session_state.pdf_file = None
    st.session_state.pdf_path = None
    st.session_state.extracted_data = None
    st.session_state.pipeline = None
    st.session_state.config = None
    st.session_state.upload_results = None
    st.session_state.data_modified = False
    st.session_state.monday_boards = None
    st.session_state.selected_board_id = None


# =============================================================================
# STAGE 1: CONFIGURATION AND UPLOAD
# =============================================================================

def render_stage_1():
    """Render configuration and file upload stage."""
    st.title("📊 Insurance Commission Data Pipeline")
    st.markdown("---")

    st.header("📁 Étape 1: Configuration et Upload")

    # Create tabs for different workflows
    tab1, tab2 = st.tabs(["📄 Extraction PDF", "🔄 Conversion Monday.com"])

    # =========================================================================
    # TAB 1: PDF EXTRACTION (UV, IDC, ASSOMPTION)
    # =========================================================================
    with tab1:
        st.info("""
        **📄 Extraction depuis fichiers PDF**

        Ce mode extrait les données de commissions depuis des fichiers PDF pour les sources:
        - **UV Assurance**: Rapports de rémunération
        - **IDC**: Rapports de propositions
        - **IDC Statement**: Rapports de frais de suivi (trailing fees)
        - **Assomption Vie**: Rapports de rémunération

        Les données sont extraites, standardisées et prêtes à être uploadées vers Monday.com.
        """)

        st.markdown("---")

        # Load boards section (outside form) - More prominent
        st.subheader("3️⃣ Chargement des Boards Monday.com")

        # Get API key from session state
        pdf_monday_api_key = st.session_state.monday_api_key

        if pdf_monday_api_key:
            st.info("""
            **📋 Gestion des Boards**

            Chargez vos boards Monday.com pour pouvoir sélectionner un board existant
            ou vérifier les boards disponibles avant d'en créer un nouveau.
            """)

            col_load, col_status, col_refresh = st.columns([1, 2, 1])

            with col_load:
                load_boards_btn_pdf = st.button(
                    "📥 Charger mes boards",
                    use_container_width=True,
                    type="primary",
                    key="pdf_load_boards_btn"
                )

            with col_status:
                if st.session_state.monday_boards is not None:
                    st.success(f"✅ {len(st.session_state.monday_boards)} boards disponibles")
                else:
                    st.info("ℹ️ Cliquez pour charger vos boards")

            with col_refresh:
                if st.session_state.monday_boards is not None:
                    if st.button("🔄 Rafraîchir", use_container_width=True, key="pdf_refresh_boards"):
                        st.session_state.monday_boards = None
                        st.rerun()

            # Load boards when button clicked
            if load_boards_btn_pdf:
                try:
                    from monday_automation import MondayClient

                    with st.spinner("Chargement de vos boards Monday.com..."):
                        client = MondayClient(api_key=pdf_monday_api_key)
                        boards = client.list_boards()

                        # Store in session state
                        st.session_state.monday_boards = boards

                        st.success(f"✅ {len(boards)} boards chargés avec succès!")
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Erreur lors du chargement des boards: {e}")
                    st.session_state.monday_boards = None
        else:
            st.warning("⚠️ **Veuillez d'abord entrer votre clé API Monday.com dans la barre latérale** pour pouvoir charger vos boards.")

        st.markdown("---")

        # Board Selection Mode (outside form so it's reactive)
        st.subheader("4️⃣ Sélection du Board")

        # Choose between new or existing board
        board_mode = st.radio(
            "Mode de sélection du board",
            options=["Créer un nouveau board", "Utiliser un board existant"],
            index=1,  # Default to "Utiliser un board existant"
            help="Choisissez si vous voulez créer un nouveau board ou utiliser un board existant",
            key="pdf_board_mode"
        )

        # Search box for boards (outside form for reactivity) - only show for existing board mode
        if board_mode == "Utiliser un board existant" and st.session_state.monday_boards:
            st.text_input(
                "🔍 Rechercher un board",
                value="",
                placeholder="Tapez pour filtrer les boards par nom...",
                help="Filtrez la liste des boards par nom (la recherche est instantanée)",
                key="pdf_board_search"
            )

        st.markdown("---")

        with st.form("pdf_extraction_form"):
            # Source Selection
            st.subheader("5️⃣ Source des Données PDF")
            source = st.selectbox(
                "Sélectionnez la source d'assurance",
                options=["UV", "IDC", "IDC Statement", "ASSOMPTION"],
                help="Type de document PDF à traiter"
            )

            st.markdown("---")

            # PDF Upload
            st.subheader("6️⃣ Upload du PDF")
            uploaded_file = st.file_uploader(
                "Déposez ou sélectionnez votre fichier PDF",
                type=['pdf'],
                help="Fichier PDF contenant les données de commissions"
            )

            # No Monday.com source fields for PDF extraction
            source_board_id = None
            source_group_id = None

            st.markdown("---")

            # Board Configuration (content depends on board_mode selected above)
            st.subheader("7️⃣ Configuration du Board")

            selected_board_id_pdf = None
            board_name_input = None

            if board_mode == "Créer un nouveau board":
                # New board mode
                st.info("""
                **📝 Mode Nouveau Board**

                Créez un nouveau board Monday.com ou réutilisez un board existant avec le même nom.
                """)

                col1, col2 = st.columns(2)

                with col1:
                    board_name_input = st.text_input(
                        "Nom du Nouveau Board",
                        placeholder=f"Ex: Commissions {source}",
                        help="Nom du board Monday.com qui sera créé. Laissez vide pour utiliser le nom par défaut.",
                        key="pdf_board_name"
                    )

                    # Show what will be used
                    if board_name_input and board_name_input.strip():
                        st.caption(f"📋 Nom du board: **{board_name_input.strip()}**")
                    else:
                        st.caption(f"📋 Nom par défaut sera utilisé: **Commissions {source}**")

                with col2:
                    col_reuse1, col_reuse2 = st.columns(2)
                    with col_reuse1:
                        reuse_board = st.checkbox(
                            "Réutiliser si existe",
                            value=True,
                            help="Si coché, utilisera le board existant avec le même nom au lieu d'en créer un nouveau",
                            key="pdf_reuse_board"
                        )
                    with col_reuse2:
                        reuse_group = st.checkbox(
                            "Réutiliser groupe",
                            value=True,
                            help="Si coché, utilisera le groupe existant avec le même nom",
                            key="pdf_reuse_group"
                        )

            else:
                # Existing board mode
                if st.session_state.monday_boards is not None and len(st.session_state.monday_boards) > 0:
                    st.success(f"✅ {len(st.session_state.monday_boards)} boards disponibles pour sélection")

                    # Search box for filtering boards (outside form for reactivity)
                    st.caption("🔍 **Rechercher un board par nom:**")

                    # Note: This search is inside the form, so it will filter on form rerun
                    # For better UX, we store the search in session state
                    if 'pdf_board_search' not in st.session_state:
                        st.session_state.pdf_board_search = ""

                    # Sort and filter boards with priority and search
                    sorted_boards = sort_and_filter_boards(
                        st.session_state.monday_boards,
                        search_query=st.session_state.get('pdf_board_search', '')
                    )

                    # Show filter info
                    if st.session_state.get('pdf_board_search', ''):
                        st.info(f"🔎 {len(sorted_boards)} boards trouvés pour \"{st.session_state.pdf_board_search}\"")
                    else:
                        st.caption("ℹ️ Les boards \"Paiements Historiques\" et \"Ventes/Production\" sont affichés en premier")

                    if sorted_boards:
                        # Create options with board name and ID
                        board_options = {
                            f"{board['name']} (ID: {board['id']})": board['id']
                            for board in sorted_boards
                        }

                        selected_board_option = st.selectbox(
                            "Sélectionnez le board où uploader les données",
                            options=list(board_options.keys()),
                            help="Choisissez le board où les données PDF seront uploadées",
                            key="pdf_selected_board"
                        )

                        # Get the board ID and name from selection
                        selected_board_id_pdf = board_options[selected_board_option]

                        # Extract board name from the selected board
                        selected_board = next(b for b in st.session_state.monday_boards if b['id'] == selected_board_id_pdf)
                        board_name_input = selected_board['name']

                        # Show board info in an expander
                        with st.expander("ℹ️ Détails du board sélectionné", expanded=False):
                            st.write(f"**Nom du board:** {board_name_input}")
                            st.write(f"**ID du board:** {selected_board_id_pdf}")
                            st.write(f"**Type:** {selected_board.get('board_kind', 'N/A')}")
                            st.write(f"**État:** {selected_board.get('state', 'N/A')}")
                    else:
                        st.warning(f"⚠️ Aucun board trouvé pour \"{st.session_state.get('pdf_board_search', '')}\"")
                        selected_board_id_pdf = None
                        board_name_input = None

                    # Force reuse_board and reuse_group to True for existing boards
                    reuse_board = True
                    reuse_group = True

                else:
                    st.error("❌ **Aucun board chargé**")
                    st.warning("""
                    **Action requise:**

                    1. Retournez à la section **"3️⃣ Chargement des Boards Monday.com"** ci-dessus
                    2. Cliquez sur le bouton **"📥 Charger mes boards"**
                    3. Attendez que vos boards soient chargés
                    4. Revenez ici pour sélectionner votre board

                    *Si vous n'avez pas encore entré votre clé API, allez dans la barre latérale.*
                    """)
                    reuse_board = True
                    reuse_group = True

            st.markdown("---")

            # Group Configuration
            st.subheader("8️⃣ Configuration du Groupe")

            month_group = st.text_input(
                "Groupe de Mois (optionnel)",
                value="",
                placeholder="Ex: Octobre 2025",
                help="Nom du groupe pour organiser les données (optionnel)",
                key="pdf_month_group"
            )

            st.markdown("---")

            # Target Board Type Selection
            st.subheader("9️⃣ Type de Table Cible")
            target_board_type_option = st.selectbox(
                "Type de board Monday.com",
                options=["Paiements Historiques", "Ventes et Production"],
                index=0,
                help="Sélectionnez le type de table Monday.com où les données seront uploadées",
                key="pdf_target_board_type"
            )

            st.info(f"""
            **📋 Type de table sélectionné: {target_board_type_option}**

            - **Paiements Historiques**: Pour les paiements reçus et vérifiés
            - **Ventes et Production**: Pour les ventes avec suivi de complétion et reçus
            """)

            st.markdown("---")

            # Data Processing Options
            st.subheader("🔟 Options de Traitement")
            aggregate_by_contract = st.checkbox(
                "Agréger par numéro de contrat",
                value=True,
                help="Si coché, les lignes avec le même numéro de contrat seront agrégées (somme des montants, moyenne des taux). Décochez pour garder toutes les lignes séparées.",
                key="pdf_aggregate_by_contract"
            )

            st.markdown("---")

            # Submit button
            submitted = st.form_submit_button(
                "🚀 Extraire les données du PDF",
                use_container_width=True,
                type="primary"
            )

            if submitted:
                # Validation
                errors = []

                if not uploaded_file:
                    errors.append("❌ Veuillez uploader un fichier PDF")

                # Get API key from session state
                monday_api_key = st.session_state.monday_api_key

                if not monday_api_key:
                    errors.append("❌ Veuillez fournir une clé API Monday.com dans la barre latérale")

                # Get board mode from session state
                board_mode_from_state = st.session_state.get('pdf_board_mode', 'Créer un nouveau board')

                # Validate board selection for existing board mode
                if board_mode_from_state == "Utiliser un board existant":
                    if not selected_board_id_pdf:
                        errors.append("❌ Veuillez charger vos boards et sélectionner un board existant")

                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    # Save uploaded file
                    pdf_path = save_uploaded_file(uploaded_file)

                    # Determine final board name
                    if board_mode_from_state == "Utiliser un board existant":
                        # Use the name from selected board
                        final_board_name = board_name_input
                        # For existing boards, force reuse
                        final_reuse_board = True
                        final_reuse_group = True
                    else:
                        # New board mode
                        board_name_from_state = st.session_state.get('pdf_board_name', '')

                        if board_name_from_state and board_name_from_state.strip():
                            final_board_name = board_name_from_state.strip()
                        else:
                            final_board_name = f"Commissions {source}"

                        final_reuse_board = reuse_board
                        final_reuse_group = reuse_group

                    # Create configuration
                    try:
                        # Convert display name to enum value
                        source_enum_value = source.replace(" ", "_").upper()

                        # Convert target board type option to BoardType enum
                        target_board_type_from_state = st.session_state.get('pdf_target_board_type', 'Paiements Historiques')
                        if target_board_type_from_state == "Ventes et Production":
                            target_board_type = BoardType.SALES_PRODUCTION
                        else:
                            target_board_type = BoardType.HISTORICAL_PAYMENTS

                        config = PipelineConfig(
                            source=InsuranceSource(source_enum_value),
                            pdf_path=pdf_path,
                            month_group=month_group if month_group else None,
                            board_name=final_board_name,
                            monday_api_key=monday_api_key,
                            output_dir="./results",
                            reuse_board=final_reuse_board,
                            reuse_group=final_reuse_group,
                            aggregate_by_contract=aggregate_by_contract,
                            source_board_id=None,
                            source_group_id=None,
                            target_board_type=target_board_type
                        )

                        # Store in session state
                        st.session_state.pdf_file = uploaded_file
                        st.session_state.pdf_path = pdf_path
                        st.session_state.config = config

                        # Move to next stage
                        st.session_state.stage = 2
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Erreur de configuration: {e}")

    # =========================================================================
    # TAB 2: MONDAY.COM CONVERSION (MONDAY_LEGACY)
    # =========================================================================
    with tab2:
        st.warning("""
        ⚠️ **Fonctionnalité Spéciale - Conversion de Board**

        Cette fonction est conçue pour être utilisée **une seule fois** lors de la migration
        d'un ancien format de board Monday.com vers le nouveau format standardisé.
        """)

        st.info("""
        **🔄 Conversion Monday.com Legacy → Nouveau Format**

        Cette fonctionnalité convertit les données d'un ancien tableau Monday.com vers le nouveau format standardisé.

        **Colonnes converties automatiquement:**
        - `# de Police` → `contract_number`
        - `Compagnie` → `insurer_name`
        - `PA` → `policy_premium`
        - `Com` → `commission`
        - `Boni` → `bonus_amount`
        - `Sur-Com` → `on_commission`
        - Et plus...

        **Constantes appliquées:**
        - sharing_rate = 0.4 (40%)
        - commission_rate = 0.5 (50%)
        - bonus_rate = 1.75 (175%)
        - on_commission_rate = 0.75 (75%)
        """)

        # Load boards button
        st.subheader("1️⃣ Sélection du Board Source")

        # Get API key from session state
        monday_api_key_legacy = st.session_state.monday_api_key

        if monday_api_key_legacy:
            col_load, col_status = st.columns([1, 3])

            with col_load:
                load_boards_btn = st.button(
                    "📥 Charger mes boards",
                    use_container_width=True,
                    type="secondary"
                )

            with col_status:
                if st.session_state.monday_boards is not None:
                    st.success(f"✅ {len(st.session_state.monday_boards)} boards chargés")
                elif load_boards_btn:
                    st.info("⏳ Chargement en cours...")

            # Load boards when button clicked
            if load_boards_btn:
                try:
                    from monday_automation import MondayClient

                    with st.spinner("Chargement de vos boards Monday.com..."):
                        client = MondayClient(api_key=monday_api_key_legacy)
                        boards = client.list_boards()

                        # Store in session state
                        st.session_state.monday_boards = boards

                        st.success(f"✅ {len(boards)} boards chargés avec succès!")
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Erreur lors du chargement des boards: {e}")
                    st.session_state.monday_boards = None
        else:
            st.warning("⚠️ Veuillez d'abord entrer votre clé API Monday.com dans la barre latérale")

        st.markdown("---")

        # Search box for legacy boards (outside form for reactivity)
        if st.session_state.monday_boards:
            st.text_input(
                "🔍 Rechercher un board source",
                value="",
                placeholder="Tapez pour filtrer les boards par nom...",
                help="Filtrez la liste des boards par nom (la recherche est instantanée)",
                key="legacy_board_search"
            )

        with st.form("monday_conversion_form"):
            # Source Board Configuration
            st.subheader("2️⃣ Board à Convertir")

            # Board selection dropdown
            if st.session_state.monday_boards is not None and len(st.session_state.monday_boards) > 0:
                # Sort and filter boards with priority and search
                sorted_boards = sort_and_filter_boards(
                    st.session_state.monday_boards,
                    search_query=st.session_state.get('legacy_board_search', '')
                )

                # Show filter info
                if st.session_state.get('legacy_board_search', ''):
                    st.info(f"🔎 {len(sorted_boards)} boards trouvés pour \"{st.session_state.legacy_board_search}\"")
                else:
                    st.caption("ℹ️ Les boards \"Paiements Historiques\" et \"Ventes/Production\" sont affichés en premier")

                if sorted_boards:
                    # Create options with board name and ID
                    board_options = {
                        f"{board['name']} (ID: {board['id']})": board['id']
                        for board in sorted_boards
                    }

                    selected_board_option = st.selectbox(
                        "Sélectionnez le board à convertir",
                        options=list(board_options.keys()),
                        help="Choisissez le board contenant les données à convertir (ancien format)"
                    )

                    # Get the board ID from selection
                    source_board_id = board_options[selected_board_option]
                else:
                    st.warning(f"⚠️ Aucun board trouvé pour \"{st.session_state.get('legacy_board_search', '')}\"")
                    source_board_id = None

                # Show board info only if a board is selected
                if source_board_id:
                    st.caption(f"📋 Board sélectionné - ID: **{source_board_id}**")

            else:
                st.warning("⚠️ Veuillez d'abord charger vos boards avec le bouton ci-dessus")
                source_board_id = None

            st.markdown("---")

            # Target Board Configuration
            st.subheader("3️⃣ Configuration du Nouveau Board")

            board_name_input_legacy = st.text_input(
                "Nom du Nouveau Board",
                placeholder="Ex: Commissions - Nouveau Format",
                help="Nom du board Monday.com qui sera créé avec le nouveau format",
                key="legacy_board_name"
            )

            col_reuse1, col_reuse2 = st.columns(2)
            with col_reuse1:
                reuse_board_legacy = st.checkbox(
                    "Réutiliser board existant",
                    value=True,
                    help="Si coché, utilisera le board existant avec le même nom",
                    key="legacy_reuse_board"
                )
            with col_reuse2:
                reuse_group_legacy = st.checkbox(
                    "Réutiliser groupes existants",
                    value=True,
                    help="Si coché, réutilisera les groupes existants (structure de groupes préservée)",
                    key="legacy_reuse_group"
                )

            st.info("""
            **📌 Note importante sur les groupes:**

            La structure de groupes du board source sera automatiquement préservée.
            Si votre board source contient des groupes "Septembre" et "Octobre",
            ces mêmes groupes seront créés dans le nouveau board.
            """)

            st.markdown("---")

            # Data Processing Options
            st.subheader("4️⃣ Options de Traitement")
            aggregate_by_contract_legacy = st.checkbox(
                "Agréger par numéro de contrat",
                value=False,
                help="Si coché, les lignes avec le même numéro de contrat seront agrégées (somme des montants, moyenne des taux). Normalement désactivé pour préserver la structure originale du board.",
                key="legacy_aggregate_by_contract"
            )

            st.markdown("---")

            # Submit button
            submitted_legacy = st.form_submit_button(
                "🔄 Convertir le Board Monday.com",
                use_container_width=True,
                type="primary"
            )

            if submitted_legacy:
                # Validation
                errors = []

                # Get API key from session state
                api_key_from_state = st.session_state.monday_api_key

                if not source_board_id:
                    errors.append("❌ Veuillez sélectionner un board source")

                if not api_key_from_state:
                    errors.append("❌ Veuillez fournir une clé API Monday.com dans la barre latérale")

                if not board_name_input_legacy or not board_name_input_legacy.strip():
                    errors.append("❌ Veuillez fournir un nom pour le nouveau board")

                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    # Determine final board name
                    board_name_from_state = st.session_state.get('legacy_board_name', '')

                    if board_name_from_state and board_name_from_state.strip():
                        final_board_name = board_name_from_state.strip()
                    else:
                        final_board_name = "Commissions - Nouveau Format"

                    # Create configuration for MONDAY_LEGACY
                    try:
                        config = PipelineConfig(
                            source=InsuranceSource.MONDAY_LEGACY,
                            pdf_path=None,  # No PDF for Monday.com source
                            month_group=None,  # Groups are preserved from source board
                            board_name=final_board_name,
                            monday_api_key=api_key_from_state,
                            output_dir="./results/monday_legacy",
                            reuse_board=reuse_board_legacy,
                            reuse_group=reuse_group_legacy,
                            aggregate_by_contract=aggregate_by_contract_legacy,
                            source_board_id=int(source_board_id),
                            source_group_id=None,  # Always extract ALL groups (entire board)
                            target_board_type=None  # Auto-detected from source board
                        )

                        # Store in session state
                        st.session_state.pdf_file = None
                        st.session_state.pdf_path = None
                        st.session_state.config = config

                        # Move to next stage
                        st.session_state.stage = 2
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Erreur de configuration: {e}")


# =============================================================================
# STAGE 2: EXTRACTION AND PREVIEW
# =============================================================================

def render_stage_2():
    """Render data extraction and preview stage."""
    st.title("📊 Insurance Commission Data Pipeline")
    st.markdown("---")

    st.header("🔍 Étape 2: Extraction et Prévisualisation")

    # Show configuration summary
    config = st.session_state.config

    with st.expander("📋 Résumé de la Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Source", config.source.value)
            if config.source == InsuranceSource.MONDAY_LEGACY:
                st.metric("Board Source ID", config.source_board_id)
            else:
                st.metric("Fichier PDF", Path(config.pdf_path).name if config.pdf_path else "N/A")
        with col2:
            st.metric("Board Monday.com", config.board_name)
            st.metric("Groupe de Mois", config.month_group or "Aucun")
        with col3:
            st.metric("Réutiliser Board", "✅" if config.reuse_board else "❌")
            st.metric("Réutiliser Groupe", "✅" if config.reuse_group else "❌")

    st.markdown("---")

    # Extract data if not already done
    if st.session_state.extracted_data is None:
        source_type = "PDF" if config.source != InsuranceSource.MONDAY_LEGACY else "Monday.com"
        with st.spinner(f"🔄 Extraction des données en cours depuis {source_type}..."):
            try:
                # Create pipeline
                pipeline = InsuranceCommissionPipeline(config)

                # Execute Steps 1 and 2
                success_step1 = pipeline._step1_extract_data()
                if not success_step1:
                    st.error(f"❌ Échec de l'extraction des données depuis {source_type}")
                    if st.button("🔄 Recommencer"):
                        reset_pipeline()
                        st.rerun()
                    return

                success_step2 = pipeline._step2_process_data()
                if not success_step2:
                    st.error("❌ Échec du traitement des données")
                    if st.button("🔄 Recommencer"):
                        reset_pipeline()
                        st.rerun()
                    return

                # Store results
                st.session_state.extracted_data = pipeline.final_data
                st.session_state.pipeline = pipeline

                st.success("✅ Extraction réussie!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Erreur lors de l'extraction: {e}")
                with st.expander("Détails de l'erreur"):
                    st.exception(e)
                if st.button("🔄 Recommencer"):
                    reset_pipeline()
                    st.rerun()
                return

    # Display extracted data
    df = st.session_state.extracted_data

    if df is not None and not df.empty:
        # Show modification status if data was modified
        if st.session_state.data_modified:
            st.info("ℹ️ **Données modifiées** - Vous utilisez un fichier Excel uploadé au lieu des données extraites du PDF.")

        # Statistics
        st.subheader("📊 Statistiques")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Nombre de lignes", len(df))
        with col2:
            st.metric("Nombre de colonnes", len(df.columns))
        with col3:
            # Count non-null values in key columns
            if 'contract_number' in df.columns:
                non_null = df['contract_number'].notna().sum()
                st.metric("Contrats valides", non_null)
        with col4:
            # Check for duplicates
            duplicates = df.duplicated().sum()
            st.metric("Doublons", duplicates)

        st.markdown("---")

        # Show groups for MONDAY_LEGACY source
        if config.source == InsuranceSource.MONDAY_LEGACY:
            st.subheader("📁 Groupes du Board Source")

            try:
                # Get Monday.com client
                from monday_automation import MondayClient
                monday_client = MondayClient(api_key=config.monday_api_key)

                # List groups from source board
                with st.spinner("Chargement des groupes du board source..."):
                    all_groups = monday_client.list_groups(board_id=config.source_board_id)

                    # Filter out default "Group Title" groups
                    groups = [g for g in all_groups if g['title'] != 'Group Title']
                    filtered_count = len(all_groups) - len(groups)

                if groups and len(groups) > 0:
                    st.success(f"✅ {len(groups)} groupes trouvés dans le board source")

                    st.info("""
                    **📋 Ces groupes seront recréés dans le nouveau board:**

                    Les noms de groupes ci-dessous proviennent du board source Monday.com.
                    Chaque groupe sera automatiquement recréé avec le même nom dans le nouveau board,
                    et les items seront placés dans leur groupe d'origine respectif.
                    """)

                    # Prepare data for display
                    groups_display = pd.DataFrame([
                        {
                            "Nom du Groupe": group['title'],
                            "ID": group['id'],
                            "Couleur": group.get('color', 'N/A')
                        }
                        for group in groups
                    ])

                    # Display groups table
                    st.dataframe(
                        groups_display,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Nom du Groupe": st.column_config.TextColumn(
                                "Nom du Groupe",
                                help="Nom du groupe tel qu'il apparaît dans Monday.com",
                                width="large"
                            ),
                            "ID": st.column_config.TextColumn(
                                "ID",
                                help="Identifiant unique du groupe",
                                width="small"
                            ),
                            "Couleur": st.column_config.TextColumn(
                                "Couleur",
                                help="Code couleur du groupe",
                                width="small"
                            )
                        }
                    )

                    # Summary metric
                    st.metric("📂 Nombre de groupes à copier", len(groups))

                    # Show info about filtered groups if any
                    if filtered_count > 0:
                        st.caption(f"ℹ️ {filtered_count} groupe(s) par défaut 'Group Title' non affiché(s) (les items seront copiés dans le groupe par défaut du nouveau board)")

                else:
                    st.warning("⚠️ Aucun groupe trouvé dans le board source.")
                    st.info("Le board source ne contient aucun groupe, ou l'API n'a pas pu les récupérer.")

            except Exception as e:
                st.error(f"❌ Erreur lors de la récupération des groupes: {e}")
                st.info("Impossible de charger les groupes du board source.")

            st.markdown("---")

        st.markdown("---")

        # Data preview
        st.subheader("📋 Aperçu des Données")

        # Show column info
        with st.expander("ℹ️ Information sur les Colonnes"):
            col_info = pd.DataFrame({
                'Colonne': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.notna().sum(),
                'Null': df.isna().sum()
            })
            st.dataframe(col_info, use_container_width=True)

        # Interactive data viewer
        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )

        # Download option
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Télécharger les données (CSV)",
            data=csv,
            file_name=f"commissions_{config.source.value}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        st.markdown("---")

        # Excel Upload Option for manual corrections
        with st.expander("📤 Uploader un fichier Excel modifié (optionnel)", expanded=False):
            st.info("""
            **Modifier les données avant l'upload**

            Vous pouvez télécharger les données en CSV, les modifier dans Excel,
            puis uploader le fichier modifié ici. Le fichier uploadé remplacera
            les données extraites avant l'upload vers Monday.com.

            ⚠️ **Important**: Le fichier Excel doit contenir toutes les colonnes du tableau ci-dessus.
            """)

            excel_file = st.file_uploader(
                "Sélectionnez votre fichier Excel modifié",
                type=['xlsx', 'xls', 'csv'],
                help="Fichier Excel ou CSV avec les données corrigées",
                key="excel_upload"
            )

            if excel_file is not None:
                try:
                    # Read the uploaded file
                    if excel_file.name.endswith('.csv'):
                        uploaded_df = pd.read_csv(excel_file)
                    else:
                        uploaded_df = pd.read_excel(excel_file)

                    st.success(f"✅ Fichier chargé: {excel_file.name}")

                    # Validate columns
                    required_columns = set(df.columns)
                    uploaded_columns = set(uploaded_df.columns)

                    missing_columns = required_columns - uploaded_columns
                    extra_columns = uploaded_columns - required_columns

                    if missing_columns:
                        st.error(f"❌ Colonnes manquantes: {', '.join(missing_columns)}")
                    elif extra_columns:
                        st.warning(f"⚠️ Colonnes supplémentaires (seront ignorées): {', '.join(extra_columns)}")
                        # Keep only required columns
                        uploaded_df = uploaded_df[list(required_columns)]

                    # Show preview
                    st.subheader("📋 Aperçu du fichier uploadé")
                    st.dataframe(uploaded_df.head(10), use_container_width=True)

                    # Statistics comparison
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Lignes - Original", len(df))
                        st.metric("Lignes - Modifié", len(uploaded_df), delta=len(uploaded_df) - len(df))
                    with col2:
                        st.metric("Colonnes - Original", len(df.columns))
                        st.metric("Colonnes - Modifié", len(uploaded_df.columns))
                    with col3:
                        if not missing_columns:
                            st.success("✅ Structure valide")
                        else:
                            st.error("❌ Structure invalide")

                    # Button to replace data
                    if not missing_columns:
                        if st.button("✅ Utiliser ce fichier pour l'upload", type="primary", use_container_width=True):
                            st.session_state.extracted_data = uploaded_df
                            st.session_state.pipeline.final_data = uploaded_df
                            st.session_state.data_modified = True
                            st.success("✅ Données remplacées par le fichier uploadé!")
                            st.rerun()

                except Exception as e:
                    st.error(f"❌ Erreur lors de la lecture du fichier: {e}")

        st.markdown("---")

        # Action buttons
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.write("")  # Spacer

        with col2:
            if st.button("⬅️ Retour", use_container_width=True):
                reset_pipeline()
                st.rerun()

        with col3:
            if st.button(
                    "➡️ Continuer vers Monday.com",
                    use_container_width=True,
                    type="primary"
            ):
                st.session_state.stage = 3
                st.rerun()

    else:
        st.error("❌ Aucune donnée extraite")
        if st.button("🔄 Recommencer"):
            reset_pipeline()
            st.rerun()


# =============================================================================
# STAGE 3: UPLOAD TO MONDAY.COM
# =============================================================================

def render_stage_3():
    """Render Monday.com upload stage."""
    st.title("📊 Insurance Commission Data Pipeline")
    st.markdown("---")

    st.header("☁️ Étape 3: Upload vers Monday.com")

    # Show data summary
    df = st.session_state.extracted_data
    config = st.session_state.config

    # Show modification status if data was modified
    if st.session_state.data_modified:
        st.warning("⚠️ **Attention** - Vous allez uploader des données modifiées (fichier Excel uploadé)")

    st.subheader("📊 Résumé")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lignes à uploader", len(df))
    with col2:
        st.metric("Board cible", config.board_name)
    with col3:
        st.metric("Groupe", config.month_group or "Défaut")

    st.markdown("---")

    # Upload process
    if st.session_state.upload_results is None:
        st.warning("⚠️ Les données vont être uploadées vers Monday.com. Cette opération est irréversible.")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.stage = 2
                st.rerun()

        with col2:
            if st.button(
                    "🚀 Uploader vers Monday.com",
                    use_container_width=True,
                    type="primary"
            ):
                with st.spinner("☁️ Upload en cours vers Monday.com..."):
                    try:
                        pipeline = st.session_state.pipeline

                        # Execute Steps 3 and 4
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Step 3: Setup Monday board
                        status_text.text("Configuration du board Monday.com...")
                        progress_bar.progress(25)

                        try:
                            success_step3 = pipeline._step3_setup_monday_board()

                            if not success_step3:
                                st.error("❌ Échec de la configuration du board")
                                st.error("Vérifiez que votre clé API Monday.com est valide et que vous avez les permissions nécessaires.")
                                return
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la configuration du board: {e}")
                            with st.expander("Détails de l'erreur"):
                                st.exception(e)
                            return

                        progress_bar.progress(50)

                        # Step 4: Upload data using pipeline method (handles sequential group creation)
                        status_text.text("Upload des données vers Monday.com...")

                        # Check if this is a Monday.com conversion with multiple groups
                        is_monday_legacy = config.source == InsuranceSource.MONDAY_LEGACY
                        has_groups = hasattr(pipeline, 'groups_to_create') and pipeline.groups_to_create

                        if is_monday_legacy and has_groups:
                            # Sequential group creation and upload
                            total_groups = len(pipeline.groups_to_create)

                            # Execute step 4 which handles sequential group creation
                            success_step4 = pipeline._step4_upload_to_monday()

                            if not success_step4:
                                st.error("❌ Échec de l'upload des données")
                                return

                            # Get results from pipeline
                            results = []
                            if hasattr(pipeline, 'upload_results'):
                                results = pipeline.upload_results

                            progress_bar.progress(100)
                            status_text.text(f"Upload terminé - {total_groups} groupes créés")

                        else:
                            # Standard upload for PDF sources (single group)
                            items_to_create = pipeline._prepare_monday_items(df)
                            total_items = len(items_to_create)

                            # Upload in batches with progress updates
                            batch_size = 10
                            results = []

                            for i in range(0, total_items, batch_size):
                                batch = items_to_create[i:i + batch_size]

                                # Update status
                                status_text.text(f"Upload vers Monday.com... ({i + len(batch)}/{total_items} items)")

                                # Upload batch
                                batch_results = pipeline.monday_client.create_items_batch(
                                    board_id=pipeline.board_id,
                                    items=batch,
                                    group_id=pipeline.group_id
                                )
                                results.extend(batch_results)

                                # Update progress bar: 50% to 100%
                                progress_percent = 50 + int(50 * (i + len(batch)) / total_items)
                                progress_bar.progress(min(progress_percent, 100))

                            progress_bar.progress(100)

                        status_text.empty()

                        # Analyze results
                        successful = sum(1 for r in results if r.success)
                        failed = len(results) - successful
                        total_uploaded = len(results)

                        if successful > 0:
                            st.session_state.upload_results = {
                                'success': True,
                                'board_id': pipeline.board_id,
                                'group_id': pipeline.group_id,
                                'items_uploaded': successful,
                                'items_failed': failed
                            }
                            st.success(f"✅ Upload réussi! ({successful}/{total_uploaded} items)")
                            if failed > 0:
                                st.warning(f"⚠️ {failed} items ont échoué")
                            st.rerun()
                        else:
                            st.error("❌ Échec de l'upload des données - Aucun item n'a été uploadé")

                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'upload: {e}")
                        with st.expander("Détails de l'erreur"):
                            st.exception(e)

    else:
        # Show results
        results = st.session_state.upload_results

        if results['success']:
            st.success("✅ Upload terminé avec succès!")

            # Results details
            st.subheader("📈 Résultats")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Items uploadés", results['items_uploaded'])
            with col2:
                failed_count = results.get('items_failed', 0)
                st.metric("Items échoués", failed_count, delta=None if failed_count == 0 else -failed_count)
            with col3:
                st.metric("Board ID", results['board_id'])
            with col4:
                st.metric("Group ID", results['group_id'] or "Défaut")

            st.markdown("---")

            # Success message
            st.balloons()

            # Success message with details
            failed_count = results.get('items_failed', 0)
            if failed_count == 0:
                st.info(f"""
                🎉 **Upload réussi!**

                Les données ont été uploadées vers Monday.com avec succès.
                - Board: **{config.board_name}**
                - Groupe: **{config.month_group or 'Groupe par défaut'}**
                - Items créés: **{results['items_uploaded']}**

                Vous pouvez maintenant consulter vos données dans Monday.com.
                """)
            else:
                st.warning(f"""
                ⚠️ **Upload partiellement réussi**

                Les données ont été uploadées vers Monday.com avec quelques erreurs.
                - Board: **{config.board_name}**
                - Groupe: **{config.month_group or 'Groupe par défaut'}**
                - Items créés avec succès: **{results['items_uploaded']}**
                - Items échoués: **{failed_count}**

                Vérifiez vos données et permissions Monday.com.
                """)

            st.markdown("---")

            # Action buttons
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🔄 Nouveau Pipeline", use_container_width=True, type="primary"):
                    reset_pipeline()
                    st.rerun()

            with col2:
                # Link to Monday.com (if board_id is available)
                if results['board_id']:
                    monday_url = f"https://monday.com/boards/{results['board_id']}"
                    st.markdown(f"[🔗 Ouvrir dans Monday.com]({monday_url})")

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
    # Initialize session state
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        st.markdown("---")

        # Global API Key Configuration
        st.subheader("🔑 Configuration Monday.com")

        # Check if API key is already stored
        if st.session_state.monday_api_key:
            st.success("✅ Clé API configurée")

            # Button to change API key
            if st.button("🔄 Modifier la clé API", use_container_width=True):
                st.session_state.monday_api_key = None
                st.session_state.monday_boards = None  # Reset boards when changing API key
                st.rerun()
        else:
            # Input for API key
            api_key_input = st.text_input(
                "Clé API Monday.com",
                type="password",
                help="Votre clé API Monday.com pour l'authentification",
                key="global_monday_api_key_input"
            )

            # Save button
            if api_key_input:
                if st.button("💾 Enregistrer la clé API", use_container_width=True, type="primary"):
                    st.session_state.monday_api_key = api_key_input
                    st.success("✅ Clé API enregistrée!")
                    st.rerun()
            else:
                st.info("ℹ️ Entrez votre clé API Monday.com pour commencer")

        st.markdown("---")

        # Current stage indicator
        stage_names = {
            1: "📁 Configuration",
            2: "🔍 Prévisualisation",
            3: "☁️ Upload"
        }

        st.subheader("Étapes du Pipeline")
        for stage_num, stage_name in stage_names.items():
            if stage_num == st.session_state.stage:
                st.markdown(f"**➡️ {stage_name}**")
            elif stage_num < st.session_state.stage:
                st.markdown(f"✅ {stage_name}")
            else:
                st.markdown(f"⚪ {stage_name}")

        st.markdown("---")

        # Information section
        st.subheader("ℹ️ Informations")
        st.info("""
        **Pipeline de Commissions d'Assurance**

        Cette application permet de:
        1. Extraire les données de PDF ou Monday.com
        2. Visualiser les données extraites
        3. Uploader vers Monday.com

        **Sources supportées:**
        - UV Assurance (PDF)
        - IDC (PDF)
        - IDC Statement (PDF - Frais de suivi)
        - Assomption Vie (PDF)
        - Monday.com Legacy (conversion de board)
        """)

        st.markdown("---")

        # Reset button
        if st.button("🔄 Réinitialiser", use_container_width=True):
            reset_pipeline()
            st.rerun()

    # Render appropriate stage
    if st.session_state.stage == 1:
        render_stage_1()
    elif st.session_state.stage == 2:
        render_stage_2()
    elif st.session_state.stage == 3:
        render_stage_3()


if __name__ == "__main__":
    main()