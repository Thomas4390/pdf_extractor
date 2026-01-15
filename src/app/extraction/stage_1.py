"""
Stage 1: Configuration - PDF source selection and file upload.

Provides the configuration interface for selecting PDF sources,
uploading files, and choosing Monday.com destination boards.
"""

import streamlit as st

from src.pipeline import SourceType
from src.utils.data_unifier import BoardType

from src.app.state import get_pipeline
from src.app.utils.navigation import render_stepper, render_breadcrumb
from src.app.utils.board_utils import sort_and_filter_boards, detect_board_type_from_name, load_boards_async
from src.app.utils.date_utils import detect_date_from_filename
from src.app.advisor.management import render_advisor_management_tab


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
