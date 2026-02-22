"""
Stage 1: Configuration - PDF source selection and file upload.

Provides the configuration interface for selecting PDF sources,
uploading files, and choosing Monday.com destination boards.
"""

import streamlit as st

from src.pipeline import SourceType
from src.utils.data_unifier import BoardType
from src.utils.model_registry import get_available_models, get_default_vision_model

from src.app.state import get_pipeline
from src.app.utils.navigation import render_stepper, render_breadcrumb
from src.app.utils.board_utils import sort_and_filter_boards, load_boards_async
from src.app.utils.date_utils import detect_date_from_filename
from src.app.advisor.management import render_advisor_management_tab
from src.app.column_conversion.mode import render_column_conversion_mode


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
    tab1, tab2, tab3 = st.tabs(["Extraction PDF", "Gestion Conseillers", "Conversion Colonne"])

    with tab1:
        render_pdf_extraction_tab()

    with tab2:
        render_advisor_management_tab()

    with tab3:
        render_column_conversion_mode()


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

    # Board type selection FIRST
    st.markdown("### ⚙️ Type de table")

    type_options = ["Paiements Historiques", "Ventes et Production"]
    target_type = st.selectbox(
        "Type de table",
        options=type_options,
        key="pdf_target_type"
    )

    new_board_type = (
        BoardType.SALES_PRODUCTION if target_type == "Ventes et Production"
        else BoardType.HISTORICAL_PAYMENTS
    )

    # Default board names per type
    _DEFAULT_BOARD_FOR_TYPE = {
        "Paiements Historiques": "Paiement Historique",
        "Ventes et Production": "Ventes/Production",
    }

    # On board type change or first render, set the default board in session state
    if new_board_type != st.session_state.selected_board_type or "pdf_board_select" not in st.session_state:
        default_name = _DEFAULT_BOARD_FOR_TYPE.get(target_type)
        if default_name:
            st.session_state.pdf_board_select = default_name

    st.session_state.selected_board_type = new_board_type

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
            board_names = list(board_options.keys())

            selected_name = st.selectbox(
                "Board de destination",
                options=board_names,
                key="pdf_board_select"
            )
            st.session_state.selected_board_id = board_options[selected_name]
            st.session_state._current_board_name = selected_name
        else:
            st.warning("Aucun board trouvé")
    else:
        st.warning("⚠️ Configurez la clé API Monday.com dans la sidebar")

    # Model selection
    st.markdown("#### 🤖 Modèle d'extraction")
    available_models = get_available_models()
    default_model = get_default_vision_model()

    # Build options list with display names
    model_options = list(available_models.keys())
    model_labels = list(available_models.values())

    # Find default model index
    default_idx = model_options.index(default_model) if default_model in model_options else 0

    # Get current selection or use default
    current_model = st.session_state.get("selected_model", default_model)
    current_idx = model_options.index(current_model) if current_model in model_options else default_idx

    selected_model_idx = st.selectbox(
        "Modèle VLM",
        options=range(len(model_options)),
        format_func=lambda i: model_labels[i],
        index=current_idx,
        help="Le modèle utilisé pour l'extraction des données. Gemini 3 Flash est rapide et économique.",
        key="model_select"
    )

    st.session_state.selected_model = model_options[selected_model_idx]

    # Show model info
    if model_options[selected_model_idx] == default_model:
        st.caption("✅ Modèle par défaut - Rapide et économique")
    elif "pro" in model_options[selected_model_idx].lower():
        st.caption("🎯 Haute précision - Pour documents complexes")

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
