"""
Column Conversion Mode rendering.

Provides the UI for converting label/status columns to dropdown columns.
The original column is renamed to "{name} old" and a new dropdown column
is created with the original values migrated.

Features:
- Automatic advisor name mapping using the advisor database
- Concurrent API calls for faster migration
"""

import streamlit as st

from src.clients.monday import MondayClient, MondayError


def _get_name_mapper():
    """Get the advisor name mapper function if available.

    Returns:
        A function that maps first names to full names, or None if not configured.
    """
    try:
        from src.utils.advisor_matcher import get_advisor_matcher

        matcher = get_advisor_matcher()
        if matcher.is_configured:
            # Return the mapping function
            return matcher.match_compact_or_original
        return None
    except Exception:
        return None


def render_column_conversion_mode() -> None:
    """Main entry point for column conversion mode."""
    st.title("🔄 Conversion de Colonne")
    st.markdown("""
    Convertissez une colonne d'étiquettes (status/text) en menu déroulant (dropdown).

    **Processus:**
    1. L'ancienne colonne sera renommée en `{nom} old`
    2. Une nouvelle colonne dropdown sera créée avec le nom original
    3. Toutes les valeurs seront copiées vers la nouvelle colonne
    """)

    # Check API connection
    if not st.session_state.monday_api_key:
        st.warning("Veuillez d'abord vous connecter à Monday.com via la barre latérale.")
        return

    # Check if boards are loaded
    if not st.session_state.monday_boards:
        st.info("Chargement des boards en cours...")
        return

    st.markdown("---")

    # Step 1: Select board
    _render_board_selector()

    # Step 2: Select column (only if board is selected)
    if st.session_state.conv_board_id:
        _render_column_selector()

    # Step 3: Execute (only if column is selected)
    if st.session_state.conv_board_id and st.session_state.conv_column_id:
        _render_execution_section()

    # Show result if available
    if st.session_state.conv_result:
        _render_result()


def _render_board_selector() -> None:
    """Render the board selection section with keyword search."""
    from src.app.utils.board_utils import sort_and_filter_boards

    st.subheader("1. Sélectionner le Board")

    boards = st.session_state.monday_boards or []

    # Search filter
    search = st.text_input(
        "🔍 Rechercher un board",
        placeholder="Filtrer par nom...",
        key="conv_search_board"
    )

    # Filter and sort boards
    filtered_boards = sort_and_filter_boards(boards, search)

    if not filtered_boards:
        st.warning("Aucun board trouvé avec ce filtre.")
        return

    # Build options
    board_options = {b['name']: int(b['id']) for b in filtered_boards}

    # Find current selection in options
    current_board_name = None
    if st.session_state.conv_board_id:
        for name, bid in board_options.items():
            if bid == st.session_state.conv_board_id:
                current_board_name = name
                break

    selected = st.selectbox(
        "Board Monday.com",
        options=list(board_options.keys()),
        index=list(board_options.keys()).index(current_board_name) if current_board_name else 0,
        key="conv_board_select"
    )

    if selected:
        new_board_id = board_options[selected]
        if new_board_id != st.session_state.conv_board_id:
            st.session_state.conv_board_id = new_board_id
            st.session_state.conv_column_id = None  # Reset column on board change
            st.session_state.conv_result = None
            st.rerun()


def _render_column_selector() -> None:
    """Render the column selection section."""
    st.subheader("2. Sélectionner la Colonne")

    # Load columns for the selected board
    try:
        client = MondayClient(api_key=st.session_state.monday_api_key)
        columns = client.list_columns_sync(st.session_state.conv_board_id)
    except MondayError as e:
        st.error(f"Erreur lors du chargement des colonnes: {e}")
        return

    # Filter to columns that can be converted (exclude read-only types)
    convertible_types = {"status", "color", "text", "short_text"}
    convertible_columns = [
        col for col in columns
        if col["type"] in convertible_types
    ]

    if not convertible_columns:
        st.warning("Aucune colonne convertible trouvée sur ce board.")
        return

    # Build options
    column_options = {
        f"{col['title']} ({col['type']})": col
        for col in convertible_columns
    }

    # Default to "Conseiller" if available
    default_index = 0
    for idx, col in enumerate(convertible_columns):
        if col["title"].lower() == st.session_state.conv_column_title.lower():
            default_index = idx
            break

    selected = st.selectbox(
        "Colonne à convertir",
        options=list(column_options.keys()),
        index=default_index,
        key="conv_column_select",
        help="La colonne sera convertie en menu déroulant (dropdown)"
    )

    if selected:
        col_info = column_options[selected]
        st.session_state.conv_column_id = col_info["id"]
        st.session_state.conv_column_title = col_info["title"]

        # Show column info
        st.info(f"""
        **Colonne sélectionnée:** {col_info['title']}
        **Type actuel:** {col_info['type']}
        **ID:** {col_info['id']}
        """)


def _render_execution_section() -> None:
    """Render the execution section with preview and button."""
    st.subheader("3. Exécuter la Conversion")

    col_title = st.session_state.conv_column_title

    # Check if name mapping is available
    name_mapper = _get_name_mapper()
    mapping_available = name_mapper is not None

    # Mapping options
    st.markdown("#### Options de mapping")

    if mapping_available:
        use_mapping = st.checkbox(
            "🔗 Mapper les prénoms vers les noms complets",
            value=True,
            help="Utilise la base de données des conseillers pour convertir les prénoms (ex: Thomas → Thomas, L)"
        )

        if use_mapping:
            st.success("✅ Le mapping des noms est activé. Les prénoms seront convertis en format compact (Prénom, Initiale).")

            # Show mapping preview
            with st.expander("📋 Aperçu du mapping disponible", expanded=False):
                try:
                    from src.utils.advisor_matcher import get_advisor_matcher
                    matcher = get_advisor_matcher()
                    advisors = matcher.get_all_advisors()

                    if advisors:
                        preview_data = [
                            {"Prénom": a.first_name, "Nom complet": a.full_name, "Format compact": a.display_name_compact}
                            for a in advisors[:20]  # Show first 20
                        ]
                        st.dataframe(preview_data, use_container_width=True, hide_index=True)
                        if len(advisors) > 20:
                            st.caption(f"... et {len(advisors) - 20} autres conseillers")
                    else:
                        st.info("Aucun conseiller dans la base de données.")
                except Exception as e:
                    st.warning(f"Impossible de charger l'aperçu: {e}")
    else:
        use_mapping = False
        st.info("ℹ️ Le mapping des noms n'est pas disponible (Google Sheets non configuré). "
                "Les valeurs seront copiées telles quelles.")

    st.markdown("---")

    # Preview of changes
    st.markdown(f"""
    **Changements prévus:**
    - La colonne `{col_title}` sera renommée en `{col_title} old`
    - Une nouvelle colonne dropdown `{col_title}` sera créée
    - Toutes les valeurs existantes seront {"mappées puis " if use_mapping else ""}copiées vers la nouvelle colonne
    """)

    st.warning("⚠️ Cette opération ne peut pas être annulée automatiquement. "
               "Assurez-vous de vouloir continuer.")

    # Store mapping preference in session state
    st.session_state.conv_use_mapping = use_mapping

    # Execution button
    if st.session_state.conv_is_executing:
        st.info("Conversion en cours...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current: int, total: int, message: str) -> None:
            progress_bar.progress(current / total)
            status_text.text(message)

        try:
            client = MondayClient(api_key=st.session_state.monday_api_key)

            # Get name mapper if enabled
            active_mapper = name_mapper if st.session_state.get("conv_use_mapping", False) else None

            result = client.migrate_column_to_dropdown_sync(
                board_id=st.session_state.conv_board_id,
                source_column_id=st.session_state.conv_column_id,
                source_column_title=st.session_state.conv_column_title,
                progress_callback=update_progress,
                name_mapper=active_mapper,
                max_concurrent=10  # 10 concurrent updates for speed
            )
            st.session_state.conv_result = result
            st.session_state.conv_is_executing = False
            st.rerun()
        except Exception as e:
            st.session_state.conv_is_executing = False
            st.error(f"Erreur lors de la conversion: {e}")
    else:
        if st.button("🚀 Lancer la Conversion", type="primary", use_container_width=True):
            st.session_state.conv_result = None
            st.session_state.conv_is_executing = True
            st.rerun()


def _render_result() -> None:
    """Render the migration result."""
    st.markdown("---")
    st.subheader("Résultat de la Conversion")

    result = st.session_state.conv_result

    if result.get("success"):
        items_migrated = result.get('items_migrated', 0)
        values_mapped = result.get('values_mapped', 0)
        new_column_id = result.get('new_column_id', 'N/A')

        success_msg = f"""
        **Conversion réussie!**
        - Éléments migrés: {items_migrated}
        - Nouvelle colonne ID: {new_column_id}
        """

        if values_mapped > 0:
            success_msg += f"\n        - Noms mappés: {values_mapped}"

        st.success(success_msg)

        if values_mapped > 0:
            st.info(f"🔗 {values_mapped} prénoms ont été convertis vers leur format complet (Prénom, Initiale).")
    else:
        st.error("La conversion a rencontré des problèmes.")

    if result.get("errors"):
        with st.expander(f"Voir les erreurs ({len(result['errors'])})", expanded=False):
            for error in result["errors"]:
                st.text(f"- {error}")

    # Reset button
    if st.button("🔄 Nouvelle Conversion", use_container_width=True):
        st.session_state.conv_board_id = None
        st.session_state.conv_column_id = None
        st.session_state.conv_result = None
        st.session_state.conv_use_mapping = True
        st.rerun()
