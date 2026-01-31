"""
Column Conversion Mode rendering.

Provides the UI for converting label/status columns to dropdown columns.
The original column is renamed to "{name} old" and a new dropdown column
is created with the original values migrated.
"""

import streamlit as st

from src.clients.monday import MondayClient, MondayError


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
    """Render the board selection section."""
    st.subheader("1. Sélectionner le Board")

    boards = st.session_state.monday_boards or []
    board_options = {f"{b['name']} (ID: {b['id']})": int(b['id']) for b in boards}

    # Find current selection in options
    current_board_name = None
    if st.session_state.conv_board_id:
        for name, bid in board_options.items():
            if bid == st.session_state.conv_board_id:
                current_board_name = name
                break

    selected = st.selectbox(
        "Board Monday.com",
        options=["-- Sélectionner un board --"] + list(board_options.keys()),
        index=0 if current_board_name is None else list(board_options.keys()).index(current_board_name) + 1,
        key="conv_board_select"
    )

    if selected and selected != "-- Sélectionner un board --":
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

    # Preview of changes
    st.markdown(f"""
    **Changements prévus:**
    - La colonne `{col_title}` sera renommée en `{col_title} old`
    - Une nouvelle colonne dropdown `{col_title}` sera créée
    - Toutes les valeurs existantes seront copiées vers la nouvelle colonne
    """)

    st.warning("Cette opération ne peut pas être annulée automatiquement. "
               "Assurez-vous de vouloir continuer.")

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
            result = client.migrate_column_to_dropdown_sync(
                board_id=st.session_state.conv_board_id,
                source_column_id=st.session_state.conv_column_id,
                source_column_title=st.session_state.conv_column_title,
                progress_callback=update_progress
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
        st.success(f"""
        **Conversion réussie!**
        - Éléments migrés: {result.get('items_migrated', 0)}
        - Nouvelle colonne ID: {result.get('new_column_id', 'N/A')}
        """)
    else:
        st.error("La conversion a rencontré des problèmes.")

    if result.get("errors"):
        with st.expander("Voir les erreurs", expanded=False):
            for error in result["errors"]:
                st.text(f"- {error}")

    # Reset button
    if st.button("🔄 Nouvelle Conversion", use_container_width=True):
        st.session_state.conv_board_id = None
        st.session_state.conv_column_id = None
        st.session_state.conv_result = None
        st.rerun()
