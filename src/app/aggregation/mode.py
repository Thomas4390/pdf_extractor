"""
Aggregation mode UI components.

Provides the 3-step wizard interface for data aggregation:
1. Configuration (target board + auto-load data)
2. Period & Preview (real-time period changes with live preview)
3. Execute upsert
"""

import pandas as pd
import streamlit as st

from src.utils.aggregator import (
    DatePeriod,
    SOURCE_BOARDS,
    get_group_name_for_period,
    get_period_date_range,
)
from src.app.aggregation_ui import (
    render_target_board_selection,
    render_source_data_preview,
    render_combined_preview,
    render_editable_preview,
    render_execution_summary,
    render_execution_result,
    render_navigation_buttons,
)
from src.app.aggregation.execution import (
    load_source_data,
    filter_and_aggregate_data,
    execute_aggregation_upsert,
)


def render_aggregation_stepper(current_step: int) -> None:
    """
    Render a visual progress stepper for the 3-step wizard.

    Args:
        current_step: Current step (1-3)
    """
    steps = [
        ("1", "Configuration", "⚙️"),
        ("2", "Période & Aperçu", "📊"),
        ("3", "Exécution", "🚀"),
    ]

    cols = st.columns(len(steps))
    for idx, (num, label, icon) in enumerate(steps):
        step_num = idx + 1
        with cols[idx]:
            is_current = step_num == current_step
            is_completed = step_num < current_step
            is_future = step_num > current_step

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
                <div class="step-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

            # Add clickable button for completed stages
            if is_completed:
                if st.button(f"← Retour", key=f"agg_stepper_nav_{step_num}", width="stretch"):
                    st.session_state.agg_step = step_num
                    st.rerun()

    st.markdown("---")


def render_aggregation_mode() -> None:
    """Main entry point for aggregation mode."""
    st.title("📊 Agrégation Data")
    st.markdown("Agrégez les données de plusieurs boards Monday.com par conseiller.")

    # Check prerequisites
    if not st.session_state.monday_api_key:
        st.warning("⚠️ Veuillez vous connecter à Monday.com dans la barre latérale.")
        return

    if not st.session_state.monday_boards:
        st.info("📥 Chargement des boards en cours...")
        return

    # Render stepper
    render_aggregation_stepper(st.session_state.agg_step)

    # Route to appropriate step
    if st.session_state.agg_step == 1:
        render_agg_step_1_config()
    elif st.session_state.agg_step == 2:
        render_agg_step_2_period_preview()
    else:
        render_agg_step_3_execute()


def render_agg_step_1_config() -> None:
    """Step 1: Configuration - Target board selection + auto-load data."""
    boards = st.session_state.monday_boards

    # Show which sources are configured
    st.subheader("📋 Sources configurées")

    source_info = []
    for source_key, board_id in st.session_state.agg_selected_sources.items():
        config = SOURCE_BOARDS.get(source_key)
        if config:
            # Find board name
            board_name = f"ID: {board_id}"
            for b in boards:
                if int(b["id"]) == board_id:
                    board_name = b["name"]
                    break
            source_info.append({
                "Source": config.display_name,
                "Board": board_name,
                "Colonne agrégée": config.aggregate_column,
            })

    if source_info:
        st.dataframe(
            pd.DataFrame(source_info),
            hide_index=True,
            width="stretch",
        )
    else:
        st.warning("Aucune source configurée.")
        return

    st.markdown("---")

    # Target board selection
    target_board_id = render_target_board_selection(
        boards=boards,
        current_selection=st.session_state.agg_target_board_id,
    )
    st.session_state.agg_target_board_id = target_board_id

    st.markdown("---")

    # Data loading status and controls
    st.subheader("📥 Chargement des données")

    data_loaded = st.session_state.get("agg_data_loaded", False)
    source_data = st.session_state.get("agg_source_data", {})

    if data_loaded and source_data:
        # Show data summary
        total_rows = sum(len(df) for df in source_data.values())
        st.success(f"✅ Données chargées: {total_rows} lignes au total")

        # Show per-source counts
        cols = st.columns(len(source_data))
        for idx, (source_key, df) in enumerate(source_data.items()):
            config = SOURCE_BOARDS.get(source_key)
            if config:
                with cols[idx]:
                    st.metric(config.display_name, f"{len(df)} lignes")

        # Button to reload
        if st.button("🔄 Recharger les données", type="secondary"):
            st.session_state.agg_data_loaded = False
            load_source_data()
            st.rerun()
    else:
        st.info("Les données seront chargées automatiquement.")
        # Auto-load data
        if st.session_state.agg_selected_sources:
            load_source_data()
            st.rerun()

    # Navigation
    can_proceed = (
        target_board_id is not None
        and st.session_state.get("agg_data_loaded", False)
    )

    go_back, go_next = render_navigation_buttons(
        current_step=1,
        max_step=3,
        can_proceed=can_proceed,
    )

    if go_next:
        # Set default period if not set
        if st.session_state.agg_period is None:
            st.session_state.agg_period = DatePeriod.LAST_MONTH
        # Filter and aggregate with the period
        filter_and_aggregate_data()
        st.session_state.agg_step = 2
        st.rerun()


def render_agg_step_2_period_preview() -> None:
    """Step 2: Period selection with live preview."""
    boards = st.session_state.monday_boards
    target_board_id = st.session_state.agg_target_board_id

    # Get current period (default to LAST_MONTH)
    current_period = st.session_state.agg_period
    if current_period is None:
        current_period = DatePeriod.LAST_MONTH
        st.session_state.agg_period = current_period

    # Period selection header
    st.subheader("📅 Période de filtrage")

    # Period selector as pill buttons
    period_cols = st.columns(len(DatePeriod))
    for idx, period in enumerate(DatePeriod):
        with period_cols[idx]:
            is_selected = period == current_period
            btn_type = "primary" if is_selected else "secondary"
            if st.button(
                period.display_name,
                key=f"period_btn_{period.value}",
                type=btn_type,
                width="stretch",
            ):
                if period != current_period:
                    st.session_state.agg_period = period
                    # Re-filter and aggregate with new period (instant, no API call)
                    filter_and_aggregate_data()
                    st.rerun()

    # Show date range info
    start_date, end_date = get_period_date_range(current_period)
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📆 **Du** {start_date.strftime('%d/%m/%Y')} **au** {end_date.strftime('%d/%m/%Y')}")
    with col2:
        # Group name options
        use_custom_group = st.session_state.get("agg_use_custom_group", False)
        custom_group_name = st.session_state.get("agg_custom_group_name", "")
        auto_group_name = get_group_name_for_period(current_period)

        group_mode = st.radio(
            "Groupe cible",
            options=["auto", "manual"],
            format_func=lambda x: f"🔄 {auto_group_name}" if x == "auto" else "✏️ Personnalisé",
            index=1 if use_custom_group else 0,
            key="agg_group_mode_step2",
            horizontal=True,
        )

        st.session_state.agg_use_custom_group = (group_mode == "manual")

        if group_mode == "manual":
            custom_name = st.text_input(
                "Nom du groupe",
                value=custom_group_name if custom_group_name else auto_group_name,
                key="agg_custom_group_input_step2",
                label_visibility="collapsed",
            )
            st.session_state.agg_custom_group_name = custom_name
            final_group_name = custom_name
        else:
            final_group_name = auto_group_name

    st.markdown("---")

    # Preview section
    combined_df = st.session_state.agg_combined_data

    # Create tabs for different views
    tab_combined, tab_sources, tab_upload = st.tabs([
        "📋 Résumé",
        "📊 Détail par source",
        "📤 Aperçu upload"
    ])

    # Tab 1: Combined summary (main view)
    with tab_combined:
        if combined_df is not None and not combined_df.empty:
            # Quick stats
            advisor_count = len(combined_df)
            numeric_cols = [col for col in combined_df.columns if col != "Conseiller"]

            stat_cols = st.columns(len(numeric_cols) + 1)
            with stat_cols[0]:
                st.metric("Conseillers", advisor_count)
            for idx, col in enumerate(numeric_cols):
                with stat_cols[idx + 1]:
                    total = combined_df[col].sum()
                    st.metric(f"Total {col}", f"{total:,.2f}")

            # Show info about filtered unknown advisors
            unknown_count = st.session_state.get("agg_unknown_advisors_count", 0)
            if unknown_count > 0:
                st.info(
                    f"ℹ️ **{unknown_count} ligne(s) ignorée(s)** : conseillers non trouvés dans la base de données. "
                    "Seuls les conseillers enregistrés dans la gestion des conseillers sont inclus."
                )

            st.markdown("---")
            render_combined_preview(combined_df)
        else:
            st.warning("Aucune donnée pour cette période.")

    # Tab 2: Source-by-source detail
    with tab_sources:
        for source_key in st.session_state.agg_selected_sources.keys():
            config = SOURCE_BOARDS.get(source_key)
            if not config:
                continue

            source_df = st.session_state.agg_source_data.get(source_key, pd.DataFrame())
            filtered_df = st.session_state.agg_filtered_data.get(source_key, pd.DataFrame())
            aggregated_df = st.session_state.agg_aggregated_data.get(source_key, pd.DataFrame())

            render_source_data_preview(
                source_key=source_key,
                config=config,
                df=source_df,
                filtered_df=filtered_df,
                aggregated_df=aggregated_df,
            )

    # Tab 3: Editable upload preview
    with tab_upload:
        # Get target board name
        target_board_name = "Non sélectionné"
        for board in boards:
            if int(board["id"]) == target_board_id:
                target_board_name = board["name"]
                break

        if combined_df is not None and not combined_df.empty:
            edited_df = render_editable_preview(
                combined_df=combined_df,
                target_board_name=target_board_name,
                group_name=final_group_name,
            )
            # Store edited data for use in execution
            st.session_state.agg_edited_data = edited_df
        else:
            st.warning("Aucune donnée à uploader.")

    st.markdown("---")

    # Navigation
    has_data = combined_df is not None and not combined_df.empty
    has_group = bool(final_group_name.strip()) if st.session_state.agg_use_custom_group else True

    go_back, go_next = render_navigation_buttons(
        current_step=2,
        max_step=3,
        can_proceed=has_data and has_group,
    )

    if go_back:
        st.session_state.agg_step = 1
        st.rerun()

    if go_next:
        st.session_state.agg_step = 3
        st.rerun()


def render_agg_step_3_execute() -> None:
    """Step 3: Execute upsert to target board."""
    boards = st.session_state.monday_boards
    target_board_id = st.session_state.agg_target_board_id
    period = st.session_state.agg_period
    combined_df = st.session_state.agg_combined_data

    # Get target board name
    target_board_name = "Unknown"
    for board in boards:
        if int(board["id"]) == target_board_id:
            target_board_name = board["name"]
            break

    # Get group name - use custom if specified
    if st.session_state.get("agg_use_custom_group") and st.session_state.get("agg_custom_group_name"):
        group_name = st.session_state.agg_custom_group_name
    else:
        group_name = get_group_name_for_period(period)

    advisor_count = len(combined_df) if combined_df is not None else 0
    sources_count = len(st.session_state.agg_selected_sources)

    render_execution_summary(
        target_board_name=target_board_name,
        group_name=group_name,
        advisor_count=advisor_count,
        sources_count=sources_count,
    )

    # Show previous result if exists
    if st.session_state.agg_upsert_result:
        render_execution_result(st.session_state.agg_upsert_result)

    # Navigation
    go_back, go_next = render_navigation_buttons(
        current_step=3,
        max_step=3,
        can_proceed=not st.session_state.agg_is_executing,
    )

    if go_back:
        st.session_state.agg_step = 2
        st.session_state.agg_upsert_result = None
        st.rerun()

    if go_next and not st.session_state.agg_is_executing:
        execute_aggregation_upsert()
