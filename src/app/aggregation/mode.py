"""
Aggregation mode UI components.

Provides the 4-step wizard interface for data aggregation:
1. Source and target board selection
2. Date period selection
3. Preview aggregated data
4. Execute upsert
"""

import pandas as pd
import streamlit as st

from src.utils.aggregator import (
    DatePeriod,
    SOURCE_BOARDS,
    get_group_name_for_period,
    filter_by_date,
    aggregate_by_advisor,
    combine_aggregations,
)
from src.app.aggregation_ui import (
    render_aggregation_stepper,
    render_source_selection,
    render_target_board_selection,
    render_period_selection,
    render_source_data_preview,
    render_combined_preview,
    render_editable_preview,
    render_execution_summary,
    render_execution_result,
    render_navigation_buttons,
)
from src.app.aggregation.execution import load_and_aggregate_data, execute_aggregation_upsert


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
        render_agg_step_1_sources()
    elif st.session_state.agg_step == 2:
        render_agg_step_2_period()
    elif st.session_state.agg_step == 3:
        render_agg_step_3_preview()
    else:
        render_agg_step_4_execute()


def render_agg_step_1_sources() -> None:
    """Step 1: Source and target board selection."""
    boards = st.session_state.monday_boards

    # Source selection
    updated_sources = render_source_selection(
        boards=boards,
        selected_sources=st.session_state.agg_selected_sources,
    )
    st.session_state.agg_selected_sources = updated_sources

    st.markdown("---")

    # Target board selection
    target_board_id = render_target_board_selection(
        boards=boards,
        current_selection=st.session_state.agg_target_board_id,
    )
    st.session_state.agg_target_board_id = target_board_id

    # Navigation
    can_proceed = bool(updated_sources) and target_board_id is not None
    go_back, go_next = render_navigation_buttons(
        current_step=1,
        can_proceed=can_proceed,
    )

    if go_next:
        st.session_state.agg_step = 2
        st.rerun()


def render_agg_step_2_period() -> None:
    """Step 2: Date period and group selection."""
    current_period = st.session_state.agg_period
    if current_period is None:
        current_period = DatePeriod.LAST_MONTH

    # Get current custom group settings
    use_custom_group = st.session_state.get("agg_use_custom_group", False)
    custom_group_name = st.session_state.get("agg_custom_group_name", "")

    # Render period selection with group options
    selected_period, use_custom, group_name = render_period_selection(
        current_period=current_period,
        use_custom_group=use_custom_group,
        custom_group_name=custom_group_name,
    )

    # Update session state
    st.session_state.agg_period = selected_period
    st.session_state.agg_use_custom_group = use_custom
    st.session_state.agg_custom_group_name = group_name

    # Determine if can proceed (need group name if using custom)
    can_proceed = True
    if use_custom and not group_name.strip():
        can_proceed = False

    # Navigation
    go_back, go_next = render_navigation_buttons(
        current_step=2,
        can_proceed=can_proceed,
        next_label="📥 Charger et agréger →",
    )

    if go_back:
        st.session_state.agg_step = 1
        st.rerun()

    if go_next:
        # Load and aggregate data
        load_and_aggregate_data()
        st.session_state.agg_step = 3
        st.rerun()


def render_agg_step_3_preview() -> None:
    """Step 3: Preview aggregated data with tabs."""
    st.subheader("📋 Aperçu des données agrégées")

    combined_df = st.session_state.agg_combined_data
    boards = st.session_state.monday_boards
    target_board_id = st.session_state.agg_target_board_id
    period = st.session_state.agg_period

    # Quick period selector for dynamic update
    st.markdown("**📅 Période de filtrage :**")
    period_cols = st.columns(len(DatePeriod))
    for idx, p in enumerate(DatePeriod):
        with period_cols[idx]:
            is_selected = p == period
            btn_type = "primary" if is_selected else "secondary"
            if st.button(
                p.display_name,
                key=f"step3_period_{p.value}",
                type=btn_type,
                use_container_width=True,
            ):
                if p != period:
                    st.session_state.agg_period = p
                    # Reload data with new period
                    load_and_aggregate_data()
                    st.rerun()

    st.markdown("---")

    # Get target board name
    target_board_name = "Non sélectionné"
    for board in boards:
        if int(board["id"]) == target_board_id:
            target_board_name = board["name"]
            break

    # Get group name
    if st.session_state.get("agg_use_custom_group") and st.session_state.get("agg_custom_group_name"):
        group_name = st.session_state.agg_custom_group_name
    else:
        group_name = get_group_name_for_period(period)

    # Create tabs for different views
    tab_sources, tab_combined, tab_upload = st.tabs([
        "📊 Par source",
        "📋 Vue combinée",
        "📤 Aperçu upload"
    ])

    # Tab 1: Source-by-source preview
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

    # Tab 2: Combined view (read-only)
    with tab_combined:
        if combined_df is not None and not combined_df.empty:
            render_combined_preview(combined_df)
        else:
            st.warning("Aucune donnée combinée disponible.")

    # Tab 3: Editable upload preview
    with tab_upload:
        if combined_df is not None and not combined_df.empty:
            edited_df = render_editable_preview(
                combined_df=combined_df,
                target_board_name=target_board_name,
                group_name=group_name,
            )
            # Store edited data for use in execution
            st.session_state.agg_edited_data = edited_df
        else:
            st.warning("Aucune donnée à uploader.")

    st.markdown("---")

    # Navigation
    has_data = combined_df is not None and not combined_df.empty
    go_back, go_next = render_navigation_buttons(
        current_step=3,
        can_proceed=has_data,
    )

    if go_back:
        st.session_state.agg_step = 2
        st.rerun()

    if go_next:
        st.session_state.agg_step = 4
        st.rerun()


def render_agg_step_4_execute() -> None:
    """Step 4: Execute upsert to target board."""
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
        current_step=4,
        can_proceed=not st.session_state.agg_is_executing,
    )

    if go_back:
        st.session_state.agg_step = 3
        st.session_state.agg_upsert_result = None
        st.rerun()

    if go_next and not st.session_state.agg_is_executing:
        execute_aggregation_upsert()
