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
from src.app.aggregation.charts import render_charts_tab


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
        # Set default period if not set (MONTH_1 = last month)
        if st.session_state.agg_period is None:
            st.session_state.agg_period = DatePeriod.MONTH_1
        # Filter and aggregate with the period
        filter_and_aggregate_data()
        st.session_state.agg_step = 2
        st.rerun()


def render_agg_step_2_period_preview() -> None:
    """Step 2: Period selection with live preview."""
    boards = st.session_state.monday_boards
    target_board_id = st.session_state.agg_target_board_id

    # Get current period (default to MONTH_1 = last month)
    current_period = st.session_state.agg_period
    if current_period is None:
        current_period = DatePeriod.MONTH_1
        st.session_state.agg_period = current_period

    # Period selection header
    st.subheader("📅 Sélection du mois")

    # Create month options for selectbox
    month_options = list(DatePeriod)
    month_labels = {period: period.display_name for period in month_options}

    # Find current index
    current_index = month_options.index(current_period) if current_period in month_options else 1

    # Month selector as selectbox
    col_select, col_info = st.columns([2, 3])

    with col_select:
        selected_period = st.selectbox(
            "Mois à agréger",
            options=month_options,
            index=current_index,
            format_func=lambda p: f"{p.display_name} ({p.short_label})",
            key="agg_month_selector",
            label_visibility="collapsed",
        )

        if selected_period != current_period:
            st.session_state.agg_period = selected_period
            # Re-filter and aggregate with new period (instant, no API call)
            filter_and_aggregate_data()
            st.rerun()

    with col_info:
        # Show date range info
        start_date, end_date = get_period_date_range(selected_period)
        st.info(f"📆 **Du** {start_date.strftime('%d/%m/%Y')} **au** {end_date.strftime('%d/%m/%Y')}")

    # Group name options
    use_custom_group = st.session_state.get("agg_use_custom_group", False)
    custom_group_name = st.session_state.get("agg_custom_group_name", "")
    auto_group_name = get_group_name_for_period(selected_period)

    col_group1, col_group2 = st.columns([1, 2])
    with col_group1:
        group_mode = st.radio(
            "Groupe cible",
            options=["auto", "manual"],
            format_func=lambda x: f"🔄 Auto ({auto_group_name})" if x == "auto" else "✏️ Personnalisé",
            index=1 if use_custom_group else 0,
            key="agg_group_mode_step2",
            horizontal=True,
        )
        st.session_state.agg_use_custom_group = (group_mode == "manual")

    with col_group2:
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
            st.markdown(f"**Groupe:** {final_group_name}")

    st.markdown("---")

    # Preview section
    combined_df = st.session_state.agg_combined_data

    # Create tabs for different views
    tab_combined, tab_charts, tab_advisors, tab_sources, tab_upload = st.tabs([
        "📋 Résumé",
        "📈 Graphiques",
        "👤 Détail par conseiller",
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
            unknown_advisors = st.session_state.get("agg_unknown_advisors", [])
            if unknown_advisors:
                names_list = ", ".join(f"**{name}**" for name in unknown_advisors[:10])
                if len(unknown_advisors) > 10:
                    names_list += f", ... (+{len(unknown_advisors) - 10} autres)"
                st.info(
                    f"ℹ️ **{len(unknown_advisors)} conseiller(s) ignoré(s)** (non trouvés dans la base de données) : {names_list}"
                )

            st.markdown("---")
            render_combined_preview(combined_df)
        else:
            st.warning("Aucune donnée pour cette période.")

    # Tab 2: Charts
    with tab_charts:
        render_charts_tab(
            combined_df=combined_df,
            period_name=selected_period.display_name,
        )

    # Tab 3: Detail by advisor (shows all transactions per advisor)
    with tab_advisors:
        if combined_df is not None and not combined_df.empty:
            # Get list of advisors from combined data
            advisors = sorted(combined_df["Conseiller"].unique().tolist())

            # Get list of sources
            source_names = []
            for source_key in st.session_state.agg_selected_sources.keys():
                config = SOURCE_BOARDS.get(source_key)
                if config:
                    source_names.append(config.display_name)

            # Get current selections from session state (persist across period changes)
            current_advisor = st.session_state.get("agg_detail_advisor", "Tous")
            current_source = st.session_state.get("agg_detail_source", "Toutes")

            # Validate current selections still exist in data
            if current_advisor != "Tous" and current_advisor not in advisors:
                current_advisor = "Tous"
            if current_source != "Toutes" and current_source not in source_names:
                current_source = "Toutes"

            # Filter selectors in columns
            col_advisor, col_source = st.columns(2)

            with col_advisor:
                advisor_index = (["Tous"] + advisors).index(current_advisor) if current_advisor in ["Tous"] + advisors else 0
                selected_advisor = st.selectbox(
                    "👤 Conseiller",
                    options=["Tous"] + advisors,
                    index=advisor_index,
                    key="agg_advisor_detail_selector",
                )
                st.session_state.agg_detail_advisor = selected_advisor

            with col_source:
                source_index = (["Toutes"] + source_names).index(current_source) if current_source in ["Toutes"] + source_names else 0
                selected_source = st.selectbox(
                    "📊 Source",
                    options=["Toutes"] + source_names,
                    index=source_index,
                    key="agg_source_detail_selector",
                )
                st.session_state.agg_detail_source = selected_source

            # Combine filtered data from all sources for display
            all_filtered_rows = []
            for source_key in st.session_state.agg_selected_sources.keys():
                config = SOURCE_BOARDS.get(source_key)
                if not config:
                    continue

                # Skip if source filter is active and doesn't match
                if selected_source != "Toutes" and config.display_name != selected_source:
                    continue

                filtered_df = st.session_state.agg_filtered_data.get(source_key, pd.DataFrame())
                if filtered_df.empty:
                    continue

                # Add source column for identification
                display_df = filtered_df.copy()
                display_df["_source"] = config.display_name

                # Rename advisor column to standard name if different
                advisor_col = config.advisor_column
                if advisor_col != "Conseiller" and advisor_col in display_df.columns:
                    display_df = display_df.rename(columns={advisor_col: "Conseiller"})

                all_filtered_rows.append(display_df)

            if all_filtered_rows:
                # Combine all sources
                detail_df = pd.concat(all_filtered_rows, ignore_index=True)

                # Filter by selected advisor if not "Tous"
                if selected_advisor != "Tous":
                    detail_df = detail_df[detail_df["Conseiller"] == selected_advisor]

                if not detail_df.empty:
                    # Build summary message
                    unique_advisors_in_view = detail_df["Conseiller"].nunique()
                    unique_sources_in_view = detail_df["_source"].nunique()

                    summary_parts = [f"**{len(detail_df)} transactions**"]
                    if selected_advisor == "Tous":
                        summary_parts.append(f"**{unique_advisors_in_view} conseillers**")
                    else:
                        summary_parts.append(f"**{selected_advisor}**")
                    if selected_source == "Toutes":
                        summary_parts.append(f"**{unique_sources_in_view} sources**")
                    else:
                        summary_parts.append(f"**{selected_source}**")

                    st.info(f"📊 {' · '.join(summary_parts)}")

                    # Columns to put at the end (technical/metadata columns)
                    end_columns = ["item_id", "group_id", "group_title", "Sous-éléments"]

                    # Select columns to display (prioritize important ones)
                    display_cols = ["Conseiller", "_source"]
                    # Add date columns
                    for col in detail_df.columns:
                        if "date" in col.lower() and col not in display_cols and col not in end_columns:
                            display_cols.append(col)
                    # Add value columns from configs
                    for source_key in st.session_state.agg_selected_sources.keys():
                        config = SOURCE_BOARDS.get(source_key)
                        if config and config.aggregate_column in detail_df.columns:
                            if config.aggregate_column not in display_cols:
                                display_cols.append(config.aggregate_column)
                    # Add remaining columns (except end columns and internal columns)
                    for col in detail_df.columns:
                        if col not in display_cols and not col.startswith("_") and col not in end_columns:
                            display_cols.append(col)
                    # Add end columns last
                    for col in end_columns:
                        if col in detail_df.columns and col not in display_cols:
                            display_cols.append(col)

                    # Filter to existing columns only
                    display_cols = [c for c in display_cols if c in detail_df.columns]

                    # Rename _source to Source for display
                    detail_df = detail_df.rename(columns={"_source": "Source"})
                    display_cols = ["Source" if c == "_source" else c for c in display_cols]

                    st.dataframe(
                        detail_df[display_cols],
                        hide_index=True,
                        use_container_width=True,
                        height=400,
                    )
                else:
                    st.warning("Aucune transaction pour ces filtres.")
            else:
                st.warning("Aucune donnée filtrée disponible.")
        else:
            st.warning("Aucune donnée pour cette période.")

    # Tab 4: Source-by-source detail
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

    # Tab 5: Editable upload preview
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
