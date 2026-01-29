"""
Aggregation mode UI components.

Provides the 3-step wizard interface for data aggregation:
1. Configuration (target board + auto-load data)
2. Period & Preview (real-time period changes with live preview)
3. Execute upsert
"""

import pandas as pd
import streamlit as st

from datetime import date
from src.utils.aggregator import (
    DatePeriod,
    PeriodType,
    FlexiblePeriod,
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
from src.app.utils.board_utils import (
    get_background_aggregation_status,
    apply_background_aggregation_data,
    start_background_aggregation_load,
    reset_background_aggregation_data,
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

    # Check background loading status
    bg_status = get_background_aggregation_status()
    data_loaded = st.session_state.get("agg_data_loaded", False)
    source_data = st.session_state.get("agg_source_data", {})

    # Try to apply background data if not already loaded
    if not data_loaded and not bg_status["loading"] and bg_status["data"]:
        if apply_background_aggregation_data():
            data_loaded = True
            source_data = st.session_state.agg_source_data
            st.rerun()

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
            reset_background_aggregation_data()
            load_source_data()
            st.rerun()

    elif bg_status["loading"]:
        # Show background loading progress
        progress = bg_status["progress"]
        current = progress.get("current", 0)
        total = progress.get("total", 1)
        current_source = progress.get("current_source", "")

        st.info(f"⏳ Chargement en arrière-plan... ({current}/{total})")
        if current_source:
            st.caption(f"Source actuelle: {current_source}")

        if total > 0:
            st.progress(current / total)

        # Auto-refresh to check progress
        import time
        time.sleep(0.5)
        st.rerun()

    elif bg_status["error"]:
        st.error(f"Erreur lors du chargement: {bg_status['error']}")
        if st.button("🔄 Réessayer", type="primary"):
            reset_background_aggregation_data()
            start_background_aggregation_load()
            st.rerun()

    else:
        st.info("Les données seront chargées automatiquement.")
        # Start background loading if not started yet
        if st.session_state.agg_selected_sources:
            if start_background_aggregation_load():
                st.rerun()
            else:
                # Fallback to synchronous load
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

    # Initialize flexible period if not set
    if "agg_flexible_period" not in st.session_state:
        st.session_state.agg_flexible_period = FlexiblePeriod(
            period_type=PeriodType.MONTH, months_ago=1
        )

    # Also keep legacy period for backwards compatibility
    current_period = st.session_state.agg_period
    if current_period is None:
        current_period = DatePeriod.MONTH_1
        st.session_state.agg_period = current_period

    flexible_period = st.session_state.agg_flexible_period

    # Period selection header
    st.subheader("📅 Sélection de la période")

    # Period type tabs
    period_tabs = st.tabs(["📅 Mois", "📆 Semaine", "📊 Trimestre", "📈 Année", "🎯 Personnalisé"])

    period_changed = False

    # Tab 1: Monthly
    with period_tabs[0]:
        month_options = list(DatePeriod)
        current_index = 1  # Default to last month
        if flexible_period.period_type == PeriodType.MONTH:
            try:
                current_index = flexible_period.months_ago
            except:
                current_index = 1

        selected_month = st.selectbox(
            "Mois à agréger",
            options=range(12),
            index=min(current_index, 11),
            format_func=lambda i: f"{DatePeriod(i).display_name} ({DatePeriod(i).short_label})",
            key="agg_month_selector",
        )

        if st.button("Appliquer", key="apply_month", type="primary"):
            st.session_state.agg_flexible_period = FlexiblePeriod(
                period_type=PeriodType.MONTH, months_ago=selected_month
            )
            st.session_state.agg_period = DatePeriod(selected_month)
            period_changed = True

    # Tab 2: Weekly
    with period_tabs[1]:
        week_options = list(range(8))  # Last 8 weeks
        week_labels = {
            0: "Cette semaine",
            1: "Semaine dernière",
        }
        for i in range(2, 8):
            week_labels[i] = f"Il y a {i} semaines"

        selected_week = st.selectbox(
            "Semaine à agréger",
            options=week_options,
            format_func=lambda i: week_labels.get(i, f"-{i} sem."),
            key="agg_week_selector",
        )

        if st.button("Appliquer", key="apply_week", type="primary"):
            st.session_state.agg_flexible_period = FlexiblePeriod(
                period_type=PeriodType.WEEK, weeks_ago=selected_week
            )
            period_changed = True

    # Tab 3: Quarterly
    with period_tabs[2]:
        quarter_options = list(range(8))  # Last 8 quarters
        quarter_labels = {
            0: "Ce trimestre",
            1: "Trimestre dernier",
        }
        for i in range(2, 8):
            quarter_labels[i] = f"Il y a {i} trimestres"

        selected_quarter = st.selectbox(
            "Trimestre à agréger",
            options=quarter_options,
            format_func=lambda i: quarter_labels.get(i, f"-{i} trim."),
            key="agg_quarter_selector",
        )

        if st.button("Appliquer", key="apply_quarter", type="primary"):
            st.session_state.agg_flexible_period = FlexiblePeriod(
                period_type=PeriodType.QUARTER, quarters_ago=selected_quarter
            )
            period_changed = True

    # Tab 4: Yearly
    with period_tabs[3]:
        year_options = list(range(5))  # Last 5 years
        current_year = date.today().year

        selected_year = st.selectbox(
            "Année à agréger",
            options=year_options,
            format_func=lambda i: f"{current_year - i}" if i > 0 else f"{current_year} (en cours)",
            key="agg_year_selector",
        )

        if st.button("Appliquer", key="apply_year", type="primary"):
            st.session_state.agg_flexible_period = FlexiblePeriod(
                period_type=PeriodType.YEAR, years_ago=selected_year
            )
            period_changed = True

    # Tab 5: Custom
    with period_tabs[4]:
        col_start, col_end = st.columns(2)

        # Default dates
        default_end = date.today()
        default_start = date(default_end.year, default_end.month, 1)

        with col_start:
            custom_start = st.date_input(
                "Date de début",
                value=flexible_period.custom_start or default_start,
                key="agg_custom_start",
            )

        with col_end:
            custom_end = st.date_input(
                "Date de fin",
                value=flexible_period.custom_end or default_end,
                key="agg_custom_end",
            )

        if st.button("Appliquer la période personnalisée", key="apply_custom", type="primary"):
            st.session_state.agg_flexible_period = FlexiblePeriod(
                period_type=PeriodType.CUSTOM,
                custom_start=custom_start,
                custom_end=custom_end,
            )
            period_changed = True

    # If period changed, re-filter data
    if period_changed:
        filter_and_aggregate_data()
        st.rerun()

    # Show current period info
    flexible_period = st.session_state.agg_flexible_period
    start_date, end_date = flexible_period.get_date_range()
    st.info(f"📆 **Période sélectionnée:** {flexible_period.display_name}\n\n**Du** {start_date.strftime('%d/%m/%Y')} **au** {end_date.strftime('%d/%m/%Y')}")

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
                        width="stretch",
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
    flexible_period = st.session_state.get("agg_flexible_period")
    combined_df = st.session_state.agg_combined_data

    # Get target board name
    target_board_name = "Unknown"
    for board in boards:
        if int(board["id"]) == target_board_id:
            target_board_name = board["name"]
            break

    # Get group name - use custom if specified, otherwise flexible period, otherwise legacy
    if st.session_state.get("agg_use_custom_group") and st.session_state.get("agg_custom_group_name"):
        group_name = st.session_state.agg_custom_group_name
    elif flexible_period is not None:
        group_name = flexible_period.get_group_name()
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

    # Data preview before upload
    if combined_df is not None and not combined_df.empty:
        with st.expander("📋 Aperçu des données à envoyer", expanded=True):
            st.dataframe(combined_df, width="stretch", height=300)
            st.caption(f"📊 {len(combined_df)} lignes × {len(combined_df.columns)} colonnes")

    st.markdown("---")

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
