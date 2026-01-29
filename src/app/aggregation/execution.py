"""
Aggregation execution logic.

Provides functions for loading data from Monday.com boards,
filtering by date period, aggregating by advisor, and upserting
to target boards.
"""

import pandas as pd
import streamlit as st

from src.utils.aggregator import (
    SOURCE_BOARDS,
    get_group_name_for_period,
    filter_by_date,
    filter_by_flexible_period,
    aggregate_by_advisor,
    combine_aggregations,
    FlexiblePeriod,
    PeriodType,
)
from src.app.utils.async_helpers import run_async


def load_source_data() -> bool:
    """
    Load raw data from all selected source boards.

    This only loads the raw data from Monday.com. Filtering and aggregation
    are done separately to allow real-time period changes without reloading.

    Returns:
        True if data was loaded successfully, False otherwise.
    """
    from src.clients.monday import MondayClient

    client = MondayClient(api_key=st.session_state.monday_api_key)

    source_data = {}
    selected_sources = st.session_state.agg_selected_sources
    total_sources = len(selected_sources)

    if total_sources == 0:
        return False

    # Progress bar for loading
    progress_bar = st.progress(0)
    status_text = st.empty()

    success = True
    for idx, (source_key, board_id) in enumerate(selected_sources.items()):
        config = SOURCE_BOARDS.get(source_key)
        if not config:
            continue

        # Update progress
        progress = idx / total_sources
        progress_bar.progress(progress)
        status_text.text(f"📥 Chargement de {config.display_name}...")

        try:
            # Extract board data
            items = client.extract_board_data_sync(board_id)
            df = client.board_items_to_dataframe(items)
            source_data[source_key] = df
        except Exception as e:
            st.error(f"Erreur lors du chargement de {config.display_name}: {e}")
            source_data[source_key] = pd.DataFrame()
            success = False

    # Complete progress
    progress_bar.progress(1.0)
    status_text.text("✅ Données chargées!")

    # Store raw source data
    st.session_state.agg_source_data = source_data
    st.session_state.agg_data_loaded = True

    # Clear progress indicators after a short delay
    import time
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    return success


def filter_and_aggregate_data() -> None:
    """
    Filter and aggregate the loaded source data based on the selected period.

    This function uses the raw data already loaded in session state and applies
    filtering/aggregation. It's designed to be fast for real-time period changes.
    Supports both legacy DatePeriod and new FlexiblePeriod.
    """
    # Check for flexible period first (new approach)
    flexible_period = st.session_state.get("agg_flexible_period")
    legacy_period = st.session_state.agg_period
    source_data = st.session_state.get("agg_source_data", {})

    if not source_data:
        return

    filtered_data = {}
    aggregated_data = {}
    all_unknown_names = []

    for source_key, df in source_data.items():
        config = SOURCE_BOARDS.get(source_key)
        if not config or df.empty:
            filtered_data[source_key] = pd.DataFrame()
            aggregated_data[source_key] = pd.DataFrame()
            continue

        # Filter by date using flexible period if available, otherwise legacy
        if flexible_period is not None:
            filtered_df = filter_by_flexible_period(
                df=df,
                period=flexible_period,
                date_column=config.date_column,
            )
        else:
            filtered_df = filter_by_date(
                df=df,
                period=legacy_period,
                date_column=config.date_column,
            )
        filtered_data[source_key] = filtered_df

        # Aggregate by advisor (using config's advisor_column)
        # Returns tuple: (aggregated_df, list of unknown advisor names)
        aggregated_df, unknown_names = aggregate_by_advisor(
            df=filtered_df,
            value_column=config.aggregate_column,
            advisor_column=config.advisor_column,
        )
        aggregated_data[source_key] = aggregated_df
        # Ensure unknown_names is a list (defensive coding for edge cases)
        if not isinstance(unknown_names, list):
            unknown_names = []
        all_unknown_names.extend(unknown_names)

    # Remove duplicates while preserving order
    unique_unknown_names = list(dict.fromkeys(all_unknown_names))

    st.session_state.agg_filtered_data = filtered_data
    st.session_state.agg_aggregated_data = aggregated_data
    st.session_state.agg_unknown_advisors = unique_unknown_names

    # Combine aggregations
    st.session_state.agg_combined_data = combine_aggregations(aggregated_data)


def load_and_aggregate_data() -> None:
    """
    Legacy function for backward compatibility.
    Loads data and aggregates in one step.
    """
    load_source_data()
    filter_and_aggregate_data()


def execute_aggregation_upsert() -> None:
    """Execute the upsert operation to Monday.com."""
    from src.clients.monday import MondayClient

    st.session_state.agg_is_executing = True

    try:
        client = MondayClient(api_key=st.session_state.monday_api_key)
        target_board_id = st.session_state.agg_target_board_id
        period = st.session_state.agg_period

        # Use edited data if available, otherwise use combined data
        data_df = st.session_state.get("agg_edited_data")
        if data_df is None or (hasattr(data_df, 'empty') and data_df.empty):
            data_df = st.session_state.get("agg_combined_data")

        if data_df is None or data_df.empty:
            st.error("Pas de données à upserter.")
            st.session_state.agg_is_executing = False
            return

        # Get group name - use custom if specified
        if st.session_state.get("agg_use_custom_group") and st.session_state.get("agg_custom_group_name"):
            group_name = st.session_state.agg_custom_group_name
        else:
            group_name = get_group_name_for_period(period)

        # Get or create target group
        with st.spinner(f"Création/récupération du groupe '{group_name}'..."):
            group_result = client.get_or_create_group_sync(
                board_id=target_board_id,
                group_name=group_name,
            )
            if not group_result.success:
                st.error(f"Erreur lors de la création du groupe: {group_result.error}")
                st.session_state.agg_is_executing = False
                return
            group_id = group_result.id

        # Get columns from target board
        with st.spinner("Récupération des colonnes..."):
            columns = run_async(client.list_columns(target_board_id))
            column_id_map = {}
            column_type_map = {}
            advisor_column_id = None
            advisor_column_name_actual = None  # Store actual column name from Monday.com

            for col in columns:
                col_title = col["title"]
                col_id = col["id"]
                col_type = col["type"]

                column_id_map[col_title] = col_id
                column_type_map[col_title] = col_type

                if col_title.lower() == "conseiller":
                    advisor_column_id = col_id
                    advisor_column_name_actual = col_title  # Keep actual case

        # Execute upsert
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(current: int, total: int, action: str) -> None:
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"Traitement: {current}/{total} conseillers...")

        # Rename DataFrame column to match actual Monday.com column name if different
        if advisor_column_name_actual and advisor_column_name_actual != "Conseiller":
            if "Conseiller" in data_df.columns:
                data_df = data_df.rename(columns={"Conseiller": advisor_column_name_actual})

        # Choose upsert method based on whether a Conseiller column exists
        if advisor_column_id:
            # Use column-based upsert (Conseiller is a separate column)
            result = client.upsert_by_advisor_sync(
                board_id=target_board_id,
                group_id=group_id,
                advisor_column_id=advisor_column_id,
                data=data_df,
                column_id_map=column_id_map,
                column_type_map=column_type_map,
                advisor_column_name=advisor_column_name_actual or "Conseiller",
                progress_callback=progress_callback,
            )
        else:
            # Use item name-based upsert (advisor name = item name / "Élément")
            result = client.upsert_by_item_name_sync(
                board_id=target_board_id,
                group_id=group_id,
                data=data_df,
                column_id_map=column_id_map,
                column_type_map=column_type_map,
                advisor_column_name="Conseiller",
                progress_callback=progress_callback,
            )

        progress_bar.empty()
        status_text.empty()

        st.session_state.agg_upsert_result = result
        st.session_state.agg_is_executing = False
        st.rerun()

    except Exception as e:
        st.error(f"Erreur lors de l'upsert: {e}")
        st.session_state.agg_is_executing = False
