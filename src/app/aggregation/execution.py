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
    METRICS_BOARD_CONFIG,
    get_group_name_for_period,
    filter_by_date,
    filter_by_flexible_period,
    aggregate_by_advisor,
    combine_aggregations,
    merge_metrics_with_aggregation,
    calculate_derived_metrics,
    FlexiblePeriod,
    PeriodType,
)
from src.app.utils.async_helpers import run_async


def _add_advisor_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advisor status (Active/New/Inactive) from the advisor matcher.

    Args:
        df: DataFrame with 'Conseiller' column

    Returns:
        DataFrame with 'Advisor_Status' column added
    """
    try:
        from src.utils.advisor_matcher import get_advisor_matcher
        matcher = get_advisor_matcher()

        if not matcher.is_configured:
            # No advisor data available, default to "Active"
            df["Advisor_Status"] = "Active"
            return df

        advisors = matcher.get_all_advisors()

        # Create a mapping of advisor name variations to status
        status_map = {}
        for advisor in advisors:
            # Map full name (primary format)
            status_map[advisor.full_name] = advisor.status
            # Also map first name only
            status_map[advisor.first_name] = advisor.status
            # Map compact format for backward compatibility
            status_map[advisor.display_name_compact] = advisor.status

        # Apply status to each advisor in the DataFrame
        def get_status(name):
            if name in status_map:
                return status_map[name]
            # Try matching via the matcher (returns full name)
            matched = matcher.match_full_name(name)
            if matched and matched in status_map:
                return status_map[matched]
            # Default to "Active" if not found
            return "Active"

        df["Advisor_Status"] = df["Conseiller"].apply(get_status)
        return df

    except Exception as e:
        # If anything fails, log the error and default to "Active"
        import logging
        logging.warning(f"Failed to add advisor status: {e}")
        df["Advisor_Status"] = "Active"
        return df


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
    combined_df = combine_aggregations(aggregated_data)

    # Add advisor status from advisor_matcher
    if not combined_df.empty and "Conseiller" in combined_df.columns:
        combined_df = _add_advisor_status(combined_df)

    st.session_state.agg_combined_data = combined_df

    # Reset metrics loaded flag when period changes
    st.session_state.agg_metrics_loaded = False
    st.session_state.agg_metrics_group = ""


def load_and_aggregate_data() -> None:
    """
    Legacy function for backward compatibility.
    Loads data and aggregates in one step.
    """
    load_source_data()
    filter_and_aggregate_data()


def load_metrics_for_period(board_id: int, group_name: str, silent: bool = False) -> pd.DataFrame:
    """
    Load metrics data from a specific group in a board.

    The group should match the month name (e.g., "Janvier 2026").

    Args:
        board_id: Monday.com board ID containing metrics
        group_name: Name of the group to load (e.g., "Janvier 2026")
        silent: If True, don't show warnings

    Returns:
        DataFrame with metrics columns (Conseiller, Coût, Dépenses par Conseiller, Leads, Bonus, Récompenses)
    """
    from src.clients.monday import MondayClient

    api_key = st.session_state.get("monday_api_key")
    if not api_key:
        return pd.DataFrame()

    try:
        client = MondayClient(api_key=api_key)

        # Get groups from the board to find the matching group ID
        groups = run_async(client.list_groups(board_id))

        group_id = None
        for group in groups:
            if group["title"] == group_name:
                group_id = group["id"]
                break

        if group_id is None:
            if not silent:
                st.warning(f"Groupe '{group_name}' non trouvé dans le board de métriques.")
            return pd.DataFrame()

        # Load items from the specific group
        items = client.extract_board_data_sync(board_id, group_id=group_id)
        df = client.board_items_to_dataframe(items)

        if df.empty:
            return pd.DataFrame()

        # Keep only the relevant columns
        config = METRICS_BOARD_CONFIG
        columns_to_keep = [
            config.advisor_column,
            config.cost_column,
            config.expenses_column,
            config.leads_column,
            config.bonus_column,
            config.rewards_column,
        ]

        # Filter to existing columns
        existing_cols = [col for col in columns_to_keep if col in df.columns]
        if config.advisor_column not in existing_cols:
            # Try to use item name as advisor (board_items_to_dataframe uses "item_name")
            if "item_name" in df.columns:
                df[config.advisor_column] = df["item_name"]
                existing_cols = [config.advisor_column] + [c for c in existing_cols if c != config.advisor_column]
            elif "name" in df.columns:
                df[config.advisor_column] = df["name"]
                existing_cols = [config.advisor_column] + [c for c in existing_cols if c != config.advisor_column]

        df = df[existing_cols]

        # Convert numeric columns
        numeric_cols = [config.cost_column, config.expenses_column, config.leads_column,
                        config.bonus_column, config.rewards_column]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    except Exception as e:
        if not silent:
            st.error(f"Erreur lors du chargement des métriques: {e}")
        return pd.DataFrame()


def apply_metrics_to_aggregation(silent: bool = False) -> bool:
    """
    Load metrics and merge with aggregated data, then calculate derived columns.

    Only works for MONTH period type. If metrics are not available,
    adds columns with default values (0) and still calculates derived metrics.

    Args:
        silent: If True, don't show success/info messages

    Returns:
        True if metrics were loaded, False if using default values
    """
    flexible_period = st.session_state.get("agg_flexible_period")

    if flexible_period is None:
        return False

    # Get current combined data
    combined_df = st.session_state.get("agg_combined_data")
    if combined_df is None or combined_df.empty:
        return False

    # For non-MONTH periods, just add empty metric columns with defaults
    if flexible_period.period_type != PeriodType.MONTH:
        final_df = _add_default_metric_columns(combined_df)
        st.session_state.agg_combined_data = final_df
        st.session_state.agg_metrics_loaded = True
        st.session_state.agg_metrics_group = "N/A (période non mensuelle)"
        return False

    # Get the group name for the selected month
    group_name = flexible_period.get_group_name()

    # Load metrics from the configured board
    metrics_board_id = st.session_state.get("agg_metrics_board_id", METRICS_BOARD_CONFIG.board_id)

    try:
        metrics_df = load_metrics_for_period(metrics_board_id, group_name, silent=silent)
    except Exception:
        metrics_df = pd.DataFrame()

    if metrics_df.empty:
        # No metrics available - add default columns with 0 values
        final_df = _add_default_metric_columns(combined_df)
        st.session_state.agg_combined_data = final_df
        st.session_state.agg_metrics_loaded = True
        st.session_state.agg_metrics_group = f"{group_name} (données non disponibles)"
        if not silent:
            st.info(f"ℹ️ Aucune métrique disponible pour {group_name}. Colonnes ajoutées avec valeurs par défaut.")
        return False

    # Merge metrics with aggregated data
    merged_df = merge_metrics_with_aggregation(
        combined_df,
        metrics_df,
        advisor_column="Conseiller",
    )

    # Calculate derived metrics
    final_df = calculate_derived_metrics(merged_df)

    # Store the result
    st.session_state.agg_combined_data = final_df
    st.session_state.agg_metrics_loaded = True
    st.session_state.agg_metrics_group = group_name

    if not silent:
        st.success(f"✅ Métriques importées depuis {group_name}")

    return True


def _add_default_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add default metric columns with 0 values when no metrics data is available.

    Args:
        df: DataFrame to add columns to

    Returns:
        DataFrame with added metric columns
    """
    df = df.copy()

    # Add base metric columns with 0 values
    default_columns = {
        "Coût": 0.0,
        "Dépenses par Conseiller": 0.0,
        "Leads": 0,
        "Bonus": 0.0,
        "Récompenses": 0.0,
    }

    for col, default_val in default_columns.items():
        if col not in df.columns:
            df[col] = default_val

    # Calculate derived metrics (will produce 0s due to 0 inputs)
    df = calculate_derived_metrics(df)

    return df


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
