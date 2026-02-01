"""
Test script to debug the duplicate column issue in merged data.

This investigates why columns like "Dépenses par Conseiller_y" appear.
"""

import os
import sys
from datetime import date
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import pandas as pd

from src.clients.monday import MondayClient
from src.utils.aggregator import (
    SOURCE_BOARDS,
    METRICS_BOARD_CONFIG,
    FlexiblePeriod,
    PeriodType,
    filter_by_flexible_period,
    aggregate_by_advisor,
    combine_aggregations,
    merge_metrics_with_aggregation,
    calculate_derived_metrics,
)


def test_aggregation_columns():
    """Check what columns the aggregation produces."""
    print("=" * 80)
    print("TEST: Aggregation Columns")
    print("=" * 80)

    api_key = os.environ.get("MONDAY_API_KEY")
    if not api_key:
        print("ERROR: MONDAY_API_KEY not set")
        return None

    client = MondayClient(api_key=api_key)

    # Create period for December 2025
    today = date.today()
    months_ago = (today.year - 2025) * 12 + (today.month - 12)
    period = FlexiblePeriod(
        period_type=PeriodType.MONTH,
        months_ago=months_ago,
        reference_date=today,
    )

    aggregated_data = {}

    for source_key, config in SOURCE_BOARDS.items():
        if config.board_id is None:
            continue

        print(f"\nLoading {config.display_name}...")
        items = client.extract_board_data_sync(config.board_id)
        df = client.board_items_to_dataframe(items)

        # Filter by period
        filtered_df = filter_by_flexible_period(df, period, config.date_column)

        # Aggregate
        agg_df, _ = aggregate_by_advisor(
            filtered_df,
            config.aggregate_column,
            config.advisor_column,
        )

        print(f"  Aggregated columns: {list(agg_df.columns)}")
        aggregated_data[source_key] = agg_df

    # Combine
    combined_df = combine_aggregations(aggregated_data)
    print(f"\nCombined columns: {list(combined_df.columns)}")

    return combined_df


def test_metrics_columns():
    """Check what columns the metrics loading produces."""
    print("\n" + "=" * 80)
    print("TEST: Metrics Columns")
    print("=" * 80)

    api_key = os.environ.get("MONDAY_API_KEY")
    if not api_key:
        print("ERROR: MONDAY_API_KEY not set")
        return None

    from src.clients.monday import MondayClient
    from src.app.utils.async_helpers import run_async

    client = MondayClient(api_key=api_key)
    board_id = METRICS_BOARD_CONFIG.board_id
    group_name = "Décembre 2025"

    print(f"Loading from board {board_id}, group '{group_name}'...")

    # Get the group ID
    groups = run_async(client.list_groups(board_id))
    group_id = None
    for group in groups:
        if group["title"] == group_name:
            group_id = group["id"]
            break

    if not group_id:
        print(f"Group '{group_name}' not found")
        return None

    # Load items from group
    items = client.extract_board_data_sync(board_id, group_id=group_id)
    df = client.board_items_to_dataframe(items)

    print(f"Raw board columns ({len(df.columns)}): {list(df.columns)}")

    # Filter to metrics columns
    config = METRICS_BOARD_CONFIG
    columns_to_keep = [
        config.advisor_column,
        config.cost_column,
        config.expenses_column,
        config.leads_column,
        config.bonus_column,
        config.rewards_column,
    ]

    existing_cols = [col for col in columns_to_keep if col in df.columns]

    # Handle advisor column from item_name if needed
    if config.advisor_column not in existing_cols:
        if "item_name" in df.columns:
            df[config.advisor_column] = df["item_name"]
            existing_cols = [config.advisor_column] + [c for c in existing_cols if c != config.advisor_column]

    df = df[existing_cols]
    print(f"Filtered metrics columns: {list(df.columns)}")

    return df


def test_merge_result(combined_df, metrics_df):
    """Test what happens when we merge."""
    print("\n" + "=" * 80)
    print("TEST: Merge Result")
    print("=" * 80)

    if combined_df is None or metrics_df is None:
        print("Missing data, cannot test merge")
        return

    print(f"Combined columns BEFORE merge: {list(combined_df.columns)}")
    print(f"Metrics columns BEFORE merge: {list(metrics_df.columns)}")

    # Do the merge
    merged_df = merge_metrics_with_aggregation(
        combined_df,
        metrics_df,
        advisor_column="Conseiller",
    )

    print(f"\nMerged columns AFTER merge: {list(merged_df.columns)}")

    # Check for duplicates
    duplicate_cols = [c for c in merged_df.columns if '_x' in c or '_y' in c]
    if duplicate_cols:
        print(f"\n⚠️ FOUND DUPLICATE COLUMNS: {duplicate_cols}")
    else:
        print("\n✅ No duplicate columns")

    # Calculate derived metrics
    final_df = calculate_derived_metrics(merged_df)
    print(f"\nFinal columns after calculate_derived_metrics: {list(final_df.columns)}")

    # Check for duplicates again
    duplicate_cols = [c for c in final_df.columns if '_x' in c or '_y' in c]
    if duplicate_cols:
        print(f"\n⚠️ FOUND DUPLICATE COLUMNS IN FINAL: {duplicate_cols}")
    else:
        print("\n✅ No duplicate columns in final")

    return final_df


def main():
    """Run all tests."""
    combined_df = test_aggregation_columns()
    metrics_df = test_metrics_columns()
    test_merge_result(combined_df, metrics_df)

    print("\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
