"""
Test script to trace column duplication in the aggregation flow.

This traces each step to find where duplicate columns appear.
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
from src.app.utils.async_helpers import run_async
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


def check_duplicate_columns(df: pd.DataFrame, step_name: str) -> None:
    """Check if DataFrame has duplicate columns (with _x or _y suffix)."""
    cols = list(df.columns)
    duplicates = [c for c in cols if '_x' in c or '_y' in c]

    if duplicates:
        print(f"  ⚠️ DUPLICATES FOUND at {step_name}: {duplicates}")
    else:
        print(f"  ✅ No duplicates at {step_name}")

    print(f"     Columns: {cols}")


def test_aggregation_flow():
    """Test the full aggregation flow step by step."""
    print("=" * 80)
    print("TESTING AGGREGATION FLOW FOR COLUMN DUPLICATION")
    print("=" * 80)

    # Use the API key that has access to all boards including Data
    api_key = "eyJhbGciOiJIUzI1NiJ9.eyJ0aWQiOjU5MDY4NzI0NiwiYWFpIjoxMSwidWlkIjo2ODQ2NTcyMCwiaWFkIjoiMjAyNS0xMS0yNVQxOTo0MzozOC4wMDBaIiwicGVyIjoibWU6d3JpdGUiLCJhY3RpZCI6MjY0NjQxNDIsInJnbiI6InVzZTEifQ.qtNuWfDl6XVJ4Djj00E-c5a1pmzbZjNiNzsXeSMqXuY"

    client = MondayClient(api_key=api_key)

    # Create period for December 2025
    today = date.today()
    months_ago = (today.year - 2025) * 12 + (today.month - 12)
    period = FlexiblePeriod(
        period_type=PeriodType.MONTH,
        months_ago=months_ago,
        reference_date=today,
    )

    print(f"\nPeriod: {period.display_name}")
    print(f"Group name: {period.get_group_name()}")

    # STEP 1: Load and aggregate source data
    print("\n" + "-" * 40)
    print("STEP 1: Load and aggregate source data")
    print("-" * 40)

    aggregated_data = {}

    for source_key, config in SOURCE_BOARDS.items():
        if config.board_id is None:
            continue

        print(f"\n  Loading {config.display_name} (board {config.board_id})...")

        try:
            items = client.extract_board_data_sync(config.board_id)
            df = client.board_items_to_dataframe(items)
            print(f"    Raw data: {len(df)} rows, columns: {list(df.columns)[:5]}...")

            # Filter by period
            filtered_df = filter_by_flexible_period(df, period, config.date_column)
            print(f"    After filter: {len(filtered_df)} rows")

            # Aggregate
            agg_df, unknown = aggregate_by_advisor(
                filtered_df,
                config.aggregate_column,
                config.advisor_column,
            )
            print(f"    After aggregate: {len(agg_df)} advisors")
            check_duplicate_columns(agg_df, f"{config.display_name} aggregated")

            aggregated_data[source_key] = agg_df
        except Exception as e:
            print(f"    ERROR: {e}")
            aggregated_data[source_key] = pd.DataFrame()

    # STEP 2: Combine aggregations
    print("\n" + "-" * 40)
    print("STEP 2: Combine aggregations")
    print("-" * 40)

    combined_df = combine_aggregations(aggregated_data)
    print(f"  Combined: {len(combined_df)} advisors")
    check_duplicate_columns(combined_df, "combined_aggregations")

    # STEP 3: Load metrics from Data board
    print("\n" + "-" * 40)
    print("STEP 3: Load metrics from Data board")
    print("-" * 40)

    metrics_board_id = METRICS_BOARD_CONFIG.board_id
    group_name = period.get_group_name()

    print(f"  Board ID: {metrics_board_id}")
    print(f"  Group: {group_name}")

    # Get group ID
    groups = run_async(client.list_groups(metrics_board_id))
    group_id = None
    for g in groups:
        if g["title"] == group_name:
            group_id = g["id"]
            break

    if not group_id:
        print(f"  ERROR: Group '{group_name}' not found")
        return

    # Load metrics
    items = client.extract_board_data_sync(metrics_board_id, group_id=group_id)
    metrics_raw = client.board_items_to_dataframe(items)
    print(f"  Raw metrics: {len(metrics_raw)} rows")
    print(f"  Raw columns: {list(metrics_raw.columns)}")
    check_duplicate_columns(metrics_raw, "metrics_raw")

    # Filter to only metrics columns
    config = METRICS_BOARD_CONFIG
    columns_to_keep = [
        config.advisor_column,
        config.cost_column,
        config.expenses_column,
        config.leads_column,
        config.bonus_column,
        config.rewards_column,
    ]

    existing_cols = [col for col in columns_to_keep if col in metrics_raw.columns]

    # Handle advisor column from item_name if needed
    if config.advisor_column not in existing_cols:
        if "item_name" in metrics_raw.columns:
            metrics_raw[config.advisor_column] = metrics_raw["item_name"]
            existing_cols = [config.advisor_column] + [c for c in existing_cols if c != config.advisor_column]

    metrics_df = metrics_raw[existing_cols].copy()
    print(f"  Filtered metrics columns: {list(metrics_df.columns)}")
    check_duplicate_columns(metrics_df, "metrics_filtered")

    # STEP 4: Merge metrics with combined data
    print("\n" + "-" * 40)
    print("STEP 4: Merge metrics with combined data")
    print("-" * 40)

    print(f"  Before merge - combined columns: {list(combined_df.columns)}")
    print(f"  Before merge - metrics columns: {list(metrics_df.columns)}")

    # Check for overlapping columns
    combined_cols = set(combined_df.columns)
    metrics_cols = set(metrics_df.columns) - {"Conseiller"}
    overlapping = combined_cols & metrics_cols

    if overlapping:
        print(f"  ⚠️ OVERLAPPING COLUMNS: {overlapping}")
        print("     These will cause _x/_y suffixes!")
    else:
        print(f"  ✅ No overlapping columns")

    merged_df = merge_metrics_with_aggregation(
        combined_df,
        metrics_df,
        advisor_column="Conseiller",
    )
    print(f"  After merge: {len(merged_df)} rows")
    check_duplicate_columns(merged_df, "after_merge")

    # STEP 5: Calculate derived metrics
    print("\n" + "-" * 40)
    print("STEP 5: Calculate derived metrics")
    print("-" * 40)

    final_df = calculate_derived_metrics(merged_df)
    print(f"  Final: {len(final_df)} rows")
    check_duplicate_columns(final_df, "final")

    # STEP 6: Simulate calling merge again (what might happen on re-render)
    print("\n" + "-" * 40)
    print("STEP 6: Simulate second merge (re-render scenario)")
    print("-" * 40)

    print(f"  Merging again with final_df that already has metrics...")
    print(f"  final_df columns: {list(final_df.columns)}")

    second_merge = merge_metrics_with_aggregation(
        final_df,
        metrics_df,
        advisor_column="Conseiller",
    )
    print(f"  After second merge: {len(second_merge)} rows")
    check_duplicate_columns(second_merge, "second_merge")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_aggregation_flow()
