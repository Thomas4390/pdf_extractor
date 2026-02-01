"""
Test script to debug why Hassam shows as "Loss" instead of "New" for December 2025.

This traces the full flow to find where the issue occurs.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import pandas as pd
from src.clients.monday import MondayClient
from src.utils.advisor_status import (
    AdvisorStatusCalculator,
    load_advisor_history,
    get_advisor_status,
    clear_advisor_status_cache,
)
from src.utils.aggregator import (
    METRICS_BOARD_CONFIG,
    SOURCE_BOARDS,
    FlexiblePeriod,
    PeriodType,
    filter_by_flexible_period,
    aggregate_by_advisor,
    combine_aggregations,
    merge_metrics_with_aggregation,
    calculate_derived_metrics,
)
from src.app.utils.async_helpers import run_async
from datetime import date


def test_hassam_status_flow():
    """Test the full flow for Hassam's status in December 2025."""
    print("=" * 80)
    print("DEBUGGING HASSAM STATUS FOR DECEMBER 2025")
    print("=" * 80)

    api_key = os.getenv("MONDAY_API_KEY")
    if not api_key:
        print("ERROR: MONDAY_API_KEY not found")
        return

    client = MondayClient(api_key=api_key)

    # Step 1: Clear cache and load advisor history
    print("\n" + "-" * 40)
    print("STEP 1: Load Advisor History")
    print("-" * 40)

    clear_advisor_status_cache()
    history = load_advisor_history(client, METRICS_BOARD_CONFIG.board_id)

    hassam_first = history.get("Hassam Ramadan")
    print(f"Hassam Ramadan first appearance: {hassam_first}")
    print(f"Cache loaded: {AdvisorStatusCalculator._cache_loaded}")

    # Step 2: Check status calculation
    print("\n" + "-" * 40)
    print("STEP 2: Check Status Calculation")
    print("-" * 40)

    status_dec = get_advisor_status("Hassam Ramadan", "Décembre 2025")
    status_jan = get_advisor_status("Hassam Ramadan", "Janvier 2026")
    print(f"Status for Décembre 2025: {status_dec}")
    print(f"Status for Janvier 2026: {status_jan}")

    # Step 3: Simulate the aggregation flow
    print("\n" + "-" * 40)
    print("STEP 3: Simulate Aggregation Flow")
    print("-" * 40)

    # Create period for December 2025
    today = date.today()
    # Calculate months ago for December 2025
    months_ago = (today.year - 2025) * 12 + (today.month - 12)
    period = FlexiblePeriod(
        period_type=PeriodType.MONTH,
        months_ago=months_ago,
        reference_date=today,
    )
    print(f"Period: {period.display_name}")
    print(f"Group name: {period.get_group_name()}")

    # Step 4: Load and aggregate data (simplified)
    print("\n" + "-" * 40)
    print("STEP 4: Load and Aggregate Data")
    print("-" * 40)

    aggregated_data = {}
    for source_key, config in SOURCE_BOARDS.items():
        if config.board_id is None:
            continue

        try:
            items = client.extract_board_data_sync(config.board_id)
            df = client.board_items_to_dataframe(items)
            filtered_df = filter_by_flexible_period(df, period, config.date_column)
            agg_df, _ = aggregate_by_advisor(
                filtered_df,
                config.aggregate_column,
                config.advisor_column,
            )
            aggregated_data[source_key] = agg_df
            print(f"  {config.display_name}: {len(agg_df)} advisors")
        except Exception as e:
            print(f"  {config.display_name}: ERROR - {e}")
            aggregated_data[source_key] = pd.DataFrame()

    # Step 5: Combine aggregations
    print("\n" + "-" * 40)
    print("STEP 5: Combine Aggregations")
    print("-" * 40)

    combined_df = combine_aggregations(aggregated_data)
    print(f"Combined: {len(combined_df)} advisors")
    print(f"Columns: {list(combined_df.columns)}")

    # Check if Hassam is in combined data
    hassam_rows = combined_df[combined_df["Conseiller"].str.contains("Hassam", case=False, na=False)]
    print(f"\nHassam in combined data: {len(hassam_rows)} rows")
    if not hassam_rows.empty:
        print(hassam_rows.to_string())

    # Step 6: Add advisor status
    print("\n" + "-" * 40)
    print("STEP 6: Add Advisor Status")
    print("-" * 40)

    period_month = period.get_group_name()
    print(f"Period month for status: {period_month}")

    # Add status column
    combined_df = AdvisorStatusCalculator.add_status_to_dataframe(
        combined_df,
        period_month,
        advisor_column="Conseiller",
        sync_to_cloud=False,
    )

    print(f"Columns after adding status: {list(combined_df.columns)}")
    print(f"'Advisor_Status' in columns: {'Advisor_Status' in combined_df.columns}")

    hassam_rows = combined_df[combined_df["Conseiller"].str.contains("Hassam", case=False, na=False)]
    if not hassam_rows.empty:
        print(f"\nHassam after adding status:")
        print(hassam_rows[["Conseiller", "Advisor_Status"]].to_string())

    # Step 7: Load metrics
    print("\n" + "-" * 40)
    print("STEP 7: Load Metrics")
    print("-" * 40)

    metrics_board_id = METRICS_BOARD_CONFIG.board_id
    group_name = period.get_group_name()

    groups = run_async(client.list_groups(metrics_board_id))
    group_id = None
    for g in groups:
        if g["title"] == group_name:
            group_id = g["id"]
            break

    if group_id:
        items = client.extract_board_data_sync(metrics_board_id, group_id=group_id)
        metrics_df = client.board_items_to_dataframe(items)
        print(f"Metrics loaded: {len(metrics_df)} rows")

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
        existing_cols = [col for col in columns_to_keep if col in metrics_df.columns]
        if config.advisor_column not in existing_cols and "item_name" in metrics_df.columns:
            metrics_df[config.advisor_column] = metrics_df["item_name"]
            existing_cols = [config.advisor_column] + [c for c in existing_cols if c != config.advisor_column]

        metrics_df = metrics_df[existing_cols].copy()

        # Merge metrics
        merged_df = merge_metrics_with_aggregation(
            combined_df,
            metrics_df,
            advisor_column="Conseiller",
        )
        print(f"After merge: {len(merged_df)} rows")
        print(f"Columns after merge: {list(merged_df.columns)}")
        print(f"'Advisor_Status' still in columns: {'Advisor_Status' in merged_df.columns}")

        hassam_rows = merged_df[merged_df["Conseiller"].str.contains("Hassam", case=False, na=False)]
        if not hassam_rows.empty:
            print(f"\nHassam after merge:")
            if "Advisor_Status" in merged_df.columns:
                print(hassam_rows[["Conseiller", "Advisor_Status"]].to_string())
            else:
                print("WARNING: Advisor_Status column is MISSING after merge!")
    else:
        print(f"Group '{group_name}' not found")
        merged_df = combined_df

    # Step 8: Calculate derived metrics
    print("\n" + "-" * 40)
    print("STEP 8: Calculate Derived Metrics")
    print("-" * 40)

    print(f"'Advisor_Status' before calculate_derived_metrics: {'Advisor_Status' in merged_df.columns}")

    final_df = calculate_derived_metrics(merged_df)

    print(f"Columns after calculate_derived_metrics: {list(final_df.columns)}")
    print(f"'Profitable' in columns: {'Profitable' in final_df.columns}")

    hassam_rows = final_df[final_df["Conseiller"].str.contains("Hassam", case=False, na=False)]
    if not hassam_rows.empty:
        print(f"\nHassam final result:")
        cols_to_show = ["Conseiller", "Advisor_Status", "Profitable", "Ratio Net"]
        cols_to_show = [c for c in cols_to_show if c in final_df.columns]
        print(hassam_rows[cols_to_show].to_string())

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if not hassam_rows.empty:
        advisor_status = hassam_rows["Advisor_Status"].iloc[0] if "Advisor_Status" in hassam_rows.columns else "N/A"
        profitable = hassam_rows["Profitable"].iloc[0] if "Profitable" in hassam_rows.columns else "N/A"
        print(f"Hassam Ramadan:")
        print(f"  Advisor_Status: {advisor_status}")
        print(f"  Profitable: {profitable}")
        print(f"  Expected Profitable: New (because Advisor_Status should be New)")

        if profitable != "New" and advisor_status == "New":
            print("\n  ❌ BUG: Advisor_Status is 'New' but Profitable is not 'New'!")
        elif advisor_status != "New":
            print(f"\n  ❌ BUG: Advisor_Status should be 'New' but is '{advisor_status}'!")
        else:
            print("\n  ✅ Status is correct!")
    else:
        print("Hassam not found in final data")


if __name__ == "__main__":
    test_hassam_status_flow()
