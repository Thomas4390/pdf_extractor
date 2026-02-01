"""
Debug test for aggregation issues - specifically for Ayoub Chamoumi data.

This test investigates why incorrect values appear in the aggregation.
"""

import os
import sys
from datetime import date
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import pandas as pd

from src.clients.monday import MondayClient
from src.utils.aggregator import (
    SOURCE_BOARDS,
    FlexiblePeriod,
    PeriodType,
    filter_by_flexible_period,
    aggregate_by_advisor,
    combine_aggregations,
    normalize_advisor_column,
)


def test_date_filtering():
    """Test that date filtering works correctly for December 2025."""
    print("\n" + "=" * 80)
    print("TEST: Date Filtering for December 2025")
    print("=" * 80)

    # Create a FlexiblePeriod for December 2025
    # Today is February 1, 2026, so December 2025 is 2 months ago
    today = date.today()
    print(f"Today's date: {today}")

    # Calculate months_ago for December 2025
    target_year = 2025
    target_month = 12
    months_ago = (today.year - target_year) * 12 + (today.month - target_month)
    print(f"Months ago for December 2025: {months_ago}")

    period = FlexiblePeriod(
        period_type=PeriodType.MONTH,
        months_ago=months_ago,
        reference_date=today,  # Explicitly set reference date
    )

    start_date, end_date = period.get_date_range()
    print(f"Period: {period.display_name}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Group name: {period.get_group_name()}")

    # Verify the dates are correct
    assert start_date.year == 2025, f"Expected year 2025, got {start_date.year}"
    assert start_date.month == 12, f"Expected month 12, got {start_date.month}"
    assert end_date.year == 2025, f"Expected end year 2025, got {end_date.year}"
    assert end_date.month == 12, f"Expected end month 12, got {end_date.month}"

    print("✅ Date filtering parameters are correct!")
    return period


def test_load_and_filter_source_data(period: FlexiblePeriod):
    """Load source data and filter by period."""
    print("\n" + "=" * 80)
    print("TEST: Load and Filter Source Data")
    print("=" * 80)

    api_key = os.environ.get("MONDAY_API_KEY")
    if not api_key:
        print("❌ MONDAY_API_KEY not set. Skipping API test.")
        return None

    client = MondayClient(api_key=api_key)

    all_source_data = {}
    all_filtered_data = {}

    for source_key, config in SOURCE_BOARDS.items():
        if config.board_id is None:
            print(f"Skipping {source_key} - no board_id configured")
            continue

        print(f"\n--- {config.display_name} (Board ID: {config.board_id}) ---")

        # Load data
        try:
            items = client.extract_board_data_sync(config.board_id)
            df = client.board_items_to_dataframe(items)
            print(f"Loaded {len(df)} rows")
            all_source_data[source_key] = df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            continue

        if df.empty:
            print("No data found")
            continue

        # Check columns
        print(f"Columns: {list(df.columns)}")
        print(f"Date column: {config.date_column}")
        print(f"Advisor column: {config.advisor_column}")
        print(f"Value column: {config.aggregate_column}")

        # Check for Ayoub BEFORE filtering
        if config.advisor_column in df.columns:
            ayoub_mask = df[config.advisor_column].astype(str).str.lower().str.contains('ayoub', na=False)
            ayoub_all = df[ayoub_mask]
            if not ayoub_all.empty:
                print(f"\n🔍 AYOUB - All data (before date filter): {len(ayoub_all)} rows")
                if config.date_column in ayoub_all.columns:
                    print(f"   Dates: {ayoub_all[config.date_column].tolist()}")
                if config.aggregate_column in ayoub_all.columns:
                    print(f"   Values: {ayoub_all[config.aggregate_column].tolist()}")
                    total = pd.to_numeric(ayoub_all[config.aggregate_column], errors='coerce').sum()
                    print(f"   Total (all dates): {total}")

        # Filter by date
        filtered_df = filter_by_flexible_period(
            df=df,
            period=period,
            date_column=config.date_column,
        )
        print(f"\nAfter filtering for {period.display_name}: {len(filtered_df)} rows")
        all_filtered_data[source_key] = filtered_df

        # Check for Ayoub AFTER filtering
        if config.advisor_column in filtered_df.columns:
            ayoub_mask = filtered_df[config.advisor_column].astype(str).str.lower().str.contains('ayoub', na=False)
            ayoub_filtered = filtered_df[ayoub_mask]
            if not ayoub_filtered.empty:
                print(f"\n🔍 AYOUB - After date filter: {len(ayoub_filtered)} rows")
                if config.date_column in ayoub_filtered.columns:
                    print(f"   Dates: {ayoub_filtered[config.date_column].tolist()}")
                if config.aggregate_column in ayoub_filtered.columns:
                    print(f"   Values: {ayoub_filtered[config.aggregate_column].tolist()}")
                    total = pd.to_numeric(ayoub_filtered[config.aggregate_column], errors='coerce').sum()
                    print(f"   Total (Dec 2025 only): {total}")
            else:
                print(f"\n🔍 AYOUB - No data after date filter")

    return all_source_data, all_filtered_data


def test_aggregation(filtered_data: dict):
    """Test the aggregation by advisor."""
    print("\n" + "=" * 80)
    print("TEST: Aggregation by Advisor")
    print("=" * 80)

    aggregated_data = {}

    for source_key, filtered_df in filtered_data.items():
        config = SOURCE_BOARDS.get(source_key)
        if not config or filtered_df.empty:
            continue

        print(f"\n--- Aggregating {config.display_name} ---")

        # Check Ayoub before aggregation
        if config.advisor_column in filtered_df.columns:
            ayoub_mask = filtered_df[config.advisor_column].astype(str).str.lower().str.contains('ayoub', na=False)
            if ayoub_mask.any():
                ayoub_pre = filtered_df[ayoub_mask]
                print(f"Ayoub before aggregation: {len(ayoub_pre)} rows")
                print(f"  Names: {ayoub_pre[config.advisor_column].unique().tolist()}")
                if config.aggregate_column in ayoub_pre.columns:
                    print(f"  Values: {ayoub_pre[config.aggregate_column].tolist()}")

        # Aggregate
        agg_df, unknown_names = aggregate_by_advisor(
            df=filtered_df,
            value_column=config.aggregate_column,
            advisor_column=config.advisor_column,
            normalize_names=True,
        )

        print(f"Aggregated to {len(agg_df)} advisors")
        if unknown_names:
            print(f"Unknown names filtered out: {unknown_names[:10]}...")

        # Check Ayoub after aggregation
        if "Conseiller" in agg_df.columns:
            ayoub_mask = agg_df["Conseiller"].astype(str).str.lower().str.contains('ayoub', na=False)
            if ayoub_mask.any():
                ayoub_agg = agg_df[ayoub_mask]
                print(f"\n🔍 AYOUB after aggregation:")
                print(ayoub_agg.to_string())

        aggregated_data[source_key] = agg_df

    return aggregated_data


def test_combine_aggregations(aggregated_data: dict):
    """Test combining aggregations from multiple sources."""
    print("\n" + "=" * 80)
    print("TEST: Combine Aggregations")
    print("=" * 80)

    combined_df = combine_aggregations(aggregated_data)

    print(f"Combined data: {len(combined_df)} advisors")
    print(f"Columns: {list(combined_df.columns)}")

    # Check Ayoub in combined data
    if "Conseiller" in combined_df.columns:
        ayoub_mask = combined_df["Conseiller"].astype(str).str.lower().str.contains('ayoub', na=False)
        if ayoub_mask.any():
            ayoub_combined = combined_df[ayoub_mask]
            print(f"\n🔍 AYOUB in combined data:")
            print(ayoub_combined.to_string())

            # Check if AE CA is 9547
            if "AE CA" in ayoub_combined.columns:
                ae_ca = ayoub_combined["AE CA"].values[0]
                if abs(ae_ca - 9547) < 1:
                    print(f"\n⚠️ FOUND THE BUG: Ayoub has AE CA = {ae_ca}")
                    print("This value should not exist for December 2025!")

    return combined_df


def test_advisor_name_normalization():
    """Test how Ayoub's name is being normalized."""
    print("\n" + "=" * 80)
    print("TEST: Advisor Name Normalization")
    print("=" * 80)

    from src.utils.advisor_matcher import (
        get_advisor_matcher,
        normalize_advisor_name_full,
    )

    # Test various forms of Ayoub's name
    test_names = [
        "Ayoub",
        "Ayoub Chamoumi",
        "Chamoumi",
        "ayoub",
        "AYOUB",
        "Ayoub C",
        "A. Chamoumi",
    ]

    matcher = get_advisor_matcher()
    print(f"Matcher configured: {matcher.is_configured}")
    print(f"Number of advisors: {len(matcher.advisors)}")

    # Find Ayoub in the advisor list
    for advisor in matcher.advisors:
        if 'ayoub' in advisor.first_name.lower() or 'ayoub' in advisor.last_name.lower():
            print(f"\nFound Ayoub in advisor list:")
            print(f"  First name: {advisor.first_name}")
            print(f"  Last name: {advisor.last_name}")
            print(f"  Full name: {advisor.full_name}")
            print(f"  Status: {advisor.status}")

    print("\nTesting name normalization:")
    for name in test_names:
        normalized = normalize_advisor_name_full(name)
        print(f"  '{name}' -> '{normalized}'")


def main():
    """Run all debug tests."""
    print("=" * 80)
    print("AGGREGATION DEBUG TESTS")
    print("=" * 80)

    # Test 1: Date filtering parameters
    period = test_date_filtering()

    # Test 2: Advisor name normalization
    test_advisor_name_normalization()

    # Test 3: Load and filter source data (requires API key)
    result = test_load_and_filter_source_data(period)
    if result is None:
        print("\n⚠️ Skipping API-dependent tests (no API key)")
        return

    source_data, filtered_data = result

    # Test 4: Aggregation
    aggregated_data = test_aggregation(filtered_data)

    # Test 5: Combine aggregations
    combined_df = test_combine_aggregations(aggregated_data)

    print("\n" + "=" * 80)
    print("DEBUG TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
