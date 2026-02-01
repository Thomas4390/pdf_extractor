"""
Test script to debug metrics loading from the Data board.

This investigates if there are duplicate entries or issues when
loading metrics from the 2026 Copie de Data board.
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
    METRICS_BOARD_CONFIG,
    FlexiblePeriod,
    PeriodType,
)


def test_load_metrics_board():
    """Load and analyze the metrics board data."""
    print("=" * 80)
    print("TEST: Metrics Board Loading")
    print("=" * 80)

    api_key = os.environ.get("MONDAY_API_KEY")
    if not api_key:
        print("ERROR: MONDAY_API_KEY not set")
        return

    client = MondayClient(api_key=api_key)
    board_id = METRICS_BOARD_CONFIG.board_id

    print(f"Loading from board ID: {board_id}")

    # Load all data from the board
    items = client.extract_board_data_sync(board_id)
    df = client.board_items_to_dataframe(items)

    print(f"Loaded {len(df)} rows total")
    print(f"Columns: {list(df.columns)}")

    # Check unique groups
    if "group_title" in df.columns:
        groups = df["group_title"].unique()
        print(f"\nUnique groups ({len(groups)}):")
        for g in groups:
            count = len(df[df["group_title"] == g])
            print(f"  - '{g}': {count} rows")

    # Check for "Décembre 2025" group
    dec_2025_group = "Décembre 2025"
    if "group_title" in df.columns:
        dec_df = df[df["group_title"] == dec_2025_group]
        print(f"\nFiltering for group '{dec_2025_group}': {len(dec_df)} rows")

        if not dec_df.empty:
            # Check for Ayoub
            if "Conseiller" in dec_df.columns:
                advisors = dec_df["Conseiller"].value_counts()
                print(f"\nAdvisor counts in {dec_2025_group}:")
                for advisor, count in advisors.items():
                    print(f"  - '{advisor}': {count}")

                    # If Ayoub appears multiple times, show all rows
                    if "ayoub" in str(advisor).lower() and count > 1:
                        print(f"\n⚠️ WARNING: {advisor} appears {count} times!")
                        ayoub_rows = dec_df[dec_df["Conseiller"] == advisor]
                        print(ayoub_rows.to_string())

    # Check if there are duplicates in Conseiller column
    print("\n--- Checking for duplicates ---")
    if "Conseiller" in df.columns and "group_title" in df.columns:
        for group in df["group_title"].unique():
            group_df = df[df["group_title"] == group]
            duplicates = group_df["Conseiller"].duplicated()
            if duplicates.any():
                print(f"Group '{group}' has duplicate advisors:")
                dup_names = group_df.loc[duplicates, "Conseiller"].unique()
                for name in dup_names:
                    count = len(group_df[group_df["Conseiller"] == name])
                    print(f"  - '{name}' appears {count} times")


def test_metrics_import():
    """Test the metrics import process for December 2025."""
    print("\n" + "=" * 80)
    print("TEST: Metrics Import Process")
    print("=" * 80)

    api_key = os.environ.get("MONDAY_API_KEY")
    if not api_key:
        print("ERROR: MONDAY_API_KEY not set")
        return

    from src.clients.monday import MondayClient
    from src.utils.aggregator import METRICS_BOARD_CONFIG

    client = MondayClient(api_key=api_key)
    board_id = METRICS_BOARD_CONFIG.board_id

    # Create period for December 2025
    today = date.today()
    months_ago = (today.year - 2025) * 12 + (today.month - 12)
    period = FlexiblePeriod(
        period_type=PeriodType.MONTH,
        months_ago=months_ago,
        reference_date=today,
    )
    group_name = period.get_group_name()

    print(f"Loading metrics for group: '{group_name}'")

    # Load data
    items = client.extract_board_data_sync(board_id)
    df = client.board_items_to_dataframe(items)

    # Filter by group
    if "group_title" in df.columns:
        filtered_df = df[df["group_title"] == group_name]
        print(f"Found {len(filtered_df)} rows for group '{group_name}'")

        if not filtered_df.empty and "Conseiller" in filtered_df.columns:
            print(f"\nMetrics columns available: {[c for c in filtered_df.columns if c != 'Conseiller']}")
            print("\nData for each advisor:")

            for _, row in filtered_df.iterrows():
                advisor = row.get("Conseiller", "N/A")
                cout = row.get("Coût", 0)
                depenses = row.get("Dépenses par Conseiller", 0)
                leads = row.get("Leads", 0)
                bonus = row.get("Bonus", 0)
                rewards = row.get("Récompenses", 0)

                if "ayoub" in str(advisor).lower():
                    print(f"\n🔍 AYOUB DATA:")
                    print(f"  Conseiller: {advisor}")
                    print(f"  Coût: {cout}")
                    print(f"  Dépenses par Conseiller: {depenses}")
                    print(f"  Leads: {leads}")
                    print(f"  Bonus: {bonus}")
                    print(f"  Récompenses: {rewards}")
    else:
        print("No 'group_title' column found in data")


def main():
    """Run all tests."""
    test_load_metrics_board()
    test_metrics_import()

    print("\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
