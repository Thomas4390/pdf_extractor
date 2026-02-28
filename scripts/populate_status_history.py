"""
Populate the AdvisorStatusHistory cloud database with complete history.

This script:
1. Loads all advisor first appearances from the Data board
2. Generates status records for each month from first appearance to now
3. Saves all records to Google Sheets
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

from src.clients.monday import MondayClient
from src.utils.advisor_status import (
    AdvisorStatusCalculator,
    AdvisorStatusHistoryStore,
    load_advisor_history,
    get_advisor_status,
    clear_advisor_status_cache,
)
from src.utils.aggregator import METRICS_BOARD_CONFIG


# French month names in order
MONTHS_FR = [
    "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
    "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
]


def generate_months_range(start_month: str, end_month: str) -> list[str]:
    """
    Generate a list of months from start to end (inclusive).

    Args:
        start_month: Start month (e.g., "Mars 2025")
        end_month: End month (e.g., "Janvier 2026")

    Returns:
        List of month strings
    """
    start_year, start_m = AdvisorStatusCalculator._parse_month_year(start_month)
    end_year, end_m = AdvisorStatusCalculator._parse_month_year(end_month)

    if start_year == 0 or end_year == 0:
        return []

    months = []
    year = start_year
    month = start_m

    while (year, month) <= (end_year, end_m):
        month_name = MONTHS_FR[month - 1]
        months.append(f"{month_name} {year}")

        month += 1
        if month > 12:
            month = 1
            year += 1

    return months


def get_current_month() -> str:
    """Get the current month in French format."""
    today = date.today()
    month_name = MONTHS_FR[today.month - 1]
    return f"{month_name} {today.year}"


def populate_status_history():
    """Populate the cloud database with complete status history."""
    print("=" * 80)
    print("POPULATING ADVISOR STATUS HISTORY DATABASE")
    print("=" * 80)

    # Get API key
    api_key = os.getenv("MONDAY_API_KEY")
    if not api_key:
        print("ERROR: MONDAY_API_KEY not found in environment")
        return

    client = MondayClient(api_key=api_key)
    data_board_id = METRICS_BOARD_CONFIG.board_id

    # Clear cache and load fresh data
    clear_advisor_status_cache()

    print(f"\nData Board ID: {data_board_id}")
    print("Loading advisor history from Data board...")

    first_appearances = load_advisor_history(client, data_board_id)

    print(f"\nFound {len(first_appearances)} advisors:")
    for advisor, first_month in sorted(first_appearances.items()):
        print(f"  - {advisor}: first appeared {first_month}")

    # Get the store
    store = AdvisorStatusHistoryStore.get_instance()
    if not store.is_configured:
        print(f"\nERROR: Cloud storage not configured: {store.configuration_error}")
        return

    print("\nCloud storage is configured. Generating status history...")

    # Generate status records for each advisor
    current_month = get_current_month()
    all_records = []

    for advisor_name, first_month in first_appearances.items():
        # Generate all months from first appearance to current month
        months = generate_months_range(first_month, current_month)

        print(f"\n  {advisor_name}:")
        print(f"    First appearance: {first_month}")
        print(f"    Generating {len(months)} months of history...")

        for month in months:
            status = get_advisor_status(advisor_name, month)
            all_records.append({
                'advisor_name': advisor_name,
                'month': month,
                'status': status,
                'first_appearance_month': first_month,
            })
            print(f"      {month}: {status}")

    # Save all records in batch
    print("\n" + "-" * 40)
    print(f"SAVING {len(all_records)} RECORDS TO CLOUD")
    print("-" * 40)

    saved_count = store.save_batch_status(all_records)
    print(f"\nSuccessfully saved {saved_count} records to Google Sheets")

    # Verify by loading back
    print("\n" + "-" * 40)
    print("VERIFICATION: Loading records back from cloud")
    print("-" * 40)

    # Check a sample month
    sample_month = "Janvier 2026"
    statuses_for_month = store.get_all_status_for_month(sample_month)
    print(f"\nStatuses for {sample_month}:")
    for advisor, status in sorted(statuses_for_month.items()):
        print(f"  - {advisor}: {status}")

    print("\n" + "=" * 80)
    print("DATABASE POPULATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    populate_status_history()
