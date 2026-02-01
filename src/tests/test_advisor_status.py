"""
Test script for dynamic advisor status calculation.

Tests that advisor status (New/Active/Past) is calculated correctly
based on their first appearance in the Data board.

Also tests cloud storage synchronization to Google Sheets.
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
    get_status_history_store,
    get_advisor_status_history,
    save_advisor_status_to_cloud,
)
from src.utils.aggregator import METRICS_BOARD_CONFIG


def test_advisor_status_calculation():
    """Test the full advisor status calculation flow."""
    print("=" * 80)
    print("TESTING DYNAMIC ADVISOR STATUS CALCULATION")
    print("=" * 80)

    # Use the API key from environment
    api_key = os.getenv("MONDAY_API_KEY")
    if not api_key:
        print("ERROR: MONDAY_API_KEY not found in environment")
        return

    client = MondayClient(api_key=api_key)
    data_board_id = METRICS_BOARD_CONFIG.board_id

    print(f"\nData Board ID: {data_board_id}")

    # Clear any previous cache
    clear_advisor_status_cache()

    # Step 1: Load advisor history from Data board
    print("\n" + "-" * 40)
    print("STEP 1: Load Advisor History from Data Board")
    print("-" * 40)

    first_appearances = load_advisor_history(client, data_board_id)

    print(f"\nFound {len(first_appearances)} advisors with first appearances:")
    for advisor, first_month in sorted(first_appearances.items()):
        print(f"  - {advisor}: {first_month}")

    # Step 2: Test status calculation for different periods
    print("\n" + "-" * 40)
    print("STEP 2: Test Status Calculation for Different Periods")
    print("-" * 40)

    test_periods = [
        "Décembre 2025",
        "Janvier 2026",
        "Février 2026",
    ]

    # Pick some advisors to track
    test_advisors = list(first_appearances.keys())[:5] if first_appearances else ["Test Advisor"]

    for period in test_periods:
        print(f"\n📅 Period: {period}")
        print("-" * 30)

        for advisor in test_advisors:
            status = get_advisor_status(advisor, period)
            first_month = first_appearances.get(advisor, "Unknown")
            print(f"  {advisor}: {status} (first: {first_month})")

    # Step 3: Verify status logic
    print("\n" + "-" * 40)
    print("STEP 3: Verify Status Logic")
    print("-" * 40)

    # Find an advisor who first appeared in December 2025
    dec_advisors = [a for a, m in first_appearances.items() if m == "Décembre 2025"]
    if dec_advisors:
        advisor = dec_advisors[0]
        print(f"\nAdvisor '{advisor}' first appeared in Décembre 2025:")

        # In December 2025 (first month) -> should be "New"
        status_dec = get_advisor_status(advisor, "Décembre 2025")
        print(f"  Status in Décembre 2025: {status_dec} (expected: New)")
        assert status_dec == "New", f"Expected 'New' but got '{status_dec}'"

        # In January 2026 (second month) -> should be "Active"
        status_jan = get_advisor_status(advisor, "Janvier 2026")
        print(f"  Status in Janvier 2026: {status_jan} (expected: Active)")
        assert status_jan == "Active", f"Expected 'Active' but got '{status_jan}'"

        print("  ✅ Status logic verified!")
    else:
        print("No advisor with first appearance in December 2025 found.")

    # Find an advisor who first appeared in January 2026
    jan_advisors = [a for a, m in first_appearances.items() if m == "Janvier 2026"]
    if jan_advisors:
        advisor = jan_advisors[0]
        print(f"\nAdvisor '{advisor}' first appeared in Janvier 2026:")

        # In December 2025 (before first month) -> should still be "New" (no prior history)
        status_dec = get_advisor_status(advisor, "Décembre 2025")
        print(f"  Status in Décembre 2025: {status_dec} (expected: New)")
        # Note: viewing a period before first appearance still shows "New"

        # In January 2026 (first month) -> should be "New"
        status_jan = get_advisor_status(advisor, "Janvier 2026")
        print(f"  Status in Janvier 2026: {status_jan} (expected: New)")
        assert status_jan == "New", f"Expected 'New' but got '{status_jan}'"

        # In February 2026 (second month) -> should be "Active"
        status_feb = get_advisor_status(advisor, "Février 2026")
        print(f"  Status in Février 2026: {status_feb} (expected: Active)")
        assert status_feb == "Active", f"Expected 'Active' but got '{status_feb}'"

        print("  ✅ Status logic verified!")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


def test_cloud_storage():
    """Test cloud storage synchronization for advisor status history."""
    print("=" * 80)
    print("TESTING CLOUD STORAGE SYNCHRONIZATION")
    print("=" * 80)

    # Get the status history store
    store = get_status_history_store()

    print(f"\nCloud Storage Status:")
    print(f"  Configured: {store.is_configured}")
    if store.configuration_error:
        print(f"  Error: {store.configuration_error}")
        print("\nSkipping cloud storage tests (not configured)")
        return

    # Step 1: Save a test status
    print("\n" + "-" * 40)
    print("STEP 1: Save Test Status")
    print("-" * 40)

    test_advisor = "Thomas Lussier"
    test_month = "Janvier 2026"
    test_status = "Active"

    success = save_advisor_status_to_cloud(
        advisor_name=test_advisor,
        month=test_month,
        status=test_status,
        first_appearance_month="Mars 2025",
    )
    print(f"  Saved status for {test_advisor}: {success}")

    # Step 2: Retrieve status history
    print("\n" + "-" * 40)
    print("STEP 2: Retrieve Status History")
    print("-" * 40)

    history = get_advisor_status_history(test_advisor)
    print(f"  Found {len(history)} history records for {test_advisor}:")
    for record in history:
        print(f"    - {record['month']}: {record['status']} (updated: {record.get('updated_at', 'N/A')})")

    # Step 3: Get all statuses for a month
    print("\n" + "-" * 40)
    print("STEP 3: Get All Statuses for Month")
    print("-" * 40)

    all_statuses = store.get_all_status_for_month(test_month)
    print(f"  Found {len(all_statuses)} advisors with status for {test_month}:")
    for advisor, status in list(all_statuses.items())[:5]:  # Show first 5
        print(f"    - {advisor}: {status}")
    if len(all_statuses) > 5:
        print(f"    ... and {len(all_statuses) - 5} more")

    # Step 4: Test batch saving
    print("\n" + "-" * 40)
    print("STEP 4: Test Batch Save")
    print("-" * 40)

    batch_records = [
        {'advisor_name': 'Thomas Lussier', 'month': 'Février 2026', 'status': 'Active', 'first_appearance_month': 'Mars 2025'},
        {'advisor_name': 'Ayoub Chamoumi', 'month': 'Février 2026', 'status': 'Active', 'first_appearance_month': 'Mars 2025'},
    ]

    count = store.save_batch_status(batch_records)
    print(f"  Saved {count} records in batch")

    print("\n" + "=" * 80)
    print("CLOUD STORAGE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_advisor_status_calculation()
    print("\n\n")
    test_cloud_storage()
