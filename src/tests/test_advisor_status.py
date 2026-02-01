"""
Test script for dynamic advisor status calculation.

Tests that advisor status (New/Active/Past) is calculated correctly
based on their first appearance in the Data board.
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
    load_advisor_history,
    get_advisor_status,
    clear_advisor_status_cache,
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


if __name__ == "__main__":
    test_advisor_status_calculation()
