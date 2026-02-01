"""
Test script to identify which advisor names are being incorrectly matched to Ayoub Chamoumi.

This test investigates the root cause of the data mixing issue where Ayoub's
row count doubles from 18 to 34 after normalization.
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

from src.utils.advisor_matcher import (
    get_advisor_matcher,
    normalize_advisor_name_full,
)


def test_which_names_match_ayoub():
    """Test which names get matched to Ayoub Chamoumi."""
    print("\n" + "=" * 80)
    print("TEST: Which names match Ayoub Chamoumi?")
    print("=" * 80)

    matcher = get_advisor_matcher()
    print(f"Matcher configured: {matcher.is_configured}")
    print(f"Number of advisors: {len(matcher.advisors)}")

    # Find Ayoub in the advisor list
    ayoub_advisor = None
    for advisor in matcher.advisors:
        if 'ayoub' in advisor.first_name.lower() or 'ayoub' in advisor.last_name.lower():
            ayoub_advisor = advisor
            print(f"\nFound Ayoub in advisor list:")
            print(f"  First name: {advisor.first_name}")
            print(f"  Last name: {advisor.last_name}")
            print(f"  Full name: {advisor.full_name}")
            print(f"  Variations: {advisor.variations}")
            print(f"  Status: {advisor.status}")
            break

    if not ayoub_advisor:
        print("ERROR: Ayoub not found in advisor list!")
        return

    # List all advisors for reference
    print("\n--- All advisors ---")
    for i, advisor in enumerate(matcher.advisors):
        print(f"{i+1}. {advisor.full_name} ({advisor.first_name}, {advisor.last_name})")

    # Now test every advisor's first name and last name to see what matches Ayoub
    print("\n--- Testing which advisor names normalize to Ayoub Chamoumi ---")
    matches_to_ayoub = []

    for advisor in matcher.advisors:
        # Skip Ayoub himself
        if advisor.full_name == ayoub_advisor.full_name:
            continue

        # Test first name
        result = normalize_advisor_name_full(advisor.first_name)
        if result == "Ayoub Chamoumi":
            matches_to_ayoub.append((advisor.first_name, "first_name", advisor.full_name))
            print(f"  ⚠️ '{advisor.first_name}' (first name of {advisor.full_name}) -> 'Ayoub Chamoumi'")

        # Test last name
        result = normalize_advisor_name_full(advisor.last_name)
        if result == "Ayoub Chamoumi":
            matches_to_ayoub.append((advisor.last_name, "last_name", advisor.full_name))
            print(f"  ⚠️ '{advisor.last_name}' (last name of {advisor.full_name}) -> 'Ayoub Chamoumi'")

        # Test full name
        result = normalize_advisor_name_full(advisor.full_name)
        if result == "Ayoub Chamoumi":
            matches_to_ayoub.append((advisor.full_name, "full_name", advisor.full_name))
            print(f"  ⚠️ '{advisor.full_name}' (full name) -> 'Ayoub Chamoumi'")

    if matches_to_ayoub:
        print(f"\n⚠️ Found {len(matches_to_ayoub)} names that incorrectly match to Ayoub Chamoumi:")
        for name, name_type, original_advisor in matches_to_ayoub:
            print(f"  - '{name}' ({name_type} of {original_advisor})")
    else:
        print("\n✅ No advisor names incorrectly match to Ayoub Chamoumi")


def test_load_real_data_and_check():
    """Load real data from Monday.com and check what names are matching to Ayoub."""
    print("\n" + "=" * 80)
    print("TEST: Check real data from Paiement Historique board")
    print("=" * 80)

    api_key = os.environ.get("MONDAY_API_KEY")
    if not api_key:
        print("❌ MONDAY_API_KEY not set. Skipping API test.")
        return

    from src.clients.monday import MondayClient
    from src.utils.aggregator import (
        SOURCE_BOARDS,
        FlexiblePeriod,
        PeriodType,
        filter_by_flexible_period,
    )

    client = MondayClient(api_key=api_key)
    config = SOURCE_BOARDS["paiement_historique"]

    # Load data
    print(f"Loading data from board {config.board_id}...")
    items = client.extract_board_data_sync(config.board_id)
    df = client.board_items_to_dataframe(items)
    print(f"Loaded {len(df)} rows")

    # Create period for December 2025
    today = date.today()
    months_ago = (today.year - 2025) * 12 + (today.month - 12)
    period = FlexiblePeriod(
        period_type=PeriodType.MONTH,
        months_ago=months_ago,
        reference_date=today,
    )

    # Filter by date
    filtered_df = filter_by_flexible_period(df, period, config.date_column)
    print(f"After filtering for {period.display_name}: {len(filtered_df)} rows")

    # Now check each unique advisor name and what it normalizes to
    print("\n--- Analyzing advisor name normalization ---")
    advisor_col = config.advisor_column
    unique_names = filtered_df[advisor_col].unique()
    print(f"Found {len(unique_names)} unique advisor names")

    # Create a mapping of original name -> normalized name
    name_mapping = {}
    ayoub_matches = []

    for name in unique_names:
        if pd.isna(name) or str(name).strip() == "":
            continue

        normalized = normalize_advisor_name_full(str(name))
        name_mapping[str(name)] = normalized

        if normalized == "Ayoub Chamoumi":
            ayoub_matches.append(str(name))

    print(f"\n--- Names that normalize to 'Ayoub Chamoumi' ---")
    for original_name in ayoub_matches:
        # Count rows with this name
        count = len(filtered_df[filtered_df[advisor_col] == original_name])
        value_sum = pd.to_numeric(filtered_df[filtered_df[advisor_col] == original_name][config.aggregate_column], errors='coerce').sum()
        print(f"  '{original_name}': {count} rows, total value: {value_sum:.2f}")

    if len(ayoub_matches) > 1:
        print(f"\n⚠️ BUG CONFIRMED: {len(ayoub_matches)} different names are all normalizing to 'Ayoub Chamoumi'!")
        print("This is causing data from other advisors to be attributed to Ayoub.")
    elif len(ayoub_matches) == 1:
        print(f"\n✅ Only '{ayoub_matches[0]}' normalizes to 'Ayoub Chamoumi'")
        print("The issue might be elsewhere...")

    # Show all name mappings for reference
    print("\n--- All name mappings ---")
    for original, normalized in sorted(name_mapping.items()):
        marker = " ⚠️" if normalized == "Ayoub Chamoumi" and original != "Ayoub Chamoumi" else ""
        print(f"  '{original}' -> '{normalized}'{marker}")


def test_fuzzy_matching_threshold():
    """Test the fuzzy matching threshold to see if it's too permissive."""
    print("\n" + "=" * 80)
    print("TEST: Fuzzy matching analysis")
    print("=" * 80)

    from difflib import SequenceMatcher

    matcher = get_advisor_matcher()

    # Find Ayoub
    ayoub_terms = None
    for advisor in matcher.advisors:
        if 'ayoub' in advisor.first_name.lower():
            ayoub_terms = advisor.get_all_searchable_terms()
            print(f"Ayoub's searchable terms: {ayoub_terms}")
            break

    if not ayoub_terms:
        print("ERROR: Ayoub not found")
        return

    # Check fuzzy similarity of all other advisor names against Ayoub's terms
    print(f"\nFuzzy threshold: {matcher.fuzzy_threshold}")
    print("\n--- Names that might fuzzy-match to Ayoub ---")

    problematic = []
    for advisor in matcher.advisors:
        if 'ayoub' in advisor.first_name.lower():
            continue

        for advisor_term in [advisor.first_name, advisor.last_name, advisor.full_name]:
            normalized_input = matcher._normalize_text(advisor_term)

            for ayoub_term in ayoub_terms:
                normalized_ayoub = matcher._normalize_text(ayoub_term)
                score = SequenceMatcher(None, normalized_input, normalized_ayoub).ratio()

                if score >= matcher.fuzzy_threshold:
                    problematic.append({
                        "input": advisor_term,
                        "ayoub_term": ayoub_term,
                        "score": score,
                        "advisor": advisor.full_name,
                    })
                    print(f"  '{advisor_term}' (from {advisor.full_name})")
                    print(f"    matches '{ayoub_term}' with score {score:.3f} >= {matcher.fuzzy_threshold}")

    if problematic:
        print(f"\n⚠️ Found {len(problematic)} potential fuzzy matching issues")
    else:
        print("\n✅ No fuzzy matching issues found with current threshold")


def main():
    """Run all diagnostic tests."""
    print("=" * 80)
    print("AYOUB MATCHING DIAGNOSTIC TESTS")
    print("=" * 80)

    # Test 1: Which static names match Ayoub
    test_which_names_match_ayoub()

    # Test 2: Fuzzy matching analysis
    test_fuzzy_matching_threshold()

    # Test 3: Check real data
    test_load_real_data_and_check()

    print("\n" + "=" * 80)
    print("DIAGNOSTIC TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
