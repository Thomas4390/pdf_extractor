"""
Test script to detect conflicts in advisor variations.

A conflict occurs when one advisor's variation matches another advisor's name.
This can cause data to be incorrectly attributed.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.utils.advisor_matcher import get_advisor_matcher


def find_variation_conflicts():
    """Find conflicts where one advisor's variation matches another advisor's name."""
    print("=" * 80)
    print("ADVISOR VARIATION CONFLICT DETECTION")
    print("=" * 80)

    matcher = get_advisor_matcher()
    print(f"Loaded {len(matcher.advisors)} advisors")

    conflicts = []

    for advisor in matcher.advisors:
        # Check if any variation matches another advisor's full name
        for variation in advisor.variations:
            variation_lower = variation.lower().strip()

            for other_advisor in matcher.advisors:
                if other_advisor.full_name == advisor.full_name:
                    continue  # Skip self

                # Check if variation matches other advisor's first name, last name, or full name
                other_names = [
                    other_advisor.first_name.lower(),
                    other_advisor.last_name.lower(),
                    other_advisor.full_name.lower(),
                ]

                for other_name in other_names:
                    if variation_lower == other_name:
                        conflicts.append({
                            "advisor": advisor.full_name,
                            "variation": variation,
                            "conflicts_with": other_advisor.full_name,
                            "matched_field": other_name,
                        })

    if conflicts:
        print(f"\n⚠️ Found {len(conflicts)} conflict(s):\n")
        for c in conflicts:
            print(f"  CONFLICT: '{c['advisor']}' has variation '{c['variation']}'")
            print(f"    But '{c['conflicts_with']}' exists as a separate advisor!")
            print(f"    This will cause data from '{c['conflicts_with']}' to be attributed to '{c['advisor']}'")
            print()

        print("FIX: Remove conflicting variations from the advisor database (Google Sheets)")
        print("     Access the Advisors sheet and update the 'variations' column")
    else:
        print("\n✅ No conflicts found")

    return conflicts


def main():
    conflicts = find_variation_conflicts()
    return len(conflicts) > 0


if __name__ == "__main__":
    has_conflicts = main()
    sys.exit(1 if has_conflicts else 0)
