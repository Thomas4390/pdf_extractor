"""
Fix script to remove conflicting variations from Ayoub Chamoumi's advisor record.

The issue: Ayoub has "Said Vital" as a variation, but Said Vital is a separate advisor.
This causes Said Vital's data to be incorrectly attributed to Ayoub.

Solution: Remove "Said Vital", "Vital Said", "S. Vital" from Ayoub's variations.
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


def fix_ayoub_variations():
    """Remove conflicting variations from Ayoub Chamoumi's record."""
    print("=" * 80)
    print("FIXING AYOUB CHAMOUMI VARIATIONS")
    print("=" * 80)

    matcher = get_advisor_matcher()

    if not matcher.is_configured:
        print(f"ERROR: Matcher not configured: {matcher.configuration_error}")
        return False

    # Find Ayoub
    ayoub = None
    for advisor in matcher.advisors:
        if advisor.first_name.lower() == "ayoub":
            ayoub = advisor
            break

    if not ayoub:
        print("ERROR: Ayoub not found in advisor database")
        return False

    print(f"\nCurrent Ayoub record:")
    print(f"  Name: {ayoub.full_name}")
    print(f"  Variations: {ayoub.variations}")

    # Variations to remove (those that conflict with "Said Vital" advisor)
    variations_to_remove = ["Said Vital", "Vital Said", "S. Vital"]

    # Filter out conflicting variations
    new_variations = [v for v in ayoub.variations if v not in variations_to_remove]

    if new_variations == ayoub.variations:
        print("\nNo conflicting variations to remove.")
        return True

    print(f"\nVariations to remove: {[v for v in ayoub.variations if v in variations_to_remove]}")
    print(f"New variations: {new_variations}")

    # Update the advisor
    try:
        matcher.update_advisor(ayoub, variations=new_variations)
        print("\n✅ Successfully updated Ayoub's variations in Google Sheets")

        # Verify the change
        matcher.reload()
        for advisor in matcher.advisors:
            if advisor.first_name.lower() == "ayoub":
                print(f"\nVerified - New variations: {advisor.variations}")
                break

        return True
    except Exception as e:
        print(f"\n❌ Error updating advisor: {e}")
        return False


def main():
    success = fix_ayoub_variations()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
