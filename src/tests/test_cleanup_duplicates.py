"""
Test the duplicate column cleanup function.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd


def test_cleanup_duplicate_columns():
    """Test that _cleanup_duplicate_columns works correctly."""
    from src.app.aggregation.execution import _cleanup_duplicate_columns

    print("=" * 60)
    print("TEST: _cleanup_duplicate_columns")
    print("=" * 60)

    # Create a DataFrame with duplicate columns
    df = pd.DataFrame({
        "Conseiller": ["Alice", "Bob", "Charlie"],
        "AE CA": [100, 200, 300],
        "Coût_x": [-50, -75, -100],
        "Coût_y": [-55, -80, -110],
        "Dépenses par Conseiller_x": [-10, -20, -30],
        "Dépenses par Conseiller_y": [-15, -25, -35],
        "Leads": [5, 10, 15],
    })

    print("\nBefore cleanup:")
    print(f"  Columns: {list(df.columns)}")

    # Clean up
    cleaned_df = _cleanup_duplicate_columns(df)

    print("\nAfter cleanup:")
    print(f"  Columns: {list(cleaned_df.columns)}")

    # Verify
    expected_cols = ["Conseiller", "AE CA", "Coût", "Dépenses par Conseiller", "Leads"]

    if list(cleaned_df.columns) == expected_cols:
        print("\n✅ TEST PASSED - Duplicate columns cleaned correctly")
    else:
        print(f"\n❌ TEST FAILED - Expected: {expected_cols}")

    # Verify values are from _x columns
    print("\nVerifying values:")
    print(f"  Coût: {cleaned_df['Coût'].tolist()} (should be [-50, -75, -100])")
    print(f"  Dépenses: {cleaned_df['Dépenses par Conseiller'].tolist()} (should be [-10, -20, -30])")

    return cleaned_df


if __name__ == "__main__":
    test_cleanup_duplicate_columns()
