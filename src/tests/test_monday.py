#!/usr/bin/env python3
"""
Test script for Monday.com client.

Usage:
    python -m src.tests.test_monday [command]

Commands:
    mock     - Run tests with mock data only (no API calls)
    boards   - List all boards (requires API key)
    columns  - List columns for a board (requires API key and BOARD_ID env)
    upload   - Test upload with sample data (requires API key)

Environment variables:
    MONDAY_API_KEY    - Required for API calls
    BOARD_ID_TEST     - Board ID for testing (optional)
"""

import asyncio
import os
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.clients.monday import (
    MondayClient,
    MondayError,
    ColumnType,
    CreateResult,
    UploadResult,
    get_monday_client,
    get_board_id_for_type,
)
from src.utils.data_unifier import BoardType


def create_mock_sales_dataframe() -> pd.DataFrame:
    """Create a mock Sales Production DataFrame for testing."""
    return pd.DataFrame([
        {
            "Date": "2025-10-15",
            "# de Police": "110970886",
            "Nom Client": "BALDWIN RAYMOND",
            "Compagnie": "UV Assurance",
            "Statut": "Approved",
            "Conseiller": "ACHRAF EL HAJJI",
            "Complet": True,
            "PA": 1196.00,
            "Lead/MC": "",
            "Com": 657.80,
            "Reçu 1": 0.0,
            "Boni": 1151.15,
            "Reçu 2": 0.0,
            "Sur-Com": 0.0,
            "Reçu 3": 0.0,
            "Total": 1808.95,
            "Total Reçu": 0.0,
            "Paie": "",
            "Texte": "Vie entière Valeurs Élevées",
        },
        {
            "Date": "2025-10-20",
            "# de Police": "110971504",
            "Nom Client": "MADJIGUENE SOW",
            "Compagnie": "UV Assurance",
            "Statut": "Pending",
            "Conseiller": "ACHRAF EL HAJJI",
            "Complet": False,
            "PA": 699.00,
            "Lead/MC": "",
            "Com": 153.78,
            "Reçu 1": 0.0,
            "Boni": 269.12,
            "Reçu 2": 0.0,
            "Sur-Com": 0.0,
            "Reçu 3": 0.0,
            "Total": 422.90,
            "Total Reçu": 0.0,
            "Paie": "",
            "Texte": "Temporaire 10 ans",
        },
    ])


def create_mock_historical_dataframe() -> pd.DataFrame:
    """Create a mock Historical Payments DataFrame for testing."""
    return pd.DataFrame([
        {
            "# de Police": "N894713",
            "Nom Client": "Jeanny Breault-Therrien",
            "Compagnie": "UV",
            "Statut": "Received",
            "Conseiller": "ACHRAF EL HAJJI",
            "Verifié": False,
            "PA": 0.0,
            "Com": 0.0,
            "Boni": 0.0,
            "Sur-Com": 98.76,
            "Reçu": 98.76,
            "Date": "2025-10-15",
            "Texte": "boni 75% #111011722 crt",
        },
        {
            "# de Police": "1234567",
            "Nom Client": "John Doe",
            "Compagnie": "Assomption",
            "Statut": "Pending",
            "Conseiller": "Thomas Greenberg",
            "Verifié": True,
            "PA": 0.0,
            "Com": 0.0,
            "Boni": 0.0,
            "Sur-Com": -40.00,
            "Reçu": 0.0,
            "Date": "2025-11-01",
            "Texte": "80% #123456",
        },
    ])


def test_column_type_mapping():
    """Test that column type mapping is correct."""
    print("\n[1/4] Testing column type mapping...")

    # Create client (won't connect, just testing mapping)
    # We can't create client without API key, so test mapping directly
    expected_types = {
        '# de Police': ColumnType.TEXT,
        'PA': ColumnType.NUMBERS,
        'Statut': ColumnType.STATUS,
        'Date': ColumnType.DATE,
        'Verifié': ColumnType.CHECKBOX,
        'Texte': ColumnType.LONG_TEXT,
    }

    for col_name, expected_type in expected_types.items():
        actual_type = MondayClient.COLUMN_TYPE_MAPPING.get(col_name)
        assert actual_type == expected_type, f"Mismatch for {col_name}: {actual_type} != {expected_type}"
        print(f"  ✓ {col_name} -> {expected_type.value}")

    print("  Column type mapping OK!")
    return True


def test_dataframe_creation():
    """Test that mock DataFrames are created correctly."""
    print("\n[2/4] Testing DataFrame creation...")

    df_sales = create_mock_sales_dataframe()
    df_historical = create_mock_historical_dataframe()

    print(f"  Sales DataFrame: {len(df_sales)} rows, {len(df_sales.columns)} columns")
    print(f"  Historical DataFrame: {len(df_historical)} rows, {len(df_historical.columns)} columns")

    # Verify columns
    expected_sales_cols = ['Date', '# de Police', 'Nom Client', 'Compagnie', 'Statut',
                          'Conseiller', 'Complet', 'PA', 'Lead/MC', 'Com']
    for col in expected_sales_cols:
        assert col in df_sales.columns, f"Missing column: {col}"

    expected_hist_cols = ['# de Police', 'Nom Client', 'Compagnie', 'Statut',
                         'Conseiller', 'Verifié', 'PA', 'Com', 'Boni', 'Sur-Com', 'Reçu', 'Date']
    for col in expected_hist_cols:
        assert col in df_historical.columns, f"Missing column: {col}"

    print("  DataFrame creation OK!")
    return True


def test_board_type_mapping():
    """Test board type to board ID mapping."""
    print("\n[3/4] Testing board type mapping...")

    # This tests the get_board_id_for_type function
    # Without env vars, it should return None

    # Save current env vars
    old_paiement = os.environ.pop("BOARD_ID_PAIEMENT", None)
    old_ventes = os.environ.pop("BOARD_ID_VENTES", None)

    try:
        # Test without env vars
        result = get_board_id_for_type(BoardType.HISTORICAL_PAYMENTS)
        assert result is None, f"Expected None, got {result}"
        print("  ✓ HISTORICAL_PAYMENTS without env -> None")

        result = get_board_id_for_type(BoardType.SALES_PRODUCTION)
        assert result is None, f"Expected None, got {result}"
        print("  ✓ SALES_PRODUCTION without env -> None")

        # Test with env vars
        os.environ["BOARD_ID_PAIEMENT"] = "12345"
        os.environ["BOARD_ID_VENTES"] = "67890"

        result = get_board_id_for_type(BoardType.HISTORICAL_PAYMENTS)
        assert result == 12345, f"Expected 12345, got {result}"
        print("  ✓ HISTORICAL_PAYMENTS with env -> 12345")

        result = get_board_id_for_type(BoardType.SALES_PRODUCTION)
        assert result == 67890, f"Expected 67890, got {result}"
        print("  ✓ SALES_PRODUCTION with env -> 67890")

    finally:
        # Restore env vars
        if old_paiement:
            os.environ["BOARD_ID_PAIEMENT"] = old_paiement
        else:
            os.environ.pop("BOARD_ID_PAIEMENT", None)
        if old_ventes:
            os.environ["BOARD_ID_VENTES"] = old_ventes
        else:
            os.environ.pop("BOARD_ID_VENTES", None)

    print("  Board type mapping OK!")
    return True


def test_create_result_dataclass():
    """Test CreateResult and UploadResult dataclasses."""
    print("\n[4/4] Testing result dataclasses...")

    # Test CreateResult
    result1 = CreateResult(success=True, id="123", name="Test Item")
    assert result1.success == True
    assert result1.id == "123"
    assert result1.reused == False
    print("  ✓ CreateResult basic")

    result2 = CreateResult(success=True, id="456", name="Existing", reused=True)
    assert result2.reused == True
    print("  ✓ CreateResult with reused")

    result3 = CreateResult(success=False, error="API Error")
    assert result3.success == False
    assert result3.error == "API Error"
    print("  ✓ CreateResult with error")

    # Test UploadResult
    upload = UploadResult(total=10, success=8, failed=2)
    assert upload.total == 10
    assert upload.success == 8
    assert upload.failed == 2
    assert upload.errors == []  # Default
    print("  ✓ UploadResult")

    print("  Result dataclasses OK!")
    return True


async def test_api_list_boards():
    """Test listing boards (requires API key)."""
    print("\n[API] Testing list_boards...")

    try:
        client = get_monday_client()
        boards = await client.list_boards()

        print(f"  Found {len(boards)} boards:")
        for board in boards[:5]:  # Show first 5
            print(f"    - {board['name']} (ID: {board['id']})")

        if len(boards) > 5:
            print(f"    ... and {len(boards) - 5} more")

        return True
    except MondayError as e:
        print(f"  Error: {e}")
        return False


async def test_api_list_columns():
    """Test listing columns (requires API key and BOARD_ID_TEST)."""
    print("\n[API] Testing list_columns...")

    board_id = os.getenv("BOARD_ID_TEST")
    if not board_id:
        print("  Skipping - BOARD_ID_TEST not set")
        return True

    try:
        client = get_monday_client()
        columns = await client.list_columns(int(board_id))

        print(f"  Found {len(columns)} columns:")
        for col in columns:
            print(f"    - {col['title']} (type: {col['type']}, id: {col['id']})")

        return True
    except MondayError as e:
        print(f"  Error: {e}")
        return False


async def test_api_list_groups():
    """Test listing groups (requires API key and BOARD_ID_TEST)."""
    print("\n[API] Testing list_groups...")

    board_id = os.getenv("BOARD_ID_TEST")
    if not board_id:
        print("  Skipping - BOARD_ID_TEST not set")
        return True

    try:
        client = get_monday_client()
        groups = await client.list_groups(int(board_id))

        print(f"  Found {len(groups)} groups:")
        for group in groups:
            print(f"    - {group['title']} (id: {group['id']})")

        return True
    except MondayError as e:
        print(f"  Error: {e}")
        return False


def run_mock_tests():
    """Run tests that don't require API calls."""
    print("\n" + "=" * 70)
    print("Monday.com Client - Mock Tests")
    print("=" * 70)

    tests = [
        test_column_type_mapping,
        test_dataframe_creation,
        test_board_type_mapping,
        test_create_result_dataclass,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 70)

    return passed == len(tests)


async def run_api_tests():
    """Run tests that require API calls."""
    print("\n" + "=" * 70)
    print("Monday.com Client - API Tests")
    print("=" * 70)

    api_key = os.getenv("MONDAY_API_KEY")
    if not api_key:
        print("\n  Skipping API tests - MONDAY_API_KEY not set")
        print("  Set MONDAY_API_KEY environment variable to run API tests")
        return True

    tests = [
        test_api_list_boards,
        test_api_list_columns,
        test_api_list_groups,
    ]

    passed = 0
    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\n" + "=" * 70)
    print(f"API Results: {passed}/{len(tests)} tests passed")
    print("=" * 70)

    return passed == len(tests)


async def main():
    """Main entry point."""
    command = sys.argv[1].lower() if len(sys.argv) > 1 else "mock"

    if command == "mock":
        run_mock_tests()
    elif command == "boards":
        await test_api_list_boards()
    elif command == "columns":
        await test_api_list_columns()
    elif command == "groups":
        await test_api_list_groups()
    elif command == "all":
        run_mock_tests()
        await run_api_tests()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python -m src.tests.test_monday [mock|boards|columns|groups|all]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
