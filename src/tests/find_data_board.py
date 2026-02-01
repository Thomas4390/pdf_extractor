"""
Script to find the board ID for the 'Data' board.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.clients.monday import MondayClient
from src.app.utils.async_helpers import run_async


def find_data_board():
    """Find the board named 'Data'."""
    api_key = os.environ.get("MONDAY_API_KEY")
    if not api_key:
        print("ERROR: MONDAY_API_KEY not set")
        return

    client = MondayClient(api_key=api_key)

    print("Loading boards...")
    boards = run_async(client.list_boards())

    print(f"\nFound {len(boards)} boards:\n")

    data_boards = []
    for board in boards:
        name = board.get("name", "")
        board_id = board.get("id", "")

        # Check if name contains "Data"
        if "data" in name.lower():
            data_boards.append((name, board_id))
            print(f"  ✅ '{name}' - ID: {board_id}")

    if not data_boards:
        print("\nNo boards found with 'Data' in the name.")
        print("\nAll boards:")
        for board in boards:
            print(f"  - '{board.get('name')}' - ID: {board.get('id')}")
    else:
        print(f"\n\nFound {len(data_boards)} board(s) with 'Data' in the name.")

        # Find exact match
        exact_match = [b for b in data_boards if b[0].lower() == "data"]
        if exact_match:
            print(f"\n🎯 Exact match: '{exact_match[0][0]}' - ID: {exact_match[0][1]}")


if __name__ == "__main__":
    find_data_board()
