"""
Script to retrieve label colors from Monday.com for the Profitable column.
"""

import os
import sys
import json
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# Import only what we need from monday.py without triggering other imports
import httpx

MONDAY_API_URL = "https://api.monday.com/v2"


async def get_column_settings(api_key: str, board_id: int):
    """Get column settings from Monday.com."""
    query = f"""
    query {{
        boards(ids: {board_id}) {{
            columns {{
                id
                title
                type
                settings_str
            }}
        }}
    }}
    """

    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            MONDAY_API_URL,
            json={"query": query},
            headers=headers,
            timeout=30.0,
        )
        result = response.json()

    if "errors" in result:
        print(f"Error: {result['errors']}")
        return []

    boards = result.get("data", {}).get("boards", [])
    if not boards:
        return []

    return boards[0].get("columns", [])


def main():
    api_key = os.getenv("MONDAY_API_KEY")
    if not api_key:
        print("ERROR: MONDAY_API_KEY not found")
        return

    # Data board ID
    board_id = 9142121714

    print(f"Fetching columns from board {board_id}...")
    columns = asyncio.run(get_column_settings(api_key, board_id))

    print(f"\nFound {len(columns)} columns\n")

    # Look for status columns (which have labels with colors)
    for col in columns:
        col_type = col.get("type", "")
        col_title = col.get("title", "")

        # Status columns have label colors
        if col_type in ("color", "status"):
            print(f"{'='*60}")
            print(f"Column: {col_title}")
            print(f"Type: {col_type}")
            print(f"ID: {col.get('id')}")

            settings_str = col.get("settings_str", "{}")
            try:
                settings = json.loads(settings_str)
                labels = settings.get("labels", {})
                labels_colors = settings.get("labels_colors", {})

                print(f"\nLabels:")
                for label_id, label_name in labels.items():
                    color = labels_colors.get(label_id, {})
                    color_hex = color.get("color", "N/A") if isinstance(color, dict) else color
                    print(f"  {label_id}: {label_name} -> {color_hex}")

            except json.JSONDecodeError:
                print(f"Could not parse settings: {settings_str}")

            print()


if __name__ == "__main__":
    main()
