"""
Monday.com API Client
Handles data extraction and updates with async batch processing and automatic fallback.
Includes board, group, and item creation functionality with reuse logic.

MODIFICATIONS:
- Ajout de list_columns() pour lister les colonnes d'un board
- Ajout de create_column() pour créer de nouvelles colonnes
- Ajout de get_or_create_columns() pour obtenir ou créer automatiquement les colonnes
"""

import asyncio
import json
import os
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import aiohttp
import pandas as pd
import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJ0aWQiOjU3OTYxMDI3NiwiYWFpIjoxMSwidWlkIjo5NTA2NjUzNywiaWFkIjoiMjAyNS0xMC0yOFQxNToxMDo0My40NjZaIiwicGVyIjoibWU6d3JpdGUiLCJhY3RpZCI6MjY0NjQxNDIsInJnbiI6InVzZTEifQ.q54YnC23stSJfLRnd0E9p9e4ZF8lRUK1TLgQM-13kdI"
API_URL = "https://api.monday.com/v2"

BOARD_ID_PAIEMENT = 18283488594
BOARD_ID_VENTES = 18283499297


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class UpdateResult:
    """Result of an item update operation."""
    success: bool
    item_id: str
    error: Optional[str] = None
    retries_used: int = 0


@dataclass
class BoardData:
    """Structured board data."""
    id: str
    name: str
    items: List[Dict]


@dataclass
class CreateBoardResult:
    """Result of board creation."""
    success: bool
    board_id: Optional[str] = None
    board_name: Optional[str] = None
    error: Optional[str] = None


@dataclass
class CreateGroupResult:
    """Result of group creation."""
    success: bool
    group_id: Optional[str] = None
    group_title: Optional[str] = None
    error: Optional[str] = None


@dataclass
class CreateItemResult:
    """Result of item creation."""
    success: bool
    item_id: Optional[str] = None
    item_name: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# API CLIENT
# =============================================================================

class MondayClient:
    """Client for Monday.com API operations."""

    def __init__(self, api_key: str, api_url: str = API_URL):
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': api_key
        }

    def _execute_query(self, query: str) -> Dict:
        """Execute a GraphQL query synchronously."""
        response = requests.post(
            url=self.api_url,
            headers=self.headers,
            json={'query': query}
        )

        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

        result = response.json()

        if 'errors' in result:
            raise Exception(f"GraphQL Error: {result['errors']}")

        return result

    def create_board(
        self,
        board_name: str,
        board_kind: str = "public",
        workspace_id: Optional[int] = None,
        folder_id: Optional[int] = None,
        reuse_existing: bool = True
    ) -> CreateBoardResult:
        """
        Create a new board in Monday.com, or reuse existing board with same name.

        Args:
            board_name: Name of the board to create
            board_kind: Type of board - "public", "private", or "shareable" (default: "public")
            workspace_id: Optional workspace ID to create the board in
            folder_id: Optional folder ID to create the board in
            reuse_existing: If True, reuse existing board with same name instead of creating new one (default: True)

        Returns:
            CreateBoardResult with board_id if successful
        """
        # Check if board with this name already exists
        if reuse_existing:
            try:
                print(f"🔍 Checking if board '{board_name}' already exists...")
                existing_boards = self.list_boards()

                for board in existing_boards:
                    if board['name'] == board_name:
                        print(f"✅ Board '{board_name}' already exists!")
                        print(f"   Board ID: {board['id']}")
                        print(f"   Reusing existing board instead of creating new one.")

                        return CreateBoardResult(
                            success=True,
                            board_id=board['id'],
                            board_name=board['name']
                        )

                print(f"   No existing board found with name '{board_name}'.")
                print(f"   Proceeding with board creation...")

            except Exception as e:
                print(f"⚠️  Warning: Could not check for existing boards: {e}")
                print(f"   Proceeding with board creation...")

        # Build optional arguments
        optional_args = []
        if workspace_id:
            optional_args.append(f'workspace_id: {workspace_id}')
        if folder_id:
            optional_args.append(f'folder_id: {folder_id}')

        optional_args_str = ', ' + ', '.join(optional_args) if optional_args else ''

        mutation = f"""
        mutation {{
            create_board(
                board_name: "{board_name}",
                board_kind: {board_kind}{optional_args_str}
            ) {{
                id
                name
            }}
        }}
        """

        try:
            result = self._execute_query(query=mutation)
            board_data = result['data']['create_board']

            print(f"✅ Board created successfully!")
            print(f"   Board ID: {board_data['id']}")
            print(f"   Board Name: {board_data['name']}")

            return CreateBoardResult(
                success=True,
                board_id=board_data['id'],
                board_name=board_data['name']
            )

        except Exception as e:
            print(f"❌ Failed to create board: {e}")
            return CreateBoardResult(success=False, error=str(e))

    def create_group(
        self,
        board_id: int,
        group_name: str,
        group_color: Optional[str] = None,
        relative_to: Optional[str] = None,
        position_method: Optional[str] = None,
        reuse_existing: bool = True
    ) -> CreateGroupResult:
        """
        Create a new group in a Monday.com board, or reuse existing group with same name.

        Args:
            board_id: ID of the board to create the group in
            group_name: Name of the group (e.g., "Octobre 2025")
            group_color: Optional hex color code (e.g., "#ff642e")
            relative_to: Optional group ID to position relative to
            position_method: Optional position method - "before_at" or "after_at"
            reuse_existing: If True, reuse existing group with same name instead of creating new one (default: True)

        Returns:
            CreateGroupResult with group_id if successful
        """
        # Check if group with this name already exists
        if reuse_existing:
            try:
                print(f"🔍 Checking if group '{group_name}' already exists in board {board_id}...")
                existing_groups = self.list_groups(board_id=board_id)

                for group in existing_groups:
                    if group['title'] == group_name:
                        print(f"✅ Group '{group_name}' already exists!")
                        print(f"   Group ID: {group['id']}")
                        print(f"   Reusing existing group instead of creating new one.")

                        return CreateGroupResult(
                            success=True,
                            group_id=group['id'],
                            group_title=group['title']
                        )

                print(f"   No existing group found with name '{group_name}'.")
                print(f"   Proceeding with group creation...")

            except Exception as e:
                print(f"⚠️  Warning: Could not check for existing groups: {e}")
                print(f"   Proceeding with group creation...")

        # Build optional arguments
        optional_args = []
        if group_color:
            optional_args.append(f'group_color: "{group_color}"')
        if relative_to:
            optional_args.append(f'relative_to: "{relative_to}"')
        if position_method:
            optional_args.append(f'position_relative_method: {position_method}')

        optional_args_str = ', ' + ', '.join(optional_args) if optional_args else ''

        mutation = f"""
        mutation {{
            create_group(
                board_id: {board_id},
                group_name: "{group_name}"{optional_args_str}
            ) {{
                id
                title
            }}
        }}
        """

        try:
            result = self._execute_query(query=mutation)
            group_data = result['data']['create_group']

            print(f"✅ Group created successfully!")
            print(f"   Group ID: {group_data['id']}")
            print(f"   Group Title: {group_data['title']}")

            return CreateGroupResult(
                success=True,
                group_id=group_data['id'],
                group_title=group_data['title']
            )

        except Exception as e:
            print(f"❌ Failed to create group: {e}")
            return CreateGroupResult(success=False, error=str(e))

    def create_item(
        self,
        board_id: int,
        item_name: str,
        group_id: Optional[str] = None,
        column_values: Optional[Dict] = None
    ) -> CreateItemResult:
        """
        Create a new item in a Monday.com board.

        Args:
            board_id: ID of the board to create the item in
            item_name: Name of the item to create
            group_id: Optional group ID to create the item in
            column_values: Optional dictionary of column values
                          Keys should be column IDs, values should be properly formatted
                          according to the column type

        Returns:
            CreateItemResult with item_id if successful

        Example:
            column_values = {
                "status": {"label": "Done"},
                "date4": {"date": "2025-10-30"},
                "text": "Some text value"
            }
        """
        # Build optional arguments
        optional_args = []
        if group_id:
            optional_args.append(f'group_id: "{group_id}"')
        if column_values:
            # Double JSON encoding for Monday.com API
            column_values_json = json.dumps(json.dumps(column_values))
            optional_args.append(f'column_values: {column_values_json}')

        optional_args_str = ', ' + ', '.join(optional_args) if optional_args else ''

        mutation = f"""
        mutation {{
            create_item(
                board_id: {board_id},
                item_name: "{item_name}"{optional_args_str}
            ) {{
                id
                name
            }}
        }}
        """

        try:
            result = self._execute_query(query=mutation)
            item_data = result['data']['create_item']

            return CreateItemResult(
                success=True,
                item_id=item_data['id'],
                item_name=item_data['name']
            )

        except Exception as e:
            print(f"❌ Failed to create item '{item_name}': {e}")
            return CreateItemResult(success=False, error=str(e))

    def create_items_batch(
        self,
        board_id: int,
        items: List[Dict[str, any]],
        group_id: Optional[str] = None
    ) -> List[CreateItemResult]:
        """
        Create multiple items in a Monday.com board.

        Args:
            board_id: ID of the board to create items in
            items: List of item dictionaries with 'name' and optional 'column_values'
            group_id: Optional group ID to create all items in

        Returns:
            List of CreateItemResult objects

        Example:
            items = [
                {
                    "name": "Item 1",
                    "column_values": {"status": {"label": "Working on it"}}
                },
                {
                    "name": "Item 2",
                    "column_values": {"text": "Description here"}
                }
            ]
        """
        results = []
        total = len(items)

        print(f"\n🚀 Creating {total} items in board {board_id}...")

        for i, item_data in enumerate(items, 1):
            item_name = item_data.get('name')
            column_values = item_data.get('column_values')

            result = self.create_item(
                board_id=board_id,
                item_name=item_name,
                group_id=group_id,
                column_values=column_values
            )

            results.append(result)

            if result.success:
                print(f"  [{i}/{total}] ✅ Created: {item_name}")
            else:
                print(f"  [{i}/{total}] ❌ Failed: {item_name}")

            # Small delay to avoid rate limiting
            if i < total:
                time.sleep(0.5)

        # Summary
        successful = sum(1 for r in results if r.success)
        failed = total - successful

        print(f"\n📊 Batch Creation Summary:")
        print(f"  Total items:   {total}")
        print(f"  ✅ Successful: {successful}")
        print(f"  ❌ Failed:     {failed}")

        return results

    def list_boards(self) -> List[Dict]:
        """List all boards from Monday.com account."""
        query = """
        {
            boards {
                id
                name
                description
                state
                board_kind
            }
        }
        """
        result = self._execute_query(query=query)
        return result['data']['boards']

    def list_groups(self, board_id: int) -> List[Dict]:
        """List all groups in a board."""
        query = f"""
        {{
            boards(ids: {board_id}) {{
                groups {{
                    id
                    title
                    color
                }}
            }}
        }}
        """
        result = self._execute_query(query=query)
        return result['data']['boards'][0]['groups']

    def list_items(
        self,
        board_id: int,
        group_id: Optional[str] = None,
        limit: int = 500
    ) -> List[Dict]:
        """
        List items in a board or group (returns just id and name for quick lookup).

        Args:
            board_id: The board ID
            group_id: Optional group ID to filter by
            limit: Maximum items to return (default: 500)

        Returns:
            List of dictionaries with 'id' and 'name' keys
        """
        if group_id:
            query = f"""
            {{
                boards(ids: {board_id}) {{
                    groups(ids: ["{group_id}"]) {{
                        items_page(limit: {limit}) {{
                            items {{
                                id
                                name
                            }}
                        }}
                    }}
                }}
            }}
            """
        else:
            query = f"""
            {{
                boards(ids: {board_id}) {{
                    items_page(limit: {limit}) {{
                        items {{
                            id
                            name
                        }}
                    }}
                }}
            }}
            """

        result = self._execute_query(query=query)

        if group_id:
            items = result['data']['boards'][0]['groups'][0]['items_page']['items']
        else:
            items = result['data']['boards'][0]['items_page']['items']

        return items

    def list_columns(self, board_id: int) -> List[Dict]:
        """
        List all columns in a board with their IDs, titles, and types.

        Args:
            board_id: The board ID

        Returns:
            List of dictionaries with 'id', 'title', 'type', and 'settings_str' keys

        Example return:
            [
                {'id': 'text', 'title': 'Text Column', 'type': 'text', 'settings_str': '{}'},
                {'id': 'status', 'title': 'Status', 'type': 'color', 'settings_str': '...'},
                {'id': 'date4', 'title': 'Date', 'type': 'date', 'settings_str': '{}'}
            ]
        """
        query = f"""
        {{
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

        result = self._execute_query(query=query)
        return result['data']['boards'][0]['columns']

    def create_column(
        self,
        board_id: int,
        title: str,
        column_type: str = "text",
        defaults: Optional[Dict] = None
    ) -> Dict:
        """
        Create a new column in a Monday.com board.

        Args:
            board_id: ID of the board to create the column in
            title: Title of the column to create
            column_type: Type of column (default: "text")
                Supported types:
                - "text": Simple text column
                - "long-text": Long text column
                - "numbers": Number column
                - "status": Status column (with labels)
                - "date": Date column
                - "dropdown": Dropdown column
                - "email": Email column
                - "phone": Phone column
            defaults: Optional default settings for the column (e.g., labels for status)

        Returns:
            Dictionary with column data (id, title, type)

        Example:
            # Create a text column
            column = client.create_column(board_id=123, title="Contract Number", column_type="text")

            # Create a status column with labels
            column = client.create_column(
                board_id=123,
                title="Status",
                column_type="status",
                defaults={"labels": {"0": "Active", "1": "Pending", "2": "Closed"}}
            )
        """
        # Build defaults argument if provided
        defaults_arg = ""
        if defaults:
            defaults_json = json.dumps(json.dumps(defaults))
            defaults_arg = f', defaults: {defaults_json}'

        mutation = f"""
        mutation {{
            create_column(
                board_id: {board_id},
                title: "{title}",
                column_type: {column_type}{defaults_arg}
            ) {{
                id
                title
                type
            }}
        }}
        """

        try:
            result = self._execute_query(query=mutation)
            column_data = result['data']['create_column']

            print(f"✅ Column created: {column_data['title']} (ID: {column_data['id']}, Type: {column_data['type']})")

            return column_data

        except Exception as e:
            print(f"❌ Failed to create column '{title}': {e}")
            raise

    def get_or_create_columns(
        self,
        board_id: int,
        column_names: List[str],
        column_type: str = "text"
    ) -> Dict[str, str]:
        """
        Get existing columns or create them if they don't exist.
        Returns a mapping of column_name -> column_id.

        Args:
            board_id: The board ID
            column_names: List of column names (titles) to get or create
            column_type: Default type for new columns (default: "text")

        Returns:
            Dictionary mapping column names to column IDs

        Example:
            mapping = client.get_or_create_columns(
                board_id=123,
                column_names=["contract_number", "insured_name", "commission_amount"]
            )
            # Returns: {"contract_number": "text_1", "insured_name": "text_2", "commission_amount": "text_3"}
        """
        # Get existing columns
        existing_columns = self.list_columns(board_id=board_id)

        # Create mapping of title -> id for existing columns (case-insensitive)
        existing_map = {
            col['title'].lower(): col['id']
            for col in existing_columns
        }

        # Result mapping
        column_id_map = {}

        print(f"\n🔍 Checking columns for board {board_id}...")

        for col_name in column_names:
            # Check if column already exists (case-insensitive match)
            col_name_lower = col_name.lower()

            if col_name_lower in existing_map:
                # Column exists
                column_id = existing_map[col_name_lower]
                column_id_map[col_name] = column_id
                print(f"  ✓ Column '{col_name}' exists (ID: {column_id})")
            else:
                # Column doesn't exist - create it
                print(f"  ➕ Creating column '{col_name}'...")
                try:
                    new_column = self.create_column(
                        board_id=board_id,
                        title=col_name,
                        column_type=column_type
                    )
                    column_id_map[col_name] = new_column['id']

                    # NOUVEAU: Délai après création pour laisser Monday.com traiter
                    print(f"     ⏳ Waiting 1 second for Monday.com to process...")
                    time.sleep(1.0)
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not create column '{col_name}': {e}")
                    # Skip this column
                    continue

        print(f"\n📋 Column mapping ready: {len(column_id_map)} columns available")

        return column_id_map

    def extract_board_data(
        self,
        board_id: int,
        group_id: Optional[str] = None
    ) -> BoardData:
        """
        Extract all items from a board with pagination support.

        Args:
            board_id: The board ID to extract
            group_id: Optional group ID to filter by

        Returns:
            BoardData object with all items
        """
        all_items = []
        cursor = None
        board_name = None
        board_id_str = None

        print(f"  Fetching items from board {board_id}" +
              (f", group {group_id}..." if group_id else "..."))

        while True:
            cursor_arg = f', cursor: "{cursor}"' if cursor else ''

            if group_id:
                query = self._build_group_query(
                    board_id=board_id,
                    group_id=group_id,
                    cursor_arg=cursor_arg
                )
            else:
                query = self._build_board_query(
                    board_id=board_id,
                    cursor_arg=cursor_arg
                )

            result = self._execute_query(query=query)
            board_data = result['data']['boards'][0]

            # Store metadata from first request
            if board_name is None:
                board_name = board_data['name']
                board_id_str = board_data['id']

            # Extract items based on query type
            if group_id and board_data.get('groups'):
                items_page = board_data['groups'][0]['items_page']
            else:
                items_page = board_data['items_page']

            items = items_page['items']

            if items:
                all_items.extend(items)
                print(f"    Retrieved {len(items)} items (total: {len(all_items)})")

            # Check for more pages
            cursor = items_page.get('cursor')
            if not cursor:
                break

        print(f"  ✓ Total items retrieved: {len(all_items)}")

        return BoardData(
            id=board_id_str,
            name=board_name,
            items=all_items
        )

    @staticmethod
    def _build_board_query(board_id: int, cursor_arg: str) -> str:
        """Build GraphQL query for board extraction."""
        return f"""
        {{
            boards(ids: {board_id}) {{
                id
                name
                items_page(limit: 500{cursor_arg}) {{
                    cursor
                    items {{
                        id
                        name
                        group {{
                            id
                            title
                        }}
                        column_values {{
                            id
                            column {{
                                title
                            }}
                            value
                            text
                            type
                        }}
                        subitems {{
                            id
                            name
                            column_values {{
                                id
                                column {{
                                    title
                                }}
                                value
                                text
                                type
                            }}
                        }}
                    }}
                }}
            }}
        }}
        """

    @staticmethod
    def _build_group_query(
        board_id: int,
        group_id: str,
        cursor_arg: str
    ) -> str:
        """Build GraphQL query for group extraction."""
        return f"""
        {{
            boards(ids: {board_id}) {{
                id
                name
                groups(ids: ["{group_id}"]) {{
                    id
                    title
                    items_page(limit: 500{cursor_arg}) {{
                        cursor
                        items {{
                            id
                            name
                            group {{
                                id
                                title
                            }}
                            column_values {{
                                id
                                column {{
                                    title
                                }}
                                value
                                text
                                type
                            }}
                            subitems {{
                                id
                                name
                                column_values {{
                                    id
                                    column {{
                                        title
                                    }}
                                    value
                                    text
                                    type
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
        """

    def update_item_sync(
        self,
        item_id: str,
        new_name: str,
        board_id: int
    ) -> UpdateResult:
        """
        Update a single item synchronously without retry (single attempt).
        Used internally by update_item_sync_with_retry.
        """
        mutation = f"""
        mutation {{
            change_multiple_column_values(
                item_id: {item_id},
                board_id: {board_id},
                column_values: {json.dumps(json.dumps({"name": new_name}))}
            ) {{
                id
                name
            }}
        }}
        """

        try:
            result = self._execute_query(query=mutation)
            return UpdateResult(success=True, item_id=item_id)
        except Exception as e:
            return UpdateResult(success=False, item_id=item_id, error=str(e))

    def update_item_sync_with_retry(
        self,
        item_id: str,
        new_name: str,
        board_id: int,
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> UpdateResult:
        """
        Update a single item synchronously with exponential backoff retry.

        Args:
            item_id: The item ID to update
            new_name: The new name for the item
            board_id: The board ID
            max_retries: Maximum number of retry attempts (default: 3)
            base_delay: Base delay in seconds for exponential backoff (default: 1.0)

        Returns:
            UpdateResult with success status and retry count
        """
        last_error = None

        for attempt in range(max_retries + 1):  # +1 for initial attempt
            result = self.update_item_sync(
                item_id=item_id,
                new_name=new_name,
                board_id=board_id
            )

            if result.success:
                result.retries_used = attempt
                return result

            last_error = result.error

            # If not last attempt, wait with exponential backoff
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)  # 1s, 2s, 4s, 8s...
                print(f"    ⚠️  Attempt {attempt + 1}/{max_retries + 1} failed for item {item_id}. "
                      f"Retrying in {delay}s...")
                time.sleep(delay)

        # All attempts failed
        return UpdateResult(
            success=False,
            item_id=item_id,
            error=last_error,
            retries_used=max_retries
        )

    async def _update_item_async(
        self,
        session: aiohttp.ClientSession,
        item_id: str,
        new_name: str,
        board_id: int,
        semaphore: asyncio.Semaphore
    ) -> UpdateResult:
        """Update a single item asynchronously."""
        mutation = f"""
        mutation {{
            change_multiple_column_values(
                item_id: {item_id},
                board_id: {board_id},
                column_values: {json.dumps(json.dumps({"name": new_name}))}
            ) {{
                id
                name
            }}
        }}
        """

        async with semaphore:
            try:
                async with session.post(
                    url=self.api_url,
                    headers=self.headers,
                    json={'query': mutation}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if 'errors' not in result:
                            return UpdateResult(success=True, item_id=item_id)
                        else:
                            return UpdateResult(
                                success=False,
                                item_id=item_id,
                                error=str(result['errors'])
                            )
                    else:
                        return UpdateResult(
                            success=False,
                            item_id=item_id,
                            error=f'HTTP {response.status}'
                        )
            except Exception as e:
                return UpdateResult(
                    success=False,
                    item_id=item_id,
                    error=str(e)
                )

    async def _update_items_async_batch(
        self,
        items: List[Tuple[str, str]],
        board_id: int,
        max_concurrent: int
    ) -> List[UpdateResult]:
        """Update multiple items asynchronously."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._update_item_async(
                    session=session,
                    item_id=item_id,
                    new_name=new_name,
                    board_id=board_id,
                    semaphore=semaphore
                )
                for item_id, new_name in items
            ]

            results = []
            total = len(tasks)

            for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                result = await coro
                results.append(result)

                if i % 10 == 0 or i == total:
                    success_count = sum(1 for r in results if r.success)
                    print(f"  Progress: {i}/{total} ({success_count} successful)")

            return results

    def update_items_with_fallback(
        self,
        df: pd.DataFrame,
        board_id: int,
        group_filter: Optional[str] = None,
        max_concurrent: int = 20,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> int:
        """
        Update items with async processing and automatic sync fallback with retry.

        Args:
            df: DataFrame with 'item_id' and 'item_name' columns
            board_id: The board ID to update
            group_filter: Optional group title to filter updates
            max_concurrent: Maximum concurrent async requests (default: 20)
            max_retries: Maximum retry attempts for failed items (default: 3)
            retry_delay: Base delay in seconds for exponential backoff (default: 1.0)

        Returns:
            Number of successfully updated items
        """
        # Filter by group if specified
        df_to_update = df.copy()
        if group_filter and 'group_title' in df.columns:
            df_to_update = df_to_update[df_to_update['group_title'] == group_filter]
            print(f"Filtering by group '{group_filter}': {len(df_to_update)} items to update")

        total_items = len(df_to_update)
        print(f"\n🚀 Starting async update of {total_items} items "
              f"(max {max_concurrent} concurrent)...")

        # Prepare items list
        items = [
            (row['item_id'], row['item_name'])
            for _, row in df_to_update.iterrows()
        ]

        # Execute async batch
        results = asyncio.run(
            self._update_items_async_batch(
                items=items,
                board_id=board_id,
                max_concurrent=max_concurrent
            )
        )

        # Separate successes and failures
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        # Retry failed items synchronously with retry logic
        if failed:
            print(f"\n🔄 Retrying {len(failed)} failed items synchronously "
                  f"(max {max_retries} retries per item with exponential backoff)...")

            retry_successful = []
            retry_failed = []

            for i, failed_result in enumerate(failed, 1):
                # Find the item in dataframe
                item_row = df_to_update[df_to_update['item_id'] == failed_result.item_id].iloc[0]

                print(f"  [{i}/{len(failed)}] Retrying item {failed_result.item_id}...")

                retry_result = self.update_item_sync_with_retry(
                    item_id=failed_result.item_id,
                    new_name=item_row['item_name'],
                    board_id=board_id,
                    max_retries=max_retries,
                    base_delay=retry_delay
                )

                if retry_result.success:
                    retry_successful.append(retry_result)
                    print(f"    ✅ Success after {retry_result.retries_used} retries")
                else:
                    retry_failed.append(retry_result)
                    print(f"    ❌ Failed after {max_retries} retries: {retry_result.error}")

            # Update successful list
            successful.extend(retry_successful)

            # Show retry summary
            if retry_successful:
                print(f"\n📊 Retry Summary:")
                print(f"  ✅ Recovered: {len(retry_successful)}/{len(failed)}")
                print(f"  ❌ Still failed: {len(retry_failed)}/{len(failed)}")

        # Final summary
        final_success_count = len(successful)
        final_failed_count = total_items - final_success_count

        print(f"\n{'='*80}")
        print(f"📊 Final Update Summary:")
        print(f"  Total items:      {total_items}")
        print(f"  ✅ Successful:    {final_success_count}")
        print(f"  ❌ Failed:        {final_failed_count}")
        if final_success_count > 0:
            success_rate = (final_success_count / total_items) * 100
            print(f"  📈 Success rate:  {success_rate:.1f}%")
        print(f"{'='*80}")

        return final_success_count


# =============================================================================
# DATA PROCESSING
# =============================================================================

class DataProcessor:
    """Handles data transformation and DataFrame operations."""

    @staticmethod
    def board_to_dataframe(
        board_data: BoardData,
        include_subitems: bool = True
    ) -> pd.DataFrame:
        """
        Convert board data to a structured DataFrame.

        Args:
            board_data: BoardData object from extraction
            include_subitems: Whether to include subitems as separate rows

        Returns:
            Structured DataFrame with all board data
        """
        rows = []

        for item in board_data.items:
            group_id = item.get('group', {}).get('id')
            group_title = item.get('group', {}).get('title')

            row = {
                'board_id': board_data.id,
                'board_name': board_data.name,
                'group_id': group_id,
                'group_title': group_title,
                'item_id': item['id'],
                'item_name': item['name'],
                'is_subitem': False,
                'parent_item_id': None,
                'parent_item_name': None
            }

            # Add column values
            for col_value in item['column_values']:
                column_title = (col_value['column']['title']
                              if col_value.get('column')
                              else col_value['id'])
                cell_value = col_value['text'] or col_value['value']
                row[column_title] = cell_value

            rows.append(row)

            # Process subitems
            if include_subitems and item.get('subitems'):
                for subitem in item['subitems']:
                    subitem_row = {
                        'board_id': board_data.id,
                        'board_name': board_data.name,
                        'group_id': group_id,
                        'group_title': group_title,
                        'item_id': subitem['id'],
                        'item_name': subitem['name'],
                        'is_subitem': True,
                        'parent_item_id': item['id'],
                        'parent_item_name': item['name']
                    }

                    for col_value in subitem['column_values']:
                        column_title = (col_value['column']['title']
                                      if col_value.get('column')
                                      else col_value['id'])
                        cell_value = col_value['text'] or col_value['value']
                        subitem_row[column_title] = cell_value

                    rows.append(subitem_row)

        return pd.DataFrame(rows)

    @staticmethod
    def remove_copy_from_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove all occurrences of '(copy)' from item_name column.

        Args:
            df: DataFrame with 'item_name' column

        Returns:
            DataFrame with cleaned names
        """
        original_df = df.copy()

        mask = df['item_name'].str.contains(r'\(copy\)', case=False, na=False)
        items_to_clean = mask.sum()

        if items_to_clean == 0:
            print("No items with '(copy)' found in names.")
            return df

        # Remove all (copy) occurrences and extra spaces
        df.loc[mask, 'item_name'] = df.loc[mask, 'item_name'].str.replace(
            r'(\s*\(copy\)\s*)+',
            ' ',
            case=False,
            regex=True
        ).str.strip()

        print(f"\n📝 Transformation Summary:")
        print(f"  Items modified: {items_to_clean}")
        print(f"\nExamples of changes:")

        changed_items = original_df[mask][['item_id', 'item_name']].head(5)
        for idx, row in changed_items.iterrows():
            old_name = row['item_name']
            new_name = df.loc[idx, 'item_name']
            copy_count = old_name.lower().count('(copy)')
            print(f"  '{old_name}' → '{new_name}' ({copy_count} removed)")

        return df

    @staticmethod
    def save_to_csv(df: pd.DataFrame, output_dir: str = '../results/') -> str:
        """
        Save DataFrame to CSV file.

        Args:
            df: DataFrame to save
            output_dir: Directory to save CSV

        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)

        board_name = df['board_name'].iloc[0].replace(' ', '_').replace('/', '_')
        board_id = df['board_id'].iloc[0]
        filename = os.path.join(
            output_dir,
            f"monday_board_{board_id}_{board_name}.csv"
        )

        df.to_csv(filename, index=False, encoding='utf-8-sig')
        return filename


# =============================================================================
# UTILITIES
# =============================================================================

def print_boards(boards: List[Dict]) -> None:
    """Display boards in a readable format."""
    print(f"\n{'=' * 80}")
    print(f"MONDAY.COM BOARDS ({len(boards)} found)")
    print(f"{'=' * 80}\n")

    for board in boards:
        print(f"ID:          {board['id']}")
        print(f"Name:        {board['name']}")
        print(f"State:       {board['state']}")
        print(f"Type:        {board['board_kind']}")
        if board.get('description'):
            print(f"Description: {board['description']}")
        print("-" * 80)


def configure_pandas_display() -> None:
    """Configure pandas display options for better readability."""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Main execution function."""
    try:
        print("=" * 80)
        print("MONDAY.COM DATA EXTRACTION & UPDATE")
        print("=" * 80)

        # Initialize client
        client = MondayClient(api_key=API_KEY)
        processor = DataProcessor()

        # =============================================================================
        # EXAMPLE: CREATE BOARD, GROUP, AND ITEMS (REUSE FOR BOARD & GROUP ONLY)
        # =============================================================================

        print("\n" + "=" * 80)
        print("EXAMPLE: CREATE BOARD WITH GROUP AND ITEMS")
        print("(Board and Group reuse existing, Items always created)")
        print("=" * 80)

        # 1. Create a new board (or reuse existing)
        print("\n1. Creating or reusing board...")
        board_result = client.create_board(
            board_name="Test Automation Board",
            board_kind="public",
            reuse_existing=True  # Will reuse if exists
        )

        if not board_result.success:
            print("Failed to create/reuse board. Exiting example.")
        else:
            new_board_id = int(board_result.board_id)

            # 2. Create a group in the board (or reuse existing)
            print("\n2. Creating or reusing group 'Octobre 2025'...")
            group_result = client.create_group(
                board_id=new_board_id,
                group_name="Octobre 2025",
                group_color="#ff642e",  # Orange color
                reuse_existing=True  # Will reuse if exists
            )

            if not group_result.success:
                print("Failed to create/reuse group. Continuing with default group.")
                new_group_id = None
            else:
                new_group_id = group_result.group_id

            # 3. Create items in the group
            print("\n3. Creating items in the group...")
            items_to_create = [
                {"name": "Task 1: Setup environment"},
                {"name": "Task 2: Write documentation"},
                {"name": "Task 3: Test implementation"},
                {"name": "Task 4: Deploy to production"},
                {"name": "Task 5: Monitor results"}
            ]

            item_results = client.create_items_batch(
                board_id=new_board_id,
                items=items_to_create,
                group_id=new_group_id
            )

            print(f"\n✅ Example completed! Board ID: {new_board_id}")
            print(f"   You can view it in your Monday.com account.")
            print(f"\n💡 Note: Board and group will be reused if they exist, but items are always created.")

        print("\n✅ Script completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()