"""
Monday.com API Client for PDF Extractor.

Handles GraphQL operations for uploading extracted data to Monday.com boards.
Supports automatic column creation, batch uploads with rate limiting, and
proper column type mapping.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import httpx
import pandas as pd

from ..utils.data_unifier import BoardType


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default API URL
API_URL = "https://api.monday.com/v2"

# Rate limiting
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_CONCURRENT = 5
RATE_LIMIT_DELAY = 0.3  # seconds between requests


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ColumnType(str, Enum):
    """Monday.com column types."""
    TEXT = "text"
    LONG_TEXT = "long_text"
    NUMBERS = "numbers"
    STATUS = "status"
    DATE = "date"
    CHECKBOX = "checkbox"
    EMAIL = "email"
    PHONE = "phone"
    DROPDOWN = "dropdown"


@dataclass
class MondayError(Exception):
    """Exception for Monday.com API errors."""
    message: str
    status_code: Optional[int] = None
    errors: Optional[list] = None

    def __str__(self) -> str:
        return f"MondayError: {self.message}"


@dataclass
class CreateResult:
    """Result of a create operation."""
    success: bool
    id: Optional[str] = None
    name: Optional[str] = None
    error: Optional[str] = None
    reused: bool = False


@dataclass
class UploadResult:
    """Result of a batch upload operation."""
    total: int = 0
    success: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)


# =============================================================================
# MONDAY CLIENT
# =============================================================================

class MondayClient:
    """
    Client for Monday.com GraphQL API.

    Features:
    - Board, group, and item CRUD operations
    - Automatic column creation
    - Batch upload with rate limiting
    - Async operations with httpx
    """

    # Mapping DataFrame columns to Monday.com column types
    COLUMN_TYPE_MAPPING: dict[str, ColumnType] = {
        # Text columns
        '# de Police': ColumnType.TEXT,
        'Nom Client': ColumnType.TEXT,
        'Compagnie': ColumnType.STATUS,
        'Conseiller': ColumnType.STATUS,
        'Lead/MC': ColumnType.STATUS,

        # Numeric columns
        'PA': ColumnType.NUMBERS,
        'Com': ColumnType.NUMBERS,
        'Boni': ColumnType.NUMBERS,
        'Sur-Com': ColumnType.NUMBERS,
        'Reçu': ColumnType.NUMBERS,
        'Reçu 1': ColumnType.NUMBERS,
        'Reçu 2': ColumnType.NUMBERS,
        'Reçu 3': ColumnType.NUMBERS,
        'Total': ColumnType.NUMBERS,
        'Total Reçu': ColumnType.NUMBERS,

        # Status columns
        'Statut': ColumnType.STATUS,

        # Checkbox columns
        'Verifié': ColumnType.CHECKBOX,
        'Complet': ColumnType.CHECKBOX,

        # Date columns
        'Date': ColumnType.DATE,
        'Paie': ColumnType.DATE,

        # Long text columns
        'Texte': ColumnType.LONG_TEXT,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = API_URL,
        timeout: float = 30.0
    ):
        """
        Initialize Monday.com client.

        Args:
            api_key: Monday.com API key (JWT token). If not provided,
                    reads from MONDAY_API_KEY environment variable.
            api_url: API endpoint URL.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.getenv("MONDAY_API_KEY", "")
        self.api_url = api_url
        self.timeout = timeout

        if not self.api_key:
            raise MondayError("MONDAY_API_KEY not provided or found in environment")

        self.headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
            "API-Version": "2024-01"
        }

        # Cache for column mappings
        self._column_cache: dict[str, dict[str, str]] = {}

    # -------------------------------------------------------------------------
    # Low-level API methods
    # -------------------------------------------------------------------------

    async def _execute_query(
        self,
        query: str,
        variables: Optional[dict] = None
    ) -> dict:
        """Execute a GraphQL query asynchronously."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )

            if response.status_code != 200:
                raise MondayError(
                    message=f"HTTP {response.status_code}: {response.text}",
                    status_code=response.status_code
                )

            result = response.json()

            if "errors" in result:
                raise MondayError(
                    message=str(result["errors"]),
                    errors=result["errors"]
                )

            return result

    def _execute_query_sync(self, query: str, variables: Optional[dict] = None) -> dict:
        """Execute a GraphQL query synchronously."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        response = httpx.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout
        )

        if response.status_code != 200:
            raise MondayError(
                message=f"HTTP {response.status_code}: {response.text}",
                status_code=response.status_code
            )

        result = response.json()

        if "errors" in result:
            raise MondayError(
                message=str(result["errors"]),
                errors=result["errors"]
            )

        return result

    # -------------------------------------------------------------------------
    # Board operations
    # -------------------------------------------------------------------------

    async def list_boards(self, limit: int = 100) -> list[dict]:
        """List all boards in the account."""
        all_boards = []
        page = 1

        while True:
            query = f"""
            {{
                boards(limit: {limit}, page: {page}) {{
                    id
                    name
                    description
                    state
                    board_kind
                }}
            }}
            """
            result = await self._execute_query(query)
            boards = result["data"]["boards"]

            if not boards:
                break

            all_boards.extend(boards)

            if len(boards) < limit:
                break

            page += 1

        return all_boards

    async def get_or_create_board(
        self,
        board_name: str,
        board_kind: str = "public",
        workspace_id: Optional[int] = None
    ) -> CreateResult:
        """Get existing board or create a new one."""
        # Check for existing board
        boards = await self.list_boards()
        for board in boards:
            if board["name"] == board_name:
                return CreateResult(
                    success=True,
                    id=board["id"],
                    name=board["name"],
                    reused=True
                )

        # Create new board
        optional_args = ""
        if workspace_id:
            optional_args = f", workspace_id: {workspace_id}"

        mutation = f"""
        mutation {{
            create_board(
                board_name: "{board_name}",
                board_kind: {board_kind}{optional_args}
            ) {{
                id
                name
            }}
        }}
        """

        try:
            result = await self._execute_query(mutation)
            board_data = result["data"]["create_board"]
            return CreateResult(
                success=True,
                id=board_data["id"],
                name=board_data["name"],
                reused=False
            )
        except MondayError as e:
            return CreateResult(success=False, error=str(e))

    # -------------------------------------------------------------------------
    # Group operations
    # -------------------------------------------------------------------------

    async def list_groups(self, board_id: int) -> list[dict]:
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
        result = await self._execute_query(query)
        return result["data"]["boards"][0]["groups"]

    async def get_or_create_group(
        self,
        board_id: int,
        group_name: str,
        group_color: Optional[str] = None
    ) -> CreateResult:
        """Get existing group or create a new one."""
        # Check for existing group
        groups = await self.list_groups(board_id)
        for group in groups:
            if group["title"] == group_name:
                return CreateResult(
                    success=True,
                    id=group["id"],
                    name=group["title"],
                    reused=True
                )

        # Create new group
        optional_args = ""
        if group_color:
            optional_args = f', group_color: "{group_color}"'

        mutation = f"""
        mutation {{
            create_group(
                board_id: {board_id},
                group_name: "{group_name}"{optional_args}
            ) {{
                id
                title
            }}
        }}
        """

        try:
            result = await self._execute_query(mutation)
            group_data = result["data"]["create_group"]
            return CreateResult(
                success=True,
                id=group_data["id"],
                name=group_data["title"],
                reused=False
            )
        except MondayError as e:
            return CreateResult(success=False, error=str(e))

    # -------------------------------------------------------------------------
    # Column operations
    # -------------------------------------------------------------------------

    async def list_columns(self, board_id: int) -> list[dict]:
        """List all columns in a board."""
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
        result = await self._execute_query(query)
        return result["data"]["boards"][0]["columns"]

    async def create_column(
        self,
        board_id: int,
        title: str,
        column_type: ColumnType = ColumnType.TEXT,
        defaults: Optional[dict] = None
    ) -> dict:
        """Create a new column on a board."""
        defaults_arg = ""
        if defaults:
            defaults_json = json.dumps(json.dumps(defaults))
            defaults_arg = f", defaults: {defaults_json}"

        mutation = f"""
        mutation {{
            create_column(
                board_id: {board_id},
                title: "{title}",
                column_type: {column_type.value}{defaults_arg}
            ) {{
                id
                title
                type
            }}
        }}
        """

        result = await self._execute_query(mutation)
        return result["data"]["create_column"]

    # Column types that are read-only (computed by Monday.com)
    READ_ONLY_COLUMN_TYPES = {
        "formula",
        "auto_number",
        "creation_log",
        "last_updated",
        "item_id",
        "subtasks",
        "dependency",
        "board_relation",
        "mirror",
        "button",
        "doc",
    }

    async def get_or_create_columns(
        self,
        board_id: int,
        column_names: list[str],
        default_type: ColumnType = ColumnType.TEXT
    ) -> tuple[dict[str, str], dict[str, str]]:
        """
        Get existing columns or create missing ones.

        Args:
            board_id: Target board ID
            column_names: List of column names to ensure exist
            default_type: Default type for new columns

        Returns:
            Tuple of (column_id_map, column_type_map):
            - column_id_map: column_name -> column_id (excludes read-only columns)
            - column_type_map: column_name -> actual Monday.com column type
        """
        # Get existing columns (always refresh to check types)
        existing = await self.list_columns(board_id)

        # Build maps: title -> id and title -> type
        existing_map = {col["title"].lower(): col["id"] for col in existing}
        existing_type_map = {col["title"].lower(): col["type"] for col in existing}

        column_id_map = {}
        column_type_map = {}

        for col_name in column_names:
            col_name_lower = col_name.lower()

            if col_name_lower in existing_map:
                col_id = existing_map[col_name_lower]
                col_type = existing_type_map.get(col_name_lower, "text")

                # Skip if column ID is missing (malformed data)
                if not col_id:
                    print(f"Skipping column '{col_name}' with missing ID")
                    continue

                # Skip read-only column types
                if col_type in self.READ_ONLY_COLUMN_TYPES:
                    print(f"Skipping read-only column '{col_name}' (type: {col_type})")
                    continue

                # Also skip by column ID prefix (backup check)
                if col_id.startswith(self.READ_ONLY_COLUMN_ID_PREFIXES):
                    print(f"Skipping read-only column '{col_name}' (id: {col_id})")
                    continue

                column_id_map[col_name] = col_id
                column_type_map[col_name] = col_type
            else:
                # Determine column type
                col_type_enum = self.COLUMN_TYPE_MAPPING.get(col_name, default_type)

                try:
                    new_col = await self.create_column(
                        board_id=board_id,
                        title=col_name,
                        column_type=col_type_enum
                    )
                    column_id_map[col_name] = new_col["id"]
                    column_type_map[col_name] = col_type_enum.value
                    # Wait for Monday.com to process
                    await asyncio.sleep(1.0)
                except MondayError as e:
                    print(f"Warning: Could not create column '{col_name}': {e}")
                    continue

        return column_id_map, column_type_map

    # -------------------------------------------------------------------------
    # Item operations
    # -------------------------------------------------------------------------

    async def create_item(
        self,
        board_id: int,
        item_name: str,
        group_id: Optional[str] = None,
        column_values: Optional[dict] = None,
        create_labels_if_missing: bool = True
    ) -> CreateResult:
        """Create a new item on a board."""
        # Build optional arguments
        optional_args = []
        if group_id:
            optional_args.append(f'group_id: "{group_id}"')
        if column_values:
            # Double JSON encoding for Monday.com API
            column_values_json = json.dumps(json.dumps(column_values))
            optional_args.append(f"column_values: {column_values_json}")

        optional_args.append(
            f"create_labels_if_missing: {str(create_labels_if_missing).lower()}"
        )

        optional_args_str = ", " + ", ".join(optional_args) if optional_args else ""

        # Escape item name for GraphQL
        escaped_name = item_name.replace('"', '\\"').replace('\n', ' ')

        mutation = f"""
        mutation {{
            create_item(
                board_id: {board_id},
                item_name: "{escaped_name}"{optional_args_str}
            ) {{
                id
                name
            }}
        }}
        """

        try:
            result = await self._execute_query(mutation)
            item_data = result["data"]["create_item"]
            return CreateResult(
                success=True,
                id=item_data["id"],
                name=item_data["name"]
            )
        except MondayError as e:
            return CreateResult(success=False, error=str(e))

    # -------------------------------------------------------------------------
    # Batch upload
    # -------------------------------------------------------------------------

    def _format_column_value(
        self,
        value: Any,
        column_name: str,
        actual_type: Optional[str] = None
    ) -> Any:
        """Format a value for Monday.com API based on actual column type.

        Args:
            value: The value to format
            column_name: Column name (used as fallback for type lookup)
            actual_type: Actual Monday.com column type string (e.g., "text", "numbers")
        """
        if pd.isna(value) or value is None or value == "":
            return None

        # Use actual Monday.com type if provided, otherwise fall back to our mapping
        if actual_type:
            # Map Monday.com type string to our formatting logic
            if actual_type == "numbers" or actual_type == "numeric":
                try:
                    if isinstance(value, str):
                        value = value.replace(" ", "").replace(",", ".")
                        value = value.replace("$", "").replace("%", "")
                    return float(value)
                except (ValueError, TypeError):
                    return None

            elif actual_type == "date":
                if isinstance(value, str) and len(value) >= 10:
                    return {"date": value[:10]}
                return None

            elif actual_type in ("status", "color", "dropdown"):
                # Monday.com returns "color" for status columns, not "status"
                return {"label": str(value)}

            elif actual_type == "checkbox" or actual_type == "boolean":
                if isinstance(value, bool):
                    return {"checked": "true" if value else "false"}
                if isinstance(value, str):
                    return {"checked": "true" if value.lower() in ("true", "oui", "yes", "1") else "false"}
                return None

            elif actual_type == "long_text":
                return {"text": str(value)}

            else:
                # text, short_text, and other text-like columns
                return str(value) if value else None

        # Fallback to our mapping if actual_type not provided
        col_type = self.COLUMN_TYPE_MAPPING.get(column_name, ColumnType.TEXT)

        if col_type == ColumnType.NUMBERS:
            try:
                if isinstance(value, str):
                    value = value.replace(" ", "").replace(",", ".")
                    value = value.replace("$", "").replace("%", "")
                return float(value)
            except (ValueError, TypeError):
                return None

        elif col_type == ColumnType.DATE:
            if isinstance(value, str) and len(value) >= 10:
                return {"date": value[:10]}
            return None

        elif col_type in (ColumnType.STATUS, ColumnType.DROPDOWN):
            return {"label": str(value)}

        elif col_type == ColumnType.CHECKBOX:
            if isinstance(value, bool):
                return {"checked": "true" if value else "false"}
            if isinstance(value, str):
                return {"checked": "true" if value.lower() in ("true", "oui", "yes", "1") else "false"}
            return None

        elif col_type == ColumnType.LONG_TEXT:
            return {"text": str(value)}

        else:
            # Text column
            return str(value) if value else None

    # Column ID prefixes that indicate read-only columns
    READ_ONLY_COLUMN_ID_PREFIXES = (
        "formula",
        "auto_number",
        "creation_log",
        "last_updated",
        "item_id",
        "subitems",
        "dependency",
        "board_relation",
        "mirror",
        "button",
        "doc",
    )

    def _row_to_column_values(
        self,
        row: pd.Series,
        column_id_map: dict[str, str],
        column_type_map: Optional[dict[str, str]] = None
    ) -> dict:
        """Convert a DataFrame row to Monday.com column_values format.

        Automatically skips formula columns and other read-only columns
        that cannot be set via the API.

        Args:
            row: DataFrame row to convert
            column_id_map: column_name -> column_id mapping
            column_type_map: column_name -> actual Monday.com type (optional)
        """
        column_values = {}
        column_type_map = column_type_map or {}

        for col_name, col_id in column_id_map.items():
            # Skip read-only columns (formula, auto_number, etc.)
            # These columns are auto-calculated by Monday.com
            if col_id.startswith(self.READ_ONLY_COLUMN_ID_PREFIXES):
                continue

            if col_name in row.index and col_name != "Nom Client":
                actual_type = column_type_map.get(col_name)
                formatted = self._format_column_value(row[col_name], col_name, actual_type)
                if formatted is not None:
                    column_values[col_id] = formatted

        return column_values

    async def upload_dataframe(
        self,
        df: pd.DataFrame,
        board_id: int,
        group_id: Optional[str] = None,
        item_name_column: str = "Nom Client",
        create_missing_columns: bool = True,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        progress_callback: Optional[callable] = None
    ) -> UploadResult:
        """
        Upload a DataFrame to Monday.com.

        Args:
            df: DataFrame with standardized columns
            board_id: Target board ID
            group_id: Target group ID (optional)
            item_name_column: Column to use as item name
            create_missing_columns: Auto-create missing columns
            max_concurrent: Maximum concurrent requests
            progress_callback: Optional callback(current, total)

        Returns:
            UploadResult with statistics
        """
        result = UploadResult(total=len(df))

        if df.empty:
            return result

        # Get or create columns
        columns_to_create = [
            col for col in df.columns
            if col != item_name_column
        ]

        if create_missing_columns:
            column_id_map, column_type_map = await self.get_or_create_columns(
                board_id=board_id,
                column_names=columns_to_create
            )
        else:
            existing = await self.list_columns(board_id)
            column_id_map = {
                col["title"]: col["id"]
                for col in existing
                if col["title"] in columns_to_create
            }
            column_type_map = {
                col["title"]: col["type"]
                for col in existing
                if col["title"] in columns_to_create
            }

        # Upload items with rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)

        async def upload_row(idx: int, row: pd.Series) -> CreateResult:
            async with semaphore:
                item_name = str(row.get(item_name_column, f"Item {idx}"))
                column_values = self._row_to_column_values(row, column_id_map, column_type_map)

                create_result = await self.create_item(
                    board_id=board_id,
                    item_name=item_name,
                    group_id=group_id,
                    column_values=column_values
                )

                # Rate limiting
                await asyncio.sleep(RATE_LIMIT_DELAY)

                if progress_callback:
                    progress_callback(idx + 1, result.total)

                return create_result

        # Execute uploads
        tasks = [
            upload_row(idx, row)
            for idx, row in df.iterrows()
        ]

        create_results = await asyncio.gather(*tasks)

        # Aggregate results
        for create_result in create_results:
            if create_result.success:
                result.success += 1
                if create_result.id:
                    result.item_ids.append(create_result.id)
            else:
                result.failed += 1
                if create_result.error:
                    result.errors.append(create_result.error)

        return result

    # -------------------------------------------------------------------------
    # Data extraction (read operations)
    # -------------------------------------------------------------------------

    async def extract_board_data(
        self,
        board_id: int,
        group_id: Optional[str] = None,
        limit: int = 500,
    ) -> list[dict]:
        """
        Extract all items from a board with pagination.

        Args:
            board_id: Board ID to extract from
            group_id: Optional group ID to filter by
            limit: Items per page (max 500)

        Returns:
            List of item dictionaries with column values
        """
        all_items = []
        cursor = None

        while True:
            # Build cursor argument
            cursor_arg = f', cursor: "{cursor}"' if cursor else ""

            # Build query based on whether we're filtering by group
            if group_id:
                query = f"""
                {{
                    boards(ids: {board_id}) {{
                        groups(ids: ["{group_id}"]) {{
                            items_page(limit: {limit}{cursor_arg}) {{
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
                                        text
                                        value
                                        column {{
                                            title
                                            type
                                        }}
                                    }}
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
                        items_page(limit: {limit}{cursor_arg}) {{
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
                                    text
                                    value
                                    column {{
                                        title
                                        type
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
                """

            result = await self._execute_query(query)

            # Extract items based on query structure
            if group_id:
                groups = result["data"]["boards"][0]["groups"]
                if not groups:
                    break
                items_page = groups[0]["items_page"]
            else:
                items_page = result["data"]["boards"][0]["items_page"]

            items = items_page["items"]
            cursor = items_page["cursor"]

            if not items:
                break

            all_items.extend(items)

            if not cursor:
                break

        return all_items

    def board_items_to_dataframe(
        self,
        items: list[dict],
        include_item_id: bool = True,
        include_group: bool = True,
    ) -> pd.DataFrame:
        """
        Convert Monday.com items to a pandas DataFrame.

        Args:
            items: List of items from extract_board_data
            include_item_id: Include item ID column
            include_group: Include group info columns

        Returns:
            DataFrame with item data
        """
        if not items:
            return pd.DataFrame()

        rows = []
        for item in items:
            row = {"item_name": item["name"]}

            if include_item_id:
                row["item_id"] = item["id"]

            if include_group and item.get("group"):
                row["group_id"] = item["group"]["id"]
                row["group_title"] = item["group"]["title"]

            # Extract column values
            for col_val in item.get("column_values", []):
                col_title = col_val["column"]["title"]
                col_type = col_val["column"]["type"]

                # Use text representation for most columns
                # For numbers, try to parse the value
                if col_type == "numbers" or col_type == "numeric":
                    try:
                        raw_value = col_val.get("value")
                        if raw_value:
                            import json as json_module
                            parsed = json_module.loads(raw_value)
                            row[col_title] = float(parsed) if parsed else None
                        else:
                            row[col_title] = None
                    except (ValueError, TypeError, json.JSONDecodeError):
                        row[col_title] = col_val.get("text")
                else:
                    row[col_title] = col_val.get("text")

            rows.append(row)

        return pd.DataFrame(rows)

    async def get_items_by_column_value(
        self,
        board_id: int,
        column_id: str,
        column_value: str,
        limit: int = 50,
    ) -> list[dict]:
        """
        Search for items by a specific column value.

        Args:
            board_id: Board ID to search in
            column_id: Column ID to search by
            column_value: Value to search for
            limit: Maximum items to return

        Returns:
            List of matching items
        """
        query = f"""
        {{
            items_page_by_column_values(
                board_id: {board_id},
                limit: {limit},
                columns: [{{column_id: "{column_id}", column_values: ["{column_value}"]}}]
            ) {{
                items {{
                    id
                    name
                    group {{
                        id
                        title
                    }}
                    column_values {{
                        id
                        text
                        value
                        column {{
                            title
                            type
                        }}
                    }}
                }}
            }}
        }}
        """
        result = await self._execute_query(query)
        return result["data"]["items_page_by_column_values"]["items"]

    async def update_item_column_values(
        self,
        item_id: str,
        board_id: int,
        column_values: dict,
        create_labels_if_missing: bool = True,
    ) -> bool:
        """
        Update column values for an existing item.

        Args:
            item_id: Item ID to update
            board_id: Board ID containing the item
            column_values: Dict of column_id -> value
            create_labels_if_missing: Create status labels if they don't exist

        Returns:
            True if successful
        """
        column_values_json = json.dumps(json.dumps(column_values))

        mutation = f"""
        mutation {{
            change_multiple_column_values(
                item_id: {item_id},
                board_id: {board_id},
                column_values: {column_values_json},
                create_labels_if_missing: {str(create_labels_if_missing).lower()}
            ) {{
                id
            }}
        }}
        """

        try:
            await self._execute_query(mutation)
            return True
        except MondayError:
            return False

    async def move_item_to_group(
        self,
        item_id: str,
        group_id: str,
    ) -> bool:
        """
        Move an item to a different group.

        Args:
            item_id: Item ID to move
            group_id: Target group ID

        Returns:
            True if successful
        """
        mutation = f"""
        mutation {{
            move_item_to_group(
                item_id: {item_id},
                group_id: "{group_id}"
            ) {{
                id
            }}
        }}
        """

        try:
            await self._execute_query(mutation)
            return True
        except MondayError:
            return False

    async def upsert_by_advisor(
        self,
        board_id: int,
        group_id: str,
        advisor_column_id: str,
        data: pd.DataFrame,
        column_id_map: dict[str, str],
        column_type_map: Optional[dict[str, str]] = None,
        advisor_column_name: str = "Conseiller",
        progress_callback: Optional[callable] = None,
    ) -> dict:
        """
        Upsert data by advisor - update existing items or create new ones.

        IMPORTANT: Only updates items within the target group. Items in other
        groups are NOT moved or modified, preserving historical data in
        separate period groups.

        Args:
            board_id: Target board ID
            group_id: Target group ID
            advisor_column_id: Column ID for the advisor column
            data: DataFrame with aggregated data (must have advisor_column_name)
            column_id_map: column_name -> column_id mapping
            column_type_map: column_name -> column_type mapping (optional)
            advisor_column_name: Name of advisor column in DataFrame
            progress_callback: Optional callback(current, total, action)

        Returns:
            Dict with {updated: int, created: int, errors: list}
        """
        result = {"updated": 0, "created": 0, "errors": []}
        column_type_map = column_type_map or {}
        total = len(data)

        for idx, row in data.iterrows():
            advisor_name = row.get(advisor_column_name)
            if not advisor_name or pd.isna(advisor_name):
                result["errors"].append(f"Row {idx}: missing advisor name")
                continue

            # Search for existing item with this advisor
            try:
                existing_items = await self.get_items_by_column_value(
                    board_id=board_id,
                    column_id=advisor_column_id,
                    column_value=str(advisor_name),
                    limit=50,  # Increased limit to find items across groups
                )
            except MondayError as e:
                result["errors"].append(f"Search error for {advisor_name}: {e}")
                continue

            # Filter to only items in the TARGET GROUP
            # This ensures we don't modify items in other period groups
            items_in_target_group = [
                item for item in existing_items
                if item.get("group", {}).get("id") == group_id
            ]

            # Build column values for update/create
            column_values = {}
            for col_name, col_id in column_id_map.items():
                if col_name == advisor_column_name:
                    continue  # Don't update the advisor column itself
                if col_name in row.index:
                    actual_type = column_type_map.get(col_name)
                    formatted = self._format_column_value(row[col_name], col_name, actual_type)
                    if formatted is not None:
                        column_values[col_id] = formatted

            if items_in_target_group:
                # Update existing item in target group
                item = items_in_target_group[0]
                item_id = item["id"]

                try:
                    # Update column values
                    if column_values:
                        await self.update_item_column_values(
                            item_id=item_id,
                            board_id=board_id,
                            column_values=column_values,
                        )

                    result["updated"] += 1
                except MondayError as e:
                    result["errors"].append(f"Update error for {advisor_name}: {e}")
            else:
                # Create new item in target group
                # (advisor doesn't exist in this group yet, even if in other groups)
                # Format advisor column value based on actual column type
                advisor_type = column_type_map.get(advisor_column_name)
                advisor_formatted = self._format_column_value(advisor_name, advisor_column_name, advisor_type)
                if advisor_formatted is not None:
                    column_values[advisor_column_id] = advisor_formatted

                try:
                    create_result = await self.create_item(
                        board_id=board_id,
                        item_name=str(advisor_name),
                        group_id=group_id,
                        column_values=column_values,
                    )
                    if create_result.success:
                        result["created"] += 1
                    else:
                        result["errors"].append(f"Create error for {advisor_name}: {create_result.error}")
                except MondayError as e:
                    result["errors"].append(f"Create error for {advisor_name}: {e}")

            # Rate limiting
            await asyncio.sleep(RATE_LIMIT_DELAY)

            if progress_callback:
                current = idx + 1 if isinstance(idx, int) else data.index.get_loc(idx) + 1
                progress_callback(current, total, "upsert")

        return result

    # -------------------------------------------------------------------------
    # Synchronous wrappers
    # -------------------------------------------------------------------------

    def extract_board_data_sync(
        self,
        board_id: int,
        group_id: Optional[str] = None,
    ) -> list[dict]:
        """Synchronous wrapper for extract_board_data."""
        return asyncio.run(self.extract_board_data(board_id, group_id))

    def upsert_by_advisor_sync(
        self,
        board_id: int,
        group_id: str,
        advisor_column_id: str,
        data: pd.DataFrame,
        column_id_map: dict[str, str],
        column_type_map: Optional[dict[str, str]] = None,
        advisor_column_name: str = "Conseiller",
        progress_callback: Optional[callable] = None,
    ) -> dict:
        """Synchronous wrapper for upsert_by_advisor."""
        return asyncio.run(
            self.upsert_by_advisor(
                board_id=board_id,
                group_id=group_id,
                advisor_column_id=advisor_column_id,
                data=data,
                column_id_map=column_id_map,
                column_type_map=column_type_map,
                advisor_column_name=advisor_column_name,
                progress_callback=progress_callback,
            )
        )

    def get_or_create_group_sync(
        self,
        board_id: int,
        group_name: str,
        group_color: Optional[str] = None,
    ) -> CreateResult:
        """Synchronous wrapper for get_or_create_group."""
        return asyncio.run(self.get_or_create_group(board_id, group_name, group_color))

    def upload_dataframe_sync(
        self,
        df: pd.DataFrame,
        board_id: int,
        group_id: Optional[str] = None,
        item_name_column: str = "Nom Client",
        create_missing_columns: bool = True,
        progress_callback: Optional[callable] = None
    ) -> UploadResult:
        """Synchronous wrapper for upload_dataframe."""
        return asyncio.run(
            self.upload_dataframe(
                df=df,
                board_id=board_id,
                group_id=group_id,
                item_name_column=item_name_column,
                create_missing_columns=create_missing_columns,
                progress_callback=progress_callback
            )
        )

    def list_boards_sync(self) -> list[dict]:
        """Synchronous wrapper for list_boards."""
        return asyncio.run(self.list_boards())

    def list_groups_sync(self, board_id: int) -> list[dict]:
        """Synchronous wrapper for list_groups."""
        return asyncio.run(self.list_groups(board_id))

    def list_columns_sync(self, board_id: int) -> list[dict]:
        """Synchronous wrapper for list_columns."""
        return asyncio.run(self.list_columns(board_id))


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_monday_client(api_key: Optional[str] = None) -> MondayClient:
    """
    Factory function to get a MondayClient instance.

    Args:
        api_key: Optional API key. If not provided, uses MONDAY_API_KEY env var.

    Returns:
        Configured MondayClient instance.
    """
    return MondayClient(api_key=api_key)


def get_board_id_for_type(board_type: BoardType) -> Optional[int]:
    """
    Get the default board ID for a board type from environment variables.

    Args:
        board_type: Type of board (HISTORICAL_PAYMENTS or SALES_PRODUCTION)

    Returns:
        Board ID or None if not configured.
    """
    if board_type == BoardType.HISTORICAL_PAYMENTS:
        board_id = os.getenv("BOARD_ID_PAIEMENT")
    else:
        board_id = os.getenv("BOARD_ID_VENTES")

    return int(board_id) if board_id else None
