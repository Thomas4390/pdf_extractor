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
        'Compagnie': ColumnType.TEXT,
        'Conseiller': ColumnType.TEXT,
        'Lead/MC': ColumnType.TEXT,

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

    async def get_or_create_columns(
        self,
        board_id: int,
        column_names: list[str],
        default_type: ColumnType = ColumnType.TEXT
    ) -> dict[str, str]:
        """
        Get existing columns or create missing ones.

        Args:
            board_id: Target board ID
            column_names: List of column names to ensure exist
            default_type: Default type for new columns

        Returns:
            Mapping of column_name -> column_id
        """
        # Check cache first
        cache_key = str(board_id)
        if cache_key in self._column_cache:
            cached = self._column_cache[cache_key]
            # Check if all columns are in cache
            if all(name.lower() in [k.lower() for k in cached.keys()]
                   for name in column_names):
                return {
                    name: cached[k]
                    for name in column_names
                    for k in cached.keys()
                    if k.lower() == name.lower()
                }

        # Get existing columns
        existing = await self.list_columns(board_id)
        existing_map = {col["title"].lower(): col["id"] for col in existing}

        column_id_map = {}

        for col_name in column_names:
            col_name_lower = col_name.lower()

            if col_name_lower in existing_map:
                column_id_map[col_name] = existing_map[col_name_lower]
            else:
                # Determine column type
                col_type = self.COLUMN_TYPE_MAPPING.get(col_name, default_type)

                try:
                    new_col = await self.create_column(
                        board_id=board_id,
                        title=col_name,
                        column_type=col_type
                    )
                    column_id_map[col_name] = new_col["id"]
                    # Wait for Monday.com to process
                    await asyncio.sleep(1.0)
                except MondayError as e:
                    print(f"Warning: Could not create column '{col_name}': {e}")
                    continue

        # Update cache
        self._column_cache[cache_key] = column_id_map

        return column_id_map

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
        column_name: str
    ) -> Any:
        """Format a value for Monday.com API based on column type."""
        if pd.isna(value) or value is None or value == "":
            return None

        col_type = self.COLUMN_TYPE_MAPPING.get(column_name, ColumnType.TEXT)

        if col_type == ColumnType.NUMBERS:
            try:
                # Handle string numbers with French formatting
                if isinstance(value, str):
                    value = value.replace(" ", "").replace(",", ".")
                    value = value.replace("$", "").replace("%", "")
                return float(value)
            except (ValueError, TypeError):
                return None

        elif col_type == ColumnType.DATE:
            # Expect YYYY-MM-DD format
            if isinstance(value, str) and len(value) >= 10:
                return {"date": value[:10]}
            return None

        elif col_type == ColumnType.STATUS:
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

    def _row_to_column_values(
        self,
        row: pd.Series,
        column_id_map: dict[str, str]
    ) -> dict:
        """Convert a DataFrame row to Monday.com column_values format."""
        column_values = {}

        for col_name, col_id in column_id_map.items():
            if col_name in row.index and col_name != "Nom Client":
                formatted = self._format_column_value(row[col_name], col_name)
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
            column_id_map = await self.get_or_create_columns(
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

        # Upload items with rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)

        async def upload_row(idx: int, row: pd.Series) -> CreateResult:
            async with semaphore:
                item_name = str(row.get(item_name_column, f"Item {idx}"))
                column_values = self._row_to_column_values(row, column_id_map)

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
    # Synchronous wrappers
    # -------------------------------------------------------------------------

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
