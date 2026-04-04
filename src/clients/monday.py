"""
Monday.com API Client for PDF Extractor.

Handles GraphQL operations for uploading extracted data to Monday.com boards.
Supports automatic column creation, batch uploads with rate limiting, and
proper column type mapping.
"""

import asyncio
import json
import logging
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from collections.abc import Callable
from typing import Any, Optional

import httpx
import pandas as pd

from ..utils.data_unifier import BoardType

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default API URL
API_URL = "https://api.monday.com/v2"

# Rate limiting — defaults from config, with module-level fallbacks
def _get_monday_config():
    """Get Monday.com config values from settings."""
    try:
        from ..utils.config import get_settings
        cfg = get_settings()
        return cfg
    except (ImportError, AttributeError):
        return None

_cfg = _get_monday_config()
DEFAULT_BATCH_SIZE = _cfg.monday_batch_size if _cfg else 50
DEFAULT_MAX_CONCURRENT = _cfg.monday_max_concurrent if _cfg else 3
RATE_LIMIT_DELAY = _cfg.monday_rate_limit_delay if _cfg else 0.5

# Retry configuration for transient server errors (5xx) and rate limits
MAX_RETRIES = _cfg.monday_max_retries if _cfg else 3
RETRY_BASE_DELAY = _cfg.monday_retry_base_delay if _cfg else 2.0

# Rate-limit error codes returned by Monday.com GraphQL API
_RATE_LIMIT_CODES = frozenset({
    "RATE_LIMIT_EXCEEDED",
    "FIELD_MINUTE_RATE_LIMIT_EXCEEDED",
})
_RATE_LIMIT_MESSAGES = (
    "minute limit rate exceeded",
    "rate limit",
    "too many requests",
)


def _is_rate_limited(error: "MondayError") -> bool:
    """Check whether a MondayError signals a rate-limit condition."""
    if error.status_code == 429:
        return True
    if error.errors and isinstance(error.errors, list):
        for err in error.errors:
            code = err.get("extensions", {}).get("code", "")
            if code in _RATE_LIMIT_CODES:
                return True
            msg = err.get("message", "").lower()
            if any(phrase in msg for phrase in _RATE_LIMIT_MESSAGES):
                return True
    return False


def _get_retry_after(error: "MondayError") -> float:
    """Extract a retry delay from the error, with a safe minimum."""
    api_secs = 1.0
    if error.errors and isinstance(error.errors, list):
        for err in error.errors:
            api_secs = max(
                api_secs,
                err.get("extensions", {}).get("retry_in_seconds", 1),
            )
    return max(api_secs, 5.0)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ColumnType(StrEnum):
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

    def __post_init__(self):
        super().__init__(self.message)

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
    index_to_item_id: dict[int, str] = field(default_factory=dict)


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
        'Conseiller': ColumnType.DROPDOWN,  # Dropdown supports 1000+ options (vs 40 for STATUS)
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
        'Profitable': ColumnType.STATUS,
        'Advisor_Status': ColumnType.STATUS,

        # Status label columns
        'Verifié': ColumnType.STATUS,
        'Complet': ColumnType.STATUS,

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
            "API-Version": "2025-04"
        }

        # Per-instance cache: board_id -> {column_title: column_id}.
        # Populated lazily by _resolve_column_id_by_title().
        self._column_id_cache: dict[int, dict[str, str]] = {}

    # -------------------------------------------------------------------------
    # Low-level API methods
    # -------------------------------------------------------------------------

    async def _execute_query(
        self,
        query: str,
        variables: Optional[dict] = None
    ) -> dict:
        """Execute a GraphQL query asynchronously with retry on server/rate-limit errors."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.api_url,
                        headers=self.headers,
                        json=payload
                    )

                    if response.status_code == 429:
                        raise MondayError(
                            message=f"HTTP 429: {response.text}",
                            status_code=429
                        )

                    if response.status_code >= 500:
                        raise MondayError(
                            message=f"HTTP {response.status_code}: {response.text}",
                            status_code=response.status_code
                        )

                    if response.status_code != 200:
                        raise MondayError(
                            message=f"HTTP {response.status_code}: {response.text}",
                            status_code=response.status_code
                        )

                    result = response.json()

                    if "errors" in result:
                        errors = result["errors"]
                        raise MondayError(
                            message=str(errors),
                            errors=errors
                        )

                    return result

            except MondayError as e:
                last_error = e

                # Rate-limit → always retry with backoff
                if _is_rate_limited(e) and attempt < MAX_RETRIES - 1:
                    delay = min(_get_retry_after(e) * (2 ** attempt), 120.0)
                    logger.warning(
                        f"Monday.com rate limit (attempt {attempt + 1}/{MAX_RETRIES}). "
                        f"Retrying in {delay:.1f}s... Error: {e.message[:200]}"
                    )
                    await asyncio.sleep(delay)
                    continue

                # Server error (5xx) → retry with backoff
                is_server_error = (
                    e.status_code and e.status_code >= 500
                ) or (
                    e.errors and isinstance(e.errors, list) and any(
                        err.get("extensions", {}).get("status_code") == 500
                        or "internal server error" in err.get("message", "").lower()
                        for err in e.errors
                    )
                )
                if is_server_error and attempt < MAX_RETRIES - 1:
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), 120.0)
                    logger.warning(
                        f"Monday.com server error (attempt {attempt + 1}/{MAX_RETRIES}). "
                        f"Retrying in {delay}s... Error: {e.message[:200]}"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise MondayError("No attempts were made (MAX_RETRIES may be 0)")

    def _execute_query_sync(self, query: str, variables: Optional[dict] = None) -> dict:
        """Execute a GraphQL query synchronously with retry on server/rate-limit errors."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = httpx.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 429:
                    raise MondayError(
                        message=f"HTTP 429: {response.text}",
                        status_code=429
                    )

                if response.status_code >= 500:
                    raise MondayError(
                        message=f"HTTP {response.status_code}: {response.text}",
                        status_code=response.status_code
                    )

                if response.status_code != 200:
                    raise MondayError(
                        message=f"HTTP {response.status_code}: {response.text}",
                        status_code=response.status_code
                    )

                result = response.json()

                if "errors" in result:
                    errors = result["errors"]
                    raise MondayError(message=str(errors), errors=errors)

                return result

            except MondayError as e:
                last_error = e

                # Rate-limit → always retry with backoff
                if _is_rate_limited(e) and attempt < MAX_RETRIES - 1:
                    delay = min(_get_retry_after(e) * (2 ** attempt), 120.0)
                    logger.warning(
                        f"Monday.com rate limit (attempt {attempt + 1}/{MAX_RETRIES}). "
                        f"Retrying in {delay:.1f}s... Error: {e.message[:200]}"
                    )
                    time.sleep(delay)
                    continue

                # Server error (5xx) → retry with backoff
                is_server_error = (
                    e.status_code and e.status_code >= 500
                ) or (
                    e.errors and isinstance(e.errors, list) and any(
                        err.get("extensions", {}).get("status_code") == 500
                        or "internal server error" in err.get("message", "").lower()
                        for err in e.errors
                    )
                )
                if is_server_error and attempt < MAX_RETRIES - 1:
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), 120.0)
                    logger.warning(
                        f"Monday.com server error (attempt {attempt + 1}/{MAX_RETRIES}). "
                        f"Retrying in {delay}s... Error: {e.message[:200]}"
                    )
                    time.sleep(delay)
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise MondayError("No attempts were made (MAX_RETRIES may be 0)")

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
        boards = result["data"]["boards"]
        if not boards:
            raise MondayError(f"Board {board_id} not found or not accessible")
        return boards[0]["groups"]

    async def list_groups_with_item_count(self, board_id: int) -> list[dict]:
        """List all groups in a board with item count information.

        Uses items_page(limit: 1) to efficiently check if each group has items.

        Args:
            board_id: Board ID to list groups from

        Returns:
            List of group dicts with id, title, color, items_count, and is_empty
        """
        query = f"""
        {{
            boards(ids: {board_id}) {{
                groups {{
                    id
                    title
                    color
                    items_page(limit: 1) {{
                        cursor
                        items {{
                            id
                        }}
                    }}
                }}
            }}
        }}
        """
        result = await self._execute_query(query)
        boards = result["data"]["boards"]
        if not boards:
            raise MondayError(f"Board {board_id} not found or not accessible")

        groups = []
        for group in boards[0]["groups"]:
            items = group.get("items_page", {}).get("items", [])
            has_cursor = group.get("items_page", {}).get("cursor") is not None
            # If there are items or a cursor (meaning more items exist), it's not empty
            items_count = len(items)
            is_empty = items_count == 0 and not has_cursor
            groups.append({
                "id": group["id"],
                "title": group["title"],
                "color": group.get("color"),
                "items_count": items_count,
                "is_empty": is_empty,
            })
        return groups

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
        boards = result["data"]["boards"]
        if not boards:
            raise MondayError(f"Board {board_id} not found or not accessible")
        return boards[0]["columns"]

    async def _resolve_column_id_by_title(
        self, board_id: int, column_title: str,
    ) -> Optional[str]:
        """Resolve a column's ID from its title, with per-board caching.

        Returns None if the title is not found on the board.
        The cache is populated on first call for each board.
        """
        title_map = self._column_id_cache.get(board_id)
        if title_map is None:
            try:
                columns = await self.list_columns(board_id)
            except MondayError as e:
                logger.warning(
                    f"Could not list columns for board {board_id}: {e}"
                )
                return None
            title_map = {col["title"]: col["id"] for col in columns}
            self._column_id_cache[board_id] = title_map
        return title_map.get(column_title)

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

    async def rename_column(
        self,
        board_id: int,
        column_id: str,
        new_title: str
    ) -> dict:
        """Rename an existing column on a board.

        Args:
            board_id: Board ID containing the column
            column_id: Column ID to rename
            new_title: New title for the column

        Returns:
            Dict with column info {'id': str, 'title': str}
        """
        mutation = f"""
        mutation {{
            change_column_metadata(
                board_id: {board_id},
                column_id: "{column_id}",
                column_property: title,
                value: "{new_title}"
            ) {{
                id
                title
            }}
        }}
        """

        result = await self._execute_query(mutation)
        return result["data"]["change_column_metadata"]

    def rename_column_sync(
        self,
        board_id: int,
        column_id: str,
        new_title: str
    ) -> dict:
        """Synchronous wrapper for rename_column."""
        return asyncio.run(self.rename_column(board_id, column_id, new_title))

    async def get_column_values_for_all_items(
        self,
        board_id: int,
        column_id: str
    ) -> list[dict]:
        """Get all unique values from a column across all items.

        Args:
            board_id: Board ID
            column_id: Column ID to get values from

        Returns:
            List of dicts with {item_id: str, value: str}
        """
        items = await self.extract_board_data(board_id)
        result = []

        for item in items:
            item_id = item["id"]
            for col_val in item.get("column_values", []):
                if col_val["id"] == column_id:
                    text_value = col_val.get("text")
                    if text_value:
                        result.append({
                            "item_id": item_id,
                            "value": text_value
                        })
                    break

        return result

    async def preview_column_mapping(
        self,
        board_id: int,
        column_id: str,
        name_mapper: Optional[Callable[..., Any]] = None,
    ) -> list[dict]:
        """Preview the mapping that would be applied during migration.

        Reads all values from the column, applies the mapper, and returns
        a list of {original, mapped, changed} dicts for UI display.
        No writes are performed.

        Args:
            board_id: Board ID
            column_id: Column ID to read values from
            name_mapper: Optional function(str) -> str to transform values

        Returns:
            List of dicts with keys: original, mapped, changed
        """
        item_values = await self.get_column_values_for_all_items(board_id, column_id)
        preview = []
        for item_data in item_values:
            original = item_data["value"]
            if name_mapper:
                mapped = name_mapper(original)
            else:
                mapped = original
            # Normalize same as migration
            if mapped and isinstance(mapped, str):
                mapped = mapped.strip().title()
            preview.append({
                "original": original,
                "mapped": mapped,
                "changed": original != mapped,
            })
        return preview

    def preview_column_mapping_sync(
        self,
        board_id: int,
        column_id: str,
        name_mapper: Optional[Callable[..., Any]] = None,
    ) -> list[dict]:
        """Synchronous wrapper for preview_column_mapping."""
        return asyncio.run(self.preview_column_mapping(board_id, column_id, name_mapper))

    async def get_dropdown_label_map(
        self,
        board_id: int,
        column_id: str,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> dict[str, int]:
        """Read dropdown column settings and return {label_name: label_id}.

        Retries if the label map is empty, since Monday.com may need time
        to propagate column settings after creation.

        Args:
            board_id: Board ID containing the dropdown column
            column_id: Column ID of the dropdown column
            max_retries: Number of retries if label map is empty
            retry_delay: Seconds to wait between retries

        Returns:
            Dict mapping label name to label ID
        """
        for attempt in range(max_retries):
            columns = await self.list_columns(board_id)
            for col in columns:
                if col["id"] == column_id:
                    settings = json.loads(col.get("settings_str", "{}"))
                    labels = settings.get("labels", [])
                    if labels:
                        label_map = {}
                        if isinstance(labels, list):
                            # List format: [{"id": int, "name": str}, ...]
                            for label in labels:
                                label_map[label["name"]] = label["id"]
                        elif isinstance(labels, dict):
                            # Dict format: {"1": "Label Name", ...}
                            # or {"1": {"id": 1, "name": "Label Name"}, ...}
                            for key, val in labels.items():
                                if isinstance(val, str):
                                    label_map[val] = int(key)
                                elif isinstance(val, dict):
                                    label_map[val["name"]] = val.get("id", int(key))
                        if label_map:
                            return label_map
            # Labels not yet available, wait and retry
            if attempt < max_retries - 1:
                logger.debug("[dropdown] Label map empty on attempt %d, retrying in %.1fs...", attempt + 1, retry_delay)
                await asyncio.sleep(retry_delay)
        logger.warning("[dropdown] Label map still empty after %d attempts", max_retries)
        return {}

    async def migrate_column_to_dropdown(
        self,
        board_id: int,
        source_column_id: str,
        source_column_title: str,
        progress_callback: Optional[Callable[..., Any]] = None,
        name_mapper: Optional[Callable[..., Any]] = None,
        max_concurrent: int = 5
    ) -> dict:
        """Migrate a column (status/text) to a new dropdown column.

        This process:
        1. Reads all values and applies name mapping
        2. Creates a new dropdown column with all labels pre-created
        3. Reads back label name→ID mapping from column settings
        4. Writes each item using {"ids": [label_id]} (not by name)
        5. Verifies all values were written correctly
        6. Retries any mismatches sequentially

        Writing by label ID eliminates duplicate labels caused by
        Monday.com's unreliable name-matching under concurrent writes.

        The source column is left unchanged (not renamed or deleted).

        Args:
            board_id: Board ID
            source_column_id: Column ID to migrate
            source_column_title: Original column title
            progress_callback: Optional callback(current, total, message)
            name_mapper: Optional function(str) -> str to transform values
            max_concurrent: Maximum concurrent API calls (default 5)

        Returns:
            Dict with migration results
        """
        result = {
            "success": False,
            "old_column_id": source_column_id,
            "new_column_id": None,
            "items_migrated": 0,
            "values_mapped": 0,
            "verified": 0,
            "mismatches": 0,
            "retried": 0,
            "errors": []
        }

        try:
            # Step 1: Get all values from the source column
            if progress_callback:
                progress_callback(0, 100, "Lecture des valeurs existantes...")

            item_values = await self.get_column_values_for_all_items(
                board_id, source_column_id
            )

            if progress_callback:
                progress_callback(15, 100, f"Trouvé {len(item_values)} éléments avec des valeurs")

            # Step 2: Apply name mapping if provided
            if name_mapper:
                if progress_callback:
                    progress_callback(20, 100, "Application du mapping des noms...")

                for item_data in item_values:
                    original_value = item_data["value"]
                    mapped_value = name_mapper(original_value)
                    if mapped_value != original_value:
                        result["values_mapped"] += 1
                    item_data["mapped_value"] = mapped_value

                if progress_callback:
                    progress_callback(25, 100, f"Mappé {result['values_mapped']} noms")
            else:
                for item_data in item_values:
                    item_data["mapped_value"] = item_data["value"]

            # Normalize all mapped values to prevent duplicate labels
            for item_data in item_values:
                raw = item_data["mapped_value"]
                if raw and isinstance(raw, str):
                    item_data["mapped_value"] = raw.strip().title()

            # Step 3: Create new empty dropdown column
            if progress_callback:
                progress_callback(30, 100, f"Création de la colonne dropdown '{source_column_title}'...")

            new_column = await self.create_column(
                board_id=board_id,
                title=source_column_title,
                column_type=ColumnType.DROPDOWN,
            )

            new_column_id = new_column["id"]
            result["new_column_id"] = new_column_id

            # Deduplicate labels case-insensitively
            seen: dict[str, str] = {}
            for label in (iv["mapped_value"] for iv in item_values if iv["mapped_value"]):
                key = label.lower()
                if key not in seen:
                    seen[key] = label
            unique_labels = list(seen.values())[:200]

            # Step 4: Seed each unique label sequentially to avoid duplicates.
            # Monday.com's create_column defaults don't actually create labels,
            # so we write one item per unique label with create_labels_if_missing=True.
            if progress_callback:
                progress_callback(33, 100, f"Création de {len(unique_labels)} labels...")

            # Build a map: label -> first item_id that uses it
            label_to_first_item: dict[str, str] = {}
            for iv in item_values:
                val = iv["mapped_value"]
                if val and val not in label_to_first_item:
                    label_to_first_item[val] = iv["item_id"]

            for i, label in enumerate(unique_labels):
                seed_item_id = label_to_first_item.get(label)
                if not seed_item_id:
                    continue
                try:
                    await self.update_item_column_values(
                        item_id=seed_item_id,
                        board_id=board_id,
                        column_values={new_column_id: {"labels": [label]}},
                        create_labels_if_missing=True
                    )
                    await asyncio.sleep(0.3)
                except Exception as e:
                    result["errors"].append(f"Seed label '{label}': {e}")

                if progress_callback:
                    progress = 33 + int((i + 1) / len(unique_labels) * 5)
                    progress_callback(progress, 100, f"Label {i + 1}/{len(unique_labels)}: {label}")

            await asyncio.sleep(1)

            # Step 5: Read back label name→ID mapping
            if progress_callback:
                progress_callback(38, 100, "Lecture des labels créés...")

            label_map = await self.get_dropdown_label_map(board_id, new_column_id)

            if progress_callback:
                progress_callback(39, 100, f"Trouvé {len(label_map)} labels")

            if not label_map:
                result["errors"].append(
                    f"Impossible de lire les labels de la colonne {new_column_id}. "
                    f"Attendus: {len(unique_labels)} labels. Migration annulée."
                )
                return result

            # Step 6: Write remaining items using {"ids": [label_id]}
            # Items already seeded in step 4 are skipped.
            seeded_item_ids = set(label_to_first_item.values())
            remaining_items = [iv for iv in item_values if iv["item_id"] not in seeded_item_ids]
            # Count seeded items as already migrated
            result["items_migrated"] = len(seeded_item_ids)

            if progress_callback:
                progress_callback(40, 100, f"Copie des valeurs ({len(remaining_items)} restants)...")

            total_items = len(remaining_items)
            semaphore = asyncio.Semaphore(max_concurrent)
            completed = {"count": 0}

            async def update_single_item(item_data: dict) -> Optional[str]:
                """Update a single item with semaphore-controlled concurrency and retry with exponential backoff + jitter."""
                async with semaphore:
                    item_id = item_data["item_id"]
                    value = item_data["mapped_value"]
                    label_id = label_map.get(value)

                    if label_id is None:
                        return f"Item {item_id}: pas de label ID pour '{value}'"

                    max_retries = 5

                    for attempt in range(max_retries):
                        try:
                            column_values = {
                                new_column_id: {"ids": [label_id]}
                            }
                            await self.update_item_column_values(
                                item_id=item_id,
                                board_id=board_id,
                                column_values=column_values,
                                create_labels_if_missing=False
                            )

                            # Delay to respect rate limits
                            await asyncio.sleep(0.3)

                            completed["count"] += 1
                            if progress_callback and total_items > 0:
                                progress = 40 + int(completed["count"] / total_items * 50)
                                progress_callback(progress, 100, f"Migration {completed['count']}/{total_items}")
                            return None  # Success

                        except Exception as e:
                            if attempt < max_retries - 1:
                                # Exponential backoff + jitter to prevent thundering herd
                                delay = min(2 ** attempt, 8) + random.uniform(0, 0.5)
                                await asyncio.sleep(delay)
                                continue
                            return f"Item {item_id}: {e}"

                    return f"Item {item_id}: Échec après {max_retries} tentatives"

            # Run all updates concurrently (only non-seeded items)
            errors = await asyncio.gather(*[
                update_single_item(item_data)
                for item_data in remaining_items
            ])

            # Count successes and collect errors
            for error in errors:
                if error is None:
                    result["items_migrated"] += 1
                else:
                    result["errors"].append(error)

            # Step 6: Verify all values were written correctly
            if progress_callback:
                progress_callback(92, 100, "Vérification des valeurs...")

            actual_values = await self.get_column_values_for_all_items(
                board_id, new_column_id
            )
            actual_map = {iv["item_id"]: iv["value"] for iv in actual_values}
            expected = {iv["item_id"]: iv["mapped_value"] for iv in item_values}

            mismatches = []
            for item_id, expected_value in expected.items():
                actual_value = actual_map.get(item_id)
                if actual_value != expected_value:
                    mismatches.append(item_id)

            result["verified"] = len(expected) - len(mismatches)
            result["mismatches"] = len(mismatches)

            # Step 7: Retry mismatches sequentially (no concurrency to avoid race)
            if mismatches:
                if progress_callback:
                    progress_callback(95, 100, f"Retry de {len(mismatches)} éléments...")

                for item_id in mismatches:
                    value = expected[item_id]
                    label_id = label_map.get(value)
                    if label_id is None:
                        result["errors"].append(f"Retry item {item_id}: pas de label ID pour '{value}'")
                        continue

                    try:
                        column_values = {
                            new_column_id: {"ids": [label_id]}
                        }
                        await self.update_item_column_values(
                            item_id=item_id,
                            board_id=board_id,
                            column_values=column_values,
                            create_labels_if_missing=False
                        )
                        result["retried"] += 1
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        result["errors"].append(f"Retry item {item_id}: {e}")

            result["success"] = True
            if progress_callback:
                progress_callback(100, 100, "Migration terminée!")

        except MondayError as e:
            result["errors"].append(str(e))

        return result

    def migrate_column_to_dropdown_sync(
        self,
        board_id: int,
        source_column_id: str,
        source_column_title: str,
        progress_callback: Optional[Callable[..., Any]] = None,
        name_mapper: Optional[Callable[..., Any]] = None,
        max_concurrent: int = 5
    ) -> dict:
        """Synchronous wrapper for migrate_column_to_dropdown."""
        return asyncio.run(
            self.migrate_column_to_dropdown(
                board_id, source_column_id, source_column_title,
                progress_callback, name_mapper, max_concurrent
            )
        )

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
                    logger.debug("Skipping column '%s' with missing ID", col_name)
                    continue

                # Skip read-only column types
                if col_type in self.READ_ONLY_COLUMN_TYPES:
                    logger.debug("Skipping read-only column '%s' (type: %s)", col_name, col_type)
                    continue

                # Also skip by column ID prefix (backup check)
                if col_id.startswith(self.READ_ONLY_COLUMN_ID_PREFIXES):
                    logger.debug("Skipping read-only column '%s' (id: %s)", col_name, col_id)
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
                    logger.warning("Could not create column '%s': %s", col_name, e)
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
            logger.error(
                f"create_item failed for '{escaped_name}': {e}"
            )
            return CreateResult(success=False, error=str(e))

    async def delete_item(self, item_id: str) -> bool:
        """Delete an item from a board.

        Args:
            item_id: ID of the item to delete.

        Returns:
            True if successful.
        """
        mutation = f"mutation {{ delete_item(item_id: {item_id}) {{ id }} }}"
        await self._execute_query(mutation)
        return True

    async def ensure_dropdown_labels(
        self,
        board_id: int,
        column_id: str,
        labels: list[str],
    ) -> list[str]:
        """Add missing labels to a dropdown column.

        Since Monday.com has no direct API to add dropdown labels, this works
        by creating a temporary item, writing each new label with
        ``create_labels_if_missing=True``, then deleting the temp item.

        Args:
            board_id: Board containing the dropdown column.
            column_id: Column ID of the dropdown.
            labels: Label names to ensure exist.

        Returns:
            List of labels that were actually added (i.e. were missing).
        """
        existing = await self.get_dropdown_label_map(board_id, column_id)
        missing = [l for l in labels if l not in existing]

        if not missing:
            logger.info(
                "ensure_dropdown_labels: all labels already present on "
                "board %s column %s", board_id, column_id,
            )
            return []

        # Create a temporary item to seed labels
        temp = await self.create_item(
            board_id, "__label_seed__", create_labels_if_missing=True,
        )
        if not temp.success or not temp.id:
            raise MondayError(
                f"Failed to create temp item on board {board_id}: {temp.error}"
            )

        try:
            for label in missing:
                col_values = {column_id: {"labels": [label]}}
                await self.update_item_column_values(
                    item_id=temp.id,
                    board_id=board_id,
                    column_values=col_values,
                    create_labels_if_missing=True,
                )
        finally:
            # Always clean up the temporary item
            try:
                await self.delete_item(temp.id)
            except Exception as exc:
                logger.warning(
                    "Failed to delete temp item %s on board %s: %s",
                    temp.id, board_id, exc,
                )

        logger.info(
            "ensure_dropdown_labels: added %s to board %s column %s",
            missing, board_id, column_id,
        )
        return missing

    async def deduplicate_dropdown_labels(
        self,
        board_id: int,
        column_id: str,
        dry_run: bool = True,
    ) -> dict:
        """Find and remove duplicate dropdown labels on a board.

        Duplicate labels share the same name but have different IDs.
        For each group of duplicates, the label with the lowest ID is kept
        (canonical) and items referencing higher IDs are updated.

        Args:
            board_id: Board containing the dropdown column.
            column_id: Column ID of the dropdown.
            dry_run: If True, only report duplicates without fixing.

        Returns:
            Dict with keys: duplicates_found, items_updated, labels_removed,
            and detail (list of {name, canonical_id, duplicate_ids}).
        """
        result = {
            "duplicates_found": 0,
            "items_updated": 0,
            "labels_removed": 0,
            "detail": [],
        }

        # Step 1: Get all labels from column settings
        columns = await self.list_columns(board_id)
        settings = {}
        for col in columns:
            if col["id"] == column_id:
                settings = json.loads(col.get("settings_str", "{}"))
                break

        raw_labels = settings.get("labels", [])
        if not raw_labels:
            return result

        # Normalize to list of {id, name}
        label_entries: list[dict] = []
        if isinstance(raw_labels, list):
            for lb in raw_labels:
                label_entries.append({"id": lb["id"], "name": lb["name"]})
        elif isinstance(raw_labels, dict):
            for key, val in raw_labels.items():
                if isinstance(val, str):
                    label_entries.append({"id": int(key), "name": val})
                elif isinstance(val, dict):
                    label_entries.append({
                        "id": val.get("id", int(key)),
                        "name": val["name"],
                    })

        # Step 2: Group by name to find duplicates
        name_to_ids: dict[str, list[int]] = defaultdict(list)
        for entry in label_entries:
            name_to_ids[entry["name"]].append(entry["id"])

        duplicate_groups = {
            name: sorted(ids)
            for name, ids in name_to_ids.items()
            if len(ids) > 1
        }

        if not duplicate_groups:
            return result

        result["duplicates_found"] = sum(
            len(ids) - 1 for ids in duplicate_groups.values()
        )
        for name, ids in duplicate_groups.items():
            result["detail"].append({
                "name": name,
                "canonical_id": ids[0],
                "duplicate_ids": ids[1:],
            })

        if dry_run:
            return result

        # Step 3: Read all items and fix references to duplicate label IDs
        # Build a mapping: duplicate_id -> canonical_id
        dup_to_canonical: dict[int, int] = {}
        for name, ids in duplicate_groups.items():
            canonical = ids[0]
            for dup_id in ids[1:]:
                dup_to_canonical[dup_id] = canonical

        all_items = await self.extract_board_data(board_id)
        for item in all_items:
            for cv in item.get("column_values", []):
                if cv["id"] != column_id or not cv.get("value"):
                    continue

                try:
                    val = json.loads(cv["value"])
                except (json.JSONDecodeError, TypeError):
                    continue

                label_ids = val.get("ids", [])
                if not label_ids:
                    continue

                new_ids = []
                changed = False
                for lid in label_ids:
                    if lid in dup_to_canonical:
                        new_ids.append(dup_to_canonical[lid])
                        changed = True
                    else:
                        new_ids.append(lid)

                if changed:
                    # Deduplicate in case canonical was already present
                    new_ids = list(dict.fromkeys(new_ids))
                    await self.update_item_column_values(
                        item_id=item["id"],
                        board_id=board_id,
                        column_values={column_id: {"ids": [str(i) for i in new_ids]}},
                        create_labels_if_missing=False,
                    )
                    result["items_updated"] += 1
                    await asyncio.sleep(RATE_LIMIT_DELAY)

        result["labels_removed"] = result["duplicates_found"]

        logger.info(
            "deduplicate_dropdown_labels: board %s col %s — "
            "fixed %d items, removed %d duplicate labels",
            board_id, column_id,
            result["items_updated"], result["labels_removed"],
        )
        return result

    def deduplicate_dropdown_labels_sync(
        self,
        board_id: int,
        column_id: str,
        dry_run: bool = True,
    ) -> dict:
        """Synchronous wrapper for deduplicate_dropdown_labels."""
        return asyncio.run(
            self.deduplicate_dropdown_labels(board_id, column_id, dry_run)
        )

    async def _preseed_dropdown_labels(
        self,
        board_id: int,
        df: pd.DataFrame,
        column_id_map: dict[str, str],
        column_type_map: Optional[dict[str, str]] = None,
    ) -> dict[str, dict[str, int]]:
        """Pre-seed dropdown labels and return {col_name: {label_text: label_id}}.

        Ensures all unique dropdown values exist as labels before batch upload,
        avoiding duplicate labels from concurrent ``create_labels_if_missing``.
        """
        dropdown_label_maps: dict[str, dict[str, int]] = {}
        column_type_map = column_type_map or {}

        for col_name, col_id in column_id_map.items():
            is_dropdown = (
                column_type_map.get(col_name) == "dropdown"
                or col_name in self.DROPDOWN_COLUMNS
            )
            if not is_dropdown or col_name not in df.columns:
                continue

            unique_values = (
                df[col_name].dropna().astype(str).str.strip().unique().tolist()
            )
            unique_values = [v for v in unique_values if v]
            if not unique_values:
                continue

            await self.ensure_dropdown_labels(board_id, col_id, unique_values)
            label_map = await self.get_dropdown_label_map(board_id, col_id)
            if label_map:
                dropdown_label_maps[col_name] = label_map
            else:
                logger.warning(
                    "_preseed_dropdown_labels: label map empty for column '%s' "
                    "(board %s, col %s) — falling back to text-based labels",
                    col_name, board_id, col_id,
                )

        return dropdown_label_maps

    # -------------------------------------------------------------------------
    # Batch upload
    # -------------------------------------------------------------------------

    # Columns that should use DROPDOWN format (labels array) instead of STATUS format
    # Dropdown supports 1000+ options vs 40 for STATUS columns
    DROPDOWN_COLUMNS = {"Conseiller", "conseiller"}

    def _format_column_value(
        self,
        value: Any,
        column_name: str,
        actual_type: Optional[str] = None,
        dropdown_label_id: Optional[int] = None,
    ) -> Any:
        """Format a value for Monday.com API based on actual column type.

        Args:
            value: The value to format
            column_name: Column name (used as fallback for type lookup)
            actual_type: Actual Monday.com column type string (e.g., "text", "numbers")
            dropdown_label_id: Pre-resolved label ID for dropdown columns (avoids duplicates)
        """
        if pd.isna(value) or value is None or value == "":
            return None

        # Check if this column should use DROPDOWN format
        # Dropdown uses {"labels": ["value"]} (array) vs status {"label": "value"}
        is_dropdown_column = column_name in self.DROPDOWN_COLUMNS

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

            elif actual_type == "dropdown" or is_dropdown_column:
                if dropdown_label_id is not None:
                    return {"ids": [str(dropdown_label_id)]}
                return {"labels": [str(value)]}

            elif actual_type in ("status", "color"):
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
                return str(value) if value is not None else None

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

        elif col_type == ColumnType.DROPDOWN or is_dropdown_column:
            if dropdown_label_id is not None:
                return {"ids": [str(dropdown_label_id)]}
            return {"labels": [str(value)]}

        elif col_type == ColumnType.STATUS:
            # Status uses single label format (max 40 options)
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
            return str(value) if value is not None else None

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
        column_type_map: Optional[dict[str, str]] = None,
        dropdown_label_maps: Optional[dict[str, dict[str, int]]] = None,
    ) -> dict:
        """Convert a DataFrame row to Monday.com column_values format.

        Automatically skips formula columns and other read-only columns
        that cannot be set via the API.

        Args:
            row: DataFrame row to convert
            column_id_map: column_name -> column_id mapping
            column_type_map: column_name -> actual Monday.com type (optional)
            dropdown_label_maps: {col_name: {label_text: label_id}} for dropdown columns
        """
        column_values = {}
        column_type_map = column_type_map or {}
        dropdown_label_maps = dropdown_label_maps or {}

        for col_name, col_id in column_id_map.items():
            # Skip read-only columns (formula, auto_number, etc.)
            # These columns are auto-calculated by Monday.com
            if col_id.startswith(self.READ_ONLY_COLUMN_ID_PREFIXES):
                continue

            if col_name in row.index and col_name != "Nom Client":
                actual_type = column_type_map.get(col_name)

                # Resolve dropdown label ID if available
                label_id = None
                if col_name in dropdown_label_maps and not pd.isna(row[col_name]):
                    label_text = str(row[col_name]).strip()
                    label_id = dropdown_label_maps[col_name].get(label_text)

                formatted = self._format_column_value(
                    row[col_name], col_name, actual_type, label_id
                )
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
        progress_callback: Optional[Callable[..., Any]] = None
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

        # Pre-seed dropdown labels to avoid duplicates from concurrent creation
        dropdown_label_maps = await self._preseed_dropdown_labels(
            board_id, df, column_id_map, column_type_map
        )

        # Upload items with rate limiting and retry
        semaphore = asyncio.Semaphore(max_concurrent)
        upload_retries = 2  # up to 2 extra attempts per row

        async def upload_row(idx: int, row: pd.Series) -> CreateResult:
            async with semaphore:
                item_name = str(row.get(item_name_column, f"Item {idx}"))
                column_values = self._row_to_column_values(
                    row, column_id_map, column_type_map, dropdown_label_maps
                )

                # Throttle BEFORE sending to control actual request rate
                await asyncio.sleep(RATE_LIMIT_DELAY)

                create_result = await self.create_item(
                    board_id=board_id,
                    item_name=item_name,
                    group_id=group_id,
                    column_values=column_values
                )

                # Retry on failure with exponential backoff
                for retry in range(upload_retries):
                    if create_result.success:
                        break
                    delay = RETRY_BASE_DELAY * (2 ** retry) + random.uniform(0, 0.5)
                    logger.warning(
                        f"create_item retry {retry + 1}/{upload_retries} for "
                        f"'{item_name}' in {delay:.1f}s: {create_result.error}"
                    )
                    await asyncio.sleep(delay)
                    create_result = await self.create_item(
                        board_id=board_id,
                        item_name=item_name,
                        group_id=group_id,
                        column_values=column_values
                    )

                if progress_callback:
                    progress_callback(idx + 1, result.total)

                return create_result

        # Execute uploads
        df_indices = list(df.index)
        tasks = [
            upload_row(idx, row)
            for idx, row in df.iterrows()
        ]

        create_results = await asyncio.gather(*tasks)

        # Aggregate results — zip with original DataFrame indices
        for df_idx, create_result in zip(df_indices, create_results):
            if create_result.success:
                result.success += 1
                if create_result.id:
                    result.item_ids.append(create_result.id)
                    result.index_to_item_id[df_idx] = create_result.id
            else:
                result.failed += 1
                if create_result.error:
                    result.errors.append(create_result.error)

        return result

    # -------------------------------------------------------------------------
    # Data extraction (read operations)
    # -------------------------------------------------------------------------

    # GraphQL fragment for the item fields we always want back.
    _ITEM_FIELDS = """
        id
        name
        group {
            id
            title
        }
        column_values {
            id
            text
            value
            column {
                title
                type
            }
        }
    """

    async def extract_board_data(
        self,
        board_id: int,
        group_id: Optional[str] = None,
        limit: int = 500,
        skip_formula_enrichment: bool = False,
        date_filter: Optional[tuple[str, str, str]] = None,
    ) -> list[dict]:
        """
        Extract all items from a board with pagination.

        Args:
            board_id: Board ID to extract from
            group_id: Optional group ID to filter by
            limit: Items per page (max 500)
            skip_formula_enrichment: If True, skip the slow FormulaValue
                enrichment pass.  Callers can later call
                ``enrich_formula_columns()`` on a subset of items.
            date_filter: Optional (column_id, start_date, end_date) to filter
                server-side via items_page query_params with the between
                operator. Dates are ISO strings "YYYY-MM-DD". Ignored if
                group_id is also provided (mutually exclusive).

        Returns:
            List of item dictionaries with column values
        """
        all_items = []
        cursor: Optional[str] = None

        # date_filter is only applied at the board level; combining it with a
        # group filter is not supported in this query structure.
        effective_date_filter = date_filter if not group_id else None
        if date_filter and group_id:
            logger.warning(
                "extract_board_data: date_filter ignored when group_id is set"
            )

        query_params_clause = ""
        if effective_date_filter:
            col_id, start_iso, end_iso = effective_date_filter
            # column_id in Monday's GraphQL is serialized as a string literal.
            query_params_clause = (
                f', query_params: {{ rules: [{{ '
                f'column_id: "{col_id}", '
                f'compare_value: ["{start_iso}", "{end_iso}"], '
                f'operator: between '
                f'}}] }}'
            )

        while True:
            if cursor:
                # Subsequent pages: next_items_page preserves the original
                # query_params context from the cursor.
                query = f"""
                {{
                    next_items_page(limit: {limit}, cursor: "{cursor}") {{
                        cursor
                        items {{{self._ITEM_FIELDS}}}
                    }}
                }}
                """
            elif group_id:
                query = f"""
                {{
                    boards(ids: {board_id}) {{
                        groups(ids: ["{group_id}"]) {{
                            items_page(limit: {limit}) {{
                                cursor
                                items {{{self._ITEM_FIELDS}}}
                            }}
                        }}
                    }}
                }}
                """
            else:
                query = f"""
                {{
                    boards(ids: {board_id}) {{
                        items_page(limit: {limit}{query_params_clause}) {{
                            cursor
                            items {{{self._ITEM_FIELDS}}}
                        }}
                    }}
                }}
                """

            result = await self._execute_query(query)

            # Extract items based on query structure.
            if cursor:
                items_page = result["data"]["next_items_page"]
            else:
                boards = result["data"]["boards"]
                if not boards:
                    raise MondayError(
                        f"Board {board_id} not found or not accessible"
                    )
                if group_id:
                    groups = boards[0]["groups"]
                    if not groups:
                        break
                    items_page = groups[0]["items_page"]
                else:
                    items_page = boards[0]["items_page"]

            items = items_page["items"]
            cursor = items_page["cursor"]

            if not items:
                break

            all_items.extend(items)

            if not cursor:
                break

        # Enrich formula columns with display_value via a separate batched
        # query (FormulaValue fragment is rate-limited: 10k values/min, max 5
        # formula cols per request).  _enrich_formula_columns handles retries
        # and rate-limit back-off internally.  Non-fatal: if enrichment fails
        # the formula columns will be None and downstream code warns the user.
        if not skip_formula_enrichment:
            try:
                await self._enrich_formula_columns(all_items)
                remaining = self._count_missing_formula_display_values(all_items)
                if remaining > 0:
                    logger.warning(
                        f"Formula enrichment partial: {remaining} display_value(s) "
                        "still missing after enrichment"
                    )
                else:
                    logger.info("Formula enrichment: all display_value(s) populated")
            except Exception as e:
                logger.warning(f"Formula enrichment failed (non-fatal): {e}")

        return all_items

    @staticmethod
    def _count_missing_formula_display_values(items: list[dict]) -> int:
        """Count formula column_values that lack a display_value.

        Returns 0 when there are no formula columns or all have display_value.
        """
        missing = 0
        for item in items:
            for cv in item.get("column_values", []):
                if cv.get("column", {}).get("type") == "formula":
                    if cv.get("display_value") is None:
                        missing += 1
        return missing

    async def enrich_formula_columns(self, items: list[dict]) -> None:
        """Public wrapper: enrich formula display_values for a subset of items.

        Use this after calling ``extract_board_data(skip_formula_enrichment=True)``
        to selectively enrich only the items you need (e.g. matched police numbers).

        Mutates *items* in-place.  Non-fatal: logs a warning on partial results.
        """
        await self._enrich_formula_columns(items)
        remaining = self._count_missing_formula_display_values(items)
        if remaining > 0:
            logger.warning(
                f"Formula enrichment partial: {remaining} display_value(s) "
                "still missing after enrichment"
            )

    # Shared state tracking the last formula enrichment request across
    # all MondayClient instances.  Used to enforce a cooldown that is
    # proportional to the last batch's budget use (not a fixed 15s).
    _last_formula_request_time: float = 0.0
    _last_formula_batch_values: int = 0
    _formula_cooldown_lock: asyncio.Lock = None  # type: ignore[assignment]

    # Formula-column rate-limit policy: 10k values/min enforced by Monday,
    # we use 50% of that budget to stay safe under bursty concurrency.
    _FORMULA_RATE_LIMIT_PER_MIN: int = 10_000
    _FORMULA_SAFE_RATE: float = 10_000 * 0.5  # values per minute
    # Minimum cooldown between enrichments; guarantees we never go faster
    # than once every N seconds even with tiny batches.
    _FORMULA_MIN_COOLDOWN_S: float = 2.0

    @classmethod
    def _get_formula_lock(cls) -> asyncio.Lock:
        """Lazily create the formula cooldown lock (must be in an event loop)."""
        if cls._formula_cooldown_lock is None:
            cls._formula_cooldown_lock = asyncio.Lock()
        return cls._formula_cooldown_lock

    async def _enrich_formula_columns(
        self,
        items: list[dict],
        batch_size: int = 25,
        max_formula_cols_per_query: int = 5,
        max_retries: int = 8,
    ) -> None:
        """Fetch display_value for formula columns in small batches.

        The FormulaValue display_value field is rate-limited to 10,000
        values/minute and max 5 formula columns per request.  This
        method fetches it separately in small item batches, splitting
        formula columns into groups of 5, with inter-batch delays to
        stay within the rate limit.

        The rate limit budget is **global** (shared across boards), so
        this method uses a conservative 50% budget utilisation and a
        class-level timestamp to enforce a cooldown between successive
        board enrichments.

        Mutates items in-place by adding 'display_value' to formula
        column_values dicts.
        """
        if not items:
            return

        # Detect formula column IDs from first item
        formula_col_ids = []
        for cv in items[0].get("column_values", []):
            if cv.get("column", {}).get("type") == "formula":
                formula_col_ids.append(cv["id"])

        if not formula_col_ids:
            return

        logger.info(
            f"Enriching {len(formula_col_ids)} formula column(s) "
            f"for {len(items)} items"
        )

        # --- Adaptive global cooldown (lock-protected to prevent TOCTOU) ---
        # Required cooldown = time needed for the previous batch's budget
        # share to regenerate. Much smaller than the previous fixed 15s for
        # small boards (e.g. ~2s after date-filtered loads).
        async with self._get_formula_lock():
            now = time.monotonic()
            elapsed = now - MondayClient._last_formula_request_time
            last_values = MondayClient._last_formula_batch_values
            needed = (
                (last_values / self._FORMULA_SAFE_RATE) * 60
                if last_values else 0.0
            )
            cooldown = max(needed, self._FORMULA_MIN_COOLDOWN_S)
            if elapsed < cooldown:
                wait = cooldown - elapsed
                logger.info(
                    f"Formula enrichment: global cooldown — waiting {wait:.1f}s"
                )
                await asyncio.sleep(wait)

        # Build item_id → {col_id → col_val reference} map
        item_cv_map: dict[str, dict[str, dict]] = {}
        for item in items:
            cv_refs = {}
            for cv in item.get("column_values", []):
                if cv["id"] in formula_col_ids:
                    cv_refs[cv["id"]] = cv
            item_cv_map[item["id"]] = cv_refs

        # Split formula columns into groups of max 5 (API limit)
        col_groups = [
            formula_col_ids[i : i + max_formula_cols_per_query]
            for i in range(0, len(formula_col_ids), max_formula_cols_per_query)
        ]

        item_ids = list(item_cv_map.keys())
        enriched_count = 0

        # Rate limit: 10,000 formula values/min.
        # Each batch requests batch_size × len(col_group) values.
        # Calculate inter-batch delay to stay safely under the limit.
        # Use 50% of the budget to leave headroom for back-to-back boards.
        safe_rate = self._FORMULA_SAFE_RATE

        for col_group in col_groups:
            col_ids_str = ", ".join(f'"{cid}"' for cid in col_group)
            values_per_batch = batch_size * len(col_group)
            # Seconds to wait between batches to respect rate limit
            batch_delay = max((values_per_batch / safe_rate) * 60, 1.0)

            for i in range(0, len(item_ids), batch_size):
                batch_ids = item_ids[i : i + batch_size]
                ids_str = ", ".join(batch_ids)

                query = f"""
                {{
                    items(ids: [{ids_str}]) {{
                        id
                        column_values(ids: [{col_ids_str}]) {{
                            id
                            text
                            value
                            ... on FormulaValue {{
                                display_value
                            }}
                        }}
                    }}
                }}
                """

                result = await self._execute_query_with_formula_retry(
                    query, max_retries=max_retries
                )
                MondayClient._last_formula_request_time = time.monotonic()
                # Track actual values sent so the next enrichment's cooldown
                # can scale with this batch's budget usage.
                MondayClient._last_formula_batch_values = len(batch_ids) * len(col_group)

                if result is None:
                    continue  # Batch failed after retries, skip

                # Merge enriched fields back into items
                for item_data in result.get("data", {}).get("items", []):
                    cv_refs = item_cv_map.get(item_data["id"], {})
                    for cv in item_data.get("column_values", []):
                        ref = cv_refs.get(cv["id"])
                        if not ref:
                            continue
                        # display_value: best source
                        if cv.get("display_value") is not None:
                            ref["display_value"] = cv["display_value"]
                            enriched_count += 1
                        # Also update text/value if the enrichment
                        # got a non-empty value the main query missed
                        enrich_text = cv.get("text")
                        if enrich_text and not ref.get("text"):
                            ref["text"] = enrich_text
                        enrich_value = cv.get("value")
                        if enrich_value and not ref.get("value"):
                            ref["value"] = enrich_value

                # Delay between batches to stay within rate limit
                await asyncio.sleep(batch_delay)

        logger.info(f"Formula enrichment: {enriched_count} values populated")

    async def _execute_query_with_formula_retry(
        self,
        query: str,
        max_retries: int = 3,
    ) -> Optional[dict]:
        """Execute a query with rate-limit retry. Returns None on failure."""
        for attempt in range(max_retries):
            try:
                return await self._execute_query(query)
            except MondayError as e:
                is_rate_limit = e.errors and any(
                    err.get("extensions", {}).get("code")
                    in (
                        "FIELD_MINUTE_RATE_LIMIT_EXCEEDED",
                        "RATE_LIMIT_EXCEEDED",
                    )
                    for err in (e.errors if isinstance(e.errors, list) else [])
                )
                if is_rate_limit:
                    # Monday sometimes returns retry_in_seconds=1 which is
                    # too aggressive — enforce a minimum of 5s and add
                    # exponential back-off to avoid hammering a depleted
                    # budget.
                    api_retry_secs = 1
                    for err in (e.errors if isinstance(e.errors, list) else []):
                        api_retry_secs = max(
                            api_retry_secs,
                            err.get("extensions", {}).get("retry_in_seconds", 1),
                        )
                    retry_secs = max(api_retry_secs, 5 * (2 ** attempt))
                    logger.warning(
                        f"Formula rate limit (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {retry_secs}s..."
                    )
                    await asyncio.sleep(retry_secs)
                    continue
                # Non-rate-limit error → log and give up this batch
                logger.warning(f"Formula query error (non-fatal): {e.message[:200]}")
                return None

        logger.warning("Formula rate limit: max retries exhausted for batch")
        return None

    @staticmethod
    def _parse_formula_value(col_val: dict):
        """Parse a Monday.com formula column value as a number.

        Tries every available source — the first non-empty wins:
        1. display_value (FormulaValue fragment, most reliable)
        2. value (JSON — some formulas expose it)
        3. text (populated for some formulas in API 2024-10+)

        Returns float if numeric, raw text if not, None if empty.
        """
        # 1. display_value — most reliable for formula results
        #    Use `is not None` (not truthiness) to handle numeric 0
        display = col_val.get("display_value")
        if display is not None:
            # Already a number (some APIs return int/float directly)
            if isinstance(display, (int, float)):
                return float(display)
            s = str(display).strip()
            if s:
                parsed = MondayClient._clean_numeric_text(s)
                if parsed is not None:
                    return parsed
                return s  # Non-numeric formula → keep raw text

        # 2. JSON value field
        raw_value = col_val.get("value")
        if raw_value:
            try:
                parsed = json.loads(raw_value)
                if isinstance(parsed, (int, float)):
                    return float(parsed)
                if isinstance(parsed, dict):
                    v = parsed.get("value")
                    if v is not None:
                        return float(v)
            except (ValueError, TypeError, json.JSONDecodeError):
                pass

        # 3. text field — populated for some formula types
        text = col_val.get("text")
        if text is not None:
            s = str(text).strip()
            if s:
                parsed = MondayClient._clean_numeric_text(s)
                if parsed is not None:
                    return parsed
                return s

        return None

    @staticmethod
    def _clean_numeric_text(text: str):
        """Try to parse a text value as a float, handling various formats.

        Handles: $, spaces, non-breaking spaces, French/English separators,
        trailing %, parenthesised negatives like (123.45).

        Returns float if numeric, None otherwise.
        """
        cleaned = (
            text.strip()
            .replace("$", "")
            .replace("\u00a0", "")  # non-breaking space
            .replace(" ", "")
            .replace("%", "")
        )
        if not cleaned:
            return None
        # Parenthesised negative: "(123.45)" → "-123.45"
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        # French decimal: "1234,56" → "1234.56"
        if "," in cleaned and "." not in cleaned:
            cleaned = cleaned.replace(",", ".")
        # English thousands: "1,234.56" → "1234.56"
        elif "," in cleaned and "." in cleaned:
            cleaned = cleaned.replace(",", "")
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None

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
                if col_type in ("numbers", "numeric"):
                    try:
                        raw_value = col_val.get("value")
                        if raw_value:
                            parsed = json.loads(raw_value)
                            row[col_title] = float(parsed) if parsed else None
                        else:
                            row[col_title] = None
                    except (ValueError, TypeError, json.JSONDecodeError):
                        row[col_title] = col_val.get("text")
                elif col_type == "formula":
                    row[col_title] = self._parse_formula_value(col_val)
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
        except MondayError as e:
            logger.error("update_item_column_values failed for item %s: %s", item_id, e)
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
        progress_callback: Optional[Callable[..., Any]] = None,
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

        # Pre-seed dropdown labels to avoid duplicates
        dropdown_label_maps = await self._preseed_dropdown_labels(
            board_id, data, column_id_map, column_type_map
        )

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
                    # Skip formula columns - they are auto-calculated and can't be updated
                    if actual_type == "formula":
                        continue

                    # Resolve dropdown label ID
                    label_id = None
                    if col_name in dropdown_label_maps and not pd.isna(row[col_name]):
                        label_id = dropdown_label_maps[col_name].get(
                            str(row[col_name]).strip()
                        )

                    formatted = self._format_column_value(
                        row[col_name], col_name, actual_type, label_id
                    )
                    if formatted is not None:
                        column_values[col_id] = formatted

            if items_in_target_group:
                # Update existing item in target group
                item = items_in_target_group[0]
                item_id = item["id"]

                try:
                    # Update column values
                    if column_values:
                        success = await self.update_item_column_values(
                            item_id=item_id,
                            board_id=board_id,
                            column_values=column_values,
                        )
                        if not success:
                            result["errors"].append(f"Update failed for {advisor_name}")

                    result["updated"] += 1
                except MondayError as e:
                    result["errors"].append(f"Update error for {advisor_name}: {e}")
            else:
                # Create new item in target group
                # (advisor doesn't exist in this group yet, even if in other groups)
                # Format advisor column value based on actual column type
                advisor_type = column_type_map.get(advisor_column_name)
                advisor_label_id = None
                if advisor_column_name in dropdown_label_maps:
                    advisor_label_id = dropdown_label_maps[advisor_column_name].get(
                        str(advisor_name).strip()
                    )
                advisor_formatted = self._format_column_value(
                    advisor_name, advisor_column_name, advisor_type, advisor_label_id
                )
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

    async def upsert_by_item_name(
        self,
        board_id: int,
        group_id: str,
        data: pd.DataFrame,
        column_id_map: dict[str, str],
        column_type_map: Optional[dict[str, str]] = None,
        advisor_column_name: str = "Conseiller",
        progress_callback: Optional[Callable[..., Any]] = None,
    ) -> dict:
        """
        Upsert data using item name as the identifier (for boards without a Conseiller column).

        The advisor name from the DataFrame becomes the item name in Monday.com.
        This is used when the target board uses "Élément" (item name) for advisors.

        Args:
            board_id: Target board ID
            group_id: Target group ID
            data: DataFrame with aggregated data (must have advisor_column_name)
            column_id_map: column_name -> column_id mapping
            column_type_map: column_name -> column_type mapping (optional)
            advisor_column_name: Name of advisor column in DataFrame (used as item name)
            progress_callback: Optional callback(current, total, action)

        Returns:
            Dict with {updated: int, created: int, errors: list}
        """
        result = {"updated": 0, "created": 0, "errors": []}
        column_type_map = column_type_map or {}
        total = len(data)

        # Pre-seed dropdown labels to avoid duplicates
        dropdown_label_maps = await self._preseed_dropdown_labels(
            board_id, data, column_id_map, column_type_map
        )

        # Get all existing items in the target group
        try:
            existing_items = await self.extract_board_data(board_id, group_id)
        except MondayError as e:
            result["errors"].append(f"Failed to fetch existing items: {e}")
            return result

        # Build a map of item name -> item for quick lookup
        items_by_name = {item["name"]: item for item in existing_items}

        for idx, row in data.iterrows():
            advisor_name = row.get(advisor_column_name)
            if not advisor_name or pd.isna(advisor_name):
                result["errors"].append(f"Row {idx}: missing advisor name")
                continue

            advisor_name_str = str(advisor_name)

            # Build column values for update/create (exclude advisor column)
            column_values = {}
            for col_name, col_id in column_id_map.items():
                if col_name == advisor_column_name:
                    continue  # Advisor is the item name, not a column
                if col_name in row.index:
                    actual_type = column_type_map.get(col_name)
                    # Skip formula columns - they are auto-calculated and can't be updated
                    if actual_type == "formula":
                        continue

                    # Resolve dropdown label ID
                    label_id = None
                    if col_name in dropdown_label_maps and not pd.isna(row[col_name]):
                        label_id = dropdown_label_maps[col_name].get(
                            str(row[col_name]).strip()
                        )

                    formatted = self._format_column_value(
                        row[col_name], col_name, actual_type, label_id
                    )
                    if formatted is not None:
                        column_values[col_id] = formatted

            if advisor_name_str in items_by_name:
                # Update existing item
                item = items_by_name[advisor_name_str]
                item_id = item["id"]

                try:
                    if column_values:
                        await self.update_item_column_values(
                            item_id=item_id,
                            board_id=board_id,
                            column_values=column_values,
                        )
                    result["updated"] += 1
                except MondayError as e:
                    result["errors"].append(f"Update error for {advisor_name_str}: {e}")
            else:
                # Create new item with advisor name as item name
                try:
                    create_result = await self.create_item(
                        board_id=board_id,
                        item_name=advisor_name_str,
                        group_id=group_id,
                        column_values=column_values,
                    )
                    if create_result.success:
                        result["created"] += 1
                    else:
                        result["errors"].append(f"Create error for {advisor_name_str}: {create_result.error}")
                except MondayError as e:
                    result["errors"].append(f"Create error for {advisor_name_str}: {e}")

            # Rate limiting
            await asyncio.sleep(RATE_LIMIT_DELAY)

            if progress_callback:
                current = idx + 1 if isinstance(idx, int) else data.index.get_loc(idx) + 1
                progress_callback(current, total, "upsert")

        return result

    async def get_existing_policy_numbers(self, board_id: int) -> set[str]:
        """
        Get all existing policy numbers from a board.

        Uses extract_board_data to fetch all items with pagination,
        then extracts '# de Police' column values.

        Args:
            board_id: Board ID to read from

        Returns:
            Set of policy number strings
        """
        items = await self.extract_board_data(board_id)
        policy_numbers = set()
        for item in items:
            for col_val in item.get("column_values", []):
                if col_val.get("column", {}).get("title") == "# de Police":
                    text = col_val.get("text")
                    if text and text.strip():
                        policy_numbers.add(text.strip())
                    break
        return policy_numbers

    async def get_existing_rows(
        self, board_id: int, columns: list[str]
    ) -> list[dict[str, str]]:
        """
        Get all existing rows from a board as dicts of column values.

        Fetches all items and extracts the text value for each requested column.
        Used for full-row duplicate detection.

        Args:
            board_id: Board ID to read from
            columns: List of column titles to extract

        Returns:
            List of dicts mapping column title → normalized text value
        """
        items = await self.extract_board_data(
            board_id, skip_formula_enrichment=True
        )
        rows = []
        for item in items:
            col_values = {
                cv.get("column", {}).get("title"): (cv.get("text") or "").strip()
                for cv in item.get("column_values", [])
            }
            row = {col: col_values.get(col, "") for col in columns}
            rows.append(row)
        return rows

    # -------------------------------------------------------------------------
    # Folder operations
    # -------------------------------------------------------------------------

    async def list_folders(self, workspace_id: int) -> list[dict]:
        """List all folders in a workspace.

        Args:
            workspace_id: Workspace ID to list folders from

        Returns:
            List of folder dicts with id, name, and children
        """
        all_folders = []
        page = 1

        while True:
            query = f"""
            {{
                folders(workspace_ids: [{workspace_id}], limit: 50, page: {page}) {{
                    id
                    name
                    children {{
                        id
                        name
                    }}
                }}
            }}
            """
            result = await self._execute_query(query)
            folders = result["data"]["folders"]

            if not folders:
                break

            all_folders.extend(folders)

            if len(folders) < 50:
                break

            page += 1

        return all_folders

    def list_folders_sync(self, workspace_id: int) -> list[dict]:
        """Synchronous wrapper for list_folders."""
        return asyncio.run(self.list_folders(workspace_id))

    async def list_all_folders_in_workspace(self, workspace_id: int) -> list[dict]:
        """List all folders in a workspace with parent information.

        Uses workspace_ids filter and returns parent info for each folder,
        allowing client-side filtering by parent folder.

        Args:
            workspace_id: Workspace ID to list folders from

        Returns:
            List of folder dicts with id, name, and parent {id, name}
        """
        all_folders = []
        page = 1

        while True:
            query = f"""
            {{
                folders(workspace_ids: [{workspace_id}], limit: 50, page: {page}) {{
                    id
                    name
                    parent {{
                        id
                        name
                    }}
                }}
            }}
            """
            result = await self._execute_query(query)
            folders = result["data"]["folders"]

            if not folders:
                break

            all_folders.extend(folders)

            if len(folders) < 50:
                break

            page += 1

        return all_folders

    def list_all_folders_in_workspace_sync(self, workspace_id: int) -> list[dict]:
        """Synchronous wrapper for list_all_folders_in_workspace."""
        return asyncio.run(self.list_all_folders_in_workspace(workspace_id))

    async def create_folder(
        self,
        name: str,
        workspace_id: int,
        parent_folder_id: Optional[int] = None,
    ) -> dict:
        """Create a folder in a workspace.

        Args:
            name: Folder name
            workspace_id: Workspace ID
            parent_folder_id: Optional parent folder ID for nesting

        Returns:
            Dict with folder id
        """
        parent_arg = f", parent_folder_id: {parent_folder_id}" if parent_folder_id else ""
        mutation = f"""
        mutation {{
            create_folder(
                name: "{name}",
                workspace_id: {workspace_id}{parent_arg}
            ) {{
                id
            }}
        }}
        """
        result = await self._execute_query(mutation)
        return result["data"]["create_folder"]

    def create_folder_sync(
        self,
        name: str,
        workspace_id: int,
        parent_folder_id: Optional[int] = None,
    ) -> dict:
        """Synchronous wrapper for create_folder."""
        return asyncio.run(self.create_folder(name, workspace_id, parent_folder_id))

    async def list_boards_in_folder(self, folder_id: int) -> list[dict]:
        """List boards in a specific folder.

        Fetches all boards and filters by board_folder_id.

        Args:
            folder_id: Folder ID to list boards from

        Returns:
            List of board dicts with id and name
        """
        all_boards = []
        page = 1

        while True:
            query = f"""
            {{
                boards(limit: 200, page: {page}) {{
                    id
                    name
                    board_folder_id
                }}
            }}
            """
            result = await self._execute_query(query)
            boards = result["data"]["boards"]

            if not boards:
                break

            for board in boards:
                if str(board.get("board_folder_id", "")) == str(folder_id):
                    all_boards.append({"id": board["id"], "name": board["name"]})

            if len(boards) < 200:
                break

            page += 1

        return all_boards

    def list_boards_in_folder_sync(self, folder_id: int) -> list[dict]:
        """Synchronous wrapper for list_boards_in_folder."""
        return asyncio.run(self.list_boards_in_folder(folder_id))

    async def duplicate_board(
        self,
        board_id: int,
        board_name: str,
        folder_id: Optional[int] = None,
        workspace_id: Optional[int] = None,
    ) -> dict:
        """Duplicate a board with its structure (columns, settings, groups).

        Args:
            board_id: Source board ID to duplicate
            board_name: Name for the duplicated board
            folder_id: Optional folder ID to place the new board in
            workspace_id: Optional workspace ID

        Returns:
            Dict with board id and name
        """
        optional_args = ""
        if folder_id:
            optional_args += f", folder_id: {folder_id}"
        if workspace_id:
            optional_args += f", workspace_id: {workspace_id}"

        mutation = f"""
        mutation {{
            duplicate_board(
                board_id: {board_id},
                duplicate_type: duplicate_board_with_structure,
                board_name: "{board_name}"{optional_args}
            ) {{
                board {{
                    id
                    name
                }}
            }}
        }}
        """
        result = await self._execute_query(mutation)
        return result["data"]["duplicate_board"]["board"]

    def duplicate_board_sync(
        self,
        board_id: int,
        board_name: str,
        folder_id: Optional[int] = None,
        workspace_id: Optional[int] = None,
    ) -> dict:
        """Synchronous wrapper for duplicate_board."""
        return asyncio.run(self.duplicate_board(board_id, board_name, folder_id, workspace_id))

    async def update_board_name(self, board_id: int, new_name: str) -> dict:
        """Rename a board.

        Args:
            board_id: Board ID to rename
            new_name: New board name

        Returns:
            Dict with board id
        """
        mutation = f"""
        mutation {{
            update_board(
                board_id: {board_id},
                board_attribute: name,
                new_value: "{new_name}"
            )
        }}
        """
        result = await self._execute_query(mutation)
        return result["data"]["update_board"]

    def update_board_name_sync(self, board_id: int, new_name: str) -> dict:
        """Synchronous wrapper for update_board_name."""
        return asyncio.run(self.update_board_name(board_id, new_name))

    async def delete_group(self, board_id: int, group_id: str) -> dict:
        """Delete a group from a board.

        Args:
            board_id: Board ID containing the group
            group_id: Group ID to delete

        Returns:
            Dict with group id and deleted status
        """
        mutation = f"""
        mutation {{
            delete_group(
                board_id: {board_id},
                group_id: "{group_id}"
            ) {{
                id
                deleted
            }}
        }}
        """
        result = await self._execute_query(mutation)
        return result["data"]["delete_group"]

    def delete_group_sync(self, board_id: int, group_id: str) -> dict:
        """Synchronous wrapper for delete_group."""
        return asyncio.run(self.delete_group(board_id, group_id))

    async def invite_users(self, emails: list[str]) -> dict:
        """Invite users to Monday.com by email.

        Args:
            emails: List of email addresses to invite

        Returns:
            Dict with invited_users list and errors list
        """
        emails_str = json.dumps(emails)
        mutation = f"""
        mutation {{
            invite_users(emails: {emails_str}) {{
                invited_users {{
                    id
                    email
                }}
                errors {{
                    message
                    email
                }}
            }}
        }}
        """
        result = await self._execute_query(mutation)
        return result["data"]["invite_users"]

    def invite_users_sync(self, emails: list[str]) -> dict:
        """Synchronous wrapper for invite_users."""
        return asyncio.run(self.invite_users(emails))

    async def add_users_to_board(
        self,
        board_id: int,
        user_ids: list[int],
        kind: str = "subscriber",
    ) -> dict:
        """Add users to a board as subscribers or owners.

        Args:
            board_id: Board ID to add users to
            user_ids: List of user IDs to add
            kind: "subscriber" (standard access) or "owner"

        Returns:
            Dict with board id
        """
        user_ids_str = json.dumps(user_ids)
        mutation = f"""
        mutation {{
            add_users_to_board(
                board_id: {board_id},
                user_ids: {user_ids_str},
                kind: {kind}
            ) {{
                id
            }}
        }}
        """
        result = await self._execute_query(mutation)
        return result["data"]["add_users_to_board"]

    def add_users_to_board_sync(
        self,
        board_id: int,
        user_ids: list[int],
        kind: str = "subscriber",
    ) -> dict:
        """Synchronous wrapper for add_users_to_board."""
        return asyncio.run(self.add_users_to_board(board_id, user_ids, kind))

    async def set_board_permission(
        self,
        board_id: int,
        basic_role_name: str = "contributor",
    ) -> dict:
        """Set the default permission role for a board.

        Args:
            board_id: Board ID to update
            basic_role_name: "contributor" (edit content only),
                             "editor" (edit content + structure),
                             or "viewer" (read-only)

        Returns:
            Dict with edit_permissions and failed_actions
        """
        mutation = f"""
        mutation {{
            set_board_permission(
                board_id: {board_id},
                basic_role_name: {basic_role_name}
            ) {{
                edit_permissions
                failed_actions
            }}
        }}
        """
        result = await self._execute_query(mutation)
        return result["data"]["set_board_permission"]

    def set_board_permission_sync(
        self,
        board_id: int,
        basic_role_name: str = "contributor",
    ) -> dict:
        """Synchronous wrapper for set_board_permission."""
        return asyncio.run(self.set_board_permission(board_id, basic_role_name))

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
        progress_callback: Optional[Callable[..., Any]] = None,
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

    def upsert_by_item_name_sync(
        self,
        board_id: int,
        group_id: str,
        data: pd.DataFrame,
        column_id_map: dict[str, str],
        column_type_map: Optional[dict[str, str]] = None,
        advisor_column_name: str = "Conseiller",
        progress_callback: Optional[Callable[..., Any]] = None,
    ) -> dict:
        """Synchronous wrapper for upsert_by_item_name."""
        return asyncio.run(
            self.upsert_by_item_name(
                board_id=board_id,
                group_id=group_id,
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
        progress_callback: Optional[Callable[..., Any]] = None
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

    def list_groups_with_item_count_sync(self, board_id: int) -> list[dict]:
        """Synchronous wrapper for list_groups_with_item_count."""
        return asyncio.run(self.list_groups_with_item_count(board_id))

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
