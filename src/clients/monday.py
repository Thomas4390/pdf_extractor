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
from dataclasses import dataclass, field
from enum import StrEnum
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
    except Exception:
        return None

_cfg = _get_monday_config()
DEFAULT_BATCH_SIZE = _cfg.monday_batch_size if _cfg else 50
DEFAULT_MAX_CONCURRENT = _cfg.monday_max_concurrent if _cfg else 5
RATE_LIMIT_DELAY = _cfg.monday_rate_limit_delay if _cfg else 0.5

# Retry configuration for transient server errors (5xx)
MAX_RETRIES = _cfg.monday_max_retries if _cfg else 3
RETRY_BASE_DELAY = _cfg.monday_retry_base_delay if _cfg else 2.0


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
            "API-Version": "2025-04"
        }

    # -------------------------------------------------------------------------
    # Low-level API methods
    # -------------------------------------------------------------------------

    async def _execute_query(
        self,
        query: str,
        variables: Optional[dict] = None
    ) -> dict:
        """Execute a GraphQL query asynchronously with retry on server errors."""
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
                        # Check if this is a server-side error (retryable)
                        is_server_error = any(
                            err.get("extensions", {}).get("status_code") == 500
                            or "internal server error" in err.get("message", "").lower()
                            for err in (errors if isinstance(errors, list) else [])
                        )
                        if is_server_error and attempt < MAX_RETRIES - 1:
                            raise MondayError(
                                message=str(errors),
                                errors=errors
                            )
                        raise MondayError(
                            message=str(errors),
                            errors=errors
                        )

                    return result

            except MondayError as e:
                last_error = e
                is_retryable = (
                    e.status_code and e.status_code >= 500
                ) or (
                    e.errors and any(
                        err.get("extensions", {}).get("status_code") == 500
                        or "internal server error" in err.get("message", "").lower()
                        for err in (e.errors if isinstance(e.errors, list) else [])
                    )
                )
                if is_retryable and attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Monday.com server error (attempt {attempt + 1}/{MAX_RETRIES}). "
                        f"Retrying in {delay}s... Error: {e.message[:200]}"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

        raise last_error

    def _execute_query_sync(self, query: str, variables: Optional[dict] = None) -> dict:
        """Execute a GraphQL query synchronously with retry on server errors."""
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
                    is_server_error = any(
                        err.get("extensions", {}).get("status_code") == 500
                        or "internal server error" in err.get("message", "").lower()
                        for err in (errors if isinstance(errors, list) else [])
                    )
                    if is_server_error and attempt < MAX_RETRIES - 1:
                        raise MondayError(message=str(errors), errors=errors)
                    raise MondayError(message=str(errors), errors=errors)

                return result

            except MondayError as e:
                last_error = e
                is_retryable = (
                    e.status_code and e.status_code >= 500
                ) or (
                    e.errors and any(
                        err.get("extensions", {}).get("status_code") == 500
                        or "internal server error" in err.get("message", "").lower()
                        for err in (e.errors if isinstance(e.errors, list) else [])
                    )
                )
                if is_retryable and attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Monday.com server error (attempt {attempt + 1}/{MAX_RETRIES}). "
                        f"Retrying in {delay}s... Error: {e.message[:200]}"
                    )
                    time.sleep(delay)
                    continue
                raise

        raise last_error

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
        name_mapper: Optional[callable] = None,
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
        name_mapper: Optional[callable] = None,
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
                print(f"[dropdown] Label map empty on attempt {attempt + 1}, retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
        print(f"[dropdown] WARNING: Label map still empty after {max_retries} attempts")
        return {}

    async def migrate_column_to_dropdown(
        self,
        board_id: int,
        source_column_id: str,
        source_column_title: str,
        progress_callback: Optional[callable] = None,
        name_mapper: Optional[callable] = None,
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
                except (MondayError, Exception) as e:
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

                        except (MondayError, Exception) as e:
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
                    except (MondayError, Exception) as e:
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
        progress_callback: Optional[callable] = None,
        name_mapper: Optional[callable] = None,
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
            logger.error(
                f"create_item failed for '{escaped_name}': {e}"
            )
            return CreateResult(success=False, error=str(e))

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
                # Dropdown uses labels array format
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

        elif col_type == ColumnType.DROPDOWN or is_dropdown_column:
            # Dropdown uses labels array format (supports 1000+ options)
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
            boards = result["data"]["boards"]
            if not boards:
                raise MondayError(f"Board {board_id} not found or not accessible")

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

        return all_items

    @staticmethod
    def _parse_formula_value(col_val: dict):
        """Parse a Monday.com formula column value as a number.

        Tries value (JSON) first, then text with formatting cleanup.
        Returns float if numeric, raw text if not, None if empty.
        """
        import json as json_module

        # 1. Try JSON value field (some formulas expose it)
        raw_value = col_val.get("value")
        if raw_value:
            try:
                parsed = json_module.loads(raw_value)
                if isinstance(parsed, (int, float)):
                    return float(parsed)
                if isinstance(parsed, dict):
                    v = parsed.get("value")
                    if v is not None:
                        return float(v)
            except (ValueError, TypeError, json.JSONDecodeError):
                pass

        # 2. Try text field with format cleanup
        text = col_val.get("text", "")
        if not text or not text.strip():
            return None

        cleaned = (
            text.strip()
            .replace("$", "")
            .replace("\u00a0", "")  # non-breaking space
            .replace(" ", "")
        )
        # French decimal: "1234,56" → "1234.56"
        if "," in cleaned and "." not in cleaned:
            cleaned = cleaned.replace(",", ".")
        # English thousands: "1,234.56" → "1234.56"
        elif "," in cleaned and "." in cleaned:
            cleaned = cleaned.replace(",", "")

        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return text  # Non-numeric formula → keep raw text

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
                            import json as json_module
                            parsed = json_module.loads(raw_value)
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
                    # Skip formula columns - they are auto-calculated and can't be updated
                    if actual_type == "formula":
                        continue
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

    async def upsert_by_item_name(
        self,
        board_id: int,
        group_id: str,
        data: pd.DataFrame,
        column_id_map: dict[str, str],
        column_type_map: Optional[dict[str, str]] = None,
        advisor_column_name: str = "Conseiller",
        progress_callback: Optional[callable] = None,
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
                    formatted = self._format_column_value(row[col_name], col_name, actual_type)
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

    def upsert_by_item_name_sync(
        self,
        board_id: int,
        group_id: str,
        data: pd.DataFrame,
        column_id_map: dict[str, str],
        column_type_map: Optional[dict[str, str]] = None,
        advisor_column_name: str = "Conseiller",
        progress_callback: Optional[callable] = None,
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
