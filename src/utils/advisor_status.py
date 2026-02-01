"""
Dynamic Advisor Status Calculator

Calculates advisor status (New/Active/Past) based on their history
in the Data board, not from static settings.

Status Rules:
- New: First month the advisor appears in the Data board
- Active: Advisor has been present for more than 1 consecutive month
- Past: Manually set override (stored in advisor settings)

The status is calculated relative to the period being viewed, ensuring
historical accuracy (e.g., viewing January 2026 in March 2026 will still
show advisors as "New" if January was their first month).

Cloud Storage:
- Monthly status history is persisted to Google Sheets (AdvisorStatusHistory worksheet)
- Each advisor has their status tracked per month for historical accuracy
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional, Any, List, Dict

import pandas as pd

from src.utils.advisor_matcher import (
    normalize_advisor_name_full,
    get_gcp_credentials,
    get_secret,
    GSHEETS_AVAILABLE,
)

# Try to import gspread
try:
    import gspread
except ImportError:
    gspread = None


@dataclass
class AdvisorStatusInfo:
    """Information about an advisor's status for a given period."""
    status: str  # "New", "Active", or "Past"
    first_appearance_month: Optional[str]  # e.g., "Janvier 2026"
    months_active: int  # Number of months since first appearance
    is_manual_override: bool  # True if status was manually set


class AdvisorStatusHistoryStore:
    """
    Cloud-based storage for advisor status history using Google Sheets.

    Stores monthly status records in a dedicated worksheet "AdvisorStatusHistory"
    with columns: advisor_name, month, status, first_appearance_month, updated_at
    """

    WORKSHEET_NAME = "AdvisorStatusHistory"

    # Singleton instance
    _instance: Optional['AdvisorStatusHistoryStore'] = None
    _worksheet: Optional[Any] = None
    _initialized: bool = False
    _error: Optional[str] = None

    @classmethod
    def get_instance(cls) -> 'AdvisorStatusHistoryStore':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance."""
        cls._instance = None
        cls._worksheet = None
        cls._initialized = False
        cls._error = None

    def __init__(self):
        """Initialize the store."""
        if not AdvisorStatusHistoryStore._initialized:
            self._init_gsheets()

    def _init_gsheets(self):
        """Initialize Google Sheets connection."""
        if not GSHEETS_AVAILABLE or gspread is None:
            AdvisorStatusHistoryStore._error = "gspread library not installed"
            return

        spreadsheet_id = get_secret('GOOGLE_SHEETS_SPREADSHEET_ID')
        if not spreadsheet_id:
            AdvisorStatusHistoryStore._error = "GOOGLE_SHEETS_SPREADSHEET_ID not configured"
            return

        credentials = get_gcp_credentials()
        if not credentials:
            AdvisorStatusHistoryStore._error = "GCP credentials not found"
            return

        try:
            client = gspread.authorize(credentials)
            spreadsheet = client.open_by_key(spreadsheet_id)

            # Get or create the AdvisorStatusHistory worksheet
            try:
                AdvisorStatusHistoryStore._worksheet = spreadsheet.worksheet(self.WORKSHEET_NAME)
            except gspread.WorksheetNotFound:
                # Create the worksheet with headers
                AdvisorStatusHistoryStore._worksheet = spreadsheet.add_worksheet(
                    title=self.WORKSHEET_NAME,
                    rows=1000, cols=5
                )
                AdvisorStatusHistoryStore._worksheet.update(
                    'A1:E1',
                    [['advisor_name', 'month', 'status', 'first_appearance_month', 'updated_at']]
                )
                AdvisorStatusHistoryStore._worksheet.format('A1:E1', {'textFormat': {'bold': True}})

            AdvisorStatusHistoryStore._initialized = True
            AdvisorStatusHistoryStore._error = None

        except Exception as e:
            logging.warning(f"Could not initialize AdvisorStatusHistory sheet: {e}")
            AdvisorStatusHistoryStore._error = str(e)

    @property
    def is_configured(self) -> bool:
        """Return True if Google Sheets is properly configured."""
        return AdvisorStatusHistoryStore._initialized and AdvisorStatusHistoryStore._worksheet is not None

    @property
    def configuration_error(self) -> Optional[str]:
        """Return the configuration error message, if any."""
        return AdvisorStatusHistoryStore._error

    def get_status_history(self, advisor_name: str) -> List[Dict[str, str]]:
        """
        Get all status history records for an advisor.

        Args:
            advisor_name: The advisor's normalized name

        Returns:
            List of dicts with keys: month, status, first_appearance_month, updated_at
        """
        if not self.is_configured:
            return []

        try:
            records = AdvisorStatusHistoryStore._worksheet.get_all_records()
            return [
                {
                    'month': r['month'],
                    'status': r['status'],
                    'first_appearance_month': r.get('first_appearance_month', ''),
                    'updated_at': r.get('updated_at', ''),
                }
                for r in records
                if r.get('advisor_name') == advisor_name
            ]
        except Exception as e:
            logging.warning(f"Failed to get status history: {e}")
            return []

    def get_status_for_month(self, advisor_name: str, month: str) -> Optional[str]:
        """
        Get the stored status for an advisor for a specific month.

        Args:
            advisor_name: The advisor's normalized name
            month: The month (e.g., "Janvier 2026")

        Returns:
            Status string or None if not found
        """
        if not self.is_configured:
            return None

        try:
            records = AdvisorStatusHistoryStore._worksheet.get_all_records()
            for r in records:
                if r.get('advisor_name') == advisor_name and r.get('month') == month:
                    return r.get('status')
            return None
        except Exception as e:
            logging.warning(f"Failed to get status for month: {e}")
            return None

    def save_status(
        self,
        advisor_name: str,
        month: str,
        status: str,
        first_appearance_month: Optional[str] = None,
    ) -> bool:
        """
        Save or update the status for an advisor for a specific month.

        Args:
            advisor_name: The advisor's normalized name
            month: The month (e.g., "Janvier 2026")
            status: The status (New, Active, Past)
            first_appearance_month: The first appearance month (optional)

        Returns:
            True if saved successfully
        """
        if not self.is_configured:
            return False

        try:
            from datetime import datetime
            updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Check if record already exists
            all_values = AdvisorStatusHistoryStore._worksheet.get_all_values()
            row_index = None

            for idx, row in enumerate(all_values):
                if idx == 0:  # Skip header
                    continue
                if len(row) >= 2 and row[0] == advisor_name and row[1] == month:
                    row_index = idx + 1  # gspread uses 1-based indexing
                    break

            if row_index:
                # Update existing row
                AdvisorStatusHistoryStore._worksheet.update(
                    f'A{row_index}:E{row_index}',
                    [[advisor_name, month, status, first_appearance_month or '', updated_at]]
                )
            else:
                # Append new row
                AdvisorStatusHistoryStore._worksheet.append_row([
                    advisor_name, month, status, first_appearance_month or '', updated_at
                ])

            return True

        except Exception as e:
            logging.warning(f"Failed to save status: {e}")
            return False

    def save_batch_status(
        self,
        records: List[Dict[str, str]],
    ) -> int:
        """
        Save multiple status records in batch for efficiency.

        Args:
            records: List of dicts with keys: advisor_name, month, status, first_appearance_month

        Returns:
            Number of records saved successfully
        """
        if not self.is_configured or not records:
            return 0

        try:
            from datetime import datetime
            updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Get existing records for lookup
            all_values = AdvisorStatusHistoryStore._worksheet.get_all_values()
            existing_lookup = {}
            for idx, row in enumerate(all_values):
                if idx == 0:  # Skip header
                    continue
                if len(row) >= 2:
                    key = f"{row[0]}|{row[1]}"  # advisor_name|month
                    existing_lookup[key] = idx + 1  # 1-based row index

            updates = []
            new_rows = []

            for record in records:
                advisor_name = record.get('advisor_name', '')
                month = record.get('month', '')
                status = record.get('status', '')
                first_appearance = record.get('first_appearance_month', '')

                if not advisor_name or not month or not status:
                    continue

                key = f"{advisor_name}|{month}"
                row_data = [advisor_name, month, status, first_appearance, updated_at]

                if key in existing_lookup:
                    row_index = existing_lookup[key]
                    updates.append({
                        'range': f'A{row_index}:E{row_index}',
                        'values': [row_data]
                    })
                else:
                    new_rows.append(row_data)

            # Perform batch updates
            if updates:
                AdvisorStatusHistoryStore._worksheet.batch_update(updates)

            # Append new rows
            if new_rows:
                AdvisorStatusHistoryStore._worksheet.append_rows(new_rows)

            return len(updates) + len(new_rows)

        except Exception as e:
            logging.warning(f"Failed to save batch status: {e}")
            return 0

    def get_all_status_for_month(self, month: str) -> Dict[str, str]:
        """
        Get all advisor statuses for a specific month.

        Args:
            month: The month (e.g., "Janvier 2026")

        Returns:
            Dict mapping advisor_name to status
        """
        if not self.is_configured:
            return {}

        try:
            records = AdvisorStatusHistoryStore._worksheet.get_all_records()
            return {
                r['advisor_name']: r['status']
                for r in records
                if r.get('month') == month
            }
        except Exception as e:
            logging.warning(f"Failed to get all status for month: {e}")
            return {}


class AdvisorStatusCalculator:
    """
    Calculates advisor status based on their history in the Data board.

    The status is calculated dynamically for each period being viewed,
    ensuring historical accuracy.
    """

    # Cache for first appearance data (advisor_name -> first_month)
    _first_appearance_cache: dict[str, str] = {}
    _cache_loaded: bool = False

    # Manual overrides (advisor_name -> status)
    _manual_overrides: dict[str, str] = {}

    # Month order for comparison
    MONTHS_ORDER = {
        "Janvier": 1, "Février": 2, "Mars": 3, "Avril": 4,
        "Mai": 5, "Juin": 6, "Juillet": 7, "Août": 8,
        "Septembre": 9, "Octobre": 10, "Novembre": 11, "Décembre": 12
    }

    def __init__(self):
        """Initialize the calculator."""
        pass

    @classmethod
    def clear_cache(cls):
        """Clear the first appearance cache."""
        cls._first_appearance_cache = {}
        cls._cache_loaded = False

    @classmethod
    def set_manual_override(cls, advisor_name: str, status: str):
        """
        Set a manual status override for an advisor.

        Args:
            advisor_name: The advisor's normalized name
            status: The status to set ("Past" typically)
        """
        normalized = normalize_advisor_name_full(advisor_name)
        if normalized:
            cls._manual_overrides[normalized] = status

    @classmethod
    def remove_manual_override(cls, advisor_name: str):
        """Remove a manual override for an advisor."""
        normalized = normalize_advisor_name_full(advisor_name)
        if normalized and normalized in cls._manual_overrides:
            del cls._manual_overrides[normalized]

    @classmethod
    def get_manual_overrides(cls) -> dict[str, str]:
        """Get all manual overrides."""
        return cls._manual_overrides.copy()

    @classmethod
    def _parse_month_year(cls, month_str: str) -> tuple[int, int]:
        """
        Parse a month string like "Janvier 2026" into (year, month_num).

        Returns:
            Tuple of (year, month_number) or (0, 0) if parsing fails
        """
        try:
            parts = month_str.strip().split()
            if len(parts) != 2:
                return (0, 0)

            month_name = parts[0]
            year = int(parts[1])
            month_num = cls.MONTHS_ORDER.get(month_name, 0)

            return (year, month_num)
        except (ValueError, IndexError):
            return (0, 0)

    @classmethod
    def _compare_months(cls, month1: str, month2: str) -> int:
        """
        Compare two month strings.

        Returns:
            -1 if month1 < month2
             0 if month1 == month2
             1 if month1 > month2
        """
        y1, m1 = cls._parse_month_year(month1)
        y2, m2 = cls._parse_month_year(month2)

        if (y1, m1) < (y2, m2):
            return -1
        elif (y1, m1) > (y2, m2):
            return 1
        return 0

    @classmethod
    def _months_between(cls, from_month: str, to_month: str) -> int:
        """
        Calculate the number of months between two month strings.

        Args:
            from_month: Start month (e.g., "Janvier 2026")
            to_month: End month (e.g., "Mars 2026")

        Returns:
            Number of months difference (can be negative)
        """
        y1, m1 = cls._parse_month_year(from_month)
        y2, m2 = cls._parse_month_year(to_month)

        if y1 == 0 or y2 == 0:
            return 0

        return (y2 - y1) * 12 + (m2 - m1)

    @classmethod
    def load_first_appearances_from_data_board(
        cls,
        client,
        board_id: int,
        advisor_column: str = "Conseiller",
    ) -> dict[str, str]:
        """
        Load the first appearance month for each advisor from the Data board.

        Scans all groups in the Data board and determines the earliest
        month each advisor appears.

        Args:
            client: MondayClient instance
            board_id: The Data board ID
            advisor_column: Column name for advisor (or use item_name)

        Returns:
            Dict mapping normalized advisor name to first appearance month
        """
        from src.app.utils.async_helpers import run_async

        first_appearances = {}

        try:
            # Get all groups from the board
            groups = run_async(client.list_groups(board_id))

            # Sort groups by month (earliest first)
            sorted_groups = []
            for group in groups:
                title = group.get("title", "")
                year, month = cls._parse_month_year(title)
                if year > 0 and month > 0:
                    sorted_groups.append((year, month, group))

            sorted_groups.sort(key=lambda x: (x[0], x[1]))

            # Process groups from earliest to latest
            for year, month, group in sorted_groups:
                group_id = group["id"]
                group_title = group["title"]

                # Load items from this group
                items = client.extract_board_data_sync(board_id, group_id=group_id)
                df = client.board_items_to_dataframe(items)

                if df.empty:
                    continue

                # Get advisor names
                if advisor_column in df.columns:
                    advisors = df[advisor_column].dropna().unique()
                elif "item_name" in df.columns:
                    advisors = df["item_name"].dropna().unique()
                else:
                    continue

                # Record first appearance for each advisor
                for advisor in advisors:
                    normalized = normalize_advisor_name_full(str(advisor))
                    if normalized and normalized not in first_appearances:
                        first_appearances[normalized] = group_title

            # Update cache
            cls._first_appearance_cache = first_appearances
            cls._cache_loaded = True

            return first_appearances

        except Exception as e:
            import logging
            logging.warning(f"Failed to load first appearances: {e}")
            return {}

    @classmethod
    def get_status_for_period(
        cls,
        advisor_name: str,
        period_month: str,
    ) -> AdvisorStatusInfo:
        """
        Get the status for an advisor for a specific period.

        Args:
            advisor_name: The advisor's name (will be normalized)
            period_month: The month being viewed (e.g., "Janvier 2026")

        Returns:
            AdvisorStatusInfo with calculated status
        """
        normalized = normalize_advisor_name_full(advisor_name)
        if not normalized:
            normalized = advisor_name

        # Check for manual override first
        if normalized in cls._manual_overrides:
            return AdvisorStatusInfo(
                status=cls._manual_overrides[normalized],
                first_appearance_month=cls._first_appearance_cache.get(normalized),
                months_active=0,
                is_manual_override=True,
            )

        # Get first appearance
        first_month = cls._first_appearance_cache.get(normalized)

        if not first_month:
            # No history found in Data board - treat as Past (no longer active)
            return AdvisorStatusInfo(
                status="Past",
                first_appearance_month=None,
                months_active=0,
                is_manual_override=False,
            )

        # Calculate months between first appearance and viewed period
        months_diff = cls._months_between(first_month, period_month)

        if months_diff <= 0:
            # Viewing the first month or earlier - New
            status = "New"
        else:
            # More than 0 months since first appearance - Active
            status = "Active"

        return AdvisorStatusInfo(
            status=status,
            first_appearance_month=first_month,
            months_active=max(0, months_diff),
            is_manual_override=False,
        )

    @classmethod
    def add_status_to_dataframe(
        cls,
        df: pd.DataFrame,
        period_month: str,
        advisor_column: str = "Conseiller",
        sync_to_cloud: bool = True,
    ) -> pd.DataFrame:
        """
        Add calculated status column to a DataFrame.

        Args:
            df: DataFrame with advisor data
            period_month: The month being viewed (e.g., "Janvier 2026")
            advisor_column: Name of the advisor column
            sync_to_cloud: Whether to sync status to Google Sheets

        Returns:
            DataFrame with "Advisor_Status" column added
        """
        if df.empty or advisor_column not in df.columns:
            return df

        df = df.copy()

        # Calculate status for each advisor and prepare batch records
        statuses = []
        cloud_records = []

        for advisor in df[advisor_column]:
            advisor_str = str(advisor)
            info = cls.get_status_for_period(advisor_str, period_month)
            statuses.append(info.status)

            # Prepare record for cloud sync
            normalized = normalize_advisor_name_full(advisor_str)
            if normalized:
                cloud_records.append({
                    'advisor_name': normalized,
                    'month': period_month,
                    'status': info.status,
                    'first_appearance_month': info.first_appearance_month or '',
                })

        df["Advisor_Status"] = statuses

        # Sync to cloud in background
        if sync_to_cloud and cloud_records:
            cls.sync_status_to_cloud(cloud_records)

        return df

    @classmethod
    def sync_status_to_cloud(cls, records: List[Dict[str, str]]) -> int:
        """
        Sync status records to Google Sheets cloud storage.

        Args:
            records: List of dicts with keys: advisor_name, month, status, first_appearance_month

        Returns:
            Number of records synced
        """
        try:
            store = AdvisorStatusHistoryStore.get_instance()
            if store.is_configured:
                return store.save_batch_status(records)
            else:
                logging.debug(f"Cloud storage not configured: {store.configuration_error}")
                return 0
        except Exception as e:
            logging.warning(f"Failed to sync status to cloud: {e}")
            return 0

    @classmethod
    def load_status_from_cloud(cls, month: str) -> Dict[str, str]:
        """
        Load all advisor statuses for a month from cloud storage.

        Args:
            month: The month to load (e.g., "Janvier 2026")

        Returns:
            Dict mapping advisor_name to status
        """
        try:
            store = AdvisorStatusHistoryStore.get_instance()
            if store.is_configured:
                return store.get_all_status_for_month(month)
            return {}
        except Exception as e:
            logging.warning(f"Failed to load status from cloud: {e}")
            return {}


# Module-level instance for easy access
_calculator = AdvisorStatusCalculator()


def load_advisor_history(client, board_id: int) -> dict[str, str]:
    """
    Load advisor first appearance history from the Data board.

    Args:
        client: MondayClient instance
        board_id: The Data board ID

    Returns:
        Dict mapping advisor name to first appearance month
    """
    return AdvisorStatusCalculator.load_first_appearances_from_data_board(
        client, board_id
    )


def get_advisor_status(advisor_name: str, period_month: str) -> str:
    """
    Get the status for an advisor for a specific period.

    Args:
        advisor_name: The advisor's name
        period_month: The month being viewed (e.g., "Janvier 2026")

    Returns:
        Status string: "New", "Active", or "Past"
    """
    info = AdvisorStatusCalculator.get_status_for_period(advisor_name, period_month)
    return info.status


def set_advisor_status_override(advisor_name: str, status: str):
    """Set a manual status override (e.g., "Past")."""
    AdvisorStatusCalculator.set_manual_override(advisor_name, status)


def clear_advisor_status_cache():
    """Clear the cached first appearance data."""
    AdvisorStatusCalculator.clear_cache()


def get_status_history_store() -> AdvisorStatusHistoryStore:
    """Get the cloud status history store instance."""
    return AdvisorStatusHistoryStore.get_instance()


def get_advisor_status_history(advisor_name: str) -> List[Dict[str, str]]:
    """
    Get all status history records for an advisor from cloud storage.

    Args:
        advisor_name: The advisor's name

    Returns:
        List of status records with keys: month, status, first_appearance_month, updated_at
    """
    normalized = normalize_advisor_name_full(advisor_name)
    if not normalized:
        normalized = advisor_name

    store = AdvisorStatusHistoryStore.get_instance()
    return store.get_status_history(normalized)


def save_advisor_status_to_cloud(
    advisor_name: str,
    month: str,
    status: str,
    first_appearance_month: Optional[str] = None,
) -> bool:
    """
    Save an advisor's status for a specific month to cloud storage.

    Args:
        advisor_name: The advisor's name
        month: The month (e.g., "Janvier 2026")
        status: The status (New, Active, Past)
        first_appearance_month: The first appearance month (optional)

    Returns:
        True if saved successfully
    """
    normalized = normalize_advisor_name_full(advisor_name)
    if not normalized:
        normalized = advisor_name

    store = AdvisorStatusHistoryStore.get_instance()
    return store.save_status(normalized, month, status, first_appearance_month)


def sync_all_status_to_cloud(
    df: pd.DataFrame,
    period_month: str,
    advisor_column: str = "Conseiller",
) -> int:
    """
    Sync all advisor statuses from a DataFrame to cloud storage.

    Args:
        df: DataFrame with advisor data and Advisor_Status column
        period_month: The month being saved
        advisor_column: Name of the advisor column

    Returns:
        Number of records synced
    """
    if df.empty or advisor_column not in df.columns or "Advisor_Status" not in df.columns:
        return 0

    records = []
    for _, row in df.iterrows():
        advisor_name = str(row[advisor_column])
        status = row["Advisor_Status"]

        normalized = normalize_advisor_name_full(advisor_name)
        if normalized:
            first_appearance = AdvisorStatusCalculator._first_appearance_cache.get(normalized, '')
            records.append({
                'advisor_name': normalized,
                'month': period_month,
                'status': status,
                'first_appearance_month': first_appearance,
            })

    return AdvisorStatusCalculator.sync_status_to_cloud(records)
