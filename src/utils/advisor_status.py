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
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from src.utils.advisor_matcher import normalize_advisor_name_full


@dataclass
class AdvisorStatusInfo:
    """Information about an advisor's status for a given period."""
    status: str  # "New", "Active", or "Past"
    first_appearance_month: Optional[str]  # e.g., "Janvier 2026"
    months_active: int  # Number of months since first appearance
    is_manual_override: bool  # True if status was manually set


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
            # No history found - treat as New
            return AdvisorStatusInfo(
                status="New",
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
    ) -> pd.DataFrame:
        """
        Add calculated status column to a DataFrame.

        Args:
            df: DataFrame with advisor data
            period_month: The month being viewed (e.g., "Janvier 2026")
            advisor_column: Name of the advisor column

        Returns:
            DataFrame with "Advisor_Status" column added
        """
        if df.empty or advisor_column not in df.columns:
            return df

        df = df.copy()

        # Calculate status for each advisor
        statuses = []
        for advisor in df[advisor_column]:
            info = cls.get_status_for_period(str(advisor), period_month)
            statuses.append(info.status)

        df["Advisor_Status"] = statuses

        return df


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
