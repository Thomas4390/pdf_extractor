"""
Data aggregation utilities for Monday.com data.

Provides date period filtering and aggregation by advisor for
creating summarized data in target boards.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Optional

import pandas as pd

from src.utils.advisor_matcher import normalize_advisor_name_full


# =============================================================================
# CONSTANTS
# =============================================================================

MONTHS_FR = {
    1: "Janvier",
    2: "Février",
    3: "Mars",
    4: "Avril",
    5: "Mai",
    6: "Juin",
    7: "Juillet",
    8: "Août",
    9: "Septembre",
    10: "Octobre",
    11: "Novembre",
    12: "Décembre",
}

QUARTERS_FR = {
    1: "Q1",
    2: "Q2",
    3: "Q3",
    4: "Q4",
}


# =============================================================================
# ENUMS
# =============================================================================

class DatePeriod(Enum):
    """Available date filtering periods - month-based selection."""
    MONTH_0 = 0   # Current month
    MONTH_1 = 1   # 1 month ago
    MONTH_2 = 2   # 2 months ago
    MONTH_3 = 3   # 3 months ago
    MONTH_4 = 4   # 4 months ago
    MONTH_5 = 5   # 5 months ago
    MONTH_6 = 6   # 6 months ago
    MONTH_7 = 7   # 7 months ago
    MONTH_8 = 8   # 8 months ago
    MONTH_9 = 9   # 9 months ago
    MONTH_10 = 10  # 10 months ago
    MONTH_11 = 11  # 11 months ago

    @property
    def months_ago(self) -> int:
        """Number of months ago from current month."""
        return self.value

    @property
    def display_name(self) -> str:
        """Human-readable name showing actual month and year."""
        target_date = get_month_from_offset(self.value)
        return f"{MONTHS_FR[target_date.month]} {target_date.year}"

    @property
    def short_label(self) -> str:
        """Short label for UI buttons."""
        if self.value == 0:
            return "Ce mois"
        elif self.value == 1:
            return "Mois dernier"
        else:
            return f"-{self.value} mois"


def get_month_from_offset(months_ago: int, reference_date: Optional[date] = None) -> date:
    """
    Get the first day of a month N months ago.

    Args:
        months_ago: Number of months to go back (0 = current month)
        reference_date: Reference date (defaults to today)

    Returns:
        First day of the target month
    """
    if reference_date is None:
        reference_date = date.today()

    # Calculate target year and month
    year = reference_date.year
    month = reference_date.month - months_ago

    # Handle year rollover
    while month <= 0:
        month += 12
        year -= 1

    return date(year, month, 1)


# =============================================================================
# SOURCE BOARD CONFIGURATION
# =============================================================================

@dataclass
class SourceBoardConfig:
    """Configuration for a source board."""
    display_name: str
    aggregate_column: str
    date_column: str
    output_column_name: str  # Name of column in final output
    board_id: Optional[int] = None  # Default board ID for auto-loading
    advisor_column: str = "Conseiller"  # Column name for advisor
    use_group_as_advisor: bool = False  # If True, use group_title as advisor


SOURCE_BOARDS = {
    "paiement_historique": SourceBoardConfig(
        display_name="Paiement historique",
        aggregate_column="Reçu",
        date_column="Date",
        output_column_name="Collected",
        board_id=8553813876,  # Monday.com "Paiement Historique" board
    ),
    "vente_production": SourceBoardConfig(
        display_name="Vente et production",
        aggregate_column="PA",
        date_column="Date",
        output_column_name="PA Vendues",
        board_id=9423464449,  # Monday.com "Ventes/Production" board
    ),
    "ae_tracker": SourceBoardConfig(
        display_name="AE Tracker",
        aggregate_column="$$$ Recues",
        date_column="Date",
        output_column_name="AE CA",
        board_id=9142978904,  # Monday.com "AE Tracker" board
        advisor_column="group_title",  # Advisor name is in group title
        use_group_as_advisor=True,
    ),
}


# =============================================================================
# DATE PERIOD FUNCTIONS
# =============================================================================

def get_period_date_range(
    period: DatePeriod,
    reference_date: Optional[date] = None,
) -> tuple[date, date]:
    """
    Get start and end dates for a given period.

    Args:
        period: The date period to calculate (month-based)
        reference_date: Reference date (defaults to today)

    Returns:
        Tuple of (start_date, end_date) inclusive
    """
    if reference_date is None:
        reference_date = date.today()

    # Get the first day of the target month
    start = get_month_from_offset(period.months_ago, reference_date)

    # Calculate the last day of the month
    if start.month == 12:
        end = date(start.year + 1, 1, 1) - timedelta(days=1)
    else:
        end = date(start.year, start.month + 1, 1) - timedelta(days=1)

    return start, end


def get_group_name_for_period(
    period: DatePeriod,
    reference_date: Optional[date] = None,
) -> str:
    """
    Get the target group name for a given period.

    Since all periods are now month-based, returns "Month Year" format.

    Examples:
        - MONTH_0 (in January 2026) → "Janvier 2026"
        - MONTH_1 (in January 2026) → "Décembre 2025"
        - MONTH_2 (in January 2026) → "Novembre 2025"

    Args:
        period: The date period (month-based)
        reference_date: Reference date (defaults to today)

    Returns:
        French group name string (e.g., "Janvier 2026")
    """
    start_date, _ = get_period_date_range(period, reference_date)
    month_name = MONTHS_FR[start_date.month]
    return f"{month_name} {start_date.year}"


# =============================================================================
# DATA FILTERING & AGGREGATION
# =============================================================================

def normalize_advisor_column(
    df: pd.DataFrame,
    advisor_column: str = "Conseiller",
    filter_unknown: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Normalize advisor names in a DataFrame column to full name format.

    Uses the AdvisorMatcher to match variations like "Poirier", "Poirier D",
    "D. Poirier" to the standardized full name (e.g., "Daniel Poirier").

    Args:
        df: Input DataFrame
        advisor_column: Name of the column containing advisor names
        filter_unknown: If True, filter out rows where advisor is not in database

    Returns:
        Tuple of (DataFrame with normalized advisor names, list of unknown advisor names)
    """
    if df.empty:
        return df, []

    if advisor_column not in df.columns:
        return df, []

    # Make a copy to avoid modifying original
    df = df.copy()

    # Store original names before normalization for tracking unknowns
    df["_original_advisor"] = df[advisor_column].astype(str)

    # Apply normalization to the advisor column (returns None if not found)
    df[advisor_column] = df[advisor_column].apply(
        lambda x: normalize_advisor_name_full(str(x)) if pd.notna(x) else None
    )

    # Collect and filter out unknown advisors
    unknown_names = []
    if filter_unknown:
        # Find rows with None/empty advisor
        unknown_mask = df[advisor_column].isna() | (df[advisor_column] == "") | (df[advisor_column].astype(str).str.strip() == "")
        # Get unique original names that couldn't be matched
        unknown_names = df.loc[unknown_mask, "_original_advisor"].unique().tolist()
        # Clean up the names (remove empty/nan)
        unknown_names = [n for n in unknown_names if n and n.lower() not in ["nan", "none", ""]]
        # Filter them out
        df = df[~unknown_mask]

    # Remove temporary column
    df = df.drop(columns=["_original_advisor"], errors="ignore")

    return df, unknown_names


def filter_by_date(
    df: pd.DataFrame,
    period: DatePeriod,
    date_column: str,
    reference_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Filter DataFrame by date period.

    Args:
        df: Input DataFrame
        period: Date period to filter by
        date_column: Name of the date column
        reference_date: Reference date (defaults to today)

    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df

    if date_column not in df.columns:
        # Return unfiltered if column doesn't exist
        return df

    # Make a copy to avoid modifying original
    df = df.copy()

    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    # Get date range
    start_date, end_date = get_period_date_range(period, reference_date)

    # Convert to datetime for comparison
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    # Filter
    mask = (df[date_column] >= start_dt) & (df[date_column] <= end_dt)
    return df[mask]


def aggregate_by_advisor(
    df: pd.DataFrame,
    value_column: str,
    advisor_column: str = "Conseiller",
    normalize_names: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Aggregate values by advisor.

    Normalizes advisor names before aggregation to merge variations
    like "Poirier" and "Poirier D" into a single entry.

    Args:
        df: Input DataFrame (should be filtered first)
        value_column: Column containing values to sum
        advisor_column: Column containing advisor names
        normalize_names: If True, normalize advisor names using AdvisorMatcher

    Returns:
        Tuple of (DataFrame with columns [advisor_column, value_column] aggregated by advisor,
                  list of unknown advisor names that were filtered out)
    """
    if df.empty:
        return pd.DataFrame(columns=[advisor_column, value_column]), []

    if advisor_column not in df.columns:
        return pd.DataFrame(columns=[advisor_column, value_column]), []

    if value_column not in df.columns:
        return pd.DataFrame(columns=[advisor_column, value_column]), []

    # Make a copy
    df = df.copy()

    # Normalize advisor names before aggregation (also filters unknown advisors)
    unknown_names = []
    if normalize_names:
        df, unknown_names = normalize_advisor_column(df, advisor_column, filter_unknown=True)

    # Convert value column to numeric
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce").fillna(0)

    # Group by advisor and sum
    result = (
        df.groupby(advisor_column, as_index=False)[value_column]
        .sum()
        .sort_values(value_column, ascending=False)
    )

    return result, unknown_names


def combine_aggregations(
    aggregations: dict[str, pd.DataFrame],
    advisor_column: str = "Conseiller",
) -> pd.DataFrame:
    """
    Combine multiple aggregation DataFrames into one.

    Args:
        aggregations: Dict of {source_key: aggregated_df}
        advisor_column: Column name for advisor in output

    Returns:
        Combined DataFrame with advisor and all value columns
    """
    if not aggregations:
        return pd.DataFrame()

    # Start with first non-empty DataFrame
    result = None

    for source_key, df in aggregations.items():
        if df.empty:
            continue

        config = SOURCE_BOARDS.get(source_key)
        if config is None:
            continue

        # Get the source's advisor column name
        source_advisor_col = config.advisor_column

        # Make a copy to avoid modifying original
        df = df.copy()

        # Rename the advisor column to standard name if different
        if source_advisor_col != advisor_column and source_advisor_col in df.columns:
            df = df.rename(columns={source_advisor_col: advisor_column})

        # Rename value column to the output column name
        value_col = config.aggregate_column
        output_col = config.output_column_name
        if value_col in df.columns:
            df = df.rename(columns={value_col: output_col})

        # Only keep advisor and value columns
        cols_to_keep = [advisor_column, output_col]
        df = df[[c for c in cols_to_keep if c in df.columns]]

        if result is None:
            result = df
        else:
            # Merge on advisor column
            result = result.merge(
                df,
                on=advisor_column,
                how="outer",
            )

    if result is None:
        return pd.DataFrame()

    # Fill NaN with 0 for numeric columns
    for col in result.columns:
        if col != advisor_column:
            result[col] = result[col].fillna(0)

    # Reorder columns: Conseiller first, then PA Vendues, Collected, AE CA
    desired_order = [advisor_column, "PA Vendues", "Collected", "AE CA"]
    # Keep only columns that exist in the result
    ordered_cols = [c for c in desired_order if c in result.columns]
    # Add any remaining columns not in desired order
    remaining_cols = [c for c in result.columns if c not in ordered_cols]
    result = result[ordered_cols + remaining_cols]

    return result.sort_values(advisor_column)
