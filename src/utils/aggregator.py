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
    """Available date filtering periods."""
    CURRENT_WEEK = "current_week"
    CURRENT_MONTH = "current_month"
    LAST_MONTH = "last_month"
    CURRENT_QUARTER = "current_quarter"
    CURRENT_YEAR = "current_year"

    @property
    def display_name(self) -> str:
        """Human-readable name for UI."""
        names = {
            DatePeriod.CURRENT_WEEK: "Semaine courante",
            DatePeriod.CURRENT_MONTH: "Mois courant",
            DatePeriod.LAST_MONTH: "Mois dernier",
            DatePeriod.CURRENT_QUARTER: "Trimestre courant",
            DatePeriod.CURRENT_YEAR: "Année courante",
        }
        return names[self]


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
        period: The date period to calculate
        reference_date: Reference date (defaults to today)

    Returns:
        Tuple of (start_date, end_date) inclusive
    """
    if reference_date is None:
        reference_date = date.today()

    if period == DatePeriod.CURRENT_WEEK:
        # Monday to Sunday of current week
        start = reference_date - timedelta(days=reference_date.weekday())
        end = start + timedelta(days=6)

    elif period == DatePeriod.CURRENT_MONTH:
        # First to last day of current month
        start = reference_date.replace(day=1)
        # Go to next month, then back one day
        if reference_date.month == 12:
            end = reference_date.replace(year=reference_date.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            end = reference_date.replace(month=reference_date.month + 1, day=1) - timedelta(days=1)

    elif period == DatePeriod.LAST_MONTH:
        # First to last day of previous month
        first_of_current = reference_date.replace(day=1)
        end = first_of_current - timedelta(days=1)
        start = end.replace(day=1)

    elif period == DatePeriod.CURRENT_QUARTER:
        # First day of quarter to last day of quarter
        quarter = (reference_date.month - 1) // 3 + 1
        start_month = (quarter - 1) * 3 + 1
        start = reference_date.replace(month=start_month, day=1)
        # End of quarter
        end_month = quarter * 3
        if end_month == 12:
            end = reference_date.replace(month=12, day=31)
        else:
            end = reference_date.replace(month=end_month + 1, day=1) - timedelta(days=1)

    elif period == DatePeriod.CURRENT_YEAR:
        # Jan 1 to Dec 31 of current year
        start = reference_date.replace(month=1, day=1)
        end = reference_date.replace(month=12, day=31)

    else:
        raise ValueError(f"Unknown period: {period}")

    return start, end


def get_group_name_for_period(
    period: DatePeriod,
    reference_date: Optional[date] = None,
) -> str:
    """
    Get the target group name for a given period.

    Examples:
        - LAST_MONTH (in January 2026) → "Décembre 2025"
        - CURRENT_MONTH (in January 2026) → "Janvier 2026"
        - CURRENT_QUARTER (in Q1 2026) → "Q1 2026"
        - CURRENT_YEAR (in 2026) → "2026"
        - CURRENT_WEEK (week of Jan 13, 2026) → "Semaine 3 - 2026"

    Args:
        period: The date period
        reference_date: Reference date (defaults to today)

    Returns:
        French group name string
    """
    start_date, end_date = get_period_date_range(period, reference_date)

    if period in (DatePeriod.CURRENT_MONTH, DatePeriod.LAST_MONTH):
        # Use month name + year
        month_name = MONTHS_FR[start_date.month]
        return f"{month_name} {start_date.year}"

    elif period == DatePeriod.CURRENT_QUARTER:
        quarter = (start_date.month - 1) // 3 + 1
        return f"Q{quarter} {start_date.year}"

    elif period == DatePeriod.CURRENT_YEAR:
        return str(start_date.year)

    elif period == DatePeriod.CURRENT_WEEK:
        week_number = start_date.isocalendar()[1]
        return f"Semaine {week_number} - {start_date.year}"

    else:
        raise ValueError(f"Unknown period: {period}")


# =============================================================================
# DATA FILTERING & AGGREGATION
# =============================================================================

def normalize_advisor_column(
    df: pd.DataFrame,
    advisor_column: str = "Conseiller",
    filter_unknown: bool = True,
) -> tuple[pd.DataFrame, int]:
    """
    Normalize advisor names in a DataFrame column to full name format.

    Uses the AdvisorMatcher to match variations like "Poirier", "Poirier D",
    "D. Poirier" to the standardized full name (e.g., "Daniel Poirier").

    Args:
        df: Input DataFrame
        advisor_column: Name of the column containing advisor names
        filter_unknown: If True, filter out rows where advisor is not in database

    Returns:
        Tuple of (DataFrame with normalized advisor names, count of filtered rows)
    """
    if df.empty:
        return df, 0

    if advisor_column not in df.columns:
        return df, 0

    # Make a copy to avoid modifying original
    df = df.copy()

    # Apply normalization to the advisor column (returns None if not found)
    df[advisor_column] = df[advisor_column].apply(
        lambda x: normalize_advisor_name_full(str(x)) if pd.notna(x) else None
    )

    # Count and filter out unknown advisors
    filtered_count = 0
    if filter_unknown:
        # Count rows with None/empty advisor
        unknown_mask = df[advisor_column].isna() | (df[advisor_column] == "") | (df[advisor_column].astype(str).str.strip() == "")
        filtered_count = unknown_mask.sum()
        # Filter them out
        df = df[~unknown_mask]

    return df, filtered_count


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
) -> tuple[pd.DataFrame, int]:
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
                  count of rows filtered out due to unknown advisors)
    """
    if df.empty:
        return pd.DataFrame(columns=[advisor_column, value_column]), 0

    if advisor_column not in df.columns:
        return pd.DataFrame(columns=[advisor_column, value_column]), 0

    if value_column not in df.columns:
        return pd.DataFrame(columns=[advisor_column, value_column]), 0

    # Make a copy
    df = df.copy()

    # Normalize advisor names before aggregation (also filters unknown advisors)
    filtered_count = 0
    if normalize_names:
        df, filtered_count = normalize_advisor_column(df, advisor_column, filter_unknown=True)

    # Convert value column to numeric
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce").fillna(0)

    # Group by advisor and sum
    result = (
        df.groupby(advisor_column, as_index=False)[value_column]
        .sum()
        .sort_values(value_column, ascending=False)
    )

    return result, filtered_count


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
