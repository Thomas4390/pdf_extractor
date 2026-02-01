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

class PeriodType(Enum):
    """Types of date periods available."""
    MONTH = "month"
    WEEK = "week"
    QUARTER = "quarter"
    YEAR = "year"
    CUSTOM = "custom"


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


@dataclass
class FlexiblePeriod:
    """
    Flexible date period supporting various period types.

    Can represent monthly, weekly, quarterly, annual, or custom date ranges.

    IMPORTANT: The reference_date is stored when the period is created to ensure
    consistent date calculations even when the actual date changes (e.g., month rollover).
    """
    period_type: PeriodType
    # For MONTH type - uses months_ago offset
    months_ago: int = 0
    # For WEEK type - uses weeks_ago offset
    weeks_ago: int = 0
    # For QUARTER type - uses quarters_ago offset
    quarters_ago: int = 0
    # For YEAR type - uses years_ago offset
    years_ago: int = 0
    # For CUSTOM type - explicit date range
    custom_start: Optional[date] = None
    custom_end: Optional[date] = None
    # Reference date for calculating offsets - stored at creation time
    # This ensures the period doesn't shift when the month changes
    reference_date: Optional[date] = None

    def __post_init__(self):
        """Validate period inputs and set reference date."""
        # Store reference date at creation time if not provided
        # This ensures consistent calculations even when the month changes
        if self.reference_date is None:
            # Use object.__setattr__ for frozen-like behavior in dataclass
            object.__setattr__(self, 'reference_date', date.today())

        # Validate non-negative offsets
        if self.months_ago < 0:
            raise ValueError(f"months_ago must be >= 0, got {self.months_ago}")
        if self.weeks_ago < 0:
            raise ValueError(f"weeks_ago must be >= 0, got {self.weeks_ago}")
        if self.quarters_ago < 0:
            raise ValueError(f"quarters_ago must be >= 0, got {self.quarters_ago}")
        if self.years_ago < 0:
            raise ValueError(f"years_ago must be >= 0, got {self.years_ago}")

        # Validate reasonable upper bounds (100 years)
        if self.months_ago > 1200:
            raise ValueError(f"months_ago must be <= 1200, got {self.months_ago}")
        if self.weeks_ago > 5200:
            raise ValueError(f"weeks_ago must be <= 5200, got {self.weeks_ago}")
        if self.quarters_ago > 400:
            raise ValueError(f"quarters_ago must be <= 400, got {self.quarters_ago}")
        if self.years_ago > 100:
            raise ValueError(f"years_ago must be <= 100, got {self.years_ago}")

        # Validate custom date range
        if self.period_type == PeriodType.CUSTOM:
            if self.custom_start and self.custom_end:
                if self.custom_start > self.custom_end:
                    raise ValueError("custom_start must be <= custom_end")

    @property
    def display_name(self) -> str:
        """Human-readable name for the period."""
        # Use stored reference_date for consistent calculations
        ref_date = self.reference_date or date.today()

        if self.period_type == PeriodType.MONTH:
            target_date = get_month_from_offset(self.months_ago, ref_date)
            return f"{MONTHS_FR[target_date.month]} {target_date.year}"

        elif self.period_type == PeriodType.WEEK:
            start, end = self.get_date_range()
            return f"Semaine du {start.strftime('%d/%m')} au {end.strftime('%d/%m/%Y')}"

        elif self.period_type == PeriodType.QUARTER:
            start, _ = self.get_date_range()
            quarter = (start.month - 1) // 3 + 1
            return f"{QUARTERS_FR[quarter]} {start.year}"

        elif self.period_type == PeriodType.YEAR:
            target_year = ref_date.year - self.years_ago
            return f"Année {target_year}"

        elif self.period_type == PeriodType.CUSTOM:
            if self.custom_start and self.custom_end:
                return f"{self.custom_start.strftime('%d/%m/%Y')} - {self.custom_end.strftime('%d/%m/%Y')}"
            return "Période personnalisée"

        return "Période inconnue"

    @property
    def short_label(self) -> str:
        """Short label for UI."""
        if self.period_type == PeriodType.MONTH:
            if self.months_ago == 0:
                return "Ce mois"
            elif self.months_ago == 1:
                return "Mois dernier"
            else:
                return f"-{self.months_ago} mois"
        elif self.period_type == PeriodType.WEEK:
            if self.weeks_ago == 0:
                return "Cette semaine"
            elif self.weeks_ago == 1:
                return "Semaine dernière"
            else:
                return f"-{self.weeks_ago} sem."
        elif self.period_type == PeriodType.QUARTER:
            if self.quarters_ago == 0:
                return "Ce trimestre"
            elif self.quarters_ago == 1:
                return "Trim. dernier"
            else:
                return f"-{self.quarters_ago} trim."
        elif self.period_type == PeriodType.YEAR:
            if self.years_ago == 0:
                return "Cette année"
            elif self.years_ago == 1:
                return "Année dernière"
            else:
                return f"-{self.years_ago} ans"
        elif self.period_type == PeriodType.CUSTOM:
            return "Personnalisé"
        return ""

    def get_date_range(self, reference_date: Optional[date] = None) -> tuple[date, date]:
        """Get start and end dates for this period."""
        # Use stored reference_date by default for consistent calculations
        if reference_date is None:
            reference_date = self.reference_date or date.today()

        if self.period_type == PeriodType.MONTH:
            start = get_month_from_offset(self.months_ago, reference_date)
            if start.month == 12:
                end = date(start.year + 1, 1, 1) - timedelta(days=1)
            else:
                end = date(start.year, start.month + 1, 1) - timedelta(days=1)
            return start, end

        elif self.period_type == PeriodType.WEEK:
            # Get the Monday of the target week
            days_since_monday = reference_date.weekday()
            current_monday = reference_date - timedelta(days=days_since_monday)
            target_monday = current_monday - timedelta(weeks=self.weeks_ago)
            target_sunday = target_monday + timedelta(days=6)
            return target_monday, target_sunday

        elif self.period_type == PeriodType.QUARTER:
            # Calculate target quarter
            current_quarter = (reference_date.month - 1) // 3 + 1
            current_year = reference_date.year

            target_quarter = current_quarter - self.quarters_ago
            target_year = current_year

            while target_quarter <= 0:
                target_quarter += 4
                target_year -= 1
            while target_quarter > 4:
                target_quarter -= 4
                target_year += 1

            # Quarter start month: Q1=1, Q2=4, Q3=7, Q4=10
            start_month = (target_quarter - 1) * 3 + 1
            start = date(target_year, start_month, 1)

            # End of quarter
            end_month = start_month + 2
            if end_month == 12:
                end = date(target_year + 1, 1, 1) - timedelta(days=1)
            else:
                end = date(target_year, end_month + 1, 1) - timedelta(days=1)

            return start, end

        elif self.period_type == PeriodType.YEAR:
            target_year = reference_date.year - self.years_ago
            start = date(target_year, 1, 1)
            end = date(target_year, 12, 31)
            return start, end

        elif self.period_type == PeriodType.CUSTOM:
            if self.custom_start and self.custom_end:
                return self.custom_start, self.custom_end
            # Fallback to current month if no custom dates
            return get_month_from_offset(0, reference_date), reference_date

        # Default fallback
        return get_month_from_offset(0, reference_date), reference_date

    def get_group_name(self, reference_date: Optional[date] = None) -> str:
        """Get the target group name for this period."""
        start, end = self.get_date_range(reference_date)

        if self.period_type == PeriodType.MONTH:
            return f"{MONTHS_FR[start.month]} {start.year}"
        elif self.period_type == PeriodType.WEEK:
            return f"Semaine {start.isocalendar()[1]} - {start.year}"
        elif self.period_type == PeriodType.QUARTER:
            quarter = (start.month - 1) // 3 + 1
            return f"{QUARTERS_FR[quarter]} {start.year}"
        elif self.period_type == PeriodType.YEAR:
            return f"Année {start.year}"
        elif self.period_type == PeriodType.CUSTOM:
            return f"{start.strftime('%d/%m/%Y')} - {end.strftime('%d/%m/%Y')}"

        return f"{MONTHS_FR[start.month]} {start.year}"


def get_flexible_period_options() -> list[FlexiblePeriod]:
    """Get a list of predefined flexible period options for UI."""
    options = []

    # Monthly options (last 12 months)
    for i in range(12):
        options.append(FlexiblePeriod(period_type=PeriodType.MONTH, months_ago=i))

    # Weekly options (last 4 weeks)
    for i in range(4):
        options.append(FlexiblePeriod(period_type=PeriodType.WEEK, weeks_ago=i))

    # Quarterly options (last 4 quarters)
    for i in range(4):
        options.append(FlexiblePeriod(period_type=PeriodType.QUARTER, quarters_ago=i))

    # Yearly options (last 3 years)
    for i in range(3):
        options.append(FlexiblePeriod(period_type=PeriodType.YEAR, years_ago=i))

    return options


def get_month_from_offset(months_ago: int, reference_date: Optional[date] = None) -> date:
    """
    Get the first day of a month N months ago.

    Args:
        months_ago: Number of months to go back (0 = current month)
        reference_date: Reference date (defaults to today)

    Returns:
        First day of the target month

    Raises:
        ValueError: If months_ago is negative or exceeds 1200 (100 years)
    """
    # Validate input
    if months_ago < 0:
        raise ValueError(f"months_ago must be >= 0, got {months_ago}")
    if months_ago > 1200:
        raise ValueError(f"months_ago must be <= 1200 (100 years), got {months_ago}")

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
        output_column_name="AE CA",
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
        output_column_name="Collected",
        board_id=9142978904,  # Monday.com "AE Tracker" board
        advisor_column="group_title",  # Advisor name is in group title
        use_group_as_advisor=True,
    ),
}


# =============================================================================
# METRICS CONFIGURATION
# =============================================================================

@dataclass
class MetricsConfig:
    """Configuration for loading additional metrics from a board."""
    board_id: int
    advisor_column: str = "Conseiller"
    # Column names in the source board
    cost_column: str = "Coût"
    expenses_column: str = "Dépenses par Conseiller"
    leads_column: str = "Leads"
    bonus_column: str = "Bonus"
    rewards_column: str = "Récompenses"


# Default metrics board configuration
# The metrics board "Data" has groups named by month (e.g., "Janvier 2026")
METRICS_BOARD_CONFIG = MetricsConfig(
    board_id=18394590851,  # Monday.com "2026 Copie de Data" board
)


# =============================================================================
# METRICS CALCULATIONS
# =============================================================================

def calculate_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived metrics columns from base data.

    Calculates:
    - Total Dépenses = Coût + Dépenses par Conseiller + Bonus
    - Profit = AE CA + Récompenses + (Bonus + Dépenses par Conseiller + Coût)
    - CA/Lead = ROUND(AE CA / Leads, 2)
    - Profit/Lead = ROUND(Profit / Leads, 2)
    - Ratio Brut = ROUND((AE CA / -(Coût + Bonus + Dépenses par Conseiller)) * 100, 2)
    - Ratio Net = ROUND((Profit / -(Coût + Bonus + Dépenses par Conseiller)) * 100, 2)
    - Profitable = Win/Middle/Loss based on Ratio Net

    IMPORTANT: When "Dépenses par Conseiller" is null/zero, the advisor has no expense
    data and profitability cannot be accurately calculated. In this case:
    - Ratio Brut and Ratio Net will be 0
    - Profitable status will be "N/A" (no data)

    Uses vectorized NumPy operations for performance.

    Args:
        df: DataFrame with columns: AE CA, Coût, Dépenses par Conseiller, Leads, Bonus, Récompenses

    Returns:
        DataFrame with additional calculated columns
    """
    import numpy as np

    if df.empty:
        return df

    df = df.copy()

    # Ensure numeric columns
    numeric_cols = ["AE CA", "Coût", "Dépenses par Conseiller", "Leads", "Bonus", "Récompenses"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Calculate Total Dépenses (vectorized)
    if all(col in df.columns for col in ["Coût", "Dépenses par Conseiller", "Bonus"]):
        df["Total Dépenses"] = df["Coût"] + df["Dépenses par Conseiller"] + df["Bonus"]

    # Calculate Profit (vectorized)
    # Profit = AE CA + Récompenses + (Bonus + Dépenses par Conseiller + Coût)
    # Note: Coût, Bonus, Dépenses are typically negative values (expenses)
    if all(col in df.columns for col in ["AE CA", "Récompenses", "Bonus", "Dépenses par Conseiller", "Coût"]):
        df["Profit"] = (
            df["AE CA"] +
            df["Récompenses"] +
            df["Bonus"] +
            df["Dépenses par Conseiller"] +
            df["Coût"]
        )

    # Calculate CA/Lead (vectorized with np.where for division by zero)
    if "AE CA" in df.columns and "Leads" in df.columns:
        df["CA/Lead"] = np.where(
            df["Leads"] != 0,
            np.round(df["AE CA"] / df["Leads"], 2),
            0.0
        )

    # Calculate Profit/Lead (vectorized)
    if "Profit" in df.columns and "Leads" in df.columns:
        df["Profit/Lead"] = np.where(
            df["Leads"] != 0,
            np.round(df["Profit"] / df["Leads"], 2),
            0.0
        )

    # Check if we have expense data for profitability calculations
    # If "Dépenses par Conseiller" is 0, the advisor has no expense data
    has_expense_data = pd.Series(True, index=df.index)
    if "Dépenses par Conseiller" in df.columns:
        has_expense_data = df["Dépenses par Conseiller"] != 0

    # Calculate denominator for ratio calculations (expenses sum)
    if all(col in df.columns for col in ["Coût", "Bonus", "Dépenses par Conseiller"]):
        expenses_sum = df["Coût"] + df["Bonus"] + df["Dépenses par Conseiller"]

        # Calculate Ratio Brut (vectorized)
        # Ratio Brut = (AE CA / -expenses_sum) * 100
        # Only calculate if there's expense data
        if "AE CA" in df.columns:
            df["Ratio Brut"] = np.where(
                (expenses_sum != 0) & has_expense_data,
                np.round((df["AE CA"] / -expenses_sum) * 100, 2),
                0.0
            )

        # Calculate Ratio Net (vectorized)
        # Ratio Net = (Profit / -expenses_sum) * 100
        # Only calculate if there's expense data
        if "Profit" in df.columns:
            df["Ratio Net"] = np.where(
                (expenses_sum != 0) & has_expense_data,
                np.round((df["Profit"] / -expenses_sum) * 100, 2),
                0.0
            )

    # Calculate Profitable status based on Ratio Net (vectorized with np.select)
    # Loss: Ratio Net < 20
    # Middle: 20 <= Ratio Net <= 99
    # Win: Ratio Net > 99
    # N/A: No expense data (Dépenses par Conseiller = 0)
    if "Ratio Net" in df.columns:
        conditions = [
            ~has_expense_data,  # No expense data
            df["Ratio Net"] > 99,
            df["Ratio Net"] >= 20,
        ]
        choices = ["N/A", "Win", "Middle"]
        df["Profitable"] = np.select(conditions, choices, default="Loss")

    return df


def merge_metrics_with_aggregation(
    aggregated_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    advisor_column: str = "Conseiller",
    use_fuzzy_matching: bool = True,
) -> pd.DataFrame:
    """
    Merge metrics data with aggregated data by advisor using fuzzy name matching.

    Uses AdvisorMatcher to handle name variations (e.g., "Brandeen" vs "Brandeen G.").

    Args:
        aggregated_df: DataFrame with aggregated data (from combine_aggregations)
        metrics_df: DataFrame with metrics columns (Coût, Dépenses par Conseiller, etc.)
        advisor_column: Column name for advisor matching
        use_fuzzy_matching: If True, use AdvisorMatcher for fuzzy name matching

    Returns:
        Merged DataFrame with all columns
    """
    if aggregated_df.empty:
        return aggregated_df

    if metrics_df.empty:
        return aggregated_df

    # Ensure advisor column exists in both
    if advisor_column not in aggregated_df.columns:
        return aggregated_df
    if advisor_column not in metrics_df.columns:
        return aggregated_df

    result = aggregated_df.copy()
    metrics_df = metrics_df.copy()

    if use_fuzzy_matching:
        # Normalize advisor names in both DataFrames for better matching
        # First, normalize aggregated_df names (should already be normalized)
        result["_normalized_advisor"] = result[advisor_column].apply(
            lambda x: normalize_advisor_name_full(str(x)) if pd.notna(x) else x
        )
        # Replace None with original for fallback
        result["_normalized_advisor"] = result.apply(
            lambda row: row["_normalized_advisor"] if row["_normalized_advisor"] else row[advisor_column],
            axis=1
        )

        # Normalize metrics_df names
        metrics_df["_normalized_advisor"] = metrics_df[advisor_column].apply(
            lambda x: normalize_advisor_name_full(str(x)) if pd.notna(x) else x
        )
        # Replace None with original for fallback
        metrics_df["_normalized_advisor"] = metrics_df.apply(
            lambda row: row["_normalized_advisor"] if row["_normalized_advisor"] else row[advisor_column],
            axis=1
        )

        # Merge on normalized names
        metrics_cols_to_merge = [c for c in metrics_df.columns if c not in [advisor_column, "_normalized_advisor"]]
        metrics_subset = metrics_df[["_normalized_advisor"] + metrics_cols_to_merge]

        result = result.merge(
            metrics_subset,
            on="_normalized_advisor",
            how="left",
        )

        # Clean up temporary column
        result = result.drop(columns=["_normalized_advisor"], errors="ignore")
    else:
        # Standard exact merge
        result = result.merge(
            metrics_df,
            on=advisor_column,
            how="left",
        )

    # Fill NaN with 0 for numeric columns
    numeric_cols = ["Coût", "Dépenses par Conseiller", "Leads", "Bonus", "Récompenses"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    return result


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


def filter_by_flexible_period(
    df: pd.DataFrame,
    period: FlexiblePeriod,
    date_column: str,
    reference_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Filter DataFrame by flexible period (supports all period types).

    Args:
        df: Input DataFrame
        period: FlexiblePeriod object defining the date range
        date_column: Name of the date column
        reference_date: Reference date (defaults to today)

    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df

    if date_column not in df.columns:
        return df

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    start_date, end_date = period.get_date_range(reference_date)

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

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
