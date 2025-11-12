"""
Insurance Commission Data Unification System
=============================================

Production-ready script to extract, standardize, and unify commission data from
three insurance sources: UV Assurance, IDC, and Assomption Vie.

This script:
    1. Extracts data from PDF reports using source-specific extractors
    2. Converts all data to a standardized schema with unified column names
    3. Validates data quality and flags issues
    4. Generates individual output files (CSV and Excel) for each source
    5. Displays complete cleaned dataframes
    6. Produces detailed summary reports with statistics

Configuration:
    - PDF Paths: Configured below in PDF_PATHS dictionary
    - Output Directory: ../results/
    - Format: CSV and Excel outputs

Author: Thomas
Date: 2025-10-23
Version: 1.3.1 - Added constant sharing_rate=0.4 for Assomption
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import sys
import warnings

warnings.filterwarnings('ignore')

# Import source-specific extractors
try:
    from uv_extractor import RemunerationReportExtractor
    UV_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: uv_extractor.py not found")
    UV_AVAILABLE = False

try:
    from idc_extractor import PDFPropositionParser
    IDC_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: idc_extractor.py not found")
    IDC_AVAILABLE = False

try:
    from assomption_extractor import extract_pdf_data
    ASSOMPTION_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: assomption_extractor.py not found")
    ASSOMPTION_AVAILABLE = False

try:
    from idc_statements_extractor import PDFStatementParser
    IDC_STATEMENT_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: idc_statements_extractor.py not found")
    IDC_STATEMENT_AVAILABLE = False

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR DATA
# =============================================================================

PDF_PATHS = {
    'UV': "../pdf/rappportremun_21622_2025-10-20.pdf",
    'IDC': "../pdf/Rapport des propositions soumises.20251017_1517.pdf",
    'ASSOMPTION': "../pdf/Remuneration (61).pdf"
}

OUTPUT_DIRECTORY = "../results/"


class CommissionDataUnifier:
    """
    Unifies commission data from multiple insurance sources to a standard schema.
    """

    # Standard schema definition
    STANDARD_COLUMNS = [
        # Identifiers
        'contract_number',
        'transaction_code',
        'insurer_name',

        # Client info
        'insured_name',

        # Product info
        'product_name',
        'product_type',

        # Policy details
        'policy_status',
        'effective_date',
        'issue_date',
        'payment_frequency',
        'billing_type',

        # Financial - Premium
        'policy_premium',
        'commissionable_premium',

        # Financial - Commission
        'commission_rate',
        'sharing_rate',
        'base_commission',
        'commission',
        'commission_type',

        # Financial - Bonus
        'bonus_rate',
        'bonus_amount',

        # Financial - On Commission
        'on_commission_rate',
        'on_commission',

        # Financial - Totals
        'total_commission',
        'result_amount',

        # Coverage
        'coverage_amount',

        # Metadata
        'report_date',

        # Additional fields for Monday.com integration
        'status',
        'advisor_name',
        'is_verified',
        'comments',
        'amount_received'
    ]

    # Final columns to keep in output
    FINAL_COLUMNS = [
        'contract_number',
        'insured_name',
        'insurer_name',
        'status',
        'advisor_name',
        'is_verified',
        'policy_premium',
        'sharing_rate',
        'commission_rate',
        'commission',
        'bonus_rate',
        'bonus_amount',
        'on_commission_rate',
        'on_commission',
        'amount_received',
        'report_date',
        'comments'
    ]

    def __init__(self, output_dir: str = "./unified_data"):
        """
        Initialize the unifier.

        Args:
            output_dir: Directory to save unified data files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def _calculate_commissions(self, df: pd.DataFrame, on_commission_rate: float = 0.75) -> pd.DataFrame:
        """
        Calculate commission, bonus_amount, and on_commission for a standardized DataFrame.

        Args:
            df: DataFrame with policy_premium, sharing_rate, commission_rate, bonus_rate
            on_commission_rate: Rate for on_commission calculation (default: 0.75)

        Returns:
            DataFrame with calculated columns added
        """
        # Set on_commission_rate
        df['on_commission_rate'] = on_commission_rate

        # Calculate commission
        df['commission'] = (
            df['policy_premium'] * df['sharing_rate'] * df['commission_rate']
        )

        # Calculate bonus_amount (only if bonus_rate exists and is not None)
        if 'bonus_rate' in df.columns and df['bonus_rate'].notna().any():
            df['bonus_amount'] = (
                df['policy_premium'] * df['sharing_rate'] *
                df['commission_rate'] * df['bonus_rate']
            )

        # Calculate on_commission
        df['on_commission'] = (
            df['policy_premium'] * (1 - df['sharing_rate']) *
            df['on_commission_rate'] * df['commission_rate']
        )

        return df

    def _round_float_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Round all numeric float columns to 2 decimal places.
        Skips columns that are None or object type.

        Args:
            df: DataFrame to round

        Returns:
            DataFrame with rounded values
        """
        float_columns = ['sharing_rate', 'commission_rate', 'bonus_rate', 'commission',
                        'bonus_amount', 'on_commission_rate', 'on_commission']

        for col in float_columns:
            if col in df.columns:
                # Only round if column is numeric (skip None/object types)
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].round(2)

        return df

    def _clean_percentage(self, value) -> Optional[float]:
        """
        Clean and convert percentage values to float.

        Handles formats like:
        - "55,000 %"
        - "175,00%"
        - "40.5%"

        Args:
            value: Percentage value to clean

        Returns:
            Float value (e.g., 55.0 for 55%)
        """
        if pd.isna(value) or value == '' or value is None:
            return None

        try:
            # Convert to string and clean
            value_str = str(value).replace('%', '').replace(' ', '').replace(',', '.')
            return float(value_str)
        except (ValueError, AttributeError):
            return None

    def _clean_currency(self, value) -> Optional[float]:
        """
        Clean and convert currency values to float.

        Handles formats like:
        - "1 196,00 $"
        - "348,5 $"
        - "50 000 $"

        Args:
            value: Currency value to clean

        Returns:
            Float value without currency symbol
        """
        if pd.isna(value) or value == '' or value is None:
            return None

        try:
            # Convert to string and clean
            value_str = str(value).replace('$', '').replace(' ', '').replace(',', '.')
            return float(value_str)
        except (ValueError, AttributeError):
            return None

    def _parse_date(self, value, format_hint: str = None) -> Optional[pd.Timestamp]:
        """
        Parse date values to pandas Timestamp.

        Args:
            value: Date value to parse
            format_hint: Optional format hint ('slash' or 'dash')

        Returns:
            pandas Timestamp or None
        """
        if pd.isna(value) or value == '' or value is None:
            return None

        try:
            # If already a Timestamp, return as is
            if isinstance(value, pd.Timestamp):
                return value

            # Convert to string
            date_str = str(value).strip()

            # Try different formats
            if format_hint == 'slash' or '/' in date_str:
                # Format: YYYY/MM/DD
                return pd.to_datetime(date_str, format='%Y/%m/%d')
            else:
                # Format: YYYY-MM-DD or let pandas infer
                return pd.to_datetime(date_str)

        except (ValueError, AttributeError):
            return None

    def _format_date_uniform(self, date_value) -> Optional[str]:
        """
        Format date to uniform YYYY-MM-DD string format.

        Args:
            date_value: Date value (Timestamp, string, or other)

        Returns:
            Date string in YYYY-MM-DD format or None
        """
        if pd.isna(date_value) or date_value is None:
            return None

        try:
            # Parse if not already a Timestamp
            if not isinstance(date_value, pd.Timestamp):
                date_value = pd.to_datetime(date_value)

            # Format as YYYY-MM-DD
            return date_value.strftime('%Y-%m-%d')

        except (ValueError, AttributeError):
            return None

    def convert_uv_to_standard(self, df: pd.DataFrame, metadata: Dict = None) -> pd.DataFrame:
        """
        Convert UV Assurance data to standard schema.

        Args:
            df: UV Assurance DataFrame
            metadata: Dictionary with report_date, advisor_name, advisor_number

        Returns:
            Standardized DataFrame
        """
        if df.empty:
            return self._create_empty_standard_df()

        # Get the number of rows for consistent array creation
        n_rows = len(df)

        # Initialize with standard columns
        standard_df = pd.DataFrame(index=df.index)

        # Map columns - 'Montant de base' now goes to 'policy_premium'
        standard_df['contract_number'] = df['Contrat'].astype(str)
        standard_df['insured_name'] = df['Assuré(s)'].astype(str)
        standard_df['product_name'] = df['Protection'].astype(str)
        standard_df['policy_premium'] = df['Montant de base'].apply(self._clean_currency)
        # Convert rates to float (0.0-1.0): divide percentage by 100
        standard_df['sharing_rate'] = df['Taux de partage'].apply(self._clean_percentage) / 100
        standard_df['commission_rate'] = df['Taux de commission'].apply(self._clean_percentage) / 100
        standard_df['result_amount'] = df['Résultat'].apply(self._clean_currency)
        standard_df['commission_type'] = df['Type'].astype(str)
        standard_df['bonus_rate'] = df['Taux de Boni'].apply(self._clean_percentage) / 100
        standard_df['total_commission'] = df['Rémunération'].apply(self._clean_currency)

        # Calculate commissions using helper method
        standard_df = self._calculate_commissions(standard_df)

        # Round float columns using helper method
        standard_df = self._round_float_columns(standard_df)

        # Add insurer_name AFTER creating the DataFrame with data
        standard_df['insurer_name'] = np.repeat('UV', n_rows)

        # Add metadata if provided - format date uniformly as YYYY-MM-DD
        if metadata:
            report_date_parsed = self._parse_date(metadata.get('date'))
            report_date_formatted = self._format_date_uniform(report_date_parsed)
            if report_date_formatted:
                standard_df['report_date'] = np.repeat(report_date_formatted, n_rows)
                print(f"   ✓ Report date formatted: {report_date_formatted}")
            else:
                print(f"   ⚠️  Warning: Could not format report date from metadata: {metadata.get('date')}")
                standard_df['report_date'] = None

        return self._ensure_standard_columns(standard_df)

    def convert_idc_to_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert IDC data to standard schema.

        Args:
            df: IDC DataFrame

        Returns:
            Standardized DataFrame
        """
        if df.empty:
            return self._create_empty_standard_df()

        # Initialize with standard columns
        standard_df = pd.DataFrame(index=df.index)

        # Map columns
        standard_df['insurer_name'] = df['Assureur'].astype(str)
        standard_df['insured_name'] = df['Client'].astype(str)
        standard_df['product_type'] = df['Type de régime'].astype(str)
        standard_df['contract_number'] = df['Police'].astype(str)
        standard_df['policy_status'] = df['Statut'].astype(str)
        standard_df['effective_date'] = df['Date'].apply(self._parse_date)
        # Map 'Date' to 'report_date' and format uniformly as YYYY-MM-DD
        standard_df['report_date'] = df['Date'].apply(lambda x: self._format_date_uniform(self._parse_date(x)))
        # sharing_rate is already a decimal (0.0-1.0), keep it as is
        standard_df['sharing_rate'] = pd.to_numeric(df['Nombre'], errors='coerce')
        # Convert commission_rate to float (0.0-1.0): divide percentage by 100
        standard_df['commission_rate'] = df['Taux de CPA'].apply(self._clean_percentage) / 100
        standard_df['coverage_amount'] = df['Couverture'].apply(self._clean_currency)
        standard_df['policy_premium'] = df['Prime de la police'].apply(self._clean_currency)
        standard_df['commissionable_premium'] = df['Part prime comm.'].apply(self._clean_currency)
        standard_df['total_commission'] = df['Comm.'].apply(self._clean_currency)

        # For IDC, bonus_rate is not available in the data
        standard_df['bonus_rate'] = None
        standard_df['bonus_amount'] = None

        # Calculate commissions using helper method
        standard_df = self._calculate_commissions(standard_df)

        # Round float columns using helper method
        standard_df = self._round_float_columns(standard_df)

        # Clean insurer_name: Replace "Assumption Life (ASSUMPTI ON)" with "Assomption"
        standard_df['insurer_name'] = standard_df['insurer_name'].str.replace(
            'Assumption Life (ASSUMPTI ON)', 'Assomption', regex=False
        )

        return self._ensure_standard_columns(standard_df)

    def convert_assomption_to_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Assomption Vie data to standard schema.

        Args:
            df: Assomption Vie DataFrame

        Returns:
            Standardized DataFrame
        """
        if df.empty:
            return self._create_empty_standard_df()

        # Get the number of rows for consistent array creation
        n_rows = len(df)

        # Initialize with standard columns
        standard_df = pd.DataFrame(index=df.index)

        # Map columns
        standard_df['transaction_code'] = df['Code'].astype(str)
        standard_df['contract_number'] = df['Numéro Police'].astype(str)
        standard_df['insured_name'] = df['Nom de l\'assuré'].astype(str)
        standard_df['product_name'] = df['Produit'].astype(str)
        standard_df['issue_date'] = df['Émission'].apply(lambda x: self._parse_date(x, 'slash'))
        # Map 'Émission' to 'report_date' and format uniformly as YYYY-MM-DD
        standard_df['report_date'] = df['Émission'].apply(lambda x: self._format_date_uniform(self._parse_date(x, 'slash')))
        standard_df['payment_frequency'] = df['Fréquence paiement'].astype(str)
        standard_df['billing_type'] = df['Facturation'].astype(str)
        standard_df['policy_premium'] = pd.to_numeric(df['Prime'], errors='coerce')
        # Constant sharing_rate for Assomption in float format (0.4 = 40%)
        standard_df['sharing_rate'] = 0.4  # Constant value for Assomption
        # Convert commission_rate to float (0.0-1.0): divide percentage by 100
        standard_df['commission_rate'] = df['Taux Commission'].apply(self._clean_percentage) / 100
        standard_df['base_commission'] = pd.to_numeric(df['Commissions'], errors='coerce')

        # Handle bonus columns (may or may not exist)
        if 'Taux Boni' in df.columns:
            # Convert bonus_rate to float (0.0-1.0): divide percentage by 100
            standard_df['bonus_rate'] = df['Taux Boni'].apply(self._clean_percentage) / 100
        else:
            standard_df['bonus_rate'] = None

        # Calculate commissions using helper method
        standard_df = self._calculate_commissions(standard_df)

        # Round float columns using helper method
        standard_df = self._round_float_columns(standard_df)

        # Calculate total commission (base + bonus)
        standard_df['total_commission'] = (
            standard_df['base_commission'].fillna(0) +
            standard_df.get('bonus_amount', pd.Series(0, index=df.index)).fillna(0)
        )

        # Add insurer_name AFTER creating the DataFrame with data
        standard_df['insurer_name'] = np.repeat('Assomption', n_rows)

        return self._ensure_standard_columns(standard_df)

    def convert_idc_statement_to_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert IDC Statement data (trailing fees) to standard schema.

        IDC Statement extracts trailing fee reports with columns:
        - 'Nom du client' → insured_name
        - 'Numéro de compte' → contract_number
        - 'Compagnie' → insurer_name
        - 'Date' → report_date
        - 'Frais de suivi nets' → on_commission
        - 'Taux sur-commission' → on_commission_rate

        Args:
            df: IDC Statement DataFrame

        Returns:
            Standardized DataFrame
        """
        if df.empty:
            return self._create_empty_standard_df()

        # Initialize with standard columns
        standard_df = pd.DataFrame(index=df.index)

        # Map columns from IDC Statement format
        standard_df['insured_name'] = df['Nom du client'].astype(str)
        standard_df['contract_number'] = df['Numéro de compte'].astype(str)
        standard_df['insurer_name'] = df['Compagnie'].astype(str)

        # Parse and format dates
        standard_df['report_date'] = df['Date'].apply(lambda x: self._format_date_uniform(self._parse_date(x)))

        # Parse trailing fees - Map 'Frais de suivi nets' to 'on_commission'
        standard_df['on_commission'] = df['Frais de suivi nets'].apply(self._clean_currency)

        # Map advisor name and on-commission rate
        if 'Nom du conseiller' in df.columns:
            standard_df['advisor_name'] = df['Nom du conseiller'].astype(str)
        else:
            standard_df['advisor_name'] = None

        if 'Taux sur-commission' in df.columns:
            # on_commission_rate is already a float from extraction (e.g., 0.75)
            standard_df['on_commission_rate'] = pd.to_numeric(df['Taux sur-commission'], errors='coerce')
        else:
            standard_df['on_commission_rate'] = None

        # Replace 'Unknown' and 'None' strings with NaN
        for col in standard_df.columns:
            if standard_df[col].dtype == 'object':  # Only for string columns
                standard_df[col] = standard_df[col].replace(['Unknown', 'None', 'unknown', 'none'], np.nan)

        # Round float columns using helper method
        standard_df = self._round_float_columns(standard_df)

        return self._ensure_standard_columns(standard_df)

    def convert_monday_legacy_to_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert legacy Monday.com board data to standard schema.

        Legacy column mapping:
        - 'item_name' (colonne "Élément") → insured_name  ** NOUVEAU **
        - '# de Police' → contract_number
        - 'Compagnie' → insurer_name
        - 'Statut' → status
        - 'Conseiller' → advisor_name
        - 'Vérifié' → is_verified
        - 'PA' → policy_premium
        - 'Com' → commission
        - 'Boni' → bonus_amount
        - 'Sur-Com' → on_commission
        - 'Reçu' → amount_received
        - 'Date' → report_date

        Constants added:
        - sharing_rate = 0.4 (40%)
        - commission_rate = 0.5 (50%)
        - bonus_rate = 0.0175 (1.75%)
        - on_commission_rate = 0.75 (75%)

        Args:
            df: DataFrame from legacy Monday.com board

        Returns:
            Standardized DataFrame
        """
        if df.empty:
            return self._create_empty_standard_df()

        # Get the number of rows for consistent array creation
        n_rows = len(df)

        # Initialize with standard columns
        standard_df = pd.DataFrame(index=df.index)

        # Map columns from legacy format
        # NOUVEAU: Map item_name (colonne "Élément" dans Monday.com) vers insured_name
        if 'item_name' in df.columns:
            standard_df['insured_name'] = df['item_name'].astype(str)
            print(f"   ✓ Mapped 'item_name' (Élément) → 'insured_name': {df['item_name'].notna().sum()} values")

        # Required mappings
        if '# de Police' in df.columns:
            standard_df['contract_number'] = df['# de Police'].astype(str)

        if 'Compagnie' in df.columns:
            standard_df['insurer_name'] = df['Compagnie'].astype(str)

        if 'PA' in df.columns:
            # Convert PA (policy premium) to numeric
            standard_df['policy_premium'] = pd.to_numeric(df['PA'], errors='coerce')

        # Try to extract Com, Boni, Sur-Com from source (may be empty for formula columns)
        if 'Com' in df.columns:
            # Convert commission to numeric
            standard_df['commission'] = pd.to_numeric(df['Com'], errors='coerce')

        if 'Boni' in df.columns:
            # Convert bonus to numeric
            standard_df['bonus_amount'] = pd.to_numeric(df['Boni'], errors='coerce')

        if 'Sur-Com' in df.columns:
            # Convert on_commission to numeric
            standard_df['on_commission'] = pd.to_numeric(df['Sur-Com'], errors='coerce')

        if 'Date' in df.columns:
            # Parse and format date
            standard_df['report_date'] = df['Date'].apply(
                lambda x: self._format_date_uniform(self._parse_date(x))
            )

        # New columns from Monday.com
        if 'Statut' in df.columns:
            standard_df['status'] = df['Statut'].astype(str)

        if 'Conseiller' in df.columns:
            standard_df['advisor_name'] = df['Conseiller'].astype(str)

        if 'Verifié' in df.columns:
            standard_df['is_verified'] = df['Verifié'].astype(str)

        if 'Reçu' in df.columns:
            # Convert amount received to numeric
            standard_df['amount_received'] = pd.to_numeric(df['Reçu'], errors='coerce')

        if 'Texte' in df.columns:
            standard_df['comments'] = df['Texte'].astype(str)

        # Add constants
        standard_df['sharing_rate'] = 0.4  # 40%
        standard_df['commission_rate'] = 0.5  # 50%
        standard_df['bonus_rate'] = 1.75  # 175%
        standard_df['on_commission_rate'] = 0.75  # 75%

        # CALCUL AUTOMATIQUE: Calculate Com, Boni, Sur-Com if they are empty
        # This is necessary because Monday.com formula columns don't return values via API
        if 'policy_premium' in standard_df.columns:
            # Count how many values are missing
            com_missing = standard_df['commission'].isna().sum() if 'commission' in standard_df.columns else n_rows
            boni_missing = standard_df['bonus_amount'].isna().sum() if 'bonus_amount' in standard_df.columns else n_rows
            surcom_missing = standard_df['on_commission'].isna().sum() if 'on_commission' in standard_df.columns else n_rows

            if com_missing > 0 or boni_missing > 0 or surcom_missing > 0:
                print(f"\n   🧮 Calculating missing formula columns:")
                print(f"      - commission: {com_missing} missing values")
                print(f"      - bonus_amount: {boni_missing} missing values")
                print(f"      - on_commission: {surcom_missing} missing values")

                # Calculate commission: PA × sharing_rate × commission_rate
                if com_missing > 0:
                    mask = standard_df['commission'].isna()
                    standard_df.loc[mask, 'commission'] = (
                        standard_df.loc[mask, 'policy_premium'] *
                        standard_df.loc[mask, 'sharing_rate'] *
                        standard_df.loc[mask, 'commission_rate']
                    )
                    calculated = standard_df.loc[mask, 'commission'].notna().sum()
                    print(f"      ✓ Calculated {calculated} commission values")

                # Calculate bonus_amount: PA × sharing_rate × commission_rate × bonus_rate
                if boni_missing > 0:
                    mask = standard_df['bonus_amount'].isna()
                    standard_df.loc[mask, 'bonus_amount'] = (
                        standard_df.loc[mask, 'policy_premium'] *
                        standard_df.loc[mask, 'sharing_rate'] *
                        standard_df.loc[mask, 'commission_rate'] *
                        standard_df.loc[mask, 'bonus_rate']
                    )
                    calculated = standard_df.loc[mask, 'bonus_amount'].notna().sum()
                    print(f"      ✓ Calculated {calculated} bonus_amount values")

                # Calculate on_commission: PA × (1 - sharing_rate) × on_commission_rate × commission_rate
                if surcom_missing > 0:
                    mask = standard_df['on_commission'].isna()
                    standard_df.loc[mask, 'on_commission'] = (
                        standard_df.loc[mask, 'policy_premium'] *
                        (1 - standard_df.loc[mask, 'sharing_rate']) *
                        standard_df.loc[mask, 'on_commission_rate'] *
                        standard_df.loc[mask, 'commission_rate']
                    )
                    calculated = standard_df.loc[mask, 'on_commission'].notna().sum()
                    print(f"      ✓ Calculated {calculated} on_commission values")

        # Initialize comments as None if not already set
        if 'comments' not in standard_df.columns:
            standard_df['comments'] = None

        # Round float columns using helper method
        standard_df = self._round_float_columns(standard_df)

        # IMPORTANT: Preserve group metadata from Monday.com extraction
        if 'group_id' in df.columns:
            standard_df['group_id'] = df['group_id']
        if 'group_title' in df.columns:
            standard_df['group_title'] = df['group_title']

        print(f"✅ Converted {len(standard_df)} records from legacy Monday.com format")
        print(f"   Constants applied: sharing_rate=0.4, commission_rate=0.5, bonus_rate=1.75, on_commission_rate=0.75")

        # Check if group metadata was preserved
        if 'group_title' in standard_df.columns:
            unique_groups = standard_df['group_title'].dropna().unique()
            print(f"   ✓ Group metadata preserved: {len(unique_groups)} groups - {list(unique_groups)}")

        return self._ensure_standard_columns(standard_df)

    def filter_by_sharing_rate(self, df: pd.DataFrame, target_rate: float = 0.4) -> pd.DataFrame:
        """
        Filter DataFrame to keep only rows with a specific sharing_rate.

        Args:
            df: DataFrame with sharing_rate column
            target_rate: Target sharing rate to filter (default: 0.4)

        Returns:
            Filtered DataFrame with only rows matching the target_rate
        """
        if df.empty:
            return df

        if 'sharing_rate' not in df.columns:
            print(f"⚠️  Warning: 'sharing_rate' column not found. Returning unfiltered data.")
            return df

        initial_count = len(df)

        # Filter rows where sharing_rate equals target_rate
        # Using .isclose() to handle floating point precision issues
        filtered_df = df[pd.Series(df['sharing_rate']).apply(
            lambda x: abs(x - target_rate) < 0.0001 if pd.notna(x) else False
        )].copy()

        final_count = len(filtered_df)
        removed_count = initial_count - final_count

        print(f"📊 Sharing Rate Filter (target: {target_rate}):")
        print(f"   Initial rows:  {initial_count}")
        print(f"   Filtered rows: {final_count}")
        print(f"   Removed rows:  {removed_count}")

        return filtered_df

    def aggregate_by_contract_number(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate rows with the same contract_number.

        Aggregation rules:
        - insured_name: Join with semicolons (;)
        - policy_premium, bonus_amount, on_commission: Sum
        - commission_rate, bonus_rate, on_commission_rate: Average (mean)
        - report_date: Take the oldest date
        - Other columns: Take first value

        Args:
            df: DataFrame to aggregate

        Returns:
            Aggregated DataFrame
        """
        if df.empty:
            return df

        if 'contract_number' not in df.columns:
            print(f"⚠️  Warning: 'contract_number' column not found. Returning non-aggregated data.")
            return df

        initial_count = len(df)

        # Define aggregation rules for each column
        agg_rules = {}

        for col in df.columns:
            if col == 'contract_number':
                # Group by column, no aggregation needed
                continue
            elif col == 'insured_name':
                # Join names with semicolons
                agg_rules[col] = lambda x: '; '.join(x.dropna().astype(str).unique())
            elif col in ['policy_premium', 'bonus_amount', 'on_commission']:
                # Sum numeric columns
                agg_rules[col] = lambda x: x.sum() if x.notna().any() else None
            elif col in ['commission_rate', 'bonus_rate', 'on_commission_rate']:
                # Average (mean) for rate columns
                agg_rules[col] = lambda x: x.mean() if x.notna().any() else None
            elif col == 'report_date':
                # Take oldest date (minimum) - ensure it stays as string
                agg_rules[col] = lambda x: str(x.min()) if x.notna().any() and len(x.dropna()) > 0 else None
            else:
                # For other columns, take first non-null value
                agg_rules[col] = lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else None

        # Perform aggregation
        aggregated_df = df.groupby('contract_number', as_index=False).agg(agg_rules)

        # Ensure report_date is a string type after aggregation
        if 'report_date' in aggregated_df.columns:
            # Count non-null dates before conversion
            non_null_dates = aggregated_df['report_date'].notna().sum()
            # Convert to string and replace 'None' with actual None
            aggregated_df['report_date'] = aggregated_df['report_date'].apply(
                lambda x: str(x) if pd.notna(x) and str(x) not in ['None', 'nan', 'NaN', 'NaT'] else None
            )
            print(f"   ✓ Report dates: {non_null_dates} non-null dates found")

        final_count = len(aggregated_df)
        aggregated_rows = initial_count - final_count

        print(f"📊 Contract Number Aggregation:")
        print(f"   Initial rows:     {initial_count}")
        print(f"   Aggregated rows:  {final_count}")
        print(f"   Rows merged:      {aggregated_rows}")

        return aggregated_df

    def filter_final_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to keep only the final required columns.

        Args:
            df: Standardized DataFrame

        Returns:
            DataFrame with only FINAL_COLUMNS
        """
        if df.empty:
            return pd.DataFrame(columns=self.FINAL_COLUMNS)

        # Keep only columns that exist in both FINAL_COLUMNS and df
        available_columns = [col for col in self.FINAL_COLUMNS if col in df.columns]

        # Create filtered dataframe
        filtered_df = df[available_columns].copy()

        # Add missing columns with None
        for col in self.FINAL_COLUMNS:
            if col not in filtered_df.columns:
                filtered_df[col] = None

        # Reorder columns to match FINAL_COLUMNS
        return filtered_df[self.FINAL_COLUMNS]

    def _ensure_standard_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame has all standard columns in correct order.

        Args:
            df: DataFrame to standardize

        Returns:
            DataFrame with all standard columns (+ metadata columns if present)
        """
        # Add missing columns with None
        for col in self.STANDARD_COLUMNS:
            if col not in df.columns:
                df[col] = None

        # Preserve metadata columns (group_id, group_title, etc.)
        metadata_cols = ['group_id', 'group_title', 'item_id', 'board_id', 'board_name']
        existing_metadata = [col for col in metadata_cols if col in df.columns]

        # Reorder: standard columns first, then metadata
        column_order = self.STANDARD_COLUMNS + existing_metadata
        return df[column_order]

    def _create_empty_standard_df(self) -> pd.DataFrame:
        """Create empty DataFrame with standard schema."""
        return pd.DataFrame(columns=self.STANDARD_COLUMNS)

    def process_source(self, source: str, pdf_path: str = None, monday_df: pd.DataFrame = None,
                      aggregate_by_contract: bool = True) -> pd.DataFrame:
        """
        Process a complete source from PDF or Monday.com to final standardized DataFrame.

        This method orchestrates the entire pipeline:
        1. Extract data from PDF or Monday.com DataFrame
        2. Convert to standard format
        3. Filter by sharing_rate (0.4) - SKIPPED for MONDAY_LEGACY and IDC_STATEMENT
        4. Aggregate by contract_number - SKIPPED for MONDAY_LEGACY and IDC_STATEMENT (or if aggregate_by_contract=False)
        5. Filter to final columns

        Args:
            source: Source name - 'UV', 'IDC', 'IDC_STATEMENT', 'ASSOMPTION', or 'MONDAY_LEGACY'
            pdf_path: Path to the PDF file (required for UV, IDC, IDC_STATEMENT, ASSOMPTION)
            monday_df: DataFrame from Monday.com extraction (required for MONDAY_LEGACY)
            aggregate_by_contract: Whether to aggregate rows by contract_number (default: True)

        Returns:
            Final processed DataFrame ready for Monday.com upload
        """
        print(f"\n🔄 Processing {source}" + (f" from: {pdf_path}" if pdf_path else " from Monday.com board"))

        # Step 1: Extract data from PDF or Monday.com
        if source == 'MONDAY_LEGACY':
            # Use provided DataFrame from Monday.com
            if monday_df is None or monday_df.empty:
                print(f"❌ No Monday.com data provided for MONDAY_LEGACY source")
                return pd.DataFrame(columns=self.FINAL_COLUMNS)

            raw_df = monday_df
            metadata = None
            print(f"✅ Using {len(raw_df)} records from Monday.com board")

        elif source == 'UV':
            try:
                from uv_extractor import RemunerationReportExtractor
                extractor = RemunerationReportExtractor(pdf_path)
                data = extractor.extract_all()

                if data['activites'] is None or data['activites'].empty:
                    print(f"❌ No data extracted from {source} PDF")
                    return pd.DataFrame(columns=self.FINAL_COLUMNS)

                raw_df = data['activites']
                metadata = {
                    'date': data.get('date'),
                    'nom_conseiller': data.get('nom_conseiller'),
                    'numero_conseiller': data.get('numero_conseiller')
                }
                print(f"✅ Extracted {len(raw_df)} records from {source}")

            except ImportError:
                print(f"❌ UV extractor not available")
                return pd.DataFrame(columns=self.FINAL_COLUMNS)

        elif source == 'IDC':
            try:
                from idc_extractor import PDFPropositionParser
                parser = PDFPropositionParser(pdf_path)
                raw_df = parser.parse()
                metadata = None

                if raw_df.empty:
                    print(f"❌ No data extracted from {source} PDF")
                    return pd.DataFrame(columns=self.FINAL_COLUMNS)

                print(f"✅ Extracted {len(raw_df)} records from {source}")

            except ImportError:
                print(f"❌ IDC extractor not available")
                return pd.DataFrame(columns=self.FINAL_COLUMNS)

        elif source == 'ASSOMPTION':
            try:
                from assomption_extractor import extract_pdf_data
                raw_df = extract_pdf_data(pdf_path)
                metadata = None

                if raw_df.empty:
                    print(f"❌ No data extracted from {source} PDF")
                    return pd.DataFrame(columns=self.FINAL_COLUMNS)

                print(f"✅ Extracted {len(raw_df)} records from {source}")

            except ImportError:
                print(f"❌ Assomption extractor not available")
                return pd.DataFrame(columns=self.FINAL_COLUMNS)

        elif source == 'IDC_STATEMENT':
            try:
                from idc_statements_extractor import PDFStatementParser
                parser = PDFStatementParser(pdf_path)
                raw_df = parser.parse_trailing_fees()
                metadata = None

                if raw_df.empty:
                    print(f"❌ No data extracted from {source} PDF")
                    return pd.DataFrame(columns=self.FINAL_COLUMNS)

                print(f"✅ Extracted {len(raw_df)} records from {source}")

            except ImportError:
                print(f"❌ IDC Statement extractor not available")
                return pd.DataFrame(columns=self.FINAL_COLUMNS)

        else:
            raise ValueError(f"Unknown source: {source}. Must be 'UV', 'IDC', 'IDC_STATEMENT', 'ASSOMPTION', or 'MONDAY_LEGACY'")

        # Step 2: Convert to standard format
        if source == 'MONDAY_LEGACY':
            standardized = self.convert_monday_legacy_to_standard(raw_df)
        elif source == 'UV':
            standardized = self.convert_uv_to_standard(raw_df, metadata)
        elif source == 'IDC':
            standardized = self.convert_idc_to_standard(raw_df)
        elif source == 'IDC_STATEMENT':
            standardized = self.convert_idc_statement_to_standard(raw_df)
        elif source == 'ASSOMPTION':
            standardized = self.convert_assomption_to_standard(raw_df)

        print(f"✅ Standardized to {len(standardized)} records")

        # Step 3: Filter by sharing_rate (0.4) - SKIP for MONDAY_LEGACY and IDC_STATEMENT
        if source in ['MONDAY_LEGACY', 'IDC_STATEMENT']:
            # Skip filtering for legacy Monday.com data and IDC Statement (trailing fees)
            filtered = standardized
            print(f"⏭️  Skipping sharing_rate filter for {source} source")
        else:
            filtered = self.filter_by_sharing_rate(standardized, target_rate=0.4)

        # Step 4: Aggregate by contract_number - SKIP for MONDAY_LEGACY, IDC_STATEMENT, or if aggregate_by_contract=False
        if source in ['MONDAY_LEGACY', 'IDC_STATEMENT'] or not aggregate_by_contract:
            # Skip aggregation if:
            # - Source is MONDAY_LEGACY (preserve group structure)
            # - Source is IDC_STATEMENT (each line is a unique trailing fee payment)
            # - User explicitly disabled aggregation (aggregate_by_contract=False)
            aggregated = filtered
            if source in ['MONDAY_LEGACY', 'IDC_STATEMENT']:
                print(f"⏭️  Skipping aggregation for {source} source")
            else:
                print(f"⏭️  Skipping aggregation (aggregate_by_contract=False)")
        else:
            aggregated = self.aggregate_by_contract_number(filtered)

        # Step 5: Filter to final columns
        # For MONDAY_LEGACY, preserve group metadata
        if source == 'MONDAY_LEGACY':
            # Keep final columns but also preserve group metadata
            final = self.filter_final_columns(aggregated)

            # DEBUG: Check group columns before re-adding
            print(f"   DEBUG - Columns in aggregated: {list(aggregated.columns)}")
            print(f"   DEBUG - 'group_id' in aggregated: {'group_id' in aggregated.columns}")
            print(f"   DEBUG - 'group_title' in aggregated: {'group_title' in aggregated.columns}")

            # Re-add group metadata columns if they exist
            if 'group_id' in aggregated.columns:
                final['group_id'] = aggregated['group_id']
                print(f"   ✓ group_id preserved ({aggregated['group_id'].notna().sum()} non-null values)")
            if 'group_title' in aggregated.columns:
                final['group_title'] = aggregated['group_title']
                print(f"   ✓ group_title preserved ({aggregated['group_title'].notna().sum()} non-null values)")
                print(f"   ✓ Unique groups: {aggregated['group_title'].dropna().unique().tolist()}")

            print(f"✅ Final output: {len(final)} records ready (group metadata preserved)")
            print(f"   Final columns: {list(final.columns)}")
        else:
            final = self.filter_final_columns(aggregated)
            print(f"✅ Final output: {len(final)} records ready")

        return final

    def save_standardized_data(self, df: pd.DataFrame, source_name: str,
                               format: str = 'csv') -> Path:
        """
        Save standardized DataFrame to file.

        Args:
            df: Standardized DataFrame
            source_name: Name of the source (UV, IDC, ASSOMPTION)
            format: Output format ('csv' or 'excel')

        Returns:
            Path to saved file
        """
        filename = f"{source_name.lower()}_standardized_{self.timestamp}"

        if format == 'excel':
            filepath = self.output_dir / f"{filename}.xlsx"
            df.to_excel(filepath, index=False, engine='openpyxl')
        else:
            filepath = self.output_dir / f"{filename}.csv"
            df.to_csv(filepath, index=False, encoding='utf-8-sig')

        print(f"✅ Saved {source_name} data to: {filepath}")
        return filepath

    def generate_summary_report(self, dataframes: Dict[str, pd.DataFrame]) -> str:
        """
        Generate summary report of processed data.

        Args:
            dataframes: Dictionary of source_name -> DataFrame

        Returns:
            Summary report as string
        """
        report = []
        report.append("=" * 80)
        report.append("COMMISSION DATA UNIFICATION SUMMARY")
        report.append("=" * 80)
        report.append(f"\nProcessing timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Output directory: {self.output_dir}\n")

        total_records = 0
        total_commission = 0

        for source, df in dataframes.items():
            if df.empty:
                report.append(f"\n{source}:")
                report.append(f"  ⚠️  No data processed")
                continue

            records = len(df)
            total_records += records

            commission_sum = df['total_commission'].sum()
            if not pd.isna(commission_sum):
                total_commission += commission_sum

            report.append(f"\n{source}:")
            report.append(f"  📊 Records: {records:,}")
            report.append(f"  💰 Total Commission: ${commission_sum:,.2f}" if not pd.isna(commission_sum) else "  💰 Total Commission: N/A")

            # Unique contracts
            unique_contracts = df['contract_number'].nunique()
            report.append(f"  📝 Unique Contracts: {unique_contracts:,}")

            # Date range (if available)
            date_cols = ['effective_date', 'issue_date', 'report_date']
            for date_col in date_cols:
                if date_col in df.columns:
                    dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
                    if len(dates) > 0:
                        report.append(f"  📅 Date Range ({date_col}): {dates.min().date()} to {dates.max().date()}")
                        break

        report.append("\n" + "-" * 80)
        report.append(f"TOTAL ACROSS ALL SOURCES:")
        report.append(f"  📊 Total Records: {total_records:,}")
        report.append(f"  💰 Total Commission: ${total_commission:,.2f}")
        report.append("=" * 80)

        return "\n".join(report)

    def validate_data_quality(self, df: pd.DataFrame, source_name: str) -> Dict:
        """
        Validate data quality and return issues.

        Args:
            df: DataFrame to validate
            source_name: Name of the source

        Returns:
            Dictionary with validation results
        """
        issues = {
            'source': source_name,
            'warnings': [],
            'errors': []
        }

        if df.empty:
            issues['errors'].append("DataFrame is empty")
            return issues

        # Check for missing critical fields
        critical_fields = ['contract_number', 'insured_name', 'total_commission']
        for field in critical_fields:
            if field in df.columns:
                missing_count = df[field].isna().sum()
                if missing_count > 0:
                    pct = (missing_count / len(df)) * 100
                    issues['warnings'].append(f"{field}: {missing_count} missing values ({pct:.1f}%)")

        # Check for negative commissions (unusual but not necessarily wrong)
        if 'total_commission' in df.columns:
            negative_count = (df['total_commission'] < 0).sum()
            if negative_count > 0:
                issues['warnings'].append(f"Found {negative_count} negative commission values")

        # Check for duplicate contracts
        if 'contract_number' in df.columns:
            duplicates = df['contract_number'].duplicated().sum()
            if duplicates > 0:
                issues['warnings'].append(f"Found {duplicates} duplicate contract numbers (may be valid for multiple protections)")

        return issues


def main():
    """
    Main execution function - processes all PDF sources and creates unified datasets.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    print("=" * 100)
    print("🚀 INSURANCE COMMISSION DATA UNIFICATION SYSTEM")
    print("=" * 100)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {OUTPUT_DIRECTORY}")
    print("=" * 100)
    print()

    # Create output directory if it doesn't exist
    output_path = Path(OUTPUT_DIRECTORY)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory ready: {OUTPUT_DIRECTORY}\n")

    # Verify PDF files exist
    print("🔍 Verifying PDF files...")
    all_files_exist = True
    for source, pdf_path in PDF_PATHS.items():
        path_obj = Path(pdf_path)
        if path_obj.exists():
            size_mb = path_obj.stat().st_size / (1024 * 1024)
            print(f"   ✅ {source:12s}: {pdf_path} ({size_mb:.2f} MB)")
        else:
            print(f"   ❌ {source:12s}: NOT FOUND - {pdf_path}")
            all_files_exist = False

    if not all_files_exist:
        print("\n⚠️  Warning: Some PDF files were not found")
        print("   Processing will continue with available files\n")
    else:
        print("   ✅ All PDF files found\n")

    # Check extractor availability
    print("🔍 Checking extractor modules...")
    extractors_status = {
        'UV': UV_AVAILABLE,
        'IDC': IDC_AVAILABLE,
        'ASSOMPTION': ASSOMPTION_AVAILABLE
    }

    for source, available in extractors_status.items():
        status = "✅ Available" if available else "❌ Not Found"
        print(f"   {source:12s}: {status}")

    if not any(extractors_status.values()):
        print("\n❌ ERROR: No extractors found!")
        print("   Please ensure these files are in the same directory:")
        print("   - uv_extractor.py")
        print("   - idc_extractor.py")
        print("   - assomption_extractor.py")
        return 1

    print()

    # Initialize unifier
    unifier = CommissionDataUnifier(output_dir=OUTPUT_DIRECTORY)
    processed_data = {}

    # =========================================================================
    # PROCESS UV ASSURANCE DATA
    # =========================================================================
    print("━" * 100)
    print("📄 PROCESSING UV ASSURANCE")
    print("━" * 100)

    if not UV_AVAILABLE:
        print("❌ UV extractor not available - skipping")
        processed_data['UV'] = pd.DataFrame()
    elif not Path(PDF_PATHS['UV']).exists():
        print(f"❌ PDF file not found: {PDF_PATHS['UV']}")
        processed_data['UV'] = pd.DataFrame()
    else:
        try:
            print(f"📂 Reading: {PDF_PATHS['UV']}")

            # Extract data using UV extractor
            uv_extractor = RemunerationReportExtractor(PDF_PATHS['UV'])
            uv_data = uv_extractor.extract_all()

            # Check if extraction was successful
            if uv_data['activites'] is not None and not uv_data['activites'].empty:
                # Prepare metadata dictionary
                uv_metadata = {
                    'date': uv_data['date'],
                    'nom_conseiller': uv_data['nom_conseiller'],
                    'numero_conseiller': uv_data['numero_conseiller']
                }

                print(f"✅ Extracted {len(uv_data['activites'])} records")
                print(f"   Advisor: {uv_metadata['nom_conseiller']}")
                print(f"   Report Date: {uv_metadata['date']}")

                # Convert to standardized format
                uv_standardized = unifier.convert_uv_to_standard(
                    uv_data['activites'],
                    uv_metadata
                )

                # Filter by sharing_rate (keep only 0.4)
                uv_filtered = unifier.filter_by_sharing_rate(uv_standardized, target_rate=0.4)

                # Aggregate by contract_number
                uv_aggregated = unifier.aggregate_by_contract_number(uv_filtered)

                # Filter to keep only final columns
                uv_final = unifier.filter_final_columns(uv_aggregated)

                # ═════════════════════════════════════════════════════════
                # DISPLAY FULL CLEANED DATAFRAME
                # ═════════════════════════════════════════════════════════
                print("\n" + "═" * 100)
                print("📊 FULL CLEANED DATAFRAME - UV ASSURANCE")
                print("═" * 100)

                # Configure pandas to display full dataframe
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 1000)
                pd.set_option('display.max_rows', None)

                print(uv_final)
                print("\n" + "═" * 100)


                # Validate data quality
                print("🔍 Validating data quality...")
                validation = unifier.validate_data_quality(uv_final, 'UV')

                if validation['warnings']:
                    print("⚠️  Warnings:")
                    for warning in validation['warnings'][:5]:
                        print(f"   - {warning}")
                    if len(validation['warnings']) > 5:
                        print(f"   ... and {len(validation['warnings']) - 5} more warnings")

                # Save standardized files
                unifier.save_standardized_data(uv_final, 'UV', format='csv')
                unifier.save_standardized_data(uv_final, 'UV', format='excel')

                processed_data['UV'] = uv_final
                print(f"✅ UV processing complete: {len(uv_final)} standardized records")

            else:
                print("⚠️  No data extracted from UV PDF")
                processed_data['UV'] = pd.DataFrame()

        except Exception as e:
            print(f"❌ Error processing UV data: {str(e)}")
            import traceback
            traceback.print_exc()
            processed_data['UV'] = pd.DataFrame()

    print()

    # =========================================================================
    # PROCESS IDC DATA
    # =========================================================================
    print("━" * 100)
    print("📄 PROCESSING IDC")
    print("━" * 100)

    if not IDC_AVAILABLE:
        print("❌ IDC extractor not available - skipping")
        processed_data['IDC'] = pd.DataFrame()
    elif not Path(PDF_PATHS['IDC']).exists():
        print(f"❌ PDF file not found: {PDF_PATHS['IDC']}")
        processed_data['IDC'] = pd.DataFrame()
    else:
        try:
            print(f"📂 Reading: {PDF_PATHS['IDC']}")

            # Extract data using IDC extractor
            idc_parser = PDFPropositionParser(PDF_PATHS['IDC'])
            idc_df = idc_parser.parse()

            # Check if extraction was successful
            if not idc_df.empty:
                print(f"✅ Extracted {len(idc_df)} records")

                # Convert to standardized format
                idc_standardized = unifier.convert_idc_to_standard(idc_df)

                # Filter by sharing_rate (keep only 0.4)
                idc_filtered = unifier.filter_by_sharing_rate(idc_standardized, target_rate=0.4)

                # Aggregate by contract_number
                idc_aggregated = unifier.aggregate_by_contract_number(idc_filtered)

                # Filter to keep only final columns
                idc_final = unifier.filter_final_columns(idc_aggregated)

                # ═════════════════════════════════════════════════════════
                # DISPLAY FULL CLEANED DATAFRAME
                # ═════════════════════════════════════════════════════════
                print("\n" + "═" * 100)
                print("📊 FULL CLEANED DATAFRAME - IDC")
                print("═" * 100)

                # Configure pandas to display full dataframe
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 1000)
                pd.set_option('display.max_rows', None)

                print(idc_final)
                print("\n" + "═" * 100)


                # Validate data quality
                print("🔍 Validating data quality...")
                validation = unifier.validate_data_quality(idc_final, 'IDC')

                if validation['warnings']:
                    print("⚠️  Warnings:")
                    for warning in validation['warnings'][:5]:
                        print(f"   - {warning}")
                    if len(validation['warnings']) > 5:
                        print(f"   ... and {len(validation['warnings']) - 5} more warnings")

                # Save standardized files
                unifier.save_standardized_data(idc_final, 'IDC', format='csv')
                unifier.save_standardized_data(idc_final, 'IDC', format='excel')

                processed_data['IDC'] = idc_final
                print(f"✅ IDC processing complete: {len(idc_final)} standardized records")

            else:
                print("⚠️  No data extracted from IDC PDF")
                processed_data['IDC'] = pd.DataFrame()

        except Exception as e:
            print(f"❌ Error processing IDC data: {str(e)}")
            import traceback
            traceback.print_exc()
            processed_data['IDC'] = pd.DataFrame()

    print()

    # =========================================================================
    # PROCESS ASSOMPTION VIE DATA
    # =========================================================================
    print("━" * 100)
    print("📄 PROCESSING ASSOMPTION VIE")
    print("━" * 100)

    if not ASSOMPTION_AVAILABLE:
        print("❌ Assomption extractor not available - skipping")
        processed_data['ASSOMPTION'] = pd.DataFrame()
    elif not Path(PDF_PATHS['ASSOMPTION']).exists():
        print(f"❌ PDF file not found: {PDF_PATHS['ASSOMPTION']}")
        processed_data['ASSOMPTION'] = pd.DataFrame()
    else:
        try:
            print(f"📂 Reading: {PDF_PATHS['ASSOMPTION']}")

            # Extract data using Assomption extractor
            assomption_df = extract_pdf_data(PDF_PATHS['ASSOMPTION'])

            # Check if extraction was successful
            if not assomption_df.empty:
                print(f"✅ Extracted {len(assomption_df)} records")

                # Convert to standardized format
                assomption_standardized = unifier.convert_assomption_to_standard(assomption_df)

                # Filter by sharing_rate (keep only 0.4)
                assomption_filtered = unifier.filter_by_sharing_rate(assomption_standardized, target_rate=0.4)

                # Aggregate by contract_number
                assomption_aggregated = unifier.aggregate_by_contract_number(assomption_filtered)

                # Filter to keep only final columns
                assomption_final = unifier.filter_final_columns(assomption_aggregated)

                # ═════════════════════════════════════════════════════════
                # DISPLAY FULL CLEANED DATAFRAME
                # ═════════════════════════════════════════════════════════
                print("\n" + "═" * 100)
                print("📊 FULL CLEANED DATAFRAME - ASSOMPTION VIE")
                print("═" * 100)

                # Configure pandas to display full dataframe
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 1000)
                pd.set_option('display.max_rows', None)

                print(assomption_final)
                print("\n" + "═" * 100)


                # Validate data quality
                print("🔍 Validating data quality...")
                validation = unifier.validate_data_quality(assomption_final, 'ASSOMPTION')

                if validation['warnings']:
                    print("⚠️  Warnings:")
                    for warning in validation['warnings'][:5]:
                        print(f"   - {warning}")
                    if len(validation['warnings']) > 5:
                        print(f"   ... and {len(validation['warnings']) - 5} more warnings")

                # Save standardized files
                unifier.save_standardized_data(assomption_final, 'ASSOMPTION', format='csv')
                unifier.save_standardized_data(assomption_final, 'ASSOMPTION', format='excel')

                processed_data['ASSOMPTION'] = assomption_final
                print(f"✅ Assomption processing complete: {len(assomption_final)} standardized records")

            else:
                print("⚠️  No data extracted from Assomption PDF")
                processed_data['ASSOMPTION'] = pd.DataFrame()

        except Exception as e:
            print(f"❌ Error processing Assomption data: {str(e)}")
            import traceback
            traceback.print_exc()
            processed_data['ASSOMPTION'] = pd.DataFrame()

    print()

    # =========================================================================
    # COMPLETION
    # =========================================================================
    print("\n" + "=" * 100)
    print("✅ PROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 100)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Check output directory for results: {OUTPUT_DIRECTORY}")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    sys.exit(main())