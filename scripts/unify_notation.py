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
from enum import Enum
import sys
import warnings

warnings.filterwarnings('ignore')

# Project root directory (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent

# Import source-specific extractors
try:
    from uv_extractor import RemunerationReportExtractor
    UV_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: uv_extractor.py not found")
    UV_AVAILABLE = False

try:
    from advisor_matcher import AdvisorMatcher
    ADVISOR_MATCHER_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: advisor_matcher.py not found - using legacy patterns")
    ADVISOR_MATCHER_AVAILABLE = False

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
# BOARD TYPE ENUM
# =============================================================================


class BoardType(Enum):
    """Enum for Monday.com board types."""
    HISTORICAL_PAYMENTS = "HISTORICAL_PAYMENTS"  # Paiements historiques
    SALES_PRODUCTION = "SALES_PRODUCTION"        # Ventes et production


# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR DATA
# =============================================================================

PDF_PATHS = {
    'UV': str(PROJECT_ROOT / "pdf/uv/rappportremun_21622_2025-10-20.pdf"),
    'IDC': str(PROJECT_ROOT / "pdf/idc/Rapport des propositions soumises.20251124_1638.pdf"),
    'ASSOMPTION': str(PROJECT_ROOT / "pdf/assomption/Remuneration (61).pdf")
}

OUTPUT_DIRECTORY = str(PROJECT_ROOT / "results") + "/"


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

        # Additional fields for Monday.com integration - Historical Payments
        'status',
        'advisor_name',
        'is_verified',
        'comments',
        'amount_received',

        # Additional fields for Monday.com integration - Sales Production
        'completion',
        'sharing_type',  # Type de partage (Lead, etc.)
        'commission_receipt',
        'bonus_amount_receipt',
        'on_commission_receipt',
        'total'
    ]

    # Final columns for Historical Payments (Paiements historiques)
    # French column names matching Monday.com board
    # Note: 'Nom Client' is preserved separately for use as item_name (Élément column in Monday.com)
    FINAL_COLUMNS_HISTORICAL_PAYMENTS = [
        '# de Police',
        'Nom Client',  # Added: Client name for Monday.com Élément column
        'Compagnie',
        'Statut',
        'Conseiller',
        'Verifié',
        'PA',
        'Com',
        'Boni',
        'Sur-Com',
        'Reçu',
        'Date',
        'Texte'
    ]

    # Final columns for Sales Production (Ventes et production)
    # French column names matching Monday.com board
    # Note: 'Nom Client' is preserved separately for use as item_name (Élément column in Monday.com)
    FINAL_COLUMNS_SALES_PRODUCTION = [
        'Date',
        '# de Police',
        'Nom Client',  # Added: Client name for Monday.com Élément column
        'Compagnie',
        'Statut',
        'Conseiller',
        'Complet',
        'PA',
        'Lead/MC',  # Type de partage - toujours 'Lead' (anciennement 'Partage')
        'Com',
        'Reçu 1',
        'Boni',
        'Reçu 2',
        'Sur-Com',
        'Reçu 3',
        'Total',
        'Total Reçu',
        'Paie',
        'Texte'
    ]

    # Keep FINAL_COLUMNS as alias for backward compatibility (defaults to Historical Payments)
    FINAL_COLUMNS = FINAL_COLUMNS_HISTORICAL_PAYMENTS

    # Column mapping from internal English names to French output names
    COLUMN_MAPPING_TO_FRENCH = {
        # Common columns
        'contract_number': '# de Police',
        'insured_name': 'Nom Client',  # Client name for Monday.com Élément column
        'insurer_name': 'Compagnie',
        'status': 'Statut',
        'advisor_name': 'Conseiller',
        'policy_premium': 'PA',
        'commission': 'Com',
        'bonus_amount': 'Boni',
        'on_commission': 'Sur-Com',
        'report_date': 'Date',
        'comments': 'Texte',
        # Historical Payments specific
        'is_verified': 'Verifié',
        'amount_received': 'Reçu',
        # Sales Production specific
        'completion': 'Complet',
        'sharing_type': 'Lead/MC',  # Type de partage - toujours 'Lead'
        'commission_receipt': 'Reçu 1',
        'bonus_amount_receipt': 'Reçu 2',
        'on_commission_receipt': 'Reçu 3',
        'total': 'Total',
        'total_received': 'Total Reçu',
        'payment': 'Paie',
    }

    # Reverse mapping from French to English (for reading Monday.com data)
    COLUMN_MAPPING_TO_ENGLISH = {v: k for k, v in COLUMN_MAPPING_TO_FRENCH.items()}

    # Advisor name mapping patterns
    # Each entry: (compiled_regex_pattern, simplified_name)
    # Patterns are tested in order, first match wins
    ADVISOR_PATTERNS = None  # Will be initialized in __init__

    def __init__(self, output_dir: str = None):
        """
        Initialize the unifier.

        Args:
            output_dir: Directory to save unified data files (defaults to PROJECT_ROOT/unified_data)
        """
        if output_dir is None:
            self.output_dir = PROJECT_ROOT / "unified_data"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Initialize advisor matcher (using new robust system or legacy patterns)
        self._init_advisor_matcher()

    def _init_advisor_matcher(self):
        """Initialize advisor name matching system."""
        if ADVISOR_MATCHER_AVAILABLE:
            # Use the new robust AdvisorMatcher
            self.advisor_matcher = AdvisorMatcher()
            self.ADVISOR_PATTERNS = None  # Not used with new system
            print(f"   ✓ Using AdvisorMatcher with {len(self.advisor_matcher.advisors)} advisors")
        else:
            # Fallback to legacy regex patterns
            self.advisor_matcher = None
            self._init_legacy_advisor_patterns()

    def _init_legacy_advisor_patterns(self):
        """Initialize legacy compiled regex patterns for advisor name mapping."""
        import re

        # Define patterns for each advisor (legacy fallback)
        # Format: (pattern, simplified_name)
        # Patterns use re.IGNORECASE for case-insensitive matching
        pattern_definitions = [
            # Brandeen David Legrand - most complex, many variations
            (r'\bbrandeen\b', 'Brandeen D.'),
            (r'\bdavid\s+legrand\b', 'Brandeen D.'),
            (r'\blegrand\s*,?\s*david\b', 'Brandeen D.'),
            (r'\bd\.?\s*legrand\b', 'Brandeen D.'),
            (r'\blegrand\s*,?\s*d\.?\b', 'Brandeen D.'),
            (r'\bdl\s+brandeen\b', 'Brandeen D.'),
            (r'\bbrandeen\s+dl\b', 'Brandeen D.'),

            # Mohammed Fayçal/Faycal Guennouni
            (r'\bfay[cç]al\b', 'Faycal G.'),
            (r'\bfayÃ§al\b', 'Faycal G.'),
            (r'\bguennouni\b', 'Faycal G.'),
            (r'\bmohammed\s+f', 'Faycal G.'),

            # Mathieu Poirier (check before Derek Poirier)
            (r'\bmathieu\s+poirier\b', 'Mathieu P.'),
            (r'\bpoirier\s*,?\s*mathieu\b', 'Mathieu P.'),
            (r'\bm\.?\s+poirier\b', 'Mathieu P.'),

            # Derek Poirier
            (r'\bderek\s+poirier\b', 'Derek P.'),
            (r'\bpoirier\s*,?\s*derek\b', 'Derek P.'),
            (r'\bd\.?\s+poirier\b(?!\s*m)', 'Derek P.'),

            # Thomas Lussier
            (r'\bthomas\s+lussier\b', 'Thomas L.'),
            (r'\blussier\s*,?\s*thomas\b', 'Thomas L.'),
            (r'\bt\.?\s+lussier\b', 'Thomas L.'),
            (r'\blussier\s*,?\s*t\.?\b', 'Thomas L.'),

            # Ayoub Chamoumi
            (r'\bayoub\s+chamoumi\b', 'Ayoub C.'),
            (r'\bchamoumi\b', 'Ayoub C.'),

            # Said Vital -> Ayoub (special case)
            (r'\bsaid\s+vital\b', 'Ayoub C.'),
            (r'\bvital\s*,?\s*said\b', 'Ayoub C.'),
            (r'\bs\.?\s+vital\b', 'Ayoub C.'),

            # Alexis Bourassa
            (r'\balexis\s+bourassa\b', 'Alexis B.'),
            (r'\bbourassa\s*,?\s*alexis\b', 'Alexis B.'),
            (r'\bbourassa\b', 'Alexis B.'),

            # Jad Senhaji
            (r'\bjad\s+senhaji\b', 'Jad S.'),
            (r'\bsenhaji\b', 'Jad S.'),

            # Igor Velicico
            (r'\bigor\s+velicico\b', 'Igor V.'),
            (r'\bvelicico\b', 'Igor V.'),

            # Robinson Viaud
            (r'\brobinson\s+viaud\b', 'Robinson V.'),
            (r'\bviaud\b', 'Robinson V.'),

            # Benoît/Benoit Méthot/Methot
            (r'\bbeno[iî]t\s+m[eé]thot\b', 'Benoit M.'),
            (r'\bm[eé]thot\s*,?\s*beno[iî]t\b', 'Benoit M.'),
            (r'\bm[eé]thot\b', 'Benoit M.'),

            # Anthony Guay
            (r'\banthony\s+guay\b', 'Anthony G.'),
            (r'\bguay\s*,?\s*anthony\b', 'Anthony G.'),
            (r'\bguay\b', 'Anthony G.'),

            # Guillaume St-Pierre
            (r'\bguillaume\s+st[\-\.]?\s*pierre\b', 'Guillaume S.'),
            (r'\bst[\-\.]?\s*pierre\s*,?\s*guillaume\b', 'Guillaume S.'),
            (r'\bst[\-\.]?\s*pierre\b', 'Guillaume S.'),
        ]

        # Compile patterns
        self.ADVISOR_PATTERNS = [
            (re.compile(pattern, re.IGNORECASE), name)
            for pattern, name in pattern_definitions
        ]

    def _normalize_advisor_name(self, advisor_name) -> str:
        """
        Normalize advisor name to simplified version.

        Uses the AdvisorMatcher for robust matching, or falls back to
        regex patterns if AdvisorMatcher is not available.

        Args:
            advisor_name: The original advisor name

        Returns:
            Simplified advisor name (Prénom + First letter of last name),
            or original if no match found
        """
        if pd.isna(advisor_name) or advisor_name is None:
            return None

        name_str = str(advisor_name).strip()
        if not name_str or name_str in ['None', 'nan', 'NaN']:
            return None

        # Use AdvisorMatcher if available
        if self.advisor_matcher is not None:
            result = self.advisor_matcher.match(name_str)
            if result:
                return result
            # No match found, return original name
            return name_str

        # Fallback to legacy patterns
        for pattern, simplified_name in self.ADVISOR_PATTERNS:
            if pattern.search(name_str):
                return simplified_name

        # No match found, return original name
        return name_str

    def _normalize_advisor_column(self, df: pd.DataFrame, column_name: str = 'advisor_name') -> pd.DataFrame:
        """
        Normalize all advisor names in a DataFrame column.

        Args:
            df: DataFrame containing advisor names
            column_name: Name of the column containing advisor names

        Returns:
            DataFrame with normalized advisor names
        """
        if column_name not in df.columns:
            return df

        original_values = df[column_name].copy()
        df[column_name] = df[column_name].apply(self._normalize_advisor_name)

        # Log normalization results
        changed_mask = (original_values != df[column_name]) & original_values.notna()
        if changed_mask.any():
            changed_count = changed_mask.sum()
            print(f"   ✓ Normalized {changed_count} advisor names")
            # Show unique mappings
            unique_mappings = df.loc[changed_mask, [column_name]].drop_duplicates()
            for _, row in unique_mappings.iterrows():
                orig = original_values[unique_mappings.index[0]]
                print(f"      '{orig}' → '{row[column_name]}'")

        return df

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

    def _is_corporate_advisor(self, advisor_name: str) -> bool:
        """
        Detect if the advisor name represents a corporation (Inc) rather than a person.

        A corporate advisor name typically contains digits (like "9491-1377 QUEBEC INC").
        Personal advisor names are typically just names without digits.

        Args:
            advisor_name: The name of the advisor from the report

        Returns:
            True if the advisor appears to be a corporation, False if personal
        """
        if not advisor_name or pd.isna(advisor_name):
            return False

        # Check if the name contains any digits - corporations typically have numbers
        # like "9491-1377 QUEBEC INC" or "123456 CANADA INC"
        import re
        return bool(re.search(r'\d', str(advisor_name)))

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
        # Map 'Rémunération' to 'amount_received' (Reçu column in final output)
        # This represents the actual amount received from UV
        standard_df['amount_received'] = df['Rémunération'].apply(self._clean_currency)

        # Calculate commissions using helper method
        standard_df = self._calculate_commissions(standard_df)

        # Round float columns using helper method
        standard_df = self._round_float_columns(standard_df)

        # Determine insurer_name based on advisor type (UV Inc vs UV Perso)
        # UV Inc: when advisor name contains digits (corporation name like "9491-1377 QUEBEC INC")
        # UV Perso: when advisor name is a personal name without digits
        if metadata:
            advisor_name = metadata.get('nom_conseiller', '')
            is_corporate = self._is_corporate_advisor(advisor_name)

            if is_corporate:
                insurer_label = 'UV Inc'
                print(f"   ✓ Detected corporate advisor: '{advisor_name}' → UV Inc")
            else:
                insurer_label = 'UV Perso'
                print(f"   ✓ Detected personal advisor: '{advisor_name}' → UV Perso")
        else:
            # Default to UV Perso if no metadata
            insurer_label = 'UV Perso'
            print(f"   ⚠️  No advisor metadata - defaulting to UV Perso")

        standard_df['insurer_name'] = np.repeat(insurer_label, n_rows)

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
        # Transform status: 'Approved' → 'Approuvé', anything else → 'En attente'
        status_transformed = df['Statut'].apply(
            lambda x: 'Approuvé' if str(x).strip() == 'Approved' else 'En attente'
        )
        standard_df['policy_status'] = status_transformed
        # Also map to 'status' for FINAL_COLUMNS compatibility
        standard_df['status'] = status_transformed
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

        # For IDC, bonus_rate is constant at 175% (1.75)
        standard_df['bonus_rate'] = 1.75

        # Calculate commissions using helper method (will calculate bonus_amount)
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
        # Map 'Commissions' to 'amount_received' (Reçu column in final output)
        # This represents the actual amount received from Assomption
        standard_df['amount_received'] = pd.to_numeric(df['Commissions'], errors='coerce')

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
        # Note: 'Nom du client' is NOT mapped to insured_name to maintain uniformity with other tables
        # IDC Statement focuses on contract/account numbers, not client names
        standard_df['contract_number'] = df['Numéro de compte'].astype(str)
        standard_df['insurer_name'] = df['Compagnie'].astype(str)

        # Parse and format dates
        standard_df['report_date'] = df['Date'].apply(lambda x: self._format_date_uniform(self._parse_date(x)))

        # Parse trailing fees - Map 'Frais de suivi nets' to 'on_commission' AND 'amount_received'
        # 'on_commission' → 'Sur-Com' column
        # 'amount_received' → 'Reçu' column (used for status calculation: Payé/Charge back)
        trailing_fee = df['Frais de suivi nets'].apply(self._clean_currency)
        standard_df['on_commission'] = trailing_fee
        standard_df['amount_received'] = trailing_fee  # For status calculation in filter_final_columns

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

    def detect_board_type(self, df: pd.DataFrame) -> BoardType:
        """
        Automatically detect the board type based on column names.

        Detection logic:
        - If 'Reçu 1', 'Reçu 2', 'Reçu 3', or 'Complet' columns exist → SALES_PRODUCTION
        - Otherwise → HISTORICAL_PAYMENTS

        Args:
            df: DataFrame from Monday.com board

        Returns:
            BoardType enum value
        """
        if df.empty:
            return BoardType.HISTORICAL_PAYMENTS  # Default

        sales_production_indicators = ['Reçu 1', 'Reçu 2', 'Reçu 3', 'Complet', 'Total']

        # Check if any of the Sales Production indicators exist
        for indicator in sales_production_indicators:
            if indicator in df.columns:
                print(f"   🔍 Detected SALES_PRODUCTION board (found column: '{indicator}')")
                return BoardType.SALES_PRODUCTION

        print(f"   🔍 Detected HISTORICAL_PAYMENTS board (no Sales Production columns found)")
        return BoardType.HISTORICAL_PAYMENTS

    def convert_monday_legacy_to_standard(self, df: pd.DataFrame, board_type: BoardType = None) -> pd.DataFrame:
        """
        Convert legacy Monday.com board data to standard schema.

        Supports two board types:
        1. HISTORICAL_PAYMENTS (Paiements historiques)
        2. SALES_PRODUCTION (Ventes et production)

        Column mapping for HISTORICAL_PAYMENTS:
        - 'item_name' (colonne "Élément") → insured_name
        - '# de Police' → contract_number
        - 'Compagnie' → insurer_name
        - 'Statut' → status
        - 'Conseiller' → advisor_name
        - 'Verifié' → is_verified
        - 'PA' → policy_premium
        - 'Com' → commission
        - 'Boni' → bonus_amount
        - 'Sur-Com' → on_commission
        - 'Reçu' → amount_received
        - 'Date' → report_date

        Column mapping for SALES_PRODUCTION:
        - 'item_name' (colonne "Élément") → contract_number
        - 'item_name' → insured_name (also copied)
        - '# de Police' → contract_number (if exists)
        - 'Compagnie' → insurer_name
        - 'Statut' → status
        - 'Conseiller' → advisor_name
        - 'Complet' → completion
        - 'PA' → policy_premium
        - 'Partage' → sharing_amount (calculated if missing)
        - 'Com' → commission
        - 'Reçu 1' → commission_receipt
        - 'Boni' → bonus_amount
        - 'Reçu 2' → bonus_amount_receipt
        - 'Sur-Com' → on_commission
        - 'Reçu 3' → on_commission_receipt
        - 'Total' → total (calculated if missing)
        - 'Date' → report_date

        Constants added:
        - sharing_rate = 0.4 (40%)
        - commission_rate = 0.5 (50%)
        - bonus_rate = 1.75 (175%)
        - on_commission_rate = 0.75 (75%)

        Args:
            df: DataFrame from legacy Monday.com board
            board_type: Type of board (auto-detected if None)

        Returns:
            Standardized DataFrame
        """
        if df.empty:
            return self._create_empty_standard_df()

        # Auto-detect board type if not provided
        if board_type is None:
            board_type = self.detect_board_type(df)
            print(f"   🔍 Auto-detected board type: {board_type.value}")

        # Get the number of rows for consistent array creation
        n_rows = len(df)

        # Initialize with standard columns
        standard_df = pd.DataFrame(index=df.index)

        # =====================================================================
        # COMMON MAPPINGS (both board types)
        # =====================================================================

        # Map contract_number based on board type
        if board_type == BoardType.SALES_PRODUCTION:
            # For Sales Production: Élément contains contract_number
            if 'item_name' in df.columns:
                standard_df['contract_number'] = df['item_name'].astype(str)
                print(f"   ✓ Mapped 'item_name' (Élément) → 'contract_number': {df['item_name'].notna().sum()} values")
            # Also use '# de Police' if it exists
            if '# de Police' in df.columns:
                # Override with '# de Police' if it's not empty
                mask = df['# de Police'].notna() & (df['# de Police'] != '')
                if mask.any():
                    standard_df.loc[mask, 'contract_number'] = df.loc[mask, '# de Police'].astype(str)
                    print(f"   ✓ Overrode contract_number with '# de Police' for {mask.sum()} rows")
        else:
            # For Historical Payments: '# de Police' is contract_number
            if '# de Police' in df.columns:
                standard_df['contract_number'] = df['# de Police'].astype(str)

        # Map insured_name based on board type
        if board_type == BoardType.HISTORICAL_PAYMENTS:
            # For Historical Payments: item_name contains insured_name
            if 'item_name' in df.columns:
                standard_df['insured_name'] = df['item_name'].astype(str)
                print(f"   ✓ Mapped 'item_name' (Élément) → 'insured_name': {df['item_name'].notna().sum()} values")
        else:
            # For Sales Production: insured_name comes from original item_name
            # (which now contains contract_number, but we still need the name)
            # In this case, the name should be in item_name initially
            if 'item_name' in df.columns:
                standard_df['insured_name'] = df['item_name'].astype(str)
                print(f"   ✓ Mapped 'item_name' → 'insured_name': {df['item_name'].notna().sum()} values")

        # Common mappings
        if 'Compagnie' in df.columns:
            standard_df['insurer_name'] = df['Compagnie'].astype(str)

        if 'PA' in df.columns:
            standard_df['policy_premium'] = pd.to_numeric(df['PA'], errors='coerce')

        if 'Com' in df.columns:
            standard_df['commission'] = pd.to_numeric(df['Com'], errors='coerce')

        if 'Boni' in df.columns:
            standard_df['bonus_amount'] = pd.to_numeric(df['Boni'], errors='coerce')

        if 'Sur-Com' in df.columns:
            standard_df['on_commission'] = pd.to_numeric(df['Sur-Com'], errors='coerce')

        if 'Date' in df.columns:
            standard_df['report_date'] = df['Date'].apply(
                lambda x: self._format_date_uniform(self._parse_date(x))
            )

        if 'Statut' in df.columns:
            standard_df['status'] = df['Statut'].astype(str)

        if 'Conseiller' in df.columns:
            standard_df['advisor_name'] = df['Conseiller'].astype(str)

        if 'Texte' in df.columns:
            standard_df['comments'] = df['Texte'].astype(str)

        # =====================================================================
        # BOARD-SPECIFIC MAPPINGS
        # =====================================================================

        if board_type == BoardType.HISTORICAL_PAYMENTS:
            # Historical Payments specific columns
            if 'Verifié' in df.columns:
                standard_df['is_verified'] = df['Verifié'].astype(str)

            if 'Reçu' in df.columns:
                standard_df['amount_received'] = pd.to_numeric(df['Reçu'], errors='coerce')

        else:  # SALES_PRODUCTION
            # Sales Production specific columns
            if 'Complet' in df.columns:
                standard_df['completion'] = df['Complet'].astype(str)

            # Note: 'Partage' column is now always 'Lead' (sharing_type), set in filter_final_columns
            # Legacy 'Partage' values are ignored

            if 'Reçu 1' in df.columns:
                standard_df['commission_receipt'] = pd.to_numeric(df['Reçu 1'], errors='coerce')

            if 'Reçu 2' in df.columns:
                standard_df['bonus_amount_receipt'] = pd.to_numeric(df['Reçu 2'], errors='coerce')

            if 'Reçu 3' in df.columns:
                standard_df['on_commission_receipt'] = pd.to_numeric(df['Reçu 3'], errors='coerce')

            if 'Total' in df.columns:
                standard_df['total'] = pd.to_numeric(df['Total'], errors='coerce')

            if 'Total Reçu' in df.columns:
                standard_df['total_received'] = pd.to_numeric(df['Total Reçu'], errors='coerce')

            if 'Paie' in df.columns:
                standard_df['payment'] = pd.to_numeric(df['Paie'], errors='coerce')

        # =====================================================================
        # ADD CONSTANTS
        # =====================================================================
        standard_df['sharing_rate'] = 0.4  # 40%
        standard_df['commission_rate'] = 0.5  # 50%
        standard_df['bonus_rate'] = 1.75  # 175%
        standard_df['on_commission_rate'] = 0.75  # 75%

        # =====================================================================
        # AUTOMATIC CALCULATIONS
        # =====================================================================
        # Calculate missing formula columns (Monday.com formula columns don't return values via API)
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

            # =====================================================================
            # SALES PRODUCTION SPECIFIC CALCULATIONS
            # =====================================================================
            if board_type == BoardType.SALES_PRODUCTION:
                # Note: 'sharing_type' is always 'Lead', set in filter_final_columns

                # Calculate total if missing: commission + bonus_amount + on_commission
                if 'total' in standard_df.columns:
                    total_missing = standard_df['total'].isna().sum()
                    if total_missing > 0:
                        mask = standard_df['total'].isna()
                        standard_df.loc[mask, 'total'] = (
                            standard_df.loc[mask, 'commission'].fillna(0) +
                            standard_df.loc[mask, 'bonus_amount'].fillna(0) +
                            standard_df.loc[mask, 'on_commission'].fillna(0)
                        )
                        calculated = standard_df.loc[mask, 'total'].notna().sum()
                        print(f"      ✓ Calculated {calculated} total values")
                else:
                    # Create total column
                    standard_df['total'] = (
                        standard_df['commission'].fillna(0) +
                        standard_df['bonus_amount'].fillna(0) +
                        standard_df['on_commission'].fillna(0)
                    )
                    print(f"      ✓ Created total column")

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

        print(f"✅ Converted {len(standard_df)} records from legacy Monday.com format ({board_type.value})")
        print(f"   Constants applied: sharing_rate=0.4, commission_rate=0.5, bonus_rate=1.75, on_commission_rate=0.75")

        # Check if group metadata was preserved
        if 'group_title' in standard_df.columns:
            unique_groups = standard_df['group_title'].dropna().unique()
            print(f"   ✓ Group metadata preserved: {len(unique_groups)} groups - {list(unique_groups)}")

        # Store board_type as metadata for later use (as string for JSON serialization)
        standard_df.attrs['board_type'] = board_type.value if hasattr(board_type, 'value') else str(board_type)

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
                # Join names with ampersand (&) for different clients with same policy number
                agg_rules[col] = lambda x: ' & '.join(x.dropna().astype(str).unique())
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

    def filter_final_columns(self, df: pd.DataFrame, board_type: BoardType = None) -> pd.DataFrame:
        """
        Filter DataFrame to keep only the final required columns and rename to French.

        This method:
        1. Renames internal English column names to French output names
        2. Calculates 'Total Reçu' and 'Paie' for Sales Production if needed
        3. Filters to keep only the final columns for the board type
        4. Reorders columns to match the expected order

        Args:
            df: Standardized DataFrame (with English internal column names)
            board_type: Type of board (auto-detected from df.attrs if None)

        Returns:
            DataFrame with French column names (specific to board type)
        """
        # Auto-detect board_type from DataFrame attributes if not provided
        if board_type is None and hasattr(df, 'attrs') and 'board_type' in df.attrs:
            stored_type = df.attrs['board_type']
            # Handle both enum and string values
            if isinstance(stored_type, str):
                board_type = BoardType(stored_type) if stored_type in [bt.value for bt in BoardType] else None
            else:
                board_type = stored_type

        # Select appropriate final columns based on board type
        # Handle both enum comparison and string comparison
        is_sales_production = (
            board_type == BoardType.SALES_PRODUCTION or
            (isinstance(board_type, str) and board_type == BoardType.SALES_PRODUCTION.value)
        )
        if is_sales_production:
            final_columns = self.FINAL_COLUMNS_SALES_PRODUCTION
        else:
            # Default to Historical Payments
            final_columns = self.FINAL_COLUMNS_HISTORICAL_PAYMENTS

        if df.empty:
            return pd.DataFrame(columns=final_columns)

        # Create a copy for modification
        result_df = df.copy()

        # For Sales Production, set 'sharing_type' to 'Lead' by default
        if board_type == BoardType.SALES_PRODUCTION:
            result_df['sharing_type'] = 'Lead'

        # For Sales Production, calculate 'Total', 'Total Reçu' and 'Paie' if not present
        if board_type == BoardType.SALES_PRODUCTION:
            # Total = Com + Boni + Sur-Com (commission + bonus_amount + on_commission)
            # Always recalculate to ensure consistency
            result_df['total'] = (
                result_df.get('commission', pd.Series(0, index=result_df.index)).fillna(0) +
                result_df.get('bonus_amount', pd.Series(0, index=result_df.index)).fillna(0) +
                result_df.get('on_commission', pd.Series(0, index=result_df.index)).fillna(0)
            )

            # Total Reçu = sum of all receipt columns (Reçu 1 + Reçu 2 + Reçu 3)
            if 'total_received' not in result_df.columns:
                result_df['total_received'] = (
                    result_df.get('commission_receipt', pd.Series(0, index=result_df.index)).fillna(0) +
                    result_df.get('bonus_amount_receipt', pd.Series(0, index=result_df.index)).fillna(0) +
                    result_df.get('on_commission_receipt', pd.Series(0, index=result_df.index)).fillna(0)
                )

            # Paie = Total - Total Reçu (remaining amount to be paid)
            if 'payment' not in result_df.columns:
                result_df['payment'] = (
                    result_df['total'].fillna(0) -
                    result_df['total_received'].fillna(0)
                )

        # For Historical Payments: Set status based on amount_received (Reçu)
        # - If Reçu > 0 → "Payé" (paid)
        # - If Reçu < 0 → "Charge back" (chargeback/refund)
        # - If Reçu = 0 or None → keep existing status or None
        if not is_sales_production and 'amount_received' in result_df.columns:
            # Convert to numeric to handle any string values
            amount_received = pd.to_numeric(result_df['amount_received'], errors='coerce')

            # Create status based on amount_received
            conditions = [
                amount_received > 0,
                amount_received < 0
            ]
            choices = ['Payé', 'Charge back']

            # Apply conditions - only update where amount_received is not null/zero
            new_status = np.select(conditions, choices, default=None)

            # Update status column (create if doesn't exist)
            if 'status' not in result_df.columns:
                result_df['status'] = None

            # Only update rows where we have a valid amount_received
            mask = amount_received.notna() & (amount_received != 0)
            result_df.loc[mask, 'status'] = np.array(new_status)[mask]

            # Count updates for logging
            paid_count = (amount_received > 0).sum()
            chargeback_count = (amount_received < 0).sum()
            print(f"   ✓ Status auto-set: {paid_count} 'Payé', {chargeback_count} 'Charge back'")

        # For Historical Payments: Set is_verified based on Reçu vs Com Calculée
        # Formula: Com Calculée = ROUND((PA * 0.4) * 0.5, 2)
        # - If Reçu < 0.9 * Com Calculée → "Pas Verifié"
        # - Otherwise → "Verifié"
        if not is_sales_production and 'amount_received' in result_df.columns and 'policy_premium' in result_df.columns:
            # Convert to numeric
            amount_received = pd.to_numeric(result_df['amount_received'], errors='coerce')
            policy_premium = pd.to_numeric(result_df['policy_premium'], errors='coerce')

            # Calculate expected commission: ROUND((PA * 0.4) * 0.5, 2)
            com_calculee = (policy_premium * 0.4 * 0.5).round(2)

            # Calculate threshold: 90% of calculated commission
            threshold = com_calculee * 0.9

            # Create is_verified column
            if 'is_verified' not in result_df.columns:
                result_df['is_verified'] = None

            # Apply verification logic
            # "Pas Verifié" if Reçu < 90% of Com Calculée
            # "Verifié" otherwise (including when Reçu >= 90% of Com Calculée)
            mask_valid = amount_received.notna() & com_calculee.notna() & (com_calculee != 0)
            mask_not_verified = mask_valid & (amount_received < threshold)
            mask_verified = mask_valid & (amount_received >= threshold)

            result_df.loc[mask_not_verified, 'is_verified'] = 'Pas Verifié'
            result_df.loc[mask_verified, 'is_verified'] = 'Verifié'

            # Count updates for logging
            verified_count = mask_verified.sum()
            not_verified_count = mask_not_verified.sum()
            print(f"   ✓ Verification auto-set: {verified_count} 'Verifié', {not_verified_count} 'Pas Verifié'")

        # Normalize advisor names (before renaming to French)
        if 'advisor_name' in result_df.columns:
            result_df = self._normalize_advisor_column(result_df, 'advisor_name')

        # Rename columns from English to French
        result_df = result_df.rename(columns=self.COLUMN_MAPPING_TO_FRENCH)

        # Keep only columns that exist in both final_columns and df
        available_columns = [col for col in final_columns if col in result_df.columns]

        # Filter to available columns
        filtered_df = result_df[available_columns].copy()

        # Add missing columns with None
        for col in final_columns:
            if col not in filtered_df.columns:
                filtered_df[col] = None

        # Reorder columns to match final_columns
        filtered_df = filtered_df[final_columns]

        # Preserve board_type in attrs (as string for JSON serialization compatibility)
        if board_type is not None:
            filtered_df.attrs['board_type'] = board_type.value if hasattr(board_type, 'value') else str(board_type)

        return filtered_df

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
                      aggregate_by_contract: bool = True, target_board_type: BoardType = None) -> pd.DataFrame:
        """
        Process a complete source from PDF or Monday.com to final standardized DataFrame.

        This method orchestrates the entire pipeline:
        1. Extract data from PDF or Monday.com DataFrame
        2. Convert to standard format
        3. Filter by sharing_rate (0.4) - SKIPPED for MONDAY_LEGACY and IDC_STATEMENT
        4. Aggregate by contract_number - SKIPPED for MONDAY_LEGACY and IDC_STATEMENT (or if aggregate_by_contract=False)
        5. Filter to final columns (board-type specific)

        Args:
            source: Source name - 'UV', 'IDC', 'IDC_STATEMENT', 'ASSOMPTION', or 'MONDAY_LEGACY'
            pdf_path: Path to the PDF file (required for UV, IDC, IDC_STATEMENT, ASSOMPTION)
            monday_df: DataFrame from Monday.com extraction (required for MONDAY_LEGACY)
            aggregate_by_contract: Whether to aggregate rows by contract_number (default: True)
            target_board_type: Target board type for Monday.com upload (auto-detected for MONDAY_LEGACY,
                              defaults to HISTORICAL_PAYMENTS for PDF sources)

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

        # Determine board_type for the target
        if source == 'MONDAY_LEGACY':
            # For MONDAY_LEGACY: auto-detect source board type, use it as target
            # (same type to same type)
            if target_board_type is None:
                target_board_type = self.detect_board_type(raw_df)
        else:
            # For PDF sources: use provided target_board_type or default to HISTORICAL_PAYMENTS
            if target_board_type is None:
                target_board_type = BoardType.HISTORICAL_PAYMENTS
                print(f"   📋 Target board type not specified, defaulting to: {target_board_type.value}")
            else:
                print(f"   📋 Target board type: {target_board_type.value}")

        # Step 2: Convert to standard format
        if source == 'MONDAY_LEGACY':
            standardized = self.convert_monday_legacy_to_standard(raw_df, board_type=target_board_type)
        elif source == 'UV':
            standardized = self.convert_uv_to_standard(raw_df, metadata)
        elif source == 'IDC':
            standardized = self.convert_idc_to_standard(raw_df)
        elif source == 'IDC_STATEMENT':
            standardized = self.convert_idc_statement_to_standard(raw_df)
        elif source == 'ASSOMPTION':
            standardized = self.convert_assomption_to_standard(raw_df)

        # Store board_type in DataFrame attrs for later use (as string for JSON serialization)
        standardized.attrs['board_type'] = target_board_type.value if hasattr(target_board_type, 'value') else str(target_board_type)

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

        # Step 5: Filter to final columns (board-type specific)
        # For MONDAY_LEGACY, preserve group metadata
        if source == 'MONDAY_LEGACY':
            # Keep final columns but also preserve group metadata
            final = self.filter_final_columns(aggregated, board_type=target_board_type)

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
            final = self.filter_final_columns(aggregated, board_type=target_board_type)
            print(f"✅ Final output: {len(final)} records ready for {target_board_type.value} board")

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

            # Use French column names: 'Total' for Sales Production, 'Com' for Historical Payments
            commission_col = 'Total' if 'Total' in df.columns else 'Com'
            if commission_col in df.columns:
                commission_sum = df[commission_col].sum()
                if not pd.isna(commission_sum):
                    total_commission += commission_sum
            else:
                commission_sum = None

            report.append(f"\n{source}:")
            report.append(f"  📊 Records: {records:,}")
            if commission_sum is not None and not pd.isna(commission_sum):
                report.append(f"  💰 Total Commission: ${commission_sum:,.2f}")
            else:
                report.append(f"  💰 Total Commission: N/A")

            # Unique contracts (using French column name)
            if '# de Police' in df.columns:
                unique_contracts = df['# de Police'].nunique()
                report.append(f"  📝 Unique Contracts: {unique_contracts:,}")

            # Date range (using French column name)
            if 'Date' in df.columns:
                dates = pd.to_datetime(df['Date'], errors='coerce').dropna()
                if len(dates) > 0:
                    report.append(f"  📅 Date Range: {dates.min().date()} to {dates.max().date()}")

        report.append("\n" + "-" * 80)
        report.append(f"TOTAL ACROSS ALL SOURCES:")
        report.append(f"  📊 Total Records: {total_records:,}")
        report.append(f"  💰 Total Commission: ${total_commission:,.2f}")
        report.append("=" * 80)

        return "\n".join(report)

    def validate_data_quality(self, df: pd.DataFrame, source_name: str) -> Dict:
        """
        Validate data quality and return issues.

        Uses French column names as they appear in the final DataFrame.

        Args:
            df: DataFrame to validate (with French column names)
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

        # Check for missing critical fields (French column names)
        critical_fields = ['# de Police', 'Compagnie', 'Com']
        for field in critical_fields:
            if field in df.columns:
                missing_count = df[field].isna().sum()
                if missing_count > 0:
                    pct = (missing_count / len(df)) * 100
                    issues['warnings'].append(f"{field}: {missing_count} missing values ({pct:.1f}%)")

        # Check for negative commissions (unusual but not necessarily wrong)
        if 'Com' in df.columns:
            negative_count = (df['Com'] < 0).sum()
            if negative_count > 0:
                issues['warnings'].append(f"Found {negative_count} negative commission values")

        # Check for duplicate contracts
        if '# de Police' in df.columns:
            duplicates = df['# de Police'].duplicated().sum()
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