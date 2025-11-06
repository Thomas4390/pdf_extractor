"""
Insurance Commission Data Pipeline
===================================

Main orchestration script that:
1. Extracts commission data from insurance PDFs
2. Processes and transforms the data
3. Uploads the data to Monday.com boards

MODIFICATIONS:
- Ajout de self.column_mapping dans __init__ pour stocker les IDs des colonnes
- Modification de _step3_setup_monday_board() pour créer/obtenir les colonnes automatiquement
- Modification de _prepare_monday_items() pour remplir les column_values correctement

Author: Thomas
Date: 2025-10-30
Version: 1.1.0 (avec support colonnes Monday.com)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

import pandas as pd

# Import local modules
from unify_notation import CommissionDataUnifier, PDF_PATHS as DEFAULT_PDF_PATHS
from monday_automation import MondayClient, CreateBoardResult, CreateGroupResult, CreateItemResult


# =============================================================================
# COLORED OUTPUT
# =============================================================================


class Colors:
    """ANSI color codes for terminal output."""
    # Regular colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'

    # Reset
    RESET = '\033[0m'

    @staticmethod
    def strip_colors(text: str) -> str:
        """Remove color codes from text."""
        import re
        ansi_escape = re.compile(r'\033\[[0-9;]+m')
        return ansi_escape.sub('', text)


class ColorPrint:
    """Utility class for colored console output."""

    @staticmethod
    def header(text: str):
        """Print a header (cyan, bold)."""
        print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")

    @staticmethod
    def success(text: str):
        """Print success message (green)."""
        print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")

    @staticmethod
    def error(text: str):
        """Print error message (red)."""
        print(f"{Colors.BRIGHT_RED}❌ {text}{Colors.RESET}")

    @staticmethod
    def warning(text: str):
        """Print warning message (yellow)."""
        print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")

    @staticmethod
    def info(text: str):
        """Print info message (blue)."""
        print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")

    @staticmethod
    def step(text: str):
        """Print step header (magenta, bold)."""
        print(f"{Colors.BOLD}{Colors.MAGENTA}🔹 {text}{Colors.RESET}")

    @staticmethod
    def data(text: str):
        """Print data/content (white)."""
        print(f"{Colors.WHITE}{text}{Colors.RESET}")

    @staticmethod
    def separator(char: str = "=", length: int = 100):
        """Print a separator line."""
        print(f"{Colors.DIM}{char * length}{Colors.RESET}")

    @staticmethod
    def section(title: str):
        """Print a section header with separators."""
        print()
        ColorPrint.separator("━", 100)
        print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}📋 {title}{Colors.RESET}")
        ColorPrint.separator("━", 100)


# =============================================================================
# CONFIGURATION
# =============================================================================


class InsuranceSource(Enum):
    """Enum for supported insurance sources."""
    UV = "UV"
    IDC = "IDC"
    ASSOMPTION = "ASSOMPTION"


@dataclass
class PipelineConfig:
    """
    Configuration for the insurance commission data pipeline.

    Attributes:
        source: Insurance source to process (UV, IDC, or ASSOMPTION)
        pdf_path: Path to the PDF file to process
        month_group: Month group name (e.g., "Octobre 2025"). If None, no group is created.
        board_name: Name of the Monday.com board to create/use
        board_id: Optional existing board ID to use (overrides board_name if provided)
        monday_api_key: Monday.com API key
        output_dir: Directory for intermediate results
        reuse_board: Whether to reuse existing board with same name
        reuse_group: Whether to reuse existing group with same name
    """
    # Data source configuration
    source: InsuranceSource
    pdf_path: str

    # Monday.com configuration
    month_group: Optional[str] = None
    board_name: str = "Commission Data"
    board_id: Optional[int] = None
    monday_api_key: str = ""

    # Processing configuration
    output_dir: str = "./results"
    reuse_board: bool = True
    reuse_group: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate source
        if isinstance(self.source, str):
            self.source = InsuranceSource(self.source)

        # Validate PDF path
        if not Path(self.pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        # Validate Monday.com API key
        if not self.monday_api_key:
            raise ValueError("Monday.com API key is required")

        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# PIPELINE
# =============================================================================


class InsuranceCommissionPipeline:
    """
    Main pipeline for processing insurance commission data and uploading to Monday.com.

    This pipeline orchestrates the following steps:
    1. Data extraction from PDF
    2. Data transformation and standardization
    3. Data upload to Monday.com
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config

        # Initialize components
        self.data_unifier = CommissionDataUnifier(output_dir=config.output_dir)
        self.monday_client = MondayClient(api_key=config.monday_api_key)

        # State variables
        self.final_data: Optional[pd.DataFrame] = None

        self.board_id: Optional[int] = None
        self.group_id: Optional[str] = None
        self.column_mapping: Dict[str, str] = {}  # Maps column names to Monday.com column IDs

    def run(self) -> bool:
        """
        Run the complete pipeline.

        Returns:
            True if successful, False otherwise
        """
        ColorPrint.separator("=", 100)
        ColorPrint.header("🚀 INSURANCE COMMISSION DATA PIPELINE")
        ColorPrint.separator("=", 100)
        ColorPrint.info(f"Source: {self.config.source.value}")
        ColorPrint.info(f"PDF Path: {self.config.pdf_path}")
        ColorPrint.info(f"Board Name: {self.config.board_name}")
        ColorPrint.info(f"Month Group: {self.config.month_group or 'No group'}")
        ColorPrint.separator("=", 100)

        try:
            # Step 1: Extract and standardize data from PDF
            if not self._step1_extract_data():
                ColorPrint.error("Step 1 failed: Data extraction")
                return False

            # Step 2: Process data (placeholder for future calculations)
            if not self._step2_process_data():
                ColorPrint.error("Step 2 failed: Data processing")
                return False

            # Step 3: Setup Monday.com board and group
            if not self._step3_setup_monday_board():
                ColorPrint.error("Step 3 failed: Monday.com board setup")
                return False

            # Step 4: Upload data to Monday.com
            if not self._step4_upload_to_monday():
                ColorPrint.error("Step 4 failed: Data upload to Monday.com")
                return False

            print()
            ColorPrint.separator("=", 100)
            ColorPrint.success("PIPELINE COMPLETED SUCCESSFULLY")
            ColorPrint.separator("=", 100)

            return True

        except Exception as e:
            print()
            ColorPrint.error(f"Pipeline failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _step1_extract_data(self) -> bool:
        """
        Step 1: Extract and standardize data from PDF.

        Returns:
            True if successful, False otherwise
        """
        ColorPrint.section("STEP 1: EXTRACT & PROCESS DATA FROM PDF")

        try:
            source = self.config.source.value
            pdf_path = self.config.pdf_path

            ColorPrint.info(f"Processing {source} data from: {pdf_path}")

            # Use the unified process_source method from CommissionDataUnifier
            # This handles extraction, standardization, filtering, and aggregation
            self.final_data = self.data_unifier.process_source(source, pdf_path)

            # Check if data was processed
            if self.final_data is None or self.final_data.empty:
                ColorPrint.error("No data processed from PDF")
                return False

            ColorPrint.success(f"Processing complete: {len(self.final_data)} final records")
            ColorPrint.info(f"Columns: {list(self.final_data.columns)}")

            # Debug: Check report_date in final data
            if 'report_date' in self.final_data.columns:
                dates_final = self.final_data['report_date'].notna().sum()
                ColorPrint.info(f"Report dates: {dates_final}/{len(self.final_data)} non-null values")
                if dates_final > 0:
                    sample_dates = self.final_data['report_date'].dropna().head(3).tolist()
                    ColorPrint.info(f"Sample dates: {sample_dates}")

            # Display sample
            print()
            ColorPrint.step("Sample data (first 3 rows):")
            ColorPrint.data(self.final_data.head(3).to_string())

            return True

        except Exception as e:
            ColorPrint.error(f"Error in data processing: {e}")
            import traceback
            traceback.print_exc()
            return False


    def _step2_process_data(self) -> bool:
        """
        Step 2: Process and transform data.

        This step is a placeholder for future calculations and transformations.
        Currently, it just validates the data.

        Returns:
            True if successful, False otherwise
        """
        ColorPrint.section("STEP 2: PROCESS DATA")

        try:
            if self.final_data is None or self.final_data.empty:
                ColorPrint.error("No data to process")
                return False

            # Validate data quality
            ColorPrint.info("Validating data quality...")
            validation = self.data_unifier.validate_data_quality(
                self.final_data,
                self.config.source.value
            )

            if validation['errors']:
                ColorPrint.error("Data validation errors:")
                for error in validation['errors']:
                    print(f"  - {error}")
                return False

            if validation['warnings']:
                ColorPrint.warning("Data validation warnings:")
                for warning in validation['warnings'][:5]:
                    print(f"  - {warning}")
                if len(validation['warnings']) > 5:
                    print(f"  ... and {len(validation['warnings']) - 5} more warnings")

            # Placeholder for future calculations
            ColorPrint.success("Data processing complete (no transformations applied)")
            ColorPrint.info(f"Ready to upload {len(self.final_data)} records")

            return True

        except Exception as e:
            ColorPrint.error(f"Error in data processing: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _step3_setup_monday_board(self) -> bool:
        """
        Step 3: Setup Monday.com board, group, and columns.

        MODIFIÉ: Ajoute la configuration automatique des colonnes basée sur le DataFrame.

        Returns:
            True if successful, False otherwise
        """
        ColorPrint.section("STEP 3: SETUP MONDAY.COM BOARD & COLUMNS")

        try:
            # Create or reuse board
            if self.config.board_id:
                # Use existing board ID
                self.board_id = self.config.board_id
                ColorPrint.info(f"Using existing board ID: {self.board_id}")
            else:
                # Create or reuse board by name
                ColorPrint.info(f"Creating/reusing board: {self.config.board_name}")
                board_result = self.monday_client.create_board(
                    board_name=self.config.board_name,
                    board_kind="public",
                    reuse_existing=self.config.reuse_board
                )

                if not board_result.success:
                    ColorPrint.error(f"Failed to create/reuse board: {board_result.error}")
                    return False

                self.board_id = int(board_result.board_id)
                ColorPrint.success(f"Board ready: {board_result.board_name} (ID: {self.board_id})")

            # Create or reuse group if month_group is specified
            if self.config.month_group:
                ColorPrint.info(f"Creating/reusing group: {self.config.month_group}")
                group_result = self.monday_client.create_group(
                    board_id=self.board_id,
                    group_name=self.config.month_group,
                    group_color="#0086c0",  # Blue color
                    reuse_existing=self.config.reuse_group
                )

                if not group_result.success:
                    ColorPrint.error(f"Failed to create/reuse group: {group_result.error}")
                    return False

                self.group_id = group_result.group_id
                ColorPrint.success(f"Group ready: {group_result.group_title} (ID: {self.group_id})")
            else:
                ColorPrint.info("No month group specified - items will be added to default group")
                self.group_id = None

            # NOUVEAU: Setup columns based on DataFrame structure
            if self.final_data is not None and not self.final_data.empty:
                ColorPrint.info("Setting up columns for data structure...")

                # Get column names from DataFrame (exclude metadata columns)
                # Ces colonnes ne doivent pas être créées dans Monday car elles sont internes
                metadata_columns = ['item_id', 'item_name', 'board_id', 'board_name',
                                   'group_id', 'group_title', 'is_subitem',
                                   'parent_item_id', 'parent_item_name']

                # TOUTES les colonnes de données sont créées, y compris contract_number et insured_name
                data_columns = [col for col in self.final_data.columns
                               if col not in metadata_columns]

                ColorPrint.info(f"Found {len(data_columns)} data columns to create/map in Monday.com")
                ColorPrint.info(f"Columns: {data_columns}")

                # Define column types based on data
                # Note: Dates are now treated as text/strings for simplicity
                number_columns = ['policy_premium', 'sharing_rate', 'commission_rate',
                                 'commission', 'bonus_rate', 'bonus_amount',
                                 'on_commission_rate', 'on_commission']

                # Get or create columns with appropriate types
                self.column_mapping = {}

                # Create/get text columns (includes dates now)
                text_columns = [col for col in data_columns if col not in number_columns]
                if text_columns:
                    ColorPrint.info(f"Creating/mapping {len(text_columns)} text columns...")
                    text_mapping = self.monday_client.get_or_create_columns(
                        board_id=self.board_id,
                        column_names=text_columns,
                        column_type="text"
                    )
                    self.column_mapping.update(text_mapping)

                # Create/get number columns with proper type
                number_cols_present = [col for col in data_columns if col in number_columns]
                if number_cols_present:
                    ColorPrint.info(f"Creating/mapping {len(number_cols_present)} number columns...")
                    number_mapping = self.monday_client.get_or_create_columns(
                        board_id=self.board_id,
                        column_names=number_cols_present,
                        column_type="numbers"
                    )
                    self.column_mapping.update(number_mapping)

                ColorPrint.success(f"Columns configured: {len(self.column_mapping)} columns ready")

                # Debug: Show which columns were mapped
                ColorPrint.info("Column mapping details:")
                for col_name, col_id in self.column_mapping.items():
                    ColorPrint.data(f"  - {col_name} -> {col_id}")

                # Verify insured_name is in mapping
                if 'insured_name' in self.column_mapping:
                    ColorPrint.success("✓ insured_name column is mapped correctly")
                else:
                    ColorPrint.error("✗ insured_name column is MISSING from mapping!")
            else:
                ColorPrint.warning("No data available for column setup")
                self.column_mapping = {}

            return True

        except Exception as e:
            ColorPrint.error(f"Error in Monday.com board setup: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _step4_upload_to_monday(self) -> bool:
        """
        Step 4: Upload data to Monday.com.

        Returns:
            True if successful, False otherwise
        """
        ColorPrint.section("STEP 4: UPLOAD DATA TO MONDAY.COM")

        try:
            if self.final_data is None or self.final_data.empty:
                ColorPrint.error("No data to upload")
                return False

            # Prepare items for batch creation
            items_to_create = self._prepare_monday_items(self.final_data)

            ColorPrint.info(f"Uploading {len(items_to_create)} items to Monday.com...")

            # Upload items in batch
            results = self.monday_client.create_items_batch(
                board_id=self.board_id,
                items=items_to_create,
                group_id=self.group_id
            )

            # Analyze results
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful

            print()
            ColorPrint.step("Upload Summary:")
            ColorPrint.info(f"Total items:   {len(results)}")
            ColorPrint.success(f"Successful: {successful}")
            if failed > 0:
                ColorPrint.error(f"Failed:     {failed}")

            if failed > 0:
                ColorPrint.warning(f"Some items failed to upload. Check details above.")
                # Log first few failures
                for i, result in enumerate(results):
                    if not result.success and i < 3:
                        ColorPrint.warning(f"Failed item: {result.error}")

            return successful > 0

        except Exception as e:
            ColorPrint.error(f"Error in data upload: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _prepare_monday_items(self, df: pd.DataFrame) -> List[Dict]:
        """
        Prepare DataFrame rows as Monday.com items with proper column values.

        MODIFIÉ: Au lieu de tout mettre dans le nom de l'item, cette méthode maintenant:
        1. Crée le nom de l'item à partir de contract_number et insured_name
        2. Remplit les column_values avec toutes les colonnes du DataFrame dans l'ordre
        3. Formate correctement les colonnes de date pour l'API Monday.com

        Args:
            df: DataFrame with commission data

        Returns:
            List of item dictionaries for batch creation
        """
        items = []

        # Get column mapping
        if not hasattr(self, 'column_mapping') or not self.column_mapping:
            ColorPrint.warning("No column mapping available - items will be created with names only")
            column_mapping = {}
        else:
            column_mapping = self.column_mapping

        ColorPrint.info(f"Preparing {len(df)} items with {len(column_mapping)} mapped columns...")

        # Debug: Check report_date in DataFrame
        if 'report_date' in df.columns:
            non_null_dates = df['report_date'].notna().sum()
            ColorPrint.info(f"Debug: report_date column found in DataFrame")
            ColorPrint.info(f"       Non-null dates: {non_null_dates}/{len(df)}")
            if non_null_dates > 0:
                sample_dates = df['report_date'].dropna().head(3).tolist()
                ColorPrint.info(f"       Sample dates: {sample_dates}")
                ColorPrint.info(f"       Date types: {[type(d).__name__ for d in sample_dates]}")
        else:
            ColorPrint.warning(f"Debug: report_date column NOT found in DataFrame!")
            ColorPrint.info(f"       Available columns: {list(df.columns)}")

        # Metadata columns to skip
        metadata_columns = ['item_id', 'item_name', 'board_id', 'board_name',
                           'group_id', 'group_title', 'is_subitem',
                           'parent_item_id', 'parent_item_name']

        for idx, row in df.iterrows():
            # Create item name from contract number and insured name
            contract_num = str(row.get('contract_number', 'N/A'))
            insured_name = str(row.get('insured_name', 'Unknown'))
            item_name = f"{contract_num} - {insured_name}"

            # Prepare column values - iterate in DataFrame column order
            column_values = {}

            # Debug for first row only
            is_first_row = (idx == df.index[0])

            for col_name, col_value in row.items():
                # Skip metadata columns
                if col_name in metadata_columns:
                    continue

                # Check if we have a column ID for this column
                if col_name not in column_mapping:
                    continue

                column_id = column_mapping[col_name]

                # Debug for report_date on first row
                if is_first_row and col_name == 'report_date':
                    print(f"       DEBUG report_date:")
                    print(f"         Raw value: {repr(col_value)}")
                    print(f"         Type: {type(col_value).__name__}")
                    print(f"         Is NA: {pd.isna(col_value)}")
                    print(f"         Is None: {col_value is None}")
                    print(f"         Is empty string: {col_value == ''}")

                # Handle empty/NaN values
                if pd.isna(col_value) or col_value is None or col_value == '':
                    if is_first_row and col_name == 'report_date':
                        print(f"         → SKIPPED: Empty/NaN value")
                    # Skip empty values entirely (don't send them)
                    continue

                # Convert all values to string (including dates)
                value_str = str(col_value)

                # Debug for report_date on first row
                if is_first_row and col_name == 'report_date':
                    print(f"         Converted to string: {repr(value_str)}")

                # Skip if conversion resulted in "None" or "nan"
                if value_str in ['None', 'nan', 'NaN', 'NaT']:
                    if is_first_row and col_name == 'report_date':
                        print(f"         → SKIPPED: Invalid string value")
                    continue

                if is_first_row and col_name == 'report_date':
                    print(f"         → ADDED to column_values: {value_str}")

                column_values[column_id] = value_str

            # Create item dictionary
            item = {
                "name": item_name
            }

            # Only add column_values if we have any
            if column_values:
                item["column_values"] = column_values

            items.append(item)

        # Show preview of first item for debugging
        if items:
            ColorPrint.info(f"Example item structure:")
            ColorPrint.data(f"  Name: {items[0]['name']}")
            if 'column_values' in items[0]:
                ColorPrint.data(f"  Columns: {len(items[0]['column_values'])} values")
                # Show first few column values
                for i, (col_id, col_val) in enumerate(list(items[0]['column_values'].items())[:5]):
                    ColorPrint.data(f"    - {col_id}: {col_val[:50]}..." if len(col_val) > 50 else f"    - {col_id}: {col_val}")

                # Debug: Check if report_date is in the data
                if 'report_date' in column_mapping:
                    report_date_id = column_mapping['report_date']
                    if report_date_id in items[0].get('column_values', {}):
                        ColorPrint.success(f"  ✓ report_date found: {items[0]['column_values'][report_date_id]}")
                    else:
                        ColorPrint.warning(f"  ⚠️  report_date column ID ({report_date_id}) not in column_values!")
                else:
                    ColorPrint.warning(f"  ⚠️  report_date not in column mapping!")

        return items


# =============================================================================
# EXAMPLE CONFIGURATIONS
# =============================================================================


def create_uv_config(api_key: str) -> PipelineConfig:
    """Create configuration for UV Assurance processing."""
    return PipelineConfig(
        source=InsuranceSource.UV,
        pdf_path="../pdf/rappportremun_21622_2025-10-20.pdf",
        month_group="Octobre 2025",
        board_name="Commissions UV Assurance",
        monday_api_key=api_key,
        output_dir="./results/uv"
    )


def create_idc_config(api_key: str) -> PipelineConfig:
    """Create configuration for IDC processing."""
    return PipelineConfig(
        source=InsuranceSource.IDC,
        pdf_path="../pdf/Rapport des propositions soumises.20251017_1517.pdf",
        month_group="Octobre 2025",
        board_name="Commissions IDC",
        monday_api_key=api_key,
        output_dir="./results/idc"
    )


def create_assomption_config(api_key: str) -> PipelineConfig:
    """Create configuration for Assomption Vie processing."""
    return PipelineConfig(
        source=InsuranceSource.ASSOMPTION,
        pdf_path="../pdf/Remuneration (61).pdf",
        month_group="Octobre 2025",
        board_name="Commissions Assomption",
        monday_api_key=api_key,
        output_dir="./results/assomption"
    )


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """
    Main execution function.

    Configure your pipeline below by selecting one of the predefined
    configurations or creating your own.
    """

    # eyJhbGciOiJIUzI1NiJ9.eyJ0aWQiOjU3OTYxMDI3NiwiYWFpIjoxMSwidWlkIjo5NTA2NjUzNywiaWFkIjoiMjAyNS0xMC0yOFQxNToxMDo0My40NjZaIiwicGVyIjoibWU6d3JpdGUiLCJhY3RpZCI6MjY0NjQxNDIsInJnbiI6InVzZTEifQ.q54YnC23stSJfLRnd0E9p9e4ZF8lRUK1TLgQM-13kdI

    # Monday.com API key
    MONDAY_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJ0aWQiOjU3OTYxMDI3NiwiYWFpIjoxMSwidWlkIjo5NTA2NjUzNywiaWFkIjoiMjAyNS0xMC0yOFQxNToxMDo0My40NjZaIiwicGVyIjoibWU6d3JpdGUiLCJhY3RpZCI6MjY0NjQxNDIsInJnbiI6InVzZTEifQ.q54YnC23stSJfLRnd0E9p9e4ZF8lRUK1TLgQM-13kdI"

    # =========================================================================
    # CONFIGURATION - MODIFY THIS SECTION TO CHANGE BEHAVIOR
    # =========================================================================

    # Choose configuration by uncommenting ONE of the following:

    # Option 1: UV Assurance
    config = create_uv_config(MONDAY_API_KEY)

    # Option 2: IDC
    # config = create_idc_config(MONDAY_API_KEY)

    # Option 3: Assomption Vie
    # config = create_assomption_config(MONDAY_API_KEY)

    # Option 4: Custom configuration
    # config = PipelineConfig(
    #     source=InsuranceSource.UV,
    #     pdf_path="../pdf/your_pdf.pdf",
    #     month_group="Novembre 2025",  # Or None for no group
    #     board_name="My Custom Board",
    #     monday_api_key=MONDAY_API_KEY,
    #     output_dir="./results/custom"
    # )

    # =========================================================================
    # EXECUTION
    # =========================================================================

    print()
    ColorPrint.separator("=", 100)
    ColorPrint.header("CONFIGURATION SUMMARY")
    ColorPrint.separator("=", 100)
    ColorPrint.info(f"Source:       {config.source.value}")
    ColorPrint.info(f"PDF Path:     {config.pdf_path}")
    ColorPrint.info(f"Board Name:   {config.board_name}")
    ColorPrint.info(f"Month Group:  {config.month_group or 'None'}")
    ColorPrint.info(f"Output Dir:   {config.output_dir}")
    ColorPrint.separator("=", 100)
    print()

    # Run pipeline
    pipeline = InsuranceCommissionPipeline(config)
    success = pipeline.run()

    print()
    if success:
        ColorPrint.success("Pipeline completed successfully!")
        return 0
    else:
        ColorPrint.error("Pipeline failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())