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
    MONDAY_LEGACY = "MONDAY_LEGACY"


@dataclass
class PipelineConfig:
    """
    Configuration for the insurance commission data pipeline.

    Attributes:
        source: Insurance source to process (UV, IDC, ASSOMPTION, or MONDAY_LEGACY)
        pdf_path: Path to the PDF file to process (required for UV, IDC, ASSOMPTION)
        month_group: Month group name (e.g., "Octobre 2025"). If None, no group is created.
        board_name: Name of the Monday.com board to create/use
        board_id: Optional existing board ID to use (overrides board_name if provided)
        monday_api_key: Monday.com API key
        output_dir: Directory for intermediate results
        reuse_board: Whether to reuse existing board with same name
        reuse_group: Whether to reuse existing group with same name
        source_board_id: Board ID to extract from (required for MONDAY_LEGACY)
        source_group_id: Optional group ID to filter extraction (for MONDAY_LEGACY)
    """
    # Data source configuration
    source: InsuranceSource
    pdf_path: Optional[str] = None

    # Monday.com configuration
    month_group: Optional[str] = None
    board_name: str = "Commission Data"
    board_id: Optional[int] = None
    monday_api_key: str = ""

    # Processing configuration
    output_dir: str = "./results"
    reuse_board: bool = True
    reuse_group: bool = True

    # Monday.com source configuration (for MONDAY_LEGACY)
    source_board_id: Optional[int] = None
    source_group_id: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate source
        if isinstance(self.source, str):
            self.source = InsuranceSource(self.source)

        # Validate PDF path for PDF-based sources
        if self.source in [InsuranceSource.UV, InsuranceSource.IDC, InsuranceSource.ASSOMPTION]:
            if not self.pdf_path:
                raise ValueError(f"PDF path is required for source: {self.source.value}")
            if not Path(self.pdf_path).exists():
                raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        # Validate Monday.com source configuration for MONDAY_LEGACY
        if self.source == InsuranceSource.MONDAY_LEGACY:
            if not self.source_board_id:
                raise ValueError("source_board_id is required for MONDAY_LEGACY source")

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
        self.group_mapping: Dict[str, str] = {}  # Maps original group titles to new group IDs (for MONDAY_LEGACY)

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
        Step 1: Extract and standardize data from PDF or Monday.com.

        Returns:
            True if successful, False otherwise
        """
        source_type = "PDF" if self.config.source != InsuranceSource.MONDAY_LEGACY else "Monday.com Board"
        ColorPrint.section(f"STEP 1: EXTRACT & PROCESS DATA FROM {source_type}")

        try:
            source = self.config.source.value

            # Handle MONDAY_LEGACY source differently
            if self.config.source == InsuranceSource.MONDAY_LEGACY:
                ColorPrint.info(f"Extracting data from Monday.com board: {self.config.source_board_id}")
                ColorPrint.info(f"Extracting ALL groups to preserve board structure...")

                # Import necessary classes
                from monday_automation import DataProcessor

                # Extract data from Monday.com board (entire board, no group filtering)
                board_data = self.monday_client.extract_board_data(
                    board_id=self.config.source_board_id,
                    group_id=None  # Extract ALL groups, not just one
                )

                # Convert to DataFrame
                processor = DataProcessor()
                monday_df = processor.board_to_dataframe(board_data, include_subitems=False)

                ColorPrint.success(f"Extracted {len(monday_df)} records from Monday.com")
                ColorPrint.info(f"Columns found: {list(monday_df.columns)}")

                # Check for group information
                if 'group_title' in monday_df.columns:
                    unique_groups = monday_df['group_title'].dropna().unique()
                    ColorPrint.info(f"Groups found: {len(unique_groups)} - {list(unique_groups)}")
                else:
                    ColorPrint.warning("No group information found in extracted data")

                # Process using the unified method with Monday.com DataFrame
                self.final_data = self.data_unifier.process_source(
                    source=source,
                    monday_df=monday_df
                )

            else:
                # PDF-based sources
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

            # Handle groups based on source type
            if self.config.source == InsuranceSource.MONDAY_LEGACY:
                # For MONDAY_LEGACY: preserve original group structure
                ColorPrint.info("Detecting and creating groups from source board...")

                # Get unique groups from the data
                if 'group_title' in self.final_data.columns:
                    all_unique_groups = self.final_data['group_title'].dropna().unique()

                    # Filter out default "Group Title" groups
                    unique_groups = [g for g in all_unique_groups if g != 'Group Title']

                    # Reverse the order to maintain correct display order in Monday.com
                    # (Monday.com displays groups in reverse creation order)
                    unique_groups = list(reversed(unique_groups))

                    ColorPrint.info(f"Found {len(unique_groups)} unique groups in source data (excluding default 'Group Title')")
                    ColorPrint.info(f"Groups to create: {unique_groups}")

                    # Create each group in the new board
                    self.group_mapping = {}
                    ColorPrint.info(f"Starting group creation loop for {len(unique_groups)} groups...")
                    for idx, group_title in enumerate(unique_groups):
                        ColorPrint.info(f"[{idx+1}/{len(unique_groups)}] Creating/reusing group: {group_title}")

                        try:
                            group_result = self.monday_client.create_group(
                                board_id=self.board_id,
                                group_name=str(group_title),
                                group_color="#0086c0",  # Blue color
                                reuse_existing=self.config.reuse_group
                            )

                            if group_result.success:
                                self.group_mapping[str(group_title)] = group_result.group_id
                                ColorPrint.success(f"  ✓ Group '{group_title}' ready (ID: {group_result.group_id})")
                            else:
                                ColorPrint.warning(f"  ⚠️  Failed to create group '{group_title}': {group_result.error}")

                        except Exception as e:
                            ColorPrint.error(f"  ❌ Exception while creating group '{group_title}': {e}")
                            import traceback
                            traceback.print_exc()

                    ColorPrint.success(f"Group structure preserved: {len(self.group_mapping)} groups created")
                    self.group_id = None  # No single group, we'll use mapping
                else:
                    ColorPrint.warning("No group information found in source data")
                    self.group_id = None
                    self.group_mapping = {}

            elif self.config.month_group:
                # For PDF sources: create or reuse single month group
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

                # Import FINAL_COLUMNS from unify_notation to ensure correct order
                from unify_notation import CommissionDataUnifier
                FINAL_COLUMNS = CommissionDataUnifier.FINAL_COLUMNS

                # Use FINAL_COLUMNS order, but only keep columns that exist in DataFrame
                data_columns = [col for col in FINAL_COLUMNS
                               if col in self.final_data.columns and col not in metadata_columns]

                ColorPrint.info(f"Found {len(data_columns)} data columns to create/map in Monday.com")
                ColorPrint.info(f"Columns (in order): {data_columns}")

                # Define column types based on data
                # Note: Dates are now treated as text/strings for simplicity
                number_columns_set = {'policy_premium', 'sharing_rate', 'commission_rate',
                                     'commission', 'bonus_rate', 'bonus_amount',
                                     'on_commission_rate', 'on_commission', 'amount_received'}

                # Get or create columns with appropriate types
                # IMPORTANT: Create columns in order to preserve FINAL_COLUMNS order
                self.column_mapping = {}

                ColorPrint.info(f"Creating/mapping columns in order...")

                # Create columns one by one in the order of data_columns (which follows FINAL_COLUMNS)
                for col_name in data_columns:
                    # Determine column type
                    col_type = "numbers" if col_name in number_columns_set else "text"

                    # Create or get this single column
                    col_mapping = self.monday_client.get_or_create_columns(
                        board_id=self.board_id,
                        column_names=[col_name],
                        column_type=col_type
                    )
                    self.column_mapping.update(col_mapping)

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

            # Filter out rows with empty/invalid contract numbers
            initial_count = len(self.final_data)
            if 'contract_number' in self.final_data.columns:
                self.final_data = self.final_data[
                    self.final_data['contract_number'].notna() &
                    (self.final_data['contract_number'] != '') &
                    (self.final_data['contract_number'] != 'nan')
                ].copy()

                filtered_count = initial_count - len(self.final_data)
                if filtered_count > 0:
                    ColorPrint.warning(f"Filtered out {filtered_count} rows with empty contract numbers")

            # Check if we need to preserve group structure (MONDAY_LEGACY)
            if self.config.source == InsuranceSource.MONDAY_LEGACY and self.group_mapping:
                ColorPrint.info(f"Uploading items by group to preserve structure...")

                # Display detailed summary before upload
                ColorPrint.section("RÉSUMÉ DES DONNÉES À TRANSFÉRER")
                ColorPrint.info(f"📊 Total d'items à uploader: {len(self.final_data)}")
                ColorPrint.info(f"📁 Nombre de groupes: {len(self.group_mapping)}")

                # Display items per group
                for group_title in self.group_mapping.keys():
                    if 'group_title' in self.final_data.columns:
                        group_data = self.final_data[self.final_data['group_title'] == group_title]
                        if not group_data.empty:
                            ColorPrint.info(f"\n  📂 Groupe: {group_title}")
                            ColorPrint.info(f"     → {len(group_data)} items")
                            # Display first 5 item names (contract numbers)
                            if 'contract_number' in group_data.columns:
                                sample_items = group_data['contract_number'].head(5).tolist()
                                ColorPrint.info(f"     → Exemples: {sample_items}")
                                if len(group_data) > 5:
                                    ColorPrint.info(f"     → ... et {len(group_data) - 5} autres")

                # Show items in default "Group Title" if any
                if 'group_title' in self.final_data.columns:
                    default_items = self.final_data[self.final_data['group_title'] == 'Group Title']
                    if not default_items.empty:
                        ColorPrint.info(f"\n  📂 Groupe par défaut (non renommé)")
                        ColorPrint.info(f"     → {len(default_items)} items (seront uploadés dans le groupe par défaut)")

                ColorPrint.info("\n" + "="*60 + "\n")

                results = []

                # DEBUG: Show group mapping and data
                ColorPrint.info(f"DEBUG - Group mapping keys: {list(self.group_mapping.keys())}")
                if 'group_title' in self.final_data.columns:
                    unique_in_data = self.final_data['group_title'].unique()
                    ColorPrint.info(f"DEBUG - Unique group_title in data: {list(unique_in_data)}")
                    ColorPrint.info(f"DEBUG - Group_title value counts:")
                    for group, count in self.final_data['group_title'].value_counts().items():
                        ColorPrint.info(f"  '{group}': {count} items")

                # Upload items group by group
                for group_title, group_id in self.group_mapping.items():
                    # Skip default "Group Title" groups
                    if group_title == 'Group Title':
                        continue

                    # Filter data for this group
                    if 'group_title' in self.final_data.columns:
                        ColorPrint.info(f"DEBUG - Filtering for group: '{group_title}' (group_id: {group_id})")
                        group_data = self.final_data[
                            self.final_data['group_title'] == group_title
                        ].copy()

                        ColorPrint.info(f"DEBUG - Filtered {len(group_data)} items for group '{group_title}'")

                        if not group_data.empty:
                            ColorPrint.info(f"  Uploading {len(group_data)} items to group '{group_title}'...")

                            # Prepare items for this group
                            items_to_create = self._prepare_monday_items(group_data)

                            # Upload items to this specific group
                            group_results = self.monday_client.create_items_batch(
                                board_id=self.board_id,
                                items=items_to_create,
                                group_id=group_id
                            )
                            results.extend(group_results)
                            ColorPrint.success(f"  ✓ Uploaded {len(group_results)} items to '{group_title}'")

                # Handle items from default "Group Title" group separately
                if 'group_title' in self.final_data.columns:
                    default_group_items = self.final_data[
                        self.final_data['group_title'] == 'Group Title'
                    ].copy()

                    if not default_group_items.empty:
                        ColorPrint.info(f"  Uploading {len(default_group_items)} items from default 'Group Title' group...")
                        items_to_create = self._prepare_monday_items(default_group_items)

                        # Upload without specifying group (will go to default group)
                        default_results = self.monday_client.create_items_batch(
                            board_id=self.board_id,
                            items=items_to_create,
                            group_id=None
                        )
                        results.extend(default_results)
                        ColorPrint.success(f"  ✓ Uploaded {len(default_results)} items to default group")

            else:
                # Standard upload (PDF sources or no groups)
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
            # Create item name from contract number only (column "Élément" in Monday.com)
            contract_num = str(row.get('contract_number', 'N/A'))
            item_name = contract_num

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


def create_monday_legacy_config(
    api_key: str,
    source_board_id: int,
    target_board_name: str,
    source_group_id: Optional[str] = None,
    month_group: Optional[str] = None
) -> PipelineConfig:
    """
    Create configuration for Monday.com legacy board conversion.

    Args:
        api_key: Monday.com API key
        source_board_id: ID of the legacy board to convert FROM
        target_board_name: Name of the new board to create
        source_group_id: Optional group ID to filter extraction
        month_group: Optional month group for the new board

    Returns:
        PipelineConfig for MONDAY_LEGACY conversion
    """
    return PipelineConfig(
        source=InsuranceSource.MONDAY_LEGACY,
        pdf_path=None,  # Not needed for Monday.com source
        month_group=month_group,
        board_name=target_board_name,
        monday_api_key=api_key,
        output_dir="./results/monday_legacy",
        source_board_id=source_board_id,
        source_group_id=source_group_id,
        reuse_board=True,
        reuse_group=True
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