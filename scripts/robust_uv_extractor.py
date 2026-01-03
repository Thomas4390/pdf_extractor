"""
Robust UV Assurance Extractor

This module extends the RobustExtractorBase to provide robust extraction
for UV Assurance remuneration reports with automatic fallback to the
legacy method when needed.

Author: Thomas
Date: 2025-01
"""

import pdfplumber
import pandas as pd
import re
from typing import Dict, List, Optional, Any

from robust_extractor_base import (
    RobustExtractorBase,
    ExtractionResult,
    ExtractionMethod,
    ColumnPositionAnalyzer
)
from uv_extractor import RemunerationReportExtractor


class RobustUVExtractor(RobustExtractorBase):
    """
    Robust extractor for UV Assurance remuneration reports.

    Uses position-based table extraction with the legacy token-based
    method as verification and fallback.
    """

    # Expected columns in order
    EXPECTED_COLUMNS = [
        'Contrat',
        'Assuré(s)',
        'Protection',
        'Montant de base',
        'Taux de partage',
        'Taux de commission',
        'Résultat',
        'Type',
        'Taux de Boni',
        'Rémunération'
    ]

    # Contract number pattern
    CONTRACT_PATTERN = re.compile(r'^110\d{6}$')

    def __init__(self, pdf_path: str, debug: bool = False):
        """
        Initialize the robust UV extractor.

        Args:
            pdf_path: Path to the PDF file
            debug: Enable debug output
        """
        super().__init__(pdf_path, debug)
        self._legacy_extractor = RemunerationReportExtractor(pdf_path)

    def _extract_robust(self) -> ExtractionResult:
        """
        Extract data using robust position-based methods.

        Uses pdfplumber's table detection with multiple strategies
        and validates each row based on expected patterns.
        """
        self._log("Starting robust extraction for UV...")

        all_rows = []
        warnings = []
        errors = []
        metadata = {
            'pages_processed': 0,
            'tables_found': 0,
            'rows_extracted': 0
        }

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                metadata['total_pages'] = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages, start=1):
                    self._log(f"Processing page {page_num}...")

                    # Try multiple table extraction strategies
                    page_rows = self._extract_table_robust(page, page_num)

                    if page_rows:
                        metadata['tables_found'] += 1
                        all_rows.extend(page_rows)

                    metadata['pages_processed'] += 1

                metadata['rows_extracted'] = len(all_rows)

        except Exception as e:
            errors.append(f"Error reading PDF: {str(e)}")
            return ExtractionResult(
                data=pd.DataFrame(),
                method=ExtractionMethod.ROBUST,
                quality_score=0.0,
                metadata=metadata,
                warnings=warnings,
                errors=errors
            )

        if not all_rows:
            warnings.append("No data rows extracted")
            return ExtractionResult(
                data=pd.DataFrame(),
                method=ExtractionMethod.ROBUST,
                quality_score=0.0,
                metadata=metadata,
                warnings=warnings,
                errors=errors
            )

        # Create DataFrame
        df = pd.DataFrame(all_rows, columns=self.EXPECTED_COLUMNS)

        # Fill missing contract numbers (continuation rows)
        df = self._fill_contract_numbers(df)

        # Clean the data
        df = self._clean_dataframe(df)

        self._log(f"Robust extraction complete: {len(df)} rows")

        return ExtractionResult(
            data=df,
            method=ExtractionMethod.ROBUST,
            quality_score=0.0,  # Will be calculated later
            metadata=metadata,
            warnings=warnings,
            errors=errors
        )

    def _extract_table_robust(self, page, page_num: int) -> List[List[str]]:
        """
        Extract table from a single page using robust methods.

        Tries multiple extraction strategies and validates results.

        Args:
            page: pdfplumber page object
            page_num: Page number for logging

        Returns:
            List of valid data rows
        """
        valid_rows = []

        # Strategy 1: Use line-based detection (most accurate for UV reports)
        settings_priority = [
            {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_tolerance": 3,
                "snap_tolerance": 3,
            },
            {
                "vertical_strategy": "lines",
                "horizontal_strategy": "text",
                "intersection_tolerance": 5,
            },
            {
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "snap_tolerance": 3,
            }
        ]

        for i, settings in enumerate(settings_priority):
            try:
                table = page.extract_table(table_settings=settings)

                if table and len(table) > 0:
                    self._log(f"  Strategy {i+1}: Found table with {len(table)} rows")

                    for row in table:
                        if self._is_valid_data_row(row):
                            # Normalize row to expected column count
                            normalized = self._normalize_row(row)
                            if normalized:
                                valid_rows.append(normalized)

                    if valid_rows:
                        self._log(f"  Extracted {len(valid_rows)} valid rows with strategy {i+1}")
                        return valid_rows

            except Exception as e:
                self._log(f"  Strategy {i+1} failed: {e}")
                continue

        # Strategy 4: Use words and position analysis as fallback
        if not valid_rows:
            valid_rows = self._extract_by_position(page)

        return valid_rows

    def _extract_by_position(self, page) -> List[List[str]]:
        """
        Extract table data using word positions.

        This is a more robust fallback that analyzes word positions
        to reconstruct table structure.

        Args:
            page: pdfplumber page object

        Returns:
            List of extracted rows
        """
        self._log("  Using position-based extraction...")

        words = page.extract_words(
            keep_blank_chars=True,
            x_tolerance=3,
            y_tolerance=3
        )

        if not words:
            return []

        # Find header row to determine column positions
        header_words = self._find_header_row(words)
        if not header_words:
            self._log("  Could not find header row")
            return []

        # Create column analyzer
        analyzer = ColumnPositionAnalyzer(words, page.width)

        # Detect column boundaries from header
        header_top = min(w['top'] for w in header_words)
        header_bottom = max(w['bottom'] for w in header_words)
        columns = analyzer.detect_column_boundaries(header_top, header_bottom, min_gap=10)

        if len(columns) < 5:
            self._log(f"  Only detected {len(columns)} columns, expected 10")
            return []

        # Extract data rows (below header)
        data_words = [w for w in words if w['top'] > header_bottom + 5]
        rows = analyzer.extract_rows_by_position(data_words, row_height_threshold=12)

        valid_rows = []
        for row in rows:
            if self._is_valid_data_row(row):
                normalized = self._normalize_row(row)
                if normalized:
                    valid_rows.append(normalized)

        return valid_rows

    def _find_header_row(self, words: List[Dict]) -> List[Dict]:
        """
        Find the header row in the words list.

        Args:
            words: List of word dictionaries

        Returns:
            List of words that form the header row
        """
        # Look for key header words
        header_keywords = {'contrat', 'assuré', 'protection', 'rémunération'}

        # Group words by y position
        y_groups = {}
        for word in words:
            y_key = round(word['top'] / 10) * 10  # Group by 10-pixel bands
            if y_key not in y_groups:
                y_groups[y_key] = []
            y_groups[y_key].append(word)

        # Find the group with most header keywords
        best_group = None
        best_count = 0

        for y_key, group_words in y_groups.items():
            text = ' '.join(w['text'].lower() for w in group_words)
            count = sum(1 for kw in header_keywords if kw in text)
            if count > best_count:
                best_count = count
                best_group = group_words

        return best_group if best_count >= 2 else None

    def _is_valid_data_row(self, row: List) -> bool:
        """
        Check if a row contains valid data.

        Valid rows either:
        1. Start with a contract number (110xxxxxx)
        2. Are continuation rows (empty first cell, name in second)

        Args:
            row: List of cell values

        Returns:
            True if row is valid data
        """
        if not row or len(row) < 5:
            return False

        if not any(cell for cell in row if cell):
            return False

        first_cell = str(row[0]).strip() if row[0] else ''
        second_cell = str(row[1]).strip() if len(row) > 1 and row[1] else ''

        # Check for header keywords
        row_text = ' '.join(str(c).lower() for c in row if c)
        header_words = ['contrat', 'assuré', 'protection', 'montant', 'taux', 'total', 'page']
        if sum(1 for w in header_words if w in row_text) >= 2:
            return False

        # Contract number row
        if self.CONTRACT_PATTERN.match(first_cell):
            return True

        # Continuation row (same insured, additional protection)
        if not first_cell and second_cell and len(second_cell) > 2:
            # Should have some numeric data in later columns
            has_numbers = any(
                re.search(r'\d', str(c)) for c in row[3:] if c
            )
            return has_numbers

        return False

    def _normalize_row(self, row: List) -> Optional[List[str]]:
        """
        Normalize a row to have exactly 10 columns.

        Args:
            row: Input row of any length

        Returns:
            Row with 10 columns, or None if invalid
        """
        if not row:
            return None

        # Clean cells
        cleaned = [str(cell).strip() if cell else '' for cell in row]

        # Pad or truncate to 10 columns
        if len(cleaned) < 10:
            cleaned.extend([''] * (10 - len(cleaned)))
        elif len(cleaned) > 10:
            cleaned = cleaned[:10]

        return cleaned

    def _fill_contract_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing contract numbers with the previous number.

        Args:
            df: DataFrame with potentially missing contract numbers

        Returns:
            DataFrame with contract numbers filled
        """
        if df.empty:
            return df

        df_filled = df.copy()
        df_filled['Contrat'] = df_filled['Contrat'].replace('', None).ffill()

        return df_filled

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and format the extracted DataFrame.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df

        df_clean = df.copy()

        # Clean numeric columns
        numeric_cols = ['Montant de base', 'Résultat', 'Rémunération']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(self._clean_amount)

        # Clean percentage columns
        pct_cols = ['Taux de partage', 'Taux de commission', 'Taux de Boni']
        for col in pct_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(self._clean_percentage)

        # Clean text columns
        text_cols = ['Contrat', 'Assuré(s)', 'Protection', 'Type']
        for col in text_cols:
            if col in df_clean.columns:
                df_clean[col] = (
                    df_clean[col]
                    .astype(str)
                    .str.replace('\n', ' ', regex=False)
                    .str.replace(r'\s+', ' ', regex=True)
                    .str.strip()
                )

        return df_clean

    def _clean_amount(self, value: Any) -> float:
        """Clean and convert a monetary amount to float."""
        if not value or pd.isna(value):
            return 0.0

        amount_clean = str(value).replace('$', '').replace(' ', '').replace(',', '.')

        try:
            return float(amount_clean)
        except ValueError:
            return 0.0

    def _clean_percentage(self, value: Any) -> float:
        """Clean and convert a percentage to float."""
        if not value or pd.isna(value):
            return 0.0

        pct_clean = str(value).replace('%', '').replace(' ', '').replace(',', '.')

        try:
            return float(pct_clean)
        except ValueError:
            return 0.0

    def _extract_legacy(self) -> ExtractionResult:
        """
        Extract data using the legacy method.

        Uses the original RemunerationReportExtractor as fallback.
        """
        self._log("Starting legacy extraction for UV...")

        try:
            data = self._legacy_extractor.extract_all()

            df = data.get('activites', pd.DataFrame())

            if df is None:
                df = pd.DataFrame()

            metadata = {
                'date': data.get('date'),
                'nom_conseiller': data.get('nom_conseiller'),
                'numero_conseiller': data.get('numero_conseiller')
            }

            self._log(f"Legacy extraction complete: {len(df)} rows")

            return ExtractionResult(
                data=df,
                method=ExtractionMethod.LEGACY,
                quality_score=0.0,
                metadata=metadata,
                warnings=[],
                errors=[]
            )

        except Exception as e:
            self._log(f"Legacy extraction failed: {e}")
            return ExtractionResult(
                data=pd.DataFrame(),
                method=ExtractionMethod.LEGACY,
                quality_score=0.0,
                metadata={},
                warnings=[],
                errors=[str(e)]
            )

    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate quality score for extracted data.

        Scoring based on:
        - Valid contract numbers (30%)
        - Non-empty required fields (30%)
        - Valid numeric values (20%)
        - Expected column count (20%)

        Args:
            df: Extracted DataFrame

        Returns:
            Quality score between 0.0 and 1.0
        """
        if df.empty:
            return 0.0

        scores = []

        # Score 1: Valid contract numbers (30%)
        if 'Contrat' in df.columns:
            valid_contracts = df['Contrat'].apply(
                lambda x: bool(self.CONTRACT_PATTERN.match(str(x).strip()))
            ).sum()
            contract_score = valid_contracts / len(df)
            scores.append(('contracts', contract_score, 0.3))

        # Score 2: Non-empty required fields (30%)
        required_fields = ['Assuré(s)', 'Protection', 'Rémunération']
        non_empty_counts = []
        for field in required_fields:
            if field in df.columns:
                non_empty = (df[field].notna() & (df[field] != '') & (df[field] != 0)).sum()
                non_empty_counts.append(non_empty / len(df))

        if non_empty_counts:
            required_score = sum(non_empty_counts) / len(non_empty_counts)
            scores.append(('required_fields', required_score, 0.3))

        # Score 3: Valid numeric values (20%)
        numeric_fields = ['Montant de base', 'Résultat', 'Rémunération']
        valid_numeric_counts = []
        for field in numeric_fields:
            if field in df.columns:
                try:
                    numeric_vals = pd.to_numeric(df[field], errors='coerce')
                    valid_count = numeric_vals.notna().sum()
                    valid_numeric_counts.append(valid_count / len(df))
                except:
                    pass

        if valid_numeric_counts:
            numeric_score = sum(valid_numeric_counts) / len(valid_numeric_counts)
            scores.append(('numeric', numeric_score, 0.2))

        # Score 4: Expected column count (20%)
        expected_cols = len(self.EXPECTED_COLUMNS)
        actual_cols = len(df.columns)
        col_score = 1.0 - abs(expected_cols - actual_cols) / expected_cols
        col_score = max(0, col_score)
        scores.append(('columns', col_score, 0.2))

        # Calculate weighted average
        if not scores:
            return 0.0

        total_score = sum(score * weight for _, score, weight in scores)
        total_weight = sum(weight for _, _, weight in scores)

        final_score = total_score / total_weight if total_weight > 0 else 0.0

        self._log(f"Quality score breakdown: {[(n, f'{s:.2f}') for n, s, _ in scores]}")
        self._log(f"Final quality score: {final_score:.2f}")

        return final_score

    def extract_with_metadata(self) -> Dict[str, Any]:
        """
        Extract data with full metadata (date, advisor info).

        Returns:
            Dictionary containing:
            - data: DataFrame with extracted data
            - method: Extraction method used
            - quality_score: Quality score
            - date: Report date
            - nom_conseiller: Advisor name
            - numero_conseiller: Advisor number
        """
        result = self.extract()

        # Get metadata from legacy extractor
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                first_page = pdf.pages[0]
                text = first_page.extract_text()

                date = self._legacy_extractor.extract_report_date(text)
                advisor_name = self._legacy_extractor.extract_advisor_name(text)
                advisor_number = self._legacy_extractor.extract_advisor_number(text)
        except:
            date = None
            advisor_name = None
            advisor_number = None

        return {
            'data': result.data,
            'method': result.method.value,
            'quality_score': result.quality_score,
            'date': date,
            'nom_conseiller': advisor_name,
            'numero_conseiller': advisor_number,
            'warnings': result.warnings,
            'errors': result.errors
        }


if __name__ == "__main__":
    # Test the robust extractor
    pdf_path = "../pdf/uv/rappportremun_21622_2025-10-20.pdf"

    print("=" * 80)
    print("TESTING ROBUST UV EXTRACTOR")
    print("=" * 80)

    extractor = RobustUVExtractor(pdf_path, debug=True)
    result = extractor.extract_with_metadata()

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Method used: {result['method']}")
    print(f"Quality score: {result['quality_score']:.2f}")
    print(f"Date: {result['date']}")
    print(f"Advisor: {result['nom_conseiller']}")
    print(f"Advisor number: {result['numero_conseiller']}")
    print(f"Rows extracted: {len(result['data'])}")

    if result['warnings']:
        print(f"\nWarnings: {result['warnings']}")
    if result['errors']:
        print(f"\nErrors: {result['errors']}")

    if not result['data'].empty:
        print("\n" + "-" * 80)
        print("SAMPLE DATA (first 5 rows):")
        print("-" * 80)
        print(result['data'].head().to_string())

        print(f"\nTotal Remuneration: {result['data']['Rémunération'].sum():,.2f} $")

    print("\n" + "=" * 80)
    print("EXTRACTION LOG:")
    print("=" * 80)
    for log_entry in extractor.get_extraction_log():
        print(f"  {log_entry}")
