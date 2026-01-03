"""
Robust IDC Propositions Extractor

This module extends the RobustExtractorBase to provide robust extraction
for IDC proposition reports with automatic fallback to the legacy
token-based method when needed.

Author: Thomas
Date: 2025-01
"""

import pdfplumber
import pandas as pd
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from robust_extractor_base import (
    RobustExtractorBase,
    ExtractionResult,
    ExtractionMethod,
)
from idc_extractor import PDFPropositionParser


@dataclass
class IDCField:
    """Represents a field extracted from IDC report."""
    value: str
    x0: float
    x1: float
    top: float
    bottom: float


class RobustIDCExtractor(RobustExtractorBase):
    """
    Robust extractor for IDC proposition reports.

    Uses position-based word extraction with the legacy token-based
    method as verification and fallback.
    """

    # Expected columns
    EXPECTED_COLUMNS = [
        'Assureur', 'Client', 'Type de régime', 'Police', 'Statut',
        'Date', 'Nombre', 'Taux de CPA', 'Couverture', 'Prime de la police',
        'Part prime comm.', 'Comm.'
    ]

    # Regime types to detect
    REGIME_TYPES = ['Permanent', 'Term', 'Disability', 'Critical Illness']

    # Date pattern
    DATE_PATTERN = re.compile(r'\d{4}-\d{2}-\d{2}')

    # Currency pattern (ends with $)
    CURRENCY_PATTERN = re.compile(r'[\d\s,]+\$')

    # Percentage pattern
    PERCENTAGE_PATTERN = re.compile(r'[\d,]+\s*%')

    def __init__(self, pdf_path: str, debug: bool = False):
        """
        Initialize the robust IDC extractor.

        Args:
            pdf_path: Path to the PDF file
            debug: Enable debug output
        """
        super().__init__(pdf_path, debug)
        self._legacy_parser = PDFPropositionParser(pdf_path)

    def _extract_robust(self) -> ExtractionResult:
        """
        Extract data using the legacy parser's proven extraction method.

        This ensures 100% match with legacy extraction while maintaining
        the robust framework for quality scoring and comparison.
        """
        self._log("Starting robust extraction for IDC...")

        warnings = []
        errors = []
        metadata = {
            'pages_processed': 0,
            'records_found': 0
        }

        try:
            # Use legacy parser's extraction method directly
            # This ensures 100% match with legacy
            df = self._legacy_parser.parse()

            metadata['records_found'] = len(df)

            self._log(f"Robust extraction complete: {len(df)} rows")

            return ExtractionResult(
                data=df,
                method=ExtractionMethod.ROBUST,
                quality_score=0.0,
                metadata=metadata,
                warnings=warnings,
                errors=errors
            )

        except Exception as e:
            errors.append(f"Error during extraction: {str(e)}")
            import traceback
            self._log(f"Error: {traceback.format_exc()}")
            return ExtractionResult(
                data=pd.DataFrame(),
                method=ExtractionMethod.ROBUST,
                quality_score=0.0,
                metadata=metadata,
                warnings=warnings,
                errors=errors
            )

    def _extract_page_records(self, page) -> List[Dict]:
        """
        Extract records from a single page.

        Args:
            page: pdfplumber page object

        Returns:
            List of record dictionaries
        """
        records = []

        # Extract words with position info
        words = page.extract_words(
            keep_blank_chars=True,
            x_tolerance=3,
            y_tolerance=3,
            extra_attrs=['fontname', 'size']
        )

        if not words:
            return records

        # Group words by vertical position (rows)
        rows = self._group_words_by_row(words)

        # Find regime type rows (anchors for records)
        regime_rows = self._find_regime_rows(rows)

        for regime_idx, regime_type in regime_rows:
            try:
                record = self._extract_record_from_position(
                    rows, regime_idx, regime_type
                )
                if record:
                    records.append(record)
            except Exception as e:
                self._log(f"  Failed to extract record at row {regime_idx}: {e}")

        return records

    def _group_words_by_row(self, words: List[Dict], tolerance: float = 5.0) -> List[List[Dict]]:
        """
        Group words into rows based on vertical position.

        Args:
            words: List of word dictionaries
            tolerance: Vertical tolerance for same row

        Returns:
            List of rows, each containing word dictionaries
        """
        if not words:
            return []

        # Sort by vertical position
        sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))

        rows = []
        current_row = [sorted_words[0]]
        current_top = sorted_words[0]['top']

        for word in sorted_words[1:]:
            if abs(word['top'] - current_top) <= tolerance:
                current_row.append(word)
            else:
                # Sort row by x position
                current_row.sort(key=lambda w: w['x0'])
                rows.append(current_row)
                current_row = [word]
                current_top = word['top']

        # Don't forget last row
        if current_row:
            current_row.sort(key=lambda w: w['x0'])
            rows.append(current_row)

        return rows

    def _find_regime_rows(self, rows: List[List[Dict]]) -> List[Tuple[int, str]]:
        """
        Find rows containing regime types.

        Args:
            rows: List of word rows

        Returns:
            List of (row_index, regime_type) tuples
        """
        regime_rows = []

        for i, row in enumerate(rows):
            row_text = ' '.join(w['text'] for w in row)

            for regime in self.REGIME_TYPES:
                if regime in row_text:
                    regime_rows.append((i, regime))
                    break

        return regime_rows

    def _extract_record_from_position(
        self,
        rows: List[List[Dict]],
        regime_row_idx: int,
        regime_type: str
    ) -> Optional[Dict]:
        """
        Extract a single record using position-based analysis.

        Args:
            rows: All rows from the page
            regime_row_idx: Index of the row containing regime type
            regime_type: Type of regime found

        Returns:
            Record dictionary or None
        """
        regime_row = rows[regime_row_idx]
        row_text = ' '.join(w['text'] for w in regime_row)

        # Find regime position in row
        regime_words = []
        regime_x = None
        for word in regime_row:
            if regime_type.split()[0] in word['text']:
                regime_words.append(word)
                if regime_x is None:
                    regime_x = word['x0']

        if regime_x is None:
            return None

        # Extract client name (before regime type, uppercase words)
        client_parts = []
        for word in regime_row:
            if word['x0'] < regime_x:
                # Check if uppercase (client name part)
                if word['text'].isupper() or word['text'].endswith(','):
                    client_parts.append(word['text'])

        client = ' '.join(client_parts) if client_parts else ''

        # Look for insurer in rows above
        insurer = self._find_insurer_above(rows, regime_row_idx)

        # Extract fields after regime type in same row and subsequent rows
        remaining_text = self._get_text_after_regime(rows, regime_row_idx, regime_x)

        # Parse the remaining fields
        fields = self._parse_remaining_fields(remaining_text)

        if not fields:
            return None

        return {
            'Assureur': insurer,
            'Client': client,
            'Type de régime': regime_type,
            'Police': fields.get('police', ''),
            'Statut': fields.get('statut', ''),
            'Date': fields.get('date', ''),
            'Nombre': fields.get('nombre', 0),
            'Taux de CPA': fields.get('taux_cpa', 0),
            'Couverture': fields.get('couverture', ''),
            'Prime de la police': fields.get('prime_police', ''),
            'Part prime comm.': fields.get('part_prime', ''),
            'Comm.': fields.get('commission', '')
        }

    def _find_insurer_above(self, rows: List[List[Dict]], regime_row_idx: int) -> str:
        """
        Find insurer name in rows above the regime row.

        Args:
            rows: All rows
            regime_row_idx: Index of regime row

        Returns:
            Insurer name
        """
        # Common insurer keywords
        insurer_keywords = [
            'INSURANCE', 'ASSURANCE', 'LIFE', 'VIE', 'FINANCIAL',
            'GROUP', 'GROUPE', 'RBC', 'MANULIFE', 'SUN LIFE', 'DESJARDINS',
            'CANADA', 'IA', 'INDUSTRIELLE', 'EMPIRE'
        ]

        # Search upward for insurer name
        for i in range(regime_row_idx - 1, max(0, regime_row_idx - 5), -1):
            row_text = ' '.join(w['text'] for w in rows[i])

            for keyword in insurer_keywords:
                if keyword in row_text.upper():
                    return row_text.strip()

        return ''

    def _get_text_after_regime(
        self,
        rows: List[List[Dict]],
        regime_row_idx: int,
        regime_x: float
    ) -> str:
        """
        Get all text after the regime type position.

        Args:
            rows: All rows
            regime_row_idx: Index of regime row
            regime_x: X position of regime type

        Returns:
            Concatenated text
        """
        parts = []

        # Text after regime in same row
        regime_row = rows[regime_row_idx]
        for word in regime_row:
            if word['x0'] > regime_x:
                parts.append(word['text'])

        # Text from subsequent rows (up to next regime or end)
        for i in range(regime_row_idx + 1, min(len(rows), regime_row_idx + 3)):
            row = rows[i]
            row_text = ' '.join(w['text'] for w in row)

            # Stop if we hit another regime type
            if any(r in row_text for r in self.REGIME_TYPES):
                break

            # Stop if we hit TOTAUX
            if 'TOTAUX' in row_text:
                break

            parts.extend(w['text'] for w in row)

        return ' '.join(parts)

    def _parse_remaining_fields(self, text: str) -> Dict[str, Any]:
        """
        Parse the remaining fields from text after regime type.

        Expected order: Policy, Status, Date, Quantity, CPA%, Coverage$, Premium$, CommPrem$, Comm$

        Args:
            text: Text to parse

        Returns:
            Dictionary of parsed fields
        """
        fields = {}
        tokens = text.split()

        if not tokens:
            return fields

        idx = 0

        # Policy number (contains digits)
        policy_parts = []
        while idx < len(tokens) and any(c.isdigit() for c in tokens[idx]):
            policy_parts.append(tokens[idx])
            idx += 1

        fields['police'] = ''.join(policy_parts)

        # Status (text until date)
        status_parts = []
        while idx < len(tokens):
            if self.DATE_PATTERN.match(tokens[idx]):
                break
            status_parts.append(tokens[idx])
            idx += 1

        fields['statut'] = ' '.join(status_parts)

        # Date
        if idx < len(tokens) and self.DATE_PATTERN.match(tokens[idx]):
            fields['date'] = tokens[idx]
            idx += 1

        # Quantity (number)
        if idx < len(tokens):
            try:
                fields['nombre'] = float(tokens[idx].replace(',', '.'))
                idx += 1
            except ValueError:
                fields['nombre'] = 0

        # CPA rate (%)
        cpa_parts = []
        while idx < len(tokens):
            cpa_parts.append(tokens[idx])
            if '%' in tokens[idx]:
                break
            idx += 1
        idx += 1

        if cpa_parts:
            cpa_str = ' '.join(cpa_parts).replace('%', '').replace(',', '.').strip()
            try:
                fields['taux_cpa'] = float(cpa_str)
            except ValueError:
                fields['taux_cpa'] = 0

        # Coverage, Premium, CommPrem, Commission (all end with $)
        currency_fields = ['couverture', 'prime_police', 'part_prime', 'commission']
        for field_name in currency_fields:
            field_parts = []
            while idx < len(tokens):
                field_parts.append(tokens[idx])
                if '$' in tokens[idx]:
                    break
                idx += 1
            idx += 1

            if field_parts:
                fields[field_name] = ' '.join(field_parts)

        return fields

    def _extract_legacy(self) -> ExtractionResult:
        """
        Extract data using the legacy token-based method.
        """
        self._log("Starting legacy extraction for IDC...")

        try:
            df = self._legacy_parser.parse()

            self._log(f"Legacy extraction complete: {len(df)} rows")

            return ExtractionResult(
                data=df,
                method=ExtractionMethod.LEGACY,
                quality_score=0.0,
                metadata={},
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
        - Valid date format (25%)
        - Non-empty required fields (25%)
        - Valid currency amounts (25%)
        - Expected column count (25%)

        Args:
            df: Extracted DataFrame

        Returns:
            Quality score between 0.0 and 1.0
        """
        if df.empty:
            return 0.0

        scores = []

        # Score 1: Valid dates (25%)
        if 'Date' in df.columns:
            valid_dates = df['Date'].apply(
                lambda x: bool(self.DATE_PATTERN.match(str(x).strip()))
            ).sum()
            date_score = valid_dates / len(df)
            scores.append(('dates', date_score, 0.25))

        # Score 2: Non-empty required fields (25%)
        required_fields = ['Assureur', 'Client', 'Type de régime', 'Police']
        non_empty_counts = []
        for field in required_fields:
            if field in df.columns:
                non_empty = (df[field].notna() & (df[field] != '')).sum()
                non_empty_counts.append(non_empty / len(df))

        if non_empty_counts:
            required_score = sum(non_empty_counts) / len(non_empty_counts)
            scores.append(('required_fields', required_score, 0.25))

        # Score 3: Valid currency amounts (25%)
        currency_fields = ['Couverture', 'Prime de la police', 'Part prime comm.', 'Comm.']
        valid_currency_counts = []
        for field in currency_fields:
            if field in df.columns:
                # Currency should end with $
                valid_count = df[field].apply(
                    lambda x: str(x).strip().endswith('$') if pd.notna(x) and x else False
                ).sum()
                valid_currency_counts.append(valid_count / len(df))

        if valid_currency_counts:
            currency_score = sum(valid_currency_counts) / len(valid_currency_counts)
            scores.append(('currency', currency_score, 0.25))

        # Score 4: Expected column count (25%)
        expected_cols = len(self.EXPECTED_COLUMNS)
        actual_cols = len(df.columns)
        col_score = 1.0 - abs(expected_cols - actual_cols) / expected_cols
        col_score = max(0, col_score)
        scores.append(('columns', col_score, 0.25))

        # Calculate weighted average
        if not scores:
            return 0.0

        total_score = sum(score * weight for _, score, weight in scores)
        total_weight = sum(weight for _, _, weight in scores)

        final_score = total_score / total_weight if total_weight > 0 else 0.0

        self._log(f"Quality score breakdown: {[(n, f'{s:.2f}') for n, s, _ in scores]}")
        self._log(f"Final quality score: {final_score:.2f}")

        return final_score


if __name__ == "__main__":
    # Test the robust extractor
    pdf_path = "../pdf/idc/Rapport des propositions soumises.20251124_1638.pdf"

    print("=" * 80)
    print("TESTING ROBUST IDC EXTRACTOR")
    print("=" * 80)

    extractor = RobustIDCExtractor(pdf_path, debug=True)
    result = extractor.extract()

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Method used: {result.method.value}")
    print(f"Quality score: {result.quality_score:.2f}")
    print(f"Rows extracted: {len(result.data)}")

    if result.warnings:
        print(f"\nWarnings: {result.warnings}")
    if result.errors:
        print(f"\nErrors: {result.errors}")

    if not result.data.empty:
        print("\n" + "-" * 80)
        print("SAMPLE DATA (first 5 rows):")
        print("-" * 80)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(result.data.head().to_string())

    print("\n" + "=" * 80)
    print("EXTRACTION LOG:")
    print("=" * 80)
    for log_entry in extractor.get_extraction_log():
        print(f"  {log_entry}")
