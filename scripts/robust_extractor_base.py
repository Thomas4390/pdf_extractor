"""
Robust PDF Extractor Base Class

This module provides a base architecture for robust PDF extraction with fallback
to legacy methods when the robust extraction fails or produces inconsistent results.

Architecture:
1. Robust extraction (position-based, table-aware)
2. Legacy extraction (original method)
3. Comparison and quality scoring
4. Automatic fallback if quality is insufficient

Author: Thomas
Date: 2025-01
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import pandas as pd
import numpy as np


class ExtractionMethod(Enum):
    """Enum for extraction methods used."""
    ROBUST = "robust"
    LEGACY = "legacy"
    FALLBACK = "fallback"


@dataclass
class ExtractionResult:
    """Result of an extraction attempt."""
    data: pd.DataFrame
    method: ExtractionMethod
    quality_score: float  # 0.0 to 1.0
    metadata: Dict[str, Any]
    warnings: List[str]
    errors: List[str]


@dataclass
class ComparisonResult:
    """Result of comparing two extraction methods."""
    match_score: float  # 0.0 to 1.0
    row_count_match: bool
    column_match: bool
    value_differences: List[Dict]
    recommendation: str  # 'use_robust', 'use_legacy', 'manual_review'


class RobustExtractorBase(ABC):
    """
    Base class for robust PDF extraction with fallback capability.

    Subclasses must implement:
    - _extract_robust(): Position-based extraction
    - _extract_legacy(): Original extraction method
    - _calculate_quality_score(): Quality assessment
    """

    # Quality threshold for accepting robust extraction
    QUALITY_THRESHOLD = 0.8

    # Match threshold for comparing robust vs legacy
    MATCH_THRESHOLD = 0.9

    def __init__(self, pdf_path: str, debug: bool = False):
        """
        Initialize the extractor.

        Args:
            pdf_path: Path to the PDF file
            debug: Enable debug output
        """
        self.pdf_path = pdf_path
        self.debug = debug
        self._extraction_log: List[str] = []

    def _log(self, message: str):
        """Log a message (for debugging)."""
        self._extraction_log.append(message)
        if self.debug:
            print(f"  [DEBUG] {message}")

    @abstractmethod
    def _extract_robust(self) -> ExtractionResult:
        """
        Perform robust extraction using position-based methods.

        This method should use table detection, column position analysis,
        and other structural information from the PDF.

        Returns:
            ExtractionResult with extracted data
        """
        pass

    @abstractmethod
    def _extract_legacy(self) -> ExtractionResult:
        """
        Perform extraction using the legacy method.

        This is the original extraction method that serves as:
        1. A fallback when robust extraction fails
        2. A verification baseline for robust extraction

        Returns:
            ExtractionResult with extracted data
        """
        pass

    @abstractmethod
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate quality score for extracted data.

        Quality indicators may include:
        - Percentage of non-null values in key columns
        - Valid format for contract numbers, dates, amounts
        - Expected column count and types

        Args:
            df: Extracted DataFrame

        Returns:
            Quality score between 0.0 and 1.0
        """
        pass

    def _compare_extractions(
        self,
        robust: ExtractionResult,
        legacy: ExtractionResult
    ) -> ComparisonResult:
        """
        Compare robust and legacy extraction results.

        Args:
            robust: Result from robust extraction
            legacy: Result from legacy extraction

        Returns:
            ComparisonResult with match analysis
        """
        df_robust = robust.data
        df_legacy = legacy.data

        # Handle empty DataFrames
        if df_robust.empty and df_legacy.empty:
            return ComparisonResult(
                match_score=1.0,
                row_count_match=True,
                column_match=True,
                value_differences=[],
                recommendation='use_robust'
            )

        if df_robust.empty or df_legacy.empty:
            return ComparisonResult(
                match_score=0.0,
                row_count_match=False,
                column_match=False,
                value_differences=[{'issue': 'One extraction is empty'}],
                recommendation='use_legacy' if df_robust.empty else 'use_robust'
            )

        # Compare row counts
        row_count_match = len(df_robust) == len(df_legacy)
        row_count_diff = abs(len(df_robust) - len(df_legacy))
        row_score = 1.0 - min(row_count_diff / max(len(df_robust), len(df_legacy), 1), 1.0)

        # Compare columns
        robust_cols = set(df_robust.columns)
        legacy_cols = set(df_legacy.columns)
        common_cols = robust_cols & legacy_cols
        column_match = robust_cols == legacy_cols
        col_score = len(common_cols) / max(len(robust_cols | legacy_cols), 1)

        # Compare values in common columns (if row counts match)
        value_differences = []
        value_score = 1.0

        if row_count_match and common_cols:
            total_cells = 0
            matching_cells = 0

            for col in common_cols:
                for idx in range(min(len(df_robust), len(df_legacy))):
                    total_cells += 1
                    val_r = df_robust.iloc[idx][col]
                    val_l = df_legacy.iloc[idx][col]

                    # Handle NaN comparison
                    if pd.isna(val_r) and pd.isna(val_l):
                        matching_cells += 1
                    elif self._values_match(val_r, val_l):
                        matching_cells += 1
                    else:
                        if len(value_differences) < 10:  # Limit stored differences
                            value_differences.append({
                                'row': idx,
                                'column': col,
                                'robust': val_r,
                                'legacy': val_l
                            })

            value_score = matching_cells / max(total_cells, 1)

        # Calculate overall match score
        match_score = (row_score * 0.3 + col_score * 0.2 + value_score * 0.5)

        # Determine recommendation
        if match_score >= self.MATCH_THRESHOLD:
            recommendation = 'use_robust'
        elif robust.quality_score > legacy.quality_score + 0.1:
            recommendation = 'use_robust'
        elif legacy.quality_score > robust.quality_score + 0.1:
            recommendation = 'use_legacy'
        else:
            recommendation = 'manual_review'

        return ComparisonResult(
            match_score=match_score,
            row_count_match=row_count_match,
            column_match=column_match,
            value_differences=value_differences,
            recommendation=recommendation
        )

    def _values_match(self, val1: Any, val2: Any, tolerance: float = 0.01) -> bool:
        """
        Check if two values match (with tolerance for numeric values).

        Args:
            val1: First value
            val2: Second value
            tolerance: Tolerance for numeric comparison

        Returns:
            True if values match
        """
        # Handle None/NaN
        if pd.isna(val1) and pd.isna(val2):
            return True
        if pd.isna(val1) or pd.isna(val2):
            return False

        # Numeric comparison with tolerance
        try:
            num1 = float(val1)
            num2 = float(val2)
            return abs(num1 - num2) <= tolerance * max(abs(num1), abs(num2), 1)
        except (ValueError, TypeError):
            pass

        # String comparison (case-insensitive, whitespace-normalized)
        str1 = str(val1).strip().lower()
        str2 = str(val2).strip().lower()
        return str1 == str2

    def extract(self, use_fallback: bool = True) -> ExtractionResult:
        """
        Main extraction method with automatic fallback.

        Process:
        1. Try robust extraction
        2. Calculate quality score
        3. If quality is low or use_fallback is True, also try legacy
        4. Compare results and choose best method
        5. Return final result with method indicator

        Args:
            use_fallback: Whether to compare with legacy and potentially fallback

        Returns:
            ExtractionResult with the best extraction
        """
        self._log(f"Starting extraction for: {self.pdf_path}")

        # Step 1: Try robust extraction
        self._log("Attempting robust extraction...")
        try:
            robust_result = self._extract_robust()
            robust_result.quality_score = self._calculate_quality_score(robust_result.data)
            self._log(f"Robust extraction: {len(robust_result.data)} rows, quality={robust_result.quality_score:.2f}")
        except Exception as e:
            self._log(f"Robust extraction failed: {e}")
            robust_result = ExtractionResult(
                data=pd.DataFrame(),
                method=ExtractionMethod.ROBUST,
                quality_score=0.0,
                metadata={},
                warnings=[],
                errors=[str(e)]
            )

        # If no fallback requested and quality is acceptable, return robust
        if not use_fallback and robust_result.quality_score >= self.QUALITY_THRESHOLD:
            return robust_result

        # Step 2: Try legacy extraction for comparison
        self._log("Attempting legacy extraction for comparison...")
        try:
            legacy_result = self._extract_legacy()
            legacy_result.quality_score = self._calculate_quality_score(legacy_result.data)
            self._log(f"Legacy extraction: {len(legacy_result.data)} rows, quality={legacy_result.quality_score:.2f}")
        except Exception as e:
            self._log(f"Legacy extraction failed: {e}")
            legacy_result = ExtractionResult(
                data=pd.DataFrame(),
                method=ExtractionMethod.LEGACY,
                quality_score=0.0,
                metadata={},
                warnings=[],
                errors=[str(e)]
            )

        # Step 3: Compare and decide
        if robust_result.data.empty and legacy_result.data.empty:
            self._log("Both extractions failed!")
            return ExtractionResult(
                data=pd.DataFrame(),
                method=ExtractionMethod.FALLBACK,
                quality_score=0.0,
                metadata={'log': self._extraction_log},
                warnings=['Both extraction methods failed'],
                errors=robust_result.errors + legacy_result.errors
            )

        if robust_result.data.empty:
            self._log("Using legacy (robust failed)")
            legacy_result.method = ExtractionMethod.FALLBACK
            return legacy_result

        if legacy_result.data.empty:
            self._log("Using robust (legacy failed)")
            return robust_result

        # Compare extractions
        comparison = self._compare_extractions(robust_result, legacy_result)
        self._log(f"Comparison: match_score={comparison.match_score:.2f}, recommendation={comparison.recommendation}")

        # Decide which to use
        if comparison.recommendation == 'use_robust':
            self._log("✓ Using robust extraction")
            robust_result.metadata['comparison'] = {
                'match_score': comparison.match_score,
                'legacy_rows': len(legacy_result.data),
                'value_differences': len(comparison.value_differences)
            }
            return robust_result
        elif comparison.recommendation == 'use_legacy':
            self._log("⚠ Falling back to legacy extraction")
            legacy_result.method = ExtractionMethod.FALLBACK
            legacy_result.metadata['comparison'] = {
                'match_score': comparison.match_score,
                'robust_rows': len(robust_result.data),
                'reason': 'legacy_higher_quality'
            }
            return legacy_result
        else:
            # Manual review needed - use the one with higher quality
            self._log("⚠ Results differ significantly, using higher quality extraction")
            if robust_result.quality_score >= legacy_result.quality_score:
                robust_result.warnings.append('Results differ from legacy - manual review recommended')
                robust_result.metadata['comparison'] = {
                    'match_score': comparison.match_score,
                    'value_differences': comparison.value_differences[:5]
                }
                return robust_result
            else:
                legacy_result.method = ExtractionMethod.FALLBACK
                legacy_result.warnings.append('Robust extraction differed - using legacy')
                return legacy_result

    def get_extraction_log(self) -> List[str]:
        """Get the extraction log for debugging."""
        return self._extraction_log.copy()


class ColumnPositionAnalyzer:
    """
    Helper class to analyze and detect column positions in PDF tables.

    This helps create more robust extraction by understanding the
    table structure rather than relying on text parsing alone.
    """

    def __init__(self, words: List[Dict], page_width: float):
        """
        Initialize with words extracted from a PDF page.

        Args:
            words: List of word dictionaries from pdfplumber (with x0, x1, top, bottom, text)
            page_width: Width of the PDF page
        """
        self.words = words
        self.page_width = page_width
        self._column_boundaries: List[float] = []

    def detect_column_boundaries(
        self,
        header_row_top: float,
        header_row_bottom: float,
        min_gap: float = 5.0
    ) -> List[Tuple[float, float, str]]:
        """
        Detect column boundaries based on header row.

        Args:
            header_row_top: Top coordinate of header row
            header_row_bottom: Bottom coordinate of header row
            min_gap: Minimum gap between columns

        Returns:
            List of (x_start, x_end, header_text) tuples for each column
        """
        # Filter words in header row
        header_words = [
            w for w in self.words
            if w['top'] >= header_row_top and w['bottom'] <= header_row_bottom
        ]

        if not header_words:
            return []

        # Sort by x position
        header_words.sort(key=lambda w: w['x0'])

        # Group words into columns based on gaps
        columns = []
        current_col_words = [header_words[0]]

        for word in header_words[1:]:
            prev_word = current_col_words[-1]
            gap = word['x0'] - prev_word['x1']

            if gap > min_gap:
                # Start new column
                col_text = ' '.join(w['text'] for w in current_col_words)
                col_start = current_col_words[0]['x0']
                col_end = current_col_words[-1]['x1']
                columns.append((col_start, col_end, col_text))
                current_col_words = [word]
            else:
                current_col_words.append(word)

        # Don't forget last column
        if current_col_words:
            col_text = ' '.join(w['text'] for w in current_col_words)
            col_start = current_col_words[0]['x0']
            col_end = current_col_words[-1]['x1']
            columns.append((col_start, col_end, col_text))

        self._column_boundaries = [col[0] for col in columns] + [self.page_width]
        return columns

    def assign_word_to_column(self, word: Dict) -> int:
        """
        Assign a word to a column index based on its x position.

        Args:
            word: Word dictionary with 'x0' key

        Returns:
            Column index (0-based)
        """
        if not self._column_boundaries:
            return 0

        x_center = (word['x0'] + word.get('x1', word['x0'])) / 2

        for i, boundary in enumerate(self._column_boundaries[:-1]):
            next_boundary = self._column_boundaries[i + 1]
            if boundary <= x_center < next_boundary:
                return i

        return len(self._column_boundaries) - 2

    def extract_rows_by_position(
        self,
        words: List[Dict],
        row_height_threshold: float = 15.0
    ) -> List[List[str]]:
        """
        Group words into rows and columns based on position.

        Args:
            words: List of word dictionaries
            row_height_threshold: Maximum vertical distance for same row

        Returns:
            List of rows, where each row is a list of cell values
        """
        if not words or not self._column_boundaries:
            return []

        # Sort words by vertical position
        sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))

        # Group into rows
        rows = []
        current_row_words = []
        current_row_top = None

        for word in sorted_words:
            if current_row_top is None:
                current_row_top = word['top']
                current_row_words = [word]
            elif word['top'] - current_row_top <= row_height_threshold:
                current_row_words.append(word)
            else:
                # Process current row
                if current_row_words:
                    row_values = self._words_to_row(current_row_words)
                    rows.append(row_values)
                current_row_top = word['top']
                current_row_words = [word]

        # Process last row
        if current_row_words:
            row_values = self._words_to_row(current_row_words)
            rows.append(row_values)

        return rows

    def _words_to_row(self, words: List[Dict]) -> List[str]:
        """Convert a list of words to row values by column."""
        num_cols = len(self._column_boundaries) - 1
        cells = [[] for _ in range(num_cols)]

        for word in words:
            col_idx = self.assign_word_to_column(word)
            if 0 <= col_idx < num_cols:
                cells[col_idx].append(word['text'])

        return [' '.join(cell_words) for cell_words in cells]
