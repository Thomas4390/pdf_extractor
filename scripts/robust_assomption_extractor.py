"""
Robust Assomption Vie Extractor

This module extends the RobustExtractorBase to provide robust extraction
for Assomption Vie remuneration reports with automatic fallback to the
legacy method when needed.

Author: Thomas
Date: 2025-01
"""

import pdfplumber
import fitz  # PyMuPDF
import pandas as pd
import re
from typing import Dict, List, Optional, Any, Tuple

from robust_extractor_base import (
    RobustExtractorBase,
    ExtractionResult,
    ExtractionMethod,
)
from assomption_extractor import extract_pdf_data


class RobustAssomptionExtractor(RobustExtractorBase):
    """
    Robust extractor for Assomption Vie remuneration reports.

    Uses position-based extraction with PyMuPDF blocks and the
    legacy text-based method as verification and fallback.
    """

    # Expected columns
    EXPECTED_COLUMNS = [
        'Code', 'Numéro Police', "Nom de l'assuré", 'Produit',
        'Émission', 'Fréquence paiement', 'Facturation',
        'Prime', 'Taux Commission', 'Commissions', 'Taux Boni', 'Boni'
    ]

    # Transaction code pattern (e.g., AOH1, BCD2)
    TRANSACTION_CODE_PATTERN = re.compile(r'^[A-Z]{2,4}\d+$')

    # Policy number pattern (7 digits)
    POLICY_NUMBER_PATTERN = re.compile(r'^\d{7}$')

    # Date pattern (YYYY/MM/DD)
    DATE_PATTERN = re.compile(r'^\d{4}/\d{2}/\d{2}$')

    # Commission section keywords
    COMMISSION_KEYWORDS = ["ASSURANCE VIE INDIVIDUELLE", "Commissions de première année", "Numéro Police"]

    # Bonus section keywords
    BONUS_KEYWORDS = ["Surcommission sur la production", "Boni", "Polices"]

    def __init__(self, pdf_path: str, debug: bool = False):
        """
        Initialize the robust Assomption extractor.

        Args:
            pdf_path: Path to the PDF file
            debug: Enable debug output
        """
        super().__init__(pdf_path, debug)

    def _extract_robust(self) -> ExtractionResult:
        """
        Extract data using robust position-based methods.

        Uses PyMuPDF's block extraction for more reliable text positioning.
        """
        self._log("Starting robust extraction for Assomption...")

        warnings = []
        errors = []
        metadata = {
            'pages_processed': 0,
            'commission_records': 0,
            'bonus_records': 0
        }

        try:
            # Extract all pages text
            pages_text = self._extract_all_pages_text()
            metadata['total_pages'] = len(pages_text)

            # Find and extract commission data
            commission_section = self._find_section(pages_text, self.COMMISSION_KEYWORDS, "Commission")
            df_commissions = pd.DataFrame()

            if commission_section:
                page_num, page_text = commission_section
                self._log(f"Found commission section on page {page_num}")
                commission_data = self._parse_commission_data_robust(page_text)
                df_commissions = pd.DataFrame(commission_data)
                metadata['commission_records'] = len(df_commissions)
                self._log(f"Extracted {len(df_commissions)} commission records")

            # Find and extract bonus data
            bonus_section = self._find_section(pages_text, self.BONUS_KEYWORDS, "Bonus")
            df_bonus = pd.DataFrame()

            if bonus_section:
                page_num, page_text = bonus_section
                self._log(f"Found bonus section on page {page_num}")
                bonus_data = self._parse_bonus_data_robust(page_text)
                df_bonus = pd.DataFrame(bonus_data)
                metadata['bonus_records'] = len(df_bonus)
                self._log(f"Extracted {len(df_bonus)} bonus records")

            # Merge commission and bonus data
            df = self._merge_data(df_commissions, df_bonus)

            if df.empty:
                warnings.append("No data extracted")

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
            return ExtractionResult(
                data=pd.DataFrame(),
                method=ExtractionMethod.ROBUST,
                quality_score=0.0,
                metadata=metadata,
                warnings=warnings,
                errors=errors
            )

    def _extract_all_pages_text(self) -> Dict[int, str]:
        """
        Extract text from all PDF pages using PyMuPDF.

        Returns:
            Dictionary mapping page numbers (1-indexed) to text content
        """
        pages_text = {}

        doc = fitz.open(self.pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            pages_text[page_num + 1] = page.get_text()
        doc.close()

        return pages_text

    def _find_section(
        self,
        pages_text: Dict[int, str],
        keywords: List[str],
        section_name: str,
        min_matches: int = 2
    ) -> Optional[Tuple[int, str]]:
        """
        Find a section in the PDF based on keywords.

        Args:
            pages_text: Dictionary of page numbers to text
            keywords: Keywords to search for
            section_name: Name of section for logging
            min_matches: Minimum keywords to match

        Returns:
            Tuple of (page_number, text) or None
        """
        best_match = None
        best_count = 0

        for page_num, text in pages_text.items():
            text_lower = text.lower()
            match_count = sum(1 for kw in keywords if kw.lower() in text_lower)

            if match_count >= min_matches and match_count > best_count:
                best_match = (page_num, text)
                best_count = match_count

        if best_match:
            self._log(f"Found {section_name} section on page {best_match[0]} ({best_count}/{len(keywords)} keywords)")

        return best_match

    def _parse_commission_data_robust(self, text: str) -> List[Dict]:
        """
        Parse commission data using robust line-based extraction.

        Args:
            text: Raw text from commission page

        Returns:
            List of commission records
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        records = []

        # Find start of data section
        start_idx = None
        for idx, line in enumerate(lines):
            if idx < 20 and 'Code' in line:
                window = ' '.join(lines[idx:min(idx + 15, len(lines))]).lower()
                if 'numéro police' in window and 'commissions' in window:
                    # Find first transaction code
                    for j in range(idx, len(lines)):
                        if self.TRANSACTION_CODE_PATTERN.match(lines[j]):
                            start_idx = j
                            break
                    break

        if start_idx is None:
            return records

        # Process in blocks of 10
        i = start_idx
        while i < len(lines):
            line = lines[i]

            if "Total CPA" in line or "Total des commissions" in line:
                break

            if self.TRANSACTION_CODE_PATTERN.match(line):
                if i + 9 < len(lines):
                    try:
                        record = {
                            'Code': lines[i],
                            'Numéro Police': lines[i + 1],
                            "Nom de l'assuré": lines[i + 2],
                            'Produit': lines[i + 3],
                            'Émission': lines[i + 4],
                            'Fréquence paiement': lines[i + 5],
                            'Facturation': lines[i + 6],
                            'Prime': self._parse_float(lines[i + 7]),
                            'Taux Commission': lines[i + 8],
                            'Commissions': self._parse_float(lines[i + 9])
                        }
                        records.append(record)
                        i += 10
                    except (ValueError, IndexError) as e:
                        self._log(f"Error parsing commission at line {i}: {e}")
                        i += 1
                else:
                    break
            else:
                i += 1

        return records

    def _parse_bonus_data_robust(self, text: str) -> List[Dict]:
        """
        Parse bonus data using robust line-based extraction.

        Args:
            text: Raw text from bonus page

        Returns:
            List of bonus records
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        records = []

        # Find start of data section
        start_idx = None
        for idx, line in enumerate(lines):
            if idx + 10 < len(lines):
                window = ' '.join(lines[idx:idx + 15]).lower()
                if ('polices' in window or 'police' in window) and 'assurés' in window and 'boni' in window:
                    for j in range(idx, min(idx + 20, len(lines))):
                        if self.POLICY_NUMBER_PATTERN.match(lines[j]):
                            start_idx = j
                            break
                    if start_idx:
                        break

        if start_idx is None:
            return records

        # Process dynamically
        i = start_idx
        while i < len(lines):
            line = lines[i]

            if line == "Total" or ("Total" in line and i > start_idx + 3):
                break

            if self.POLICY_NUMBER_PATTERN.match(line):
                record = {'Numéro Police': line}
                i += 1

                # Name
                if i < len(lines):
                    record["Nom de l'assuré"] = lines[i]
                    i += 1

                # Product (1-2 lines)
                produit_parts = []
                if i < len(lines):
                    produit_parts.append(lines[i])
                    i += 1
                    if i < len(lines) and re.match(r'^[A-Z]$', lines[i]):
                        produit_parts.append(lines[i])
                        i += 1

                record['Produit'] = ' '.join(produit_parts)

                # First year commission
                if i < len(lines) and re.match(r'^-?\d+,\d+$', lines[i]):
                    record['Commissions Première Année'] = self._parse_float(lines[i])
                    i += 1

                # Bonus rate
                if i < len(lines) and '%' in lines[i]:
                    record['Taux Boni'] = lines[i]
                    i += 1

                # Bonus amount
                if i < len(lines) and re.match(r'^-?\d+,\d+$', lines[i]):
                    record['Boni'] = self._parse_float(lines[i])
                    i += 1

                if 'Boni' in record and 'Taux Boni' in record:
                    records.append(record)
            else:
                i += 1

        return records

    def _merge_data(self, df_commissions: pd.DataFrame, df_bonus: pd.DataFrame) -> pd.DataFrame:
        """
        Merge commission and bonus data.

        Args:
            df_commissions: Commission DataFrame
            df_bonus: Bonus DataFrame

        Returns:
            Merged DataFrame
        """
        if df_commissions.empty and df_bonus.empty:
            return pd.DataFrame()

        if df_commissions.empty:
            return df_bonus

        if df_bonus.empty:
            df_commissions['Taux Boni'] = None
            df_commissions['Boni'] = None
            return df_commissions

        # Merge on policy and product
        df_merged = df_commissions.copy()
        df_merged['Taux Boni'] = None
        df_merged['Boni'] = None

        for idx, comm_row in df_commissions.iterrows():
            candidates = df_bonus[
                (df_bonus['Numéro Police'] == comm_row['Numéro Police']) &
                (df_bonus['Produit'] == comm_row['Produit'])
            ]

            if len(candidates) == 1:
                bonus_row = candidates.iloc[0]
                df_merged.at[idx, 'Taux Boni'] = bonus_row['Taux Boni']
                df_merged.at[idx, 'Boni'] = bonus_row['Boni']

                if len(bonus_row["Nom de l'assuré"]) > len(comm_row["Nom de l'assuré"]):
                    df_merged.at[idx, "Nom de l'assuré"] = bonus_row["Nom de l'assuré"]

            elif len(candidates) > 1:
                comm_name = comm_row["Nom de l'assuré"]
                for _, bonus_row in candidates.iterrows():
                    bonus_name = bonus_row["Nom de l'assuré"]
                    if self._fuzzy_name_match(comm_name, bonus_name):
                        df_merged.at[idx, 'Taux Boni'] = bonus_row['Taux Boni']
                        df_merged.at[idx, 'Boni'] = bonus_row['Boni']
                        if len(bonus_name) > len(comm_name):
                            df_merged.at[idx, "Nom de l'assuré"] = bonus_name
                        break

        return df_merged

    def _fuzzy_name_match(self, name1: str, name2: str) -> bool:
        """Check if two names match, allowing for truncation."""
        return (name1.startswith(name2) or
                name2.startswith(name1) or
                name1 == name2)

    def _parse_float(self, value: str) -> float:
        """Parse float from French-formatted string."""
        try:
            return float(value.replace(',', '.').replace(' ', ''))
        except ValueError:
            return 0.0

    def _extract_legacy(self) -> ExtractionResult:
        """
        Extract data using the legacy method.
        """
        self._log("Starting legacy extraction for Assomption...")

        try:
            df = extract_pdf_data(self.pdf_path)

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
        - Valid policy numbers (25%)
        - Non-empty required fields (25%)
        - Valid numeric commission values (25%)
        - Valid date format (25%)

        Args:
            df: Extracted DataFrame

        Returns:
            Quality score between 0.0 and 1.0
        """
        if df.empty:
            return 0.0

        scores = []

        # Score 1: Valid policy numbers (25%)
        if 'Numéro Police' in df.columns:
            valid_policies = df['Numéro Police'].apply(
                lambda x: bool(self.POLICY_NUMBER_PATTERN.match(str(x).strip()))
            ).sum()
            policy_score = valid_policies / len(df)
            scores.append(('policies', policy_score, 0.25))

        # Score 2: Non-empty required fields (25%)
        required_fields = ["Nom de l'assuré", 'Produit', 'Commissions']
        non_empty_counts = []
        for field in required_fields:
            if field in df.columns:
                non_empty = (df[field].notna() & (df[field] != '') & (df[field] != 0)).sum()
                non_empty_counts.append(non_empty / len(df))

        if non_empty_counts:
            required_score = sum(non_empty_counts) / len(non_empty_counts)
            scores.append(('required_fields', required_score, 0.25))

        # Score 3: Valid commission values (25%)
        if 'Commissions' in df.columns:
            try:
                numeric_vals = pd.to_numeric(df['Commissions'], errors='coerce')
                valid_count = numeric_vals.notna().sum()
                commission_score = valid_count / len(df)
                scores.append(('commissions', commission_score, 0.25))
            except:
                scores.append(('commissions', 0.0, 0.25))

        # Score 4: Valid date format (25%)
        if 'Émission' in df.columns:
            valid_dates = df['Émission'].apply(
                lambda x: bool(self.DATE_PATTERN.match(str(x).strip()))
            ).sum()
            date_score = valid_dates / len(df)
            scores.append(('dates', date_score, 0.25))

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
    pdf_path = "../pdf/assomption/Remuneration (61).pdf"

    print("=" * 80)
    print("TESTING ROBUST ASSOMPTION EXTRACTOR")
    print("=" * 80)

    extractor = RobustAssomptionExtractor(pdf_path, debug=True)
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

        if 'Commissions' in result.data.columns:
            print(f"\nTotal Commissions: {result.data['Commissions'].sum():,.2f}")
        if 'Boni' in result.data.columns:
            total_boni = result.data['Boni'].sum() if result.data['Boni'].notna().any() else 0
            print(f"Total Boni: {total_boni:,.2f}")

    print("\n" + "=" * 80)
    print("EXTRACTION LOG:")
    print("=" * 80)
    for log_entry in extractor.get_extraction_log():
        print(f"  {log_entry}")
