"""
IDC/Propositions PDF Parser

This module parses insurance proposition reports from IDC, extracting structured
data including insurer, client, policy details, and commission information.

The parser uses token-based extraction to handle complex multi-word names
and varying document structures.

Author: Thomas
Date: 2025-10-23
"""

from PyPDF2 import PdfReader
import pandas as pd
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)


@dataclass
class PropositionRecord:
    """Data class to hold a single proposition record."""
    insurer: str
    client: str
    regime_type: str
    policy: str
    status: str
    date: str
    quantity: float
    cpa_rate: float
    coverage: str
    policy_premium: str
    commissioned_premium: str
    commission: str


class PDFPropositionParser:
    """Parser for insurance proposition PDF reports."""

    SINGLE_TOKEN_REGIMES = ['Permanent', 'Term', 'Disability']
    MULTI_TOKEN_REGIMES = [['Critical', 'Illness']]

    # Surname particles
    SURNAME_PARTICLES = {
        'DE', 'LA', 'LE', 'DU', 'DES', 'VAN', 'VON', 'DER', 'DEN',
        'TER', 'VER', 'EL', 'AL', 'BEN', 'IBN', 'MAC', 'MC', 'ST'
    }

    # Keywords indicating an insurer name (not a client name)
    INSURER_KEYWORDS = {
        'INSURANCE', 'ASSURANCE', 'LIFE', 'VIE', 'FINANCIAL',
        'FINANCIÈRE', 'GROUP', 'GROUPE', 'INC', 'LTD', 'LTÉE',
        'COMPANY', 'COMPAGNIE', 'CORP', 'CORPORATION'
    }

    MAX_SURNAME_TOKENS = 10

    COLUMNS = [
        'Assureur', 'Client', 'Type de régime', 'Police', 'Statut',
        'Date', 'Nombre', 'Taux de CPA', 'Couverture', 'Prime de la police',
        'Part prime comm.', 'Comm.'
    ]

    def __init__(self, pdf_path: str):
        """
        Initialize the parser with PDF file path.

        Args:
            pdf_path: Path to PDF file to parse
        """
        self.pdf_path = pdf_path
        self.full_text = self._extract_text()
        self.tokens = self._tokenize_text()

    def _extract_text(self) -> str:
        """Extract all text from PDF."""
        reader = PdfReader(self.pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        return full_text

    def _tokenize_text(self) -> List[str]:
        """Tokenize text into individual words/tokens."""
        text = self.full_text.replace('¶', '')
        tokens = re.split(r'\s+', text)
        tokens = [t.strip() for t in tokens if t.strip()]
        return tokens

    def _is_regime_type_at(self, idx: int) -> Tuple[bool, int, str]:
        """
        Check if there's a regime type at given token index.

        Args:
            idx: Token index to check

        Returns:
            (is_regime, tokens_consumed, regime_name)
        """
        if idx >= len(self.tokens):
            return False, 0, ""

        if self.tokens[idx] in self.SINGLE_TOKEN_REGIMES:
            return True, 1, self.tokens[idx]

        for regime_tokens in self.MULTI_TOKEN_REGIMES:
            num_tokens = len(regime_tokens)
            if idx + num_tokens > len(self.tokens):
                continue
            matches = all(
                self.tokens[idx + i] == regime_tokens[i]
                for i in range(num_tokens)
            )
            if matches:
                return True, num_tokens, ' '.join(regime_tokens)

        return False, 0, ""

    def _is_date(self, token: str) -> bool:
        """Check if token is a date in YYYY-MM-DD format."""
        return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', token))

    def _is_uppercase_word(self, token: str) -> bool:
        """Check if token is all uppercase."""
        return bool(re.match(r'^[A-ZÀ-Ÿ\-\']+$', token))

    def _is_vendor_name_token(self, idx: int) -> bool:
        """
        Check if token at index is a vendor/broker name token.

        Args:
            idx: Token index to check

        Returns:
            True if token is a vendor name format
        """
        if idx < 0 or idx + 1 >= len(self.tokens):
            return False
        current = self.tokens[idx]
        next_token = self.tokens[idx + 1]
        if not current.endswith(','):
            return False
        name_part = current[:-1]
        if not name_part or not name_part[0].isupper():
            return False
        has_lowercase = any(c.islower() for c in name_part)
        is_full_firstname = (
                len(next_token) > 1 and
                next_token[0].isupper() and
                any(c.islower() for c in next_token)
        )
        return has_lowercase and is_full_firstname

    def _parse_float(self, value: str) -> float:
        """Parse float from string with comma as decimal separator."""
        return float(value.replace(',', '.').replace(' ', ''))

    def _find_regime_indices(self) -> List[Tuple[int, str]]:
        """Find all positions where regime types appear in tokens."""
        regime_positions = []
        i = 0
        while i < len(self.tokens):
            is_regime, tokens_consumed, regime_name = self._is_regime_type_at(i)
            if is_regime:
                regime_positions.append((i, regime_name))
                i += tokens_consumed
            else:
                i += 1
        return regime_positions

    def _reconstruct_client_name(self, regime_idx: int) -> Tuple[str, int]:
        """
        Reconstruct full client name by going backwards from regime index.

        Client format: UPPERCASE WORDS ending with ", LETTER"
        Collects all uppercase tokens for the surname, but stops at insurer keywords.

        Examples:
            "BONAI LOUMLE HENRI GERVET, X" → collects all 4 name parts
            "RBC INSURANCE SMITH, J" → stops at "INSURANCE", collects only "SMITH, J"

        Args:
            regime_idx: Index of regime token

        Returns:
            (client_name, index_before_client)
        """
        client_parts = []
        idx = regime_idx - 1

        # Find single letter (initial)
        if idx >= 0 and len(self.tokens[idx]) == 1 and self.tokens[idx].isupper():
            client_parts.insert(0, self.tokens[idx])
            idx -= 1
        else:
            raise ValueError(f"Expected single uppercase letter before regime at {regime_idx}")

        # Find comma
        if idx >= 0 and self.tokens[idx].endswith(','):
            client_parts.insert(0, self.tokens[idx])
            idx -= 1
        else:
            raise ValueError(f"Expected comma before letter at {regime_idx}")

        # Collect all surname parts (all uppercase tokens)
        tokens_collected = 0

        while idx >= 0 and tokens_collected < self.MAX_SURNAME_TOKENS:
            token = self.tokens[idx]

            # Stop if not uppercase word
            if not self._is_uppercase_word(token):
                break

            # Stop if this is an insurer keyword
            token_upper = token.upper().rstrip("'")
            if token_upper in self.INSURER_KEYWORDS:
                break

            # Collect this token as part of client name
            client_parts.insert(0, token)
            tokens_collected += 1
            idx -= 1

        if not client_parts:
            raise ValueError(f"Could not find client name before regime at {regime_idx}")

        return ' '.join(client_parts), idx

    def _reconstruct_insurer(self, index_before_client: int) -> str:
        """
        Reconstruct insurer name by going backwards from client start.

        Args:
            index_before_client: Token index before client name starts

        Returns:
            Insurer name
        """
        insurer_parts = []
        idx = index_before_client
        while idx >= 0:
            token = self.tokens[idx]
            if token.endswith('$') or token == 'Comm.':
                break
            if self._is_vendor_name_token(idx):
                break
            if idx > 0 and self._is_vendor_name_token(idx - 1):
                break
            insurer_parts.insert(0, token)
            idx -= 1
        if not insurer_parts:
            raise ValueError(f"Could not find insurer name before index {index_before_client}")
        return ' '.join(insurer_parts)

    def _consume_until_terminator(self, start_idx: int, terminator: str) -> Tuple[str, int]:
        """
        Consume tokens until finding one that ends with terminator.

        Args:
            start_idx: Starting token index
            terminator: Character to search for at end of token

        Returns:
            (accumulated_string, next_index)
        """
        parts = []
        idx = start_idx
        while idx < len(self.tokens):
            token = self.tokens[idx]
            if token == 'TOTAUX':
                if parts:
                    return ' '.join(parts), idx
                else:
                    raise ValueError(f"Encountered TOTAUX before finding terminator '{terminator}'")
            parts.append(token)
            if token.endswith(terminator):
                return ' '.join(parts), idx + 1
            idx += 1
        raise ValueError(f"Terminator '{terminator}' not found from index {start_idx}")

    def _consume_until_date(self, start_idx: int) -> Tuple[str, int]:
        """
        Consume tokens until finding a date.

        Args:
            start_idx: Starting token index

        Returns:
            (accumulated_string, date_index)
        """
        parts = []
        idx = start_idx
        while idx < len(self.tokens):
            token = self.tokens[idx]
            if token == 'TOTAUX':
                if parts:
                    return ' '.join(parts), idx
                else:
                    raise ValueError("Encountered TOTAUX before finding date")
            if self._is_date(token):
                return ' '.join(parts), idx
            parts.append(token)
            idx += 1
        raise ValueError(f"Date not found from index {start_idx}")

    def _count_remaining_currency_fields(self, start_idx: int, stop_at_regime: bool = True) -> int:
        """
        Count remaining fields ending with $ from start index.

        Args:
            start_idx: Starting token index
            stop_at_regime: Whether to stop counting at next regime

        Returns:
            Number of currency fields found
        """
        count = 0
        i = start_idx
        while i < len(self.tokens):
            token = self.tokens[i]
            if token == 'TOTAUX':
                break
            if stop_at_regime:
                is_regime, _, _ = self._is_regime_type_at(i)
                if is_regime:
                    break
            if token.endswith('$'):
                count += 1
            i += 1
        return count

    def _extract_record(self, regime_idx: int, regime_name: str) -> Optional[PropositionRecord]:
        """
        Extract a single record starting at the regime index.

        Args:
            regime_idx: Token index where regime type is found
            regime_name: Name of the regime type

        Returns:
            PropositionRecord or None if extraction fails
        """
        try:
            client, index_before_client = self._reconstruct_client_name(regime_idx)
            insurer = self._reconstruct_insurer(index_before_client)
            regime_type = regime_name

            _, tokens_consumed, _ = self._is_regime_type_at(regime_idx)
            idx = regime_idx + tokens_consumed

            # Policy number (may have suffix letter)
            policy_parts = [self.tokens[idx]]
            idx += 1
            if (idx < len(self.tokens) and
                    len(self.tokens[idx]) == 1 and
                    self.tokens[idx].isupper() and
                    self.tokens[idx].isalpha()):
                policy_parts.append(self.tokens[idx])
                idx += 1
            policy = ' '.join(policy_parts)

            # Status (until date)
            status, idx = self._consume_until_date(idx)
            date = self.tokens[idx]
            idx += 1

            # Quantity (number)
            quantity = self._parse_float(self.tokens[idx])
            idx += 1

            # CPA rate (%)
            cpa_rate_str, idx = self._consume_until_terminator(idx, '%')
            cpa_rate = self._parse_float(cpa_rate_str.replace('%', '').strip())

            # Coverage amount ($)
            coverage, idx = self._consume_until_terminator(idx, '$')

            # Policy premium ($)
            policy_premium, idx = self._consume_until_terminator(idx, '$')

            # Check remaining currency fields to determine structure
            remaining_currency_count = self._count_remaining_currency_fields(idx)

            if remaining_currency_count >= 2:
                # Has commissioned premium and commission
                commissioned_premium, idx = self._consume_until_terminator(idx, '$')
                commission, idx = self._consume_until_terminator(idx, '$')
            else:
                raise ValueError(f"Expected at least 2 currency fields, found {remaining_currency_count}")

            return PropositionRecord(
                insurer=insurer,
                client=client,
                regime_type=regime_type,
                policy=policy,
                status=status,
                date=date,
                quantity=quantity,
                cpa_rate=cpa_rate,
                coverage=coverage,
                policy_premium=policy_premium,
                commissioned_premium=commissioned_premium,
                commission=commission
            )

        except (IndexError, ValueError) as e:
            # Silent fail - just return None without debug prints
            return None

    def parse(self) -> pd.DataFrame:
        """
        Parse entire PDF and return DataFrame.

        Returns:
            DataFrame with extracted data
        """
        regime_positions = self._find_regime_indices()

        records = []
        for idx, regime_name in regime_positions:
            record = self._extract_record(idx, regime_name)
            if record:
                records.append({
                    'Assureur': record.insurer,
                    'Client': record.client,
                    'Type de régime': record.regime_type,
                    'Police': record.policy,
                    'Statut': record.status,
                    'Date': record.date,
                    'Nombre': record.quantity,
                    'Taux de CPA': record.cpa_rate,
                    'Couverture': record.coverage,
                    'Prime de la police': record.policy_premium,
                    'Part prime comm.': record.commissioned_premium,
                    'Comm.': record.commission
                })

        df = pd.DataFrame(records, columns=self.COLUMNS)
        return df


if __name__ == "__main__":
    pdf_path = "../pdf/Rapport des propositions soumises.20251017_1517.pdf"

    parser = PDFPropositionParser(pdf_path)
    df = parser.parse()

    print("=" * 100)
    print("DATAFRAME CREATED SUCCESSFULLY!")
    print("=" * 100)
    print(f"Number of rows: {len(df)}\n")
    print(df)

    print("\n" + "=" * 100)
    print("DATAFRAME INFO")
    print("=" * 100)
    print(df.info())

    print("\n" + "=" * 100)
    print("STATISTICS BY REGIME TYPE")
    print("=" * 100)
    print(df['Type de régime'].value_counts())

    print("\n" + "=" * 100)
    print("FIRST AND LAST RECORDS")
    print("=" * 100)
    for i in range(min(3, len(df))):
        print(f"\nRecord {i + 1}:")
        for col in df.columns:
            print(f"  {col:20s}: {df.iloc[i][col]}")

    if len(df) > 3:
        print(f"\n... ({len(df) - 6} more records) ...\n")
        for i in range(max(0, len(df) - 3), len(df)):
            print(f"\nRecord {i + 1}:")
            for col in df.columns:
                print(f"  {col:20s}: {df.iloc[i][col]}")

    os.makedirs('../results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_filename = f"../results/propositions_parsed_{timestamp}.csv"
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n✓ CSV file saved: {output_filename}")