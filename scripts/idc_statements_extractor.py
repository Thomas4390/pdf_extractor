"""
IDC Statements PDF Parser

This module parses insurance statement reports from IDC, extracting structured
data from statement documents with "Détails des frais de suivi" tables.

The parser handles tables that span multiple pages with repeating structure.

Author: Thomas
Date: 2025-11-05
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
class PageTokens:
    """Data class to hold tokens for a single page."""
    page_number: int
    raw_text: str
    tokens: List[str]


@dataclass
class TrailingFeeRecord:
    """Data class to hold a single trailing fee record."""
    client_name: str
    account_number: str
    company: str
    product: str
    date: str
    gross_trailing_fee: str
    net_trailing_fee: str


class PDFStatementParser:
    """Parser for insurance statement PDF reports."""

    # Table trigger phrase
    TABLE_TRIGGER = ["Détails", "des", "frais", "de", "suivi"]

    # Expected column headers
    COLUMN_HEADERS = [
        "Nom", "du", "client",
        "Numéro", "de", "compte",
        "Compagnie",
        "Produit",
        "Date",
        "Concessionnaire",
        "Frais", "de", "suivi", "brut",
        "Frais", "de", "suivi", "nets"
    ]

    # Keywords that indicate start of company name
    COMPANY_KEYWORDS = {
        "WS", "Assurance", "SSQ", "Beneva", "IA", "RBC",
        "Manuvie", "Desjardins", "iA", "Industrielle",
        "Alliance", "Ivari", "Empire"
    }

    def __init__(self, pdf_path: str):
        """
        Initialize the parser with PDF file path.

        Args:
            pdf_path: Path to PDF file to parse
        """
        self.pdf_path = pdf_path
        self.pages: List[PageTokens] = []
        self._extract_pages()

        # Concatenate all tokens for easier parsing across pages
        self.all_tokens: List[Tuple[str, int]] = []  # (token, page_number)
        for page in self.pages:
            for token in page.tokens:
                self.all_tokens.append((token, page.page_number))

    def _extract_pages(self):
        """Extract text from each page and tokenize."""
        reader = PdfReader(self.pdf_path)

        for page_num, page in enumerate(reader.pages, start=1):
            # Extract text from page
            raw_text = page.extract_text()

            # Tokenize the text
            tokens = self._tokenize_text(raw_text)

            # Store page data
            self.pages.append(PageTokens(
                page_number=page_num,
                raw_text=raw_text,
                tokens=tokens
            ))

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into individual words/tokens.

        Args:
            text: Raw text to tokenize

        Returns:
            List of tokens
        """
        # Remove special characters
        text = text.replace('¶', '')

        # Split by whitespace
        tokens = re.split(r'\s+', text)

        # Clean tokens
        tokens = [t.strip() for t in tokens if t.strip()]

        return tokens

    def _is_date(self, token: str) -> bool:
        """Check if token is a date in YYYY-MM-DD format."""
        return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', token))

    def _is_currency_amount(self, idx: int) -> bool:
        """
        Check if token at index is a currency amount (number followed by $).

        Args:
            idx: Token index to check

        Returns:
            True if this is a currency amount
        """
        if idx >= len(self.all_tokens) - 1:
            return False

        token, _ = self.all_tokens[idx]
        next_token, _ = self.all_tokens[idx + 1]

        # Check for pattern: "123,45" "$"
        if next_token == "$" and re.match(r'^-?\d+,\d{2}$', token):
            return True

        return False

    def _parse_float(self, value: str) -> float:
        """Parse float from string with comma as decimal separator."""
        cleaned = value.replace(',', '.').replace('$', '').strip()
        # Handle negative values
        return float(cleaned)

    def _is_account_number(self, token: str) -> bool:
        """
        Check if token looks like an account number.

        Args:
            token: Token to check

        Returns:
            True if token appears to be an account number
        """
        # "Unknown" is a valid account number placeholder
        if token == "Unknown":
            return True

        # Pattern: 6-20 alphanumeric characters, possibly with hyphens
        if re.match(r'^[A-Z0-9\-]{6,20}$', token):
            return True

        return False

    def _find_account_separator(self, tokens: List[str]) -> Optional[int]:
        """
        Find the index that separates client name from company/product.
        This is typically either "Unknown" or a real account number,
        positioned just before company keywords.

        Args:
            tokens: List of tokens to search

        Returns:
            Index of the separator (account number), or None if not found
        """
        # Strategy 1: Look for "Unknown" - most common case
        for i, token in enumerate(tokens):
            if token == "Unknown":
                return i

        # Strategy 2: Find first company keyword, account is likely just before it
        for i, token in enumerate(tokens):
            if token in self.COMPANY_KEYWORDS:
                # Check if previous token looks like an account number
                if i > 0 and self._is_account_number(tokens[i - 1]):
                    return i - 1
                # If not, company keyword might be right after client name
                # In this case, there's no explicit account number
                return i  # Mark this as separator, account will be "Unknown"

        # Strategy 3: No clear separator found
        return None

    def _find_table_trigger_indices(self) -> List[int]:
        """
        Find all positions where 'Détails des frais de suivi' appears.

        Returns:
            List of token indices where the trigger phrase starts
        """
        trigger_indices = []
        trigger_len = len(self.TABLE_TRIGGER)

        for i in range(len(self.all_tokens) - trigger_len + 1):
            # Check if the next tokens match the trigger phrase
            match = True
            for j in range(trigger_len):
                token, _ = self.all_tokens[i + j]
                if token != self.TABLE_TRIGGER[j]:
                    match = False
                    break

            if match:
                trigger_indices.append(i)

        return trigger_indices

    def _find_client_header_at(self, idx: int) -> Optional[Tuple[str, str, int]]:
        """
        Check if there's a client header at given index.
        Format: NAME_TOKENS "-" ACCOUNT_NUMBER

        Args:
            idx: Token index to check

        Returns:
            (client_name, account_number, next_index) or None
        """
        if idx >= len(self.all_tokens) - 2:
            return None

        # Look for pattern: NAME tokens "-" ACCOUNT_NUMBER
        # Collect name tokens (uppercase, can be multi-word)
        name_parts = []
        i = idx

        # Collect name parts (stop at "-")
        while i < len(self.all_tokens):
            token, _ = self.all_tokens[i]

            if token == "-":
                # Found separator
                if i + 1 < len(self.all_tokens):
                    account_token, _ = self.all_tokens[i + 1]
                    # Account number should be alphanumeric
                    if re.match(r'^[A-Z0-9]+$', account_token):
                        client_name = ' '.join(name_parts).strip()
                        if client_name:
                            return client_name, account_token, i + 2
                break

            # Check if this looks like a name token
            if token and (token[0].isupper() or token.isupper()):
                name_parts.append(token)
                i += 1
            else:
                break

        return None

    def _extract_data_row(self, start_idx: int) -> Optional[Tuple[TrailingFeeRecord, int]]:
        """
        Extract a single data row starting from given index.

        Expected structure per line:
        [Client Name tokens] [Account Number] [Company tokens] [Product tokens] [Date] [Amount $] [Amount $]

        Args:
            start_idx: Starting token index

        Returns:
            (TrailingFeeRecord, next_index) or None if extraction fails
        """
        try:
            idx = start_idx

            # Skip special characters like "Â"
            while idx < len(self.all_tokens):
                token, _ = self.all_tokens[idx]
                if token in ["Â", "â", "Ã"]:
                    idx += 1
                else:
                    break

            # Strategy: Find account separator (Unknown or real account number)
            # Everything before = client name
            # Everything after and before date = company/product

            # Phase 1: Collect tokens until we hit end markers or have enough
            all_tokens_collected = []

            while idx < len(self.all_tokens):
                token, _ = self.all_tokens[idx]

                # Check if we hit end markers
                if token in ["Rapport", "Total", "Printed", "Fonds"]:
                    break

                # Check if we hit another table trigger
                if idx + len(self.TABLE_TRIGGER) <= len(self.all_tokens):
                    match = True
                    for j in range(len(self.TABLE_TRIGGER)):
                        check_token, _ = self.all_tokens[idx + j]
                        if check_token != self.TABLE_TRIGGER[j]:
                            match = False
                            break
                    if match:
                        break

                all_tokens_collected.append(token)
                idx += 1

                # Safety: collect up to 120 tokens
                if len(all_tokens_collected) > 120:
                    break

            if not all_tokens_collected:
                return None

            # Phase 2: Find the separator (account number) in collected tokens
            separator_idx = self._find_account_separator(all_tokens_collected)

            if separator_idx is None:
                return None

            # Extract client name (everything before separator)
            client_name_tokens = all_tokens_collected[:separator_idx]

            # Check if separator is an account number or just marks the boundary
            if self._is_account_number(all_tokens_collected[separator_idx]):
                account_number = all_tokens_collected[separator_idx]
                tokens_after_account = all_tokens_collected[separator_idx + 1:]
            else:
                # Separator is a company keyword, no explicit account
                account_number = "Unknown"
                tokens_after_account = all_tokens_collected[separator_idx:]

            if not client_name_tokens:
                return None

            # Phase 3: Find date in tokens_after_account
            company_product_tokens = []
            date_found = False
            date_value = None

            for i, token in enumerate(tokens_after_account):
                # Check for date
                if self._is_date(token):
                    date_value = token
                    date_found = True
                    # Company/product tokens are everything before the date
                    company_product_tokens = tokens_after_account[:i]
                    # Update idx to point after the date in all_tokens
                    idx = start_idx + len(all_tokens_collected) - len(tokens_after_account) + i + 1
                    break

            if not date_found:
                return None

            # Now collect the two currency amounts (gross and net)
            amounts = []
            while idx < len(self.all_tokens) and len(amounts) < 2:
                if self._is_currency_amount(idx):
                    token, _ = self.all_tokens[idx]
                    amounts.append(token)
                    idx += 2  # Skip the "$" token
                else:
                    idx += 1

                # Safety check
                if len(amounts) == 0 and idx - start_idx > 60:
                    return None

            if len(amounts) < 2:
                return None

            gross_fee = amounts[0] + " $"
            net_fee = amounts[1] + " $"

            # Now construct the record from the tokens we collected
            # client_name_tokens = everything before "Unknown"
            # company_product_tokens = everything after "Unknown" and before date

            client_name = ' '.join(client_name_tokens) if client_name_tokens else "Unknown"
            account_number = "Unknown"

            # Split company/product - simple heuristic: split in half
            if company_product_tokens:
                mid = len(company_product_tokens) // 2
                company = ' '.join(company_product_tokens[:mid]) if mid > 0 else ""
                product = ' '.join(company_product_tokens[mid:])
            else:
                company = ""
                product = ""

            record = TrailingFeeRecord(
                client_name=client_name,
                account_number=account_number,
                company=company,
                product=product,
                date=date_value,
                gross_trailing_fee=gross_fee,
                net_trailing_fee=net_fee
            )

            return record, idx

        except (IndexError, ValueError):
            return None

    def parse_trailing_fees(self) -> pd.DataFrame:
        """
        Parse all 'Détails des frais de suivi' tables and return DataFrame.

        Returns:
            DataFrame with extracted trailing fee data
        """
        records = []

        # Find all table triggers
        trigger_indices = self._find_table_trigger_indices()

        for trigger_idx in trigger_indices:
            # Skip past the trigger phrase
            idx = trigger_idx + len(self.TABLE_TRIGGER)

            # Skip column headers and section header ("Achraf El Hajji - 3449L3138")
            # Look for the section header pattern (NAME - ALPHANUMERIC)
            section_header_found = False
            while idx < len(self.all_tokens) and idx < trigger_idx + 50:
                header = self._find_client_header_at(idx)
                if header:
                    # This is the section header (e.g., "Achraf El Hajji - 3449L3138")
                    # Skip past it
                    _, _, next_idx = header
                    idx = next_idx
                    section_header_found = True
                    break
                idx += 1

            if not section_header_found:
                # If no section header found, skip forward a bit
                idx = trigger_idx + len(self.TABLE_TRIGGER) + 10

            # Now extract data rows until we hit end markers or next table
            max_idx = trigger_idx + 1500  # Safety limit
            attempts_without_success = 0

            while idx < len(self.all_tokens) and idx < max_idx:
                # Check for end markers
                token, _ = self.all_tokens[idx]
                if token in ["Rapport", "détaillé", "Total", "Printed"]:
                    break

                # Check if we hit another table trigger
                if idx + len(self.TABLE_TRIGGER) <= len(self.all_tokens):
                    match = True
                    for j in range(len(self.TABLE_TRIGGER)):
                        check_token, _ = self.all_tokens[idx + j]
                        if check_token != self.TABLE_TRIGGER[j]:
                            match = False
                            break
                    if match:
                        # Found next table
                        break

                # Try to extract a data row
                result = self._extract_data_row(idx)

                if result:
                    record, next_idx = result
                    records.append({
                        'Nom du client': record.client_name,
                        'Numéro de compte': record.account_number,
                        'Compagnie': record.company,
                        'Produit': record.product,
                        'Date': record.date,
                        'Frais de suivi brut': record.gross_trailing_fee,
                        'Frais de suivi nets': record.net_trailing_fee
                    })
                    idx = next_idx
                    attempts_without_success = 0
                else:
                    idx += 1
                    attempts_without_success += 1

                # If we've tried many times without success, stop
                if attempts_without_success > 100:
                    break

        columns = [
            'Nom du client', 'Numéro de compte', 'Compagnie', 'Produit',
            'Date', 'Frais de suivi brut', 'Frais de suivi nets'
        ]

        df = pd.DataFrame(records, columns=columns)
        return df

    def print_page_tokens(self, page_number: int = None, max_tokens: int = None):
        """
        Print tokens for a specific page or all pages.

        Args:
            page_number: Specific page to print (None = all pages)
            max_tokens: Maximum number of tokens to print per page (None = all)
        """
        if page_number is not None:
            # Print specific page
            pages_to_print = [p for p in self.pages if p.page_number == page_number]
            if not pages_to_print:
                print(f"Page {page_number} not found")
                return
        else:
            # Print all pages
            pages_to_print = self.pages

        for page in pages_to_print:
            print("=" * 100)
            print(f"PAGE {page.page_number}")
            print("=" * 100)
            print(f"Number of tokens: {len(page.tokens)}\n")

            # Print tokens
            tokens_to_show = page.tokens[:max_tokens] if max_tokens else page.tokens

            for i, token in enumerate(tokens_to_show):
                print(f"{i:4d}: {token}")

            if max_tokens and len(page.tokens) > max_tokens:
                print(f"\n... ({len(page.tokens) - max_tokens} more tokens) ...\n")

            print()

    def get_page_count(self) -> int:
        """Get total number of pages."""
        return len(self.pages)

    def get_page_tokens(self, page_number: int) -> List[str]:
        """
        Get tokens for a specific page.

        Args:
            page_number: Page number (1-indexed)

        Returns:
            List of tokens for that page
        """
        for page in self.pages:
            if page.page_number == page_number:
                return page.tokens
        return []

    def get_all_tokens(self) -> List[str]:
        """Get all tokens from all pages concatenated."""
        all_tokens = []
        for page in self.pages:
            all_tokens.extend(page.tokens)
        return all_tokens

    def print_summary(self):
        """Print a summary of the PDF."""
        print("=" * 100)
        print("PDF SUMMARY")
        print("=" * 100)
        print(f"File: {self.pdf_path}")
        print(f"Total pages: {len(self.pages)}")
        print(f"Total tokens: {sum(len(p.tokens) for p in self.pages)}")
        print()

        for page in self.pages:
            print(f"  Page {page.page_number}: {len(page.tokens)} tokens")
        print()


if __name__ == "__main__":
    # Example usage - modify the path to your actual PDF
    pdf_path = "../pdf/Statements (5).pdf"

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        print("Please update the pdf_path variable with the correct path to your PDF file.")
    else:
        # Create parser
        print("=" * 100)
        print("IDC STATEMENTS PARSER - TRAILING FEES EXTRACTION")
        print("=" * 100)
        print(f"\nProcessing: {pdf_path}\n")

        parser = PDFStatementParser(pdf_path)

        # Print summary
        parser.print_summary()

        # Print tokens for debugging
        print("\n" + "=" * 100)
        print("TOKENS BY PAGE (first 100 tokens per page)")
        print("=" * 100)
        parser.print_page_tokens(max_tokens=100)

        # Parse trailing fees
        print("\n" + "=" * 100)
        print("PARSING TRAILING FEES TABLE")
        print("=" * 100)

        df = parser.parse_trailing_fees()

        if not df.empty:
            print(f"\n✓ Successfully extracted {len(df)} records\n")
            print(df)

            # Print statistics
            print("\n" + "=" * 100)
            print("STATISTICS")
            print("=" * 100)
            print(f"Total records: {len(df)}")
            print(f"Unique clients: {df['Nom du client'].nunique()}")
            print(f"Unique accounts: {df['Numéro de compte'].nunique()}")
            print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

            # Save to CSV
            os.makedirs('../results', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"../results/trailing_fees_parsed_{timestamp}.csv"
            df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"\n✓ CSV file saved: {output_filename}")

        else:
            print("\n⚠ No records extracted. Check the PDF structure and parsing rules.")

        # Optional: Print tokens for debugging
        # print("\n" + "=" * 100)
        # print("DEBUG: TOKENS BY PAGE (first 100 tokens per page)")
        # print("=" * 100)
        # parser.print_page_tokens(max_tokens=100)
