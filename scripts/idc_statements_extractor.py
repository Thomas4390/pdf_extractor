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
        "Alliance", "Ivari", "Empire", "Assomption", "Asomption"
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

        # Real account numbers are 7-9 digits (not preceded by #)
        if re.match(r'^\d{7,9}$', token):
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

        Two formats supported:
        Format 1 (with "clt" keyword):
            [...metadata...] crt [broker] clt [CLIENT NAME] Unknown [Company] [Product] [Date] [Amounts]
        Format 2 (direct format):
            [CLIENT NAME] [ACCOUNT#] [Company] [Product] [Date] [Amounts]

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

            # Phase 1: Collect tokens until we hit end markers
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

            # Phase 2: Detect format by looking for "clt" keyword
            clt_idx = None
            client_name_tokens = []
            tokens_after_client = []
            account_number = "Unknown"

            for i, token in enumerate(all_tokens_collected):
                if token == "clt":
                    clt_idx = i
                    break

            if clt_idx is not None:
                # FORMAT 1: Contains "clt" keyword (complex format with metadata)
                # Extract ALL tokens from start until "Unknown"
                # This includes all metadata and client name
                account_number = "Unknown"

                # Find "Unknown" in the tokens
                unknown_idx = None
                for i, token in enumerate(all_tokens_collected):
                    if token == "Unknown":
                        unknown_idx = i
                        break

                if unknown_idx is not None:
                    # Everything before "Unknown" goes into client name (including metadata)
                    client_name_tokens = all_tokens_collected[:unknown_idx]
                    # Everything after "Unknown" is company/product/date
                    tokens_after_client = all_tokens_collected[unknown_idx + 1:]
                else:
                    # No "Unknown" found, try finding company keyword as separator
                    for i, token in enumerate(all_tokens_collected):
                        if token in self.COMPANY_KEYWORDS:
                            client_name_tokens = all_tokens_collected[:i]
                            tokens_after_client = all_tokens_collected[i:]
                            break

                    if not client_name_tokens:
                        return None

            else:
                # FORMAT 2: Direct format without "clt" (simple format with real account number)
                # Pattern: [Client Name] [Account Number 7-9 digits] [Company] [Product] [Date]
                account_idx = None
                account_number = "Unknown"

                # Look for a real account number (7-9 digits)
                for i, token in enumerate(all_tokens_collected):
                    if re.match(r'^\d{7,9}$', token):
                        # Check if previous token is not "#" (to avoid metadata numbers)
                        if i > 0 and all_tokens_collected[i - 1].endswith('#'):
                            continue

                        # Found real account number
                        client_name_tokens = all_tokens_collected[:i]
                        account_number = token
                        tokens_after_client = all_tokens_collected[i + 1:]
                        account_idx = i
                        break

                # If no real account number found, look for "Unknown" or company keyword
                if account_idx is None:
                    separator_idx = self._find_account_separator(all_tokens_collected)

                    if separator_idx is None:
                        return None

                    client_name_tokens = all_tokens_collected[:separator_idx]

                    if self._is_account_number(all_tokens_collected[separator_idx]):
                        account_number = all_tokens_collected[separator_idx]
                        tokens_after_client = all_tokens_collected[separator_idx + 1:]
                    else:
                        account_number = "Unknown"
                        tokens_after_client = all_tokens_collected[separator_idx:]

                        # Special case: check if there's nothing before "Unknown" except "Â"
                        # This means the client name is empty - allow it for Unknown accounts
                        if separator_idx == 0 or (separator_idx == 1 and all_tokens_collected[0] in ["Â", "â", "Ã"]):
                            client_name_tokens = []  # Empty client name - will be set to "Unknown" later

            # Only return None if client_name_tokens is empty AND account is not Unknown
            if not client_name_tokens and account_number != "Unknown":
                return None

            # Phase 3: Find date in remaining tokens
            date_found = False
            date_value = None
            company_product_tokens = []

            for i, token in enumerate(tokens_after_client):
                if self._is_date(token):
                    date_value = token
                    date_found = True
                    company_product_tokens = tokens_after_client[:i]
                    # Calculate position in all_tokens
                    tokens_before_date = len(all_tokens_collected) - len(tokens_after_client) + i
                    idx = start_idx + tokens_before_date + 1
                    break

            if not date_found:
                return None

            # Phase 4: Extract currency amounts (gross and net)
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

            # Phase 5: Build record
            client_name = ' '.join(client_name_tokens).strip() if client_name_tokens else "Unknown"

            # Special case: if client name is empty after removing special chars and account is Unknown
            # This handles the case where after "Â" there is no client name, just "Unknown"
            if account_number == "Unknown":
                # Remove special characters to check if there's actual content
                cleaned_name = client_name.replace('Â', '').replace('â', '').replace('Ã', '').strip()
                if not cleaned_name:
                    client_name = "Unknown"

            # Split company/product intelligently
            # Company name should end with ")", product often starts with a company keyword
            if company_product_tokens:
                # Strategy: find a token ending with ")" followed by a company keyword
                split_idx = None
                for i in range(len(company_product_tokens) - 1):
                    # Check if current token ends with ")" and next token is a company keyword
                    if (company_product_tokens[i].endswith(")") and
                        i + 1 < len(company_product_tokens) and
                        company_product_tokens[i + 1] in self.COMPANY_KEYWORDS):
                        split_idx = i
                        break

                if split_idx is not None:
                    # Company = everything up to and including the token ending with ")"
                    company = ' '.join(company_product_tokens[:split_idx + 1]).strip()
                    # Product = everything after (starting with company keyword)
                    product = ' '.join(company_product_tokens[split_idx + 1:]).strip()
                else:
                    # No clear split found, use the last token ending with ")"
                    last_paren_idx = None
                    for i in range(len(company_product_tokens) - 1, -1, -1):
                        if company_product_tokens[i].endswith(")"):
                            last_paren_idx = i
                            break

                    if last_paren_idx is not None:
                        # Company = everything up to and including the last token ending with ")"
                        company = ' '.join(company_product_tokens[:last_paren_idx + 1]).strip()
                        # Product = everything after
                        product = ' '.join(company_product_tokens[last_paren_idx + 1:]).strip()
                    else:
                        # No ")" found, fall back to split in half
                        mid = len(company_product_tokens) // 2
                        company = ' '.join(company_product_tokens[:mid]).strip() if mid > 0 else ""
                        product = ' '.join(company_product_tokens[mid:]).strip()
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

        except (IndexError, ValueError) as e:
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
    pdf_path = "../pdf/Statements (8).pdf"

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
