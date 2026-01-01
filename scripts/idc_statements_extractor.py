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


# =============================================================================
# ADVISOR NAME NORMALIZATION
# =============================================================================

# Compiled regex patterns for advisor name mapping
# Format: (compiled_regex, simplified_first_name)
_ADVISOR_PATTERNS = None


def _init_advisor_patterns():
    """Initialize compiled regex patterns for advisor name mapping."""
    global _ADVISOR_PATTERNS
    if _ADVISOR_PATTERNS is not None:
        return

    # Define patterns for each advisor
    # Patterns use re.IGNORECASE for case-insensitive matching
    pattern_definitions = [
        # Brandeen David Legrand - most complex, many variations
        (r'\bbrandeen\b', 'Brandeen'),
        (r'\bdavid\s+legrand\b', 'Brandeen'),
        (r'\blegrand\s*,?\s*david\b', 'Brandeen'),
        (r'\bd\.?\s*legrand\b', 'Brandeen'),
        (r'\blegrand\s*,?\s*d\.?\b', 'Brandeen'),
        (r'\bdl\s+brandeen\b', 'Brandeen'),
        (r'\bbrandeen\s+dl\b', 'Brandeen'),

        # Mohammed Fayçal/Faycal Guennouni
        (r'\bfay[cç]al\b', 'Faycal'),
        (r'\bfayÃ§al\b', 'Faycal'),  # UTF-8 encoding issue variant
        (r'\bguennouni\b', 'Faycal'),
        (r'\bmohammed\s+f', 'Faycal'),

        # Mathieu Poirier (check before Derek Poirier)
        (r'\bmathieu\s+poirier\b', 'Mathieu'),
        (r'\bpoirier\s*,?\s*mathieu\b', 'Mathieu'),
        (r'\bm\.?\s+poirier\b', 'Mathieu'),

        # Derek Poirier
        (r'\bderek\s+poirier\b', 'Derek'),
        (r'\bpoirier\s*,?\s*derek\b', 'Derek'),
        (r'\bd\.?\s+poirier\b(?!\s*m)', 'Derek'),  # D. Poirier but not if followed by M

        # Thomas Lussier
        (r'\bthomas\s+lussier\b', 'Thomas'),
        (r'\blussier\s*,?\s*thomas\b', 'Thomas'),
        (r'\bt\.?\s+lussier\b', 'Thomas'),
        (r'\blussier\s*,?\s*t\.?\b', 'Thomas'),
        (r'\blussier\b', 'Thomas'),  # Lussier alone -> Thomas

        # Ayoub Chamoumi
        (r'\bayoub\s+chamoumi\b', 'Ayoub'),
        (r'\bchamoumi\b', 'Ayoub'),

        # Said Vital -> Ayoub (special case)
        (r'\bsaid\s+vital\b', 'Ayoub'),
        (r'\bvital\s*,?\s*said\b', 'Ayoub'),
        (r'\bs\.?\s+vital\b', 'Ayoub'),

        # Alexis Bourassa
        (r'\balexis\s+bourassa\b', 'Alexis'),
        (r'\bbourassa\s*,?\s*alexis\b', 'Alexis'),
        (r'\bbourassa\b', 'Alexis'),

        # Jad Senhaji
        (r'\bjad\s+senhaji\b', 'Jad'),
        (r'\bsenhaji\b', 'Jad'),

        # Igor Velicico
        (r'\bigor\s+velicico\b', 'Igor'),
        (r'\bvelicico\b', 'Igor'),

        # Robinson Viaud
        (r'\brobinson\s+viaud\b', 'Robinson'),
        (r'\bviaud\b', 'Robinson'),

        # Benoît/Benoit Méthot/Methot
        (r'\bbeno[iî]t\s+m[eé]thot\b', 'Benoit'),
        (r'\bm[eé]thot\s*,?\s*beno[iî]t\b', 'Benoit'),
        (r'\bm[eé]thot\b', 'Benoit'),

        # Anthony Guay
        (r'\banthony\s+guay\b', 'Anthony'),
        (r'\bguay\s*,?\s*anthony\b', 'Anthony'),
        (r'\bguay\b', 'Anthony'),

        # Guillaume St-Pierre
        (r'\bguillaume\s+st[\-\.]?\s*pierre\b', 'Guillaume'),
        (r'\bst[\-\.]?\s*pierre\s*,?\s*guillaume\b', 'Guillaume'),
        (r'\bst[\-\.]?\s*pierre\b', 'Guillaume'),
    ]

    # Compile patterns
    _ADVISOR_PATTERNS = [
        (re.compile(pattern, re.IGNORECASE), name)
        for pattern, name in pattern_definitions
    ]


def normalize_advisor_name(advisor_name: str) -> str:
    """
    Normalize advisor name to simplified first name.

    Uses regex patterns to match various name formats and map them
    to simplified first names.

    Args:
        advisor_name: The original advisor name (can be full name, abbreviation, etc.)

    Returns:
        Simplified first name, or original if no match found
    """
    # Initialize patterns if not already done
    _init_advisor_patterns()

    if pd.isna(advisor_name) or advisor_name is None:
        return None

    name_str = str(advisor_name).strip()
    if not name_str or name_str in ['None', 'nan', 'NaN']:
        return None

    # Test each pattern
    for pattern, simplified_name in _ADVISOR_PATTERNS:
        if pattern.search(name_str):
            return simplified_name

    # No match found, return original name
    return name_str


# =============================================================================
# COMPANY NAME NORMALIZATION
# =============================================================================

# Mapping of company name variations to normalized names
_COMPANY_NAME_MAPPING = {
    # UV Insurance variants
    'uv insurance': 'UV',
    'uv assurance': 'UV',
    'uv': 'UV',
    # Assomption variants
    'assomption': 'Assomption',
    'asomption': 'Assomption',
    'assomption vie': 'Assomption',
    # IA / Industrielle Alliance variants
    'industrielle alliance': 'IA',
    'industrielle': 'IA',
    'ia': 'IA',
    # Beneva variants
    'beneva': 'Beneva',
    # Manuvie variants
    'manuvie': 'Manuvie',
    'manulife': 'Manuvie',
    # RBC variants
    'rbc': 'RBC',
    'rbc assurance': 'RBC',
    'rbc insurance': 'RBC',
    # SSQ variants
    'ssq': 'SSQ',
    'ssq assurance': 'SSQ',
    # Desjardins variants
    'desjardins': 'Desjardins',
    'desjardins assurance': 'Desjardins',
    # Empire variants
    'empire': 'Empire',
    'empire vie': 'Empire',
    'empire life': 'Empire',
    # Ivari variants
    'ivari': 'Ivari',
}


def normalize_company_name(company_name: str) -> str:
    """
    Normalize company name to simplified standard name.

    Args:
        company_name: The original company name

    Returns:
        Normalized company name, or original if no match found
    """
    if pd.isna(company_name) or company_name is None:
        return None

    name_str = str(company_name).strip()
    if not name_str or name_str in ['None', 'nan', 'NaN']:
        return None

    # Try exact match first (case-insensitive)
    name_lower = name_str.lower()
    if name_lower in _COMPANY_NAME_MAPPING:
        return _COMPANY_NAME_MAPPING[name_lower]

    # Try partial matching for compound names
    for pattern, normalized in _COMPANY_NAME_MAPPING.items():
        if pattern in name_lower:
            return normalized

    # No match found, return original name
    return name_str


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
    advisor_name: str = None  # Extracted from complex cases
    on_commission_rate: float = None  # Extracted from complex cases


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

    def _clean_client_name(self, client_name: str) -> str:
        """
        Clean client name by removing parentheses content, 'crt' keyword, and commas.

        Args:
            client_name: Raw client name with metadata

        Returns:
            Cleaned client name
        """
        if not client_name:
            return client_name

        # Remove content in parentheses along with the parentheses and preceding space
        cleaned = re.sub(r'\s*\([^)]*\)', '', client_name)

        # Remove 'crt' keyword if present
        cleaned = cleaned.replace(' crt ', ' ').replace('_crt_', '_').replace('_crt', '')

        # Remove commas
        cleaned = cleaned.replace(',', '')

        return cleaned.strip()

    def _is_only_metadata(self, client_name: str) -> bool:
        """
        Check if client name contains only metadata patterns and no actual client name.

        Metadata patterns include:
        - Company patterns: Assomption_..., Beneva_..., Manuvie-..., etc.
        - Account patterns: #... or _crt
        - Date patterns: 2025-..., _2025-...
        - Advisor patterns: Senhaji_..., etc. after _crt
        - Rate patterns: 75%, 75 %

        Args:
            client_name: Client name to check

        Returns:
            True if the name contains only metadata, False if it has actual client content
        """
        if not client_name or client_name == "Unknown":
            return True

        # Remove all known metadata patterns and check if anything remains
        test_name = client_name

        # Remove company prefixes with underscores and optional rate
        # Patterns: Assomption_18456_75, Assomption_8055_75%, Beneva_925800, etc.
        test_name = re.sub(r'\b(Assomption|Asomption|Beneva|Manuvie|IA|RBC|UV|iA)_[0-9_]+(%)?', '', test_name)

        # Remove account number patterns after # (including _crt suffix)
        test_name = re.sub(r'#[A-Z0-9]+(_crt)?', '', test_name)

        # Remove advisor name patterns (name after _crt followed by date)
        # Patterns: Senhaji_2025-10-01-EZ, St- Pierre_2025-10-01-EZ, Poirier M_2025-10-10-EZ
        test_name = re.sub(r'[A-Z][a-z]+(-\s?)?([A-Z]?[a-z]*)?_?\d{4}-\d{2}-\d{2}[-A-Z]*', '', test_name)

        # Remove standalone rate patterns
        test_name = re.sub(r'\b\d+\s*%', '', test_name)

        # Remove date patterns
        test_name = re.sub(r'\d{4}-\d{2}-\d{2}', '', test_name)

        # Remove special characters and whitespace
        test_name = re.sub(r'[_%#\-\s]+', '', test_name)

        # If nothing substantial remains, it's only metadata
        return len(test_name) == 0

    def _extract_company_after_special_char(self, tokens: List[str]) -> Optional[str]:
        """
        Extract company name from metadata patterns.
        Handles four cases:
        1. After 'Â': Â Beneva_... → extract "Beneva"
        2. First token with underscore: Asomption_8055_75% → extract "Asomption"
        3. First token is company name: RBC, UV → extract directly
        4. Manuvie pattern: 1305-Manuvie-32570-... → extract "Manuvie"

        Args:
            tokens: List of tokens to search

        Returns:
            Company name or None
        """
        # Case 1: After 'Â' marker
        for i, token in enumerate(tokens):
            if token in ["Â", "â", "Ã"]:
                if i + 1 < len(tokens):
                    next_token = tokens[i + 1]
                    # Extract company name until '_' or space
                    company = next_token.split('_')[0].split()[0]

                    # Normalize company names
                    if company.lower() in ['asomption', 'assomption']:
                        return 'Assomption'
                    elif company.lower() == 'beneva':
                        return 'Beneva'
                    elif company.lower() == 'ia':
                        return 'IA'
                    elif company.lower() == 'rbc':
                        return 'RBC'
                    elif company.lower() == 'uv':
                        return 'UV'
                    elif company.lower() == 'manuvie':
                        return 'Manuvie'
                    else:
                        return company

        # Case 4: Check for Manuvie pattern first (CODE-Manuvie-...)
        # Pattern: "1305-Manuvie-32570-2025-10-07-643334-El"
        for token in tokens:
            if '-Manuvie-' in token or token.startswith('Manuvie-') or '-Manuvie' in token:
                return 'Manuvie'

        # Case 2: First token contains company_CODE_RATE% pattern
        if tokens:
            first_token = tokens[0]
            # Pattern: "Asomption_8055_75%" or "Beneva_925800"
            if '_' in first_token:
                company = first_token.split('_')[0]

                # Normalize company names
                if company.lower() in ['asomption', 'assomption']:
                    return 'Assomption'
                elif company.lower() == 'beneva':
                    return 'Beneva'
                elif company.lower() == 'ia':
                    return 'IA'
                elif company.lower() == 'rbc':
                    return 'RBC'
                elif company.lower() == 'uv':
                    return 'UV'
                elif company.lower() == 'manuvie':
                    return 'Manuvie'
                # Only return if it's a known company pattern
                elif company and company[0].isupper():
                    return company

            # Case 3: First token is directly a known company (RBC, UV)
            # Pattern: "RBC 41613 2025-10-24 boni 70% #N894713"
            else:
                company = first_token.strip()
                if company.lower() in ['asomption', 'assomption']:
                    return 'Assomption'
                elif company.lower() == 'beneva':
                    return 'Beneva'
                elif company.lower() == 'ia':
                    return 'IA'
                elif company.lower() == 'rbc':
                    return 'RBC'
                elif company.lower() == 'uv':
                    return 'UV'
                elif company.lower() == 'manuvie':
                    return 'Manuvie'

        return None

    def _extract_commission_rate(self, tokens: List[str]) -> Optional[float]:
        """
        Extract commission rate from pattern before '%'.
        Examples:
        - 75% → 0.75
        - "75 %" → 0.75
        - "Assomption_18456_75 %" → 0.75

        Args:
            tokens: List of tokens to search

        Returns:
            Commission rate as float (0.75 for 75%) or None
        """
        for i, token in enumerate(tokens):
            # Pattern 1: "75%" (rate and % together)
            match = re.search(r'(\d+)%', token)
            if match:
                return float(match.group(1)) / 100

            # Pattern 2: "75 %" or "Assomption_18456_75 %" (rate and % separate)
            if token == '%' and i > 0:
                prev_token = tokens[i - 1]
                # Try to extract digits from end of previous token
                # Handle cases like "75", "Assomption_18456_75", etc.
                rate_match = re.search(r'(\d+)$', prev_token)
                if rate_match:
                    return float(rate_match.group(1)) / 100

        return None

    def _extract_account_number_from_hash(self, tokens: List[str]) -> Optional[str]:
        """
        Extract account number after '#'.
        Handles multiple formats:
        - "#1012274-CLIENT" or "#1012274" or "#N894713" (alphanumeric)
        - "#1006493-" (ends with dash)
        - "# 1012274"
        - "75%#016529104" (Beneva case)

        Args:
            tokens: List of tokens to search

        Returns:
            Account number or None
        """
        for i, token in enumerate(tokens):
            # Pattern 1: "#1012274-CLIENT" or "#1012274" or "#N894713"
            if token.startswith('#'):
                account = token[1:]  # Remove the #
                # Remove client name after '-' if present
                if '-' in account:
                    account = account.split('-')[0]
                # Remove 'crt' suffix if present
                if '_crt' in account:
                    account = account.split('_crt')[0]
                # Extract alphanumeric account (letters and digits)
                account = re.match(r'^([A-Z0-9]+)', account)
                if account:
                    return account.group(1)

            # Pattern 2: "# 1012274" (separate tokens)
            elif token == '#' and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                # Account might have '-' separator
                if '-' in next_token:
                    return next_token.split('-')[0]
                # Extract alphanumeric account
                account = re.match(r'^([A-Z0-9]+)', next_token)
                if account:
                    return account.group(1)

            # Pattern 3: "75%#016529104" (# in middle of token - Beneva case)
            elif '#' in token:
                # Extract everything after '#'
                parts = token.split('#', 1)
                if len(parts) > 1:
                    account = parts[1]
                    # Remove client name after '-' if present
                    if '-' in account:
                        account = account.split('-')[0]
                    # Remove 'crt' suffix if present
                    if '_crt' in account:
                        account = account.split('_crt')[0]
                    # Extract alphanumeric account
                    account = re.match(r'^([A-Z0-9]+)', account)
                    if account:
                        return account.group(1)

        return None

    def _extract_advisor_name(self, tokens: List[str]) -> Optional[str]:
        """
        Extract advisor name after 'crt' until '_' or 'clt'.
        Handles two cases:
        1. Separate token: "crt Lussier T_..."
        2. Suffix: "...FANGUE_crt Senhaji_..."

        Args:
            tokens: List of tokens to search

        Returns:
            Advisor name or None
        """
        for i, token in enumerate(tokens):
            # Case 1: 'crt' as separate token
            if token == 'crt':
                # Collect tokens until '_' or 'clt' or end of reasonable length
                advisor_parts = []
                j = i + 1
                while j < len(tokens) and j < i + 5:  # Max 4 tokens for name
                    next_token = tokens[j]

                    # Stop at separators
                    if next_token in ['clt', '_'] or next_token.startswith('_') or next_token.endswith('_'):
                        break

                    # Stop at dates (tokens that start with dates)
                    if re.match(r'^\d{4}-\d{2}-\d{2}', next_token):
                        break

                    # Check if token contains date after underscore (e.g., "M_2025-10-10-EZ")
                    # This handles IA cases where advisor name is like "Poirier," "M_2025-10-10-EZ"
                    if '_' in next_token:
                        token_parts = next_token.split('_')
                        # Check if any part after '_' starts with a date
                        has_date_after_underscore = any(re.match(r'^\d{4}-\d{2}-\d{2}', part) for part in token_parts[1:])

                        if has_date_after_underscore:
                            # Extract only the part before the date
                            clean_part = token_parts[0].rstrip(',').strip()
                            if clean_part:
                                advisor_parts.append(clean_part)
                            # Stop here - don't continue collecting
                            break

                    # Add to advisor name
                    clean_token = next_token.rstrip('_').rstrip(',').strip()
                    if clean_token:
                        advisor_parts.append(clean_token)
                    j += 1

                if advisor_parts:
                    advisor_name = ' '.join(advisor_parts)
                    advisor_name = advisor_name.replace('_', ' ').replace(',', '')
                    return advisor_name.strip()

            # Case 2: 'crt' as suffix in token (e.g., "#1012274-FANGUE_crt" or "MORNEAU_crt")
            elif token.endswith('_crt') and i + 1 < len(tokens):
                # Collect advisor name tokens (can be multiple: "Lussier," "T_2025...")
                advisor_parts = []
                j = i + 1
                while j < len(tokens) and j < i + 5:  # Max 4 tokens for name
                    next_token = tokens[j]

                    # Stop at separators
                    if next_token in ['clt', '_']:
                        break

                    # Stop at dates (including tokens that start with dates)
                    if re.match(r'^\d{4}-\d{2}-\d{2}', next_token):
                        break

                    # Check if token contains date after underscore (e.g., "T_2025-10-24-EZ")
                    token_parts = next_token.split('_')
                    clean_part = token_parts[0]

                    # Remove trailing punctuation (comma, etc.)
                    clean_part = clean_part.rstrip(',').strip()

                    if clean_part:
                        advisor_parts.append(clean_part)

                    # If this token ends with '_' or contains a date, stop
                    if next_token.endswith('_') or any(re.match(r'^\d{4}', part) for part in token_parts[1:]):
                        break

                    j += 1

                if advisor_parts:
                    advisor_name = ' '.join(advisor_parts)
                    # Clean up any remaining commas and underscores
                    advisor_name = advisor_name.replace('_', ' ').replace(',', '').strip()
                    return advisor_name

        return None

    def _extract_client_name_from_clt(self, tokens: List[str]) -> Optional[str]:
        """
        Extract client name after 'clt' until 'Unknown' or company keyword.
        Example: "clt Ismael Tuguhore Unknown" → "Ismael Tuguhore"

        Args:
            tokens: List of tokens to search

        Returns:
            Client name or None
        """
        for i, token in enumerate(tokens):
            if token == 'clt':
                # Collect tokens until 'Unknown' or company keyword
                client_parts = []
                j = i + 1
                while j < len(tokens) and j < i + 10:  # Max 10 tokens for client name
                    next_token = tokens[j]

                    # Stop at separators
                    if next_token in ['Unknown', 'WS', 'Assurance'] or next_token in self.COMPANY_KEYWORDS:
                        break

                    client_parts.append(next_token)
                    j += 1

                if client_parts:
                    return ' '.join(client_parts).strip()

        return None

    def _extract_manuvie_info(self, tokens: List[str]) -> Optional[dict]:
        """
        Extract information from Manuvie pattern.
        Format: CODE-Manuvie-PRODUIT-DATE-COMPTE-CLIENT-CONSEILLER
        Example: "1305-Manuvie-32570-2025-10-07-643334-El Hajji-DL"

        This can span multiple tokens due to spaces.

        Args:
            tokens: List of tokens to search

        Returns:
            Dict with account_number, client_name, advisor_name or None
        """
        # Find token containing Manuvie pattern
        for i, token in enumerate(tokens):
            if '-Manuvie-' in token or token.startswith('Manuvie-'):
                # Reconstruct the full string by joining this and following tokens
                # until we find a date pattern or run out of tokens
                full_string = token
                j = i + 1
                # Join subsequent tokens that look like they're part of the Manuvie line
                while j < len(tokens) and j < i + 5:
                    next_token = tokens[j]
                    # Stop if we hit a separator or known keyword
                    if next_token in ['Unknown', 'WS', 'Assurance'] or next_token in self.COMPANY_KEYWORDS:
                        break
                    # Add the next token with space
                    full_string += ' ' + next_token
                    j += 1

                # Split by '-' to extract fields
                parts = full_string.split('-')
                if len(parts) >= 7:
                    # parts[0]: code
                    # parts[1]: Manuvie
                    # parts[2]: product
                    # parts[3-5]: date (2025-10-07)
                    # parts[6]: account number
                    # parts[7+]: client and advisor

                    account_number = parts[6].strip()

                    # Client name and advisor: combine remaining parts
                    # Last part is advisor, everything in between is client
                    remaining = parts[7:]
                    if len(remaining) >= 2:
                        # Last part is advisor
                        advisor_name = remaining[-1].strip()
                        # Everything else is client name
                        client_name = '-'.join(remaining[:-1]).strip()
                    elif len(remaining) == 1:
                        # Only one part - could be client or advisor
                        client_name = remaining[0].strip()
                        advisor_name = None
                    else:
                        client_name = None
                        advisor_name = None

                    return {
                        'account_number': account_number,
                        'client_name': client_name,
                        'advisor_name': advisor_name
                    }

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

                # CRITICAL FIX: Stop at next 'Â' (start of next record)
                # This prevents collecting tokens from multiple records
                if token in ["Â", "â", "Ã"]:
                    break

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

                # Safety: collect up to 50 tokens per record (reduced from 120)
                if len(all_tokens_collected) > 50:
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

            # Phase 6: Clean and extract metadata from complex cases
            # Check if this is a Manuvie case first (special handling)
            is_manuvie = any('-Manuvie-' in token or token.startswith('Manuvie-') for token in all_tokens_collected)
            if is_manuvie:
                manuvie_info = self._extract_manuvie_info(all_tokens_collected)
                if manuvie_info:
                    record.company = 'Manuvie'
                    if manuvie_info['account_number']:
                        record.account_number = manuvie_info['account_number']
                    if manuvie_info['client_name']:
                        record.client_name = manuvie_info['client_name']
                    if manuvie_info['advisor_name']:
                        record.advisor_name = manuvie_info['advisor_name']
            else:
                # Check if this is a complex case (contains metadata patterns)
                is_complex = (any(marker in all_tokens_collected for marker in ['Â', 'â', 'Ã', 'crt', 'clt']) or
                             any('#' in token for token in all_tokens_collected))
                if is_complex:
                    # Extract company name from metadata (this might overwrite existing company)
                    extracted_company = self._extract_company_after_special_char(all_tokens_collected)
                    if extracted_company:
                        record.company = extracted_company

                    # Extract commission rate
                    extracted_rate = self._extract_commission_rate(all_tokens_collected)
                    if extracted_rate:
                        record.on_commission_rate = extracted_rate

                    # Extract account number from # pattern (if current account is "Unknown")
                    if record.account_number == "Unknown":
                        extracted_account = self._extract_account_number_from_hash(all_tokens_collected)
                        if extracted_account:
                            record.account_number = extracted_account

                    # Extract advisor name
                    extracted_advisor = self._extract_advisor_name(all_tokens_collected)
                    if extracted_advisor:
                        record.advisor_name = extracted_advisor

                    # Extract client name from 'clt' pattern if present
                    extracted_client_from_clt = self._extract_client_name_from_clt(all_tokens_collected)
                    if extracted_client_from_clt:
                        record.client_name = extracted_client_from_clt
                    # Otherwise, if client name has '-' after account number, extract client name
                    elif '-' in record.client_name:
                        # Pattern 1: #account-CLIENT_... (client in same token)
                        # Pattern 2: #account- then CLIENT_crt in next token
                        for i, token in enumerate(all_tokens_collected):
                            if token.startswith('#') and '-' in token:
                                parts = token.split('-', 1)
                                if len(parts) > 1 and parts[1]:
                                    # Pattern 1: Client name is in same token
                                    client_part = parts[1].split('_')[0]  # Remove trailing metadata
                                    if client_part:
                                        record.client_name = client_part
                                        break
                                elif len(parts) > 1 and not parts[1]:
                                    # Pattern 2: Token ends with '-', client is in next token
                                    # Example: '#1006493-' then 'TOUZOUHOULIA_crt'
                                    if i + 1 < len(all_tokens_collected):
                                        next_token = all_tokens_collected[i + 1]
                                        # Extract client name before '_crt'
                                        client_part = next_token.split('_crt')[0].split('_')[0]
                                        if client_part:
                                            record.client_name = client_part
                                            break

            # Clean client name (remove parentheses, 'crt' keyword, and commas)
            record.client_name = self._clean_client_name(record.client_name)

            # Check if client name contains only metadata (no actual client name)
            # This handles cases like "Assomption_18456_75 % #1012264_crt Senhaji_2025-10-01-EZ"
            # where there's no real client name, only metadata
            if self._is_only_metadata(record.client_name):
                record.client_name = "Unknown"

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
                        'Frais de suivi nets': record.net_trailing_fee,
                        'Nom du conseiller': record.advisor_name,
                        'Taux sur-commission': record.on_commission_rate
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
            'Date', 'Frais de suivi brut', 'Frais de suivi nets',
            'Nom du conseiller', 'Taux sur-commission'
        ]

        df = pd.DataFrame(records, columns=columns)

        # Normalize advisor names to first names only
        if 'Nom du conseiller' in df.columns:
            df['Nom du conseiller'] = df['Nom du conseiller'].apply(normalize_advisor_name)

        # Normalize company names (e.g., "UV Insurance" -> "UV")
        if 'Compagnie' in df.columns:
            df['Compagnie'] = df['Compagnie'].apply(normalize_company_name)

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
    pdf_path = "../pdf/idc_statement/Statements (5).pdf"

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
