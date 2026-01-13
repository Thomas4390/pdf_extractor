"""
Robust Advisor Name Matcher
===========================

This module provides a robust system for matching advisor names
from various formats and variations to standardized names.

Features:
- Persistent advisor database (Google Sheets or local JSON fallback)
- Fuzzy matching with configurable threshold
- Support for multiple name variations per advisor
- Accent/diacritic normalization
- First name + Last name initial output format

Adapted from scripts/advisor_matcher.py for use in src/
"""

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# Try to import Google Sheets client
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to import Streamlit for secrets access
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def get_secret(key: str, default: str = None) -> str:
    """
    Get a secret value from multiple sources (priority order):
    1. Streamlit secrets
    2. Environment variables
    3. Default value
    """
    # Try Streamlit secrets first
    if STREAMLIT_AVAILABLE:
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass

    # Fallback to environment variable
    value = os.environ.get(key)
    if value:
        return value

    return default


def get_gcp_credentials():
    """
    Get Google Cloud credentials from multiple sources (priority order):
    1. Streamlit secrets (gcp_service_account table)
    2. Service account JSON file
    """
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    # Try Streamlit secrets first
    if STREAMLIT_AVAILABLE:
        try:
            if 'gcp_service_account' in st.secrets:
                creds_dict = dict(st.secrets['gcp_service_account'])
                return Credentials.from_service_account_info(creds_dict, scopes=scopes)
        except Exception:
            pass

    # Fallback to file-based credentials
    credentials_file = get_secret('GOOGLE_SHEETS_CREDENTIALS_FILE')
    if credentials_file:
        # Resolve relative path from project root
        if not os.path.isabs(credentials_file):
            credentials_file = Path(__file__).parent.parent.parent / credentials_file

        if Path(credentials_file).exists():
            return Credentials.from_service_account_file(str(credentials_file), scopes=scopes)

    return None


@dataclass
class Advisor:
    """Represents an advisor with their name variations."""
    first_name: str  # Prénom principal
    last_name: str   # Nom de famille
    variations: List[str] = field(default_factory=list)  # Variations connues

    @property
    def display_name(self) -> str:
        """Return the standardized display name: 'Prénom' only (simplified)."""
        return self.first_name

    @property
    def display_name_with_initial(self) -> str:
        """Return the display name with initial: 'Prénom + First letter of last name'."""
        if self.last_name:
            return f"{self.first_name} {self.last_name[0].upper()}."
        return self.first_name

    @property
    def display_name_compact(self) -> str:
        """Return compact display name: 'Prénom, X' (e.g., 'Guillaume, S')."""
        if self.last_name:
            return f"{self.first_name}, {self.last_name[0].upper()}"
        return self.first_name

    @property
    def full_name(self) -> str:
        """Return full name for matching."""
        return f"{self.first_name} {self.last_name}"

    def get_all_searchable_terms(self) -> List[str]:
        """Get all terms that can be used to match this advisor."""
        terms = [
            self.first_name.lower(),
            self.last_name.lower(),
            self.full_name.lower(),
            f"{self.last_name} {self.first_name}".lower(),  # Reversed order
        ]
        # Add all variations
        terms.extend([v.lower() for v in self.variations])
        return terms

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'first_name': self.first_name,
            'last_name': self.last_name,
            'variations': self.variations
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Advisor':
        """Create from dictionary."""
        return cls(
            first_name=data['first_name'],
            last_name=data['last_name'],
            variations=data.get('variations', [])
        )


class AdvisorMatcher:
    """
    Robust advisor name matcher with fuzzy matching and Google Sheets storage.

    Storage backend: Google Sheets ONLY (cloud-based) - uses GOOGLE_SHEETS_* env vars
    Local storage is NOT supported - Google Sheets must be configured.

    Usage:
        matcher = AdvisorMatcher()

        # Add an advisor
        matcher.add_advisor("Thomas", "Lussier", ["Tom", "T. Lussier", "Lussier"])

        # Match a name
        result = matcher.match("Thomas Lussier")  # Returns "Thomas"
        result = matcher.match("Lussier, Thomas")  # Returns "Thomas"
        result = matcher.match("Lussier")  # Returns "Thomas"
    """

    GSHEETS_WORKSHEET_NAME = "Advisors"

    # Singleton instance
    _instance: Optional['AdvisorMatcher'] = None

    @classmethod
    def get_instance(cls, **kwargs) -> 'AdvisorMatcher':
        """Get singleton instance of AdvisorMatcher."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None

    def __init__(self, fuzzy_threshold: float = 0.85, require_gsheets: bool = False):
        """
        Initialize the advisor matcher with Google Sheets storage.

        Args:
            fuzzy_threshold: Minimum similarity ratio for fuzzy matching (0.0 to 1.0).
                            Higher = stricter matching. Default 0.85.
            require_gsheets: If True, raise error if Google Sheets unavailable.
                            If False (default), work without advisors if not configured.
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.advisors: List[Advisor] = []
        self._compiled_patterns: List[Tuple[re.Pattern, Advisor]] = []
        self._gsheets_error: Optional[str] = None

        # Initialize Google Sheets client
        self._gsheets_client: Optional[Any] = None
        self._worksheet: Optional[Any] = None
        self._use_gsheets = False

        # Try to initialize Google Sheets
        self._init_gsheets()

        if not self._use_gsheets:
            self._gsheets_error = (
                "Google Sheets non configuré. "
                "Configurez GOOGLE_SHEETS_SPREADSHEET_ID et les credentials GCP."
            )
            if require_gsheets:
                raise RuntimeError(self._gsheets_error)
            # Continue without Google Sheets - matcher will return None for all matches

        # Load existing data from Google Sheets (if available)
        if self._use_gsheets:
            self._load()

    def _init_gsheets(self):
        """Initialize Google Sheets connection."""
        if not GSHEETS_AVAILABLE:
            return

        spreadsheet_id = get_secret('GOOGLE_SHEETS_SPREADSHEET_ID')
        if not spreadsheet_id:
            return

        # Get credentials (from Streamlit secrets or file)
        credentials = get_gcp_credentials()
        if not credentials:
            return

        try:
            self._gsheets_client = gspread.authorize(credentials)
            spreadsheet = self._gsheets_client.open_by_key(spreadsheet_id)

            # Get or create the Advisors worksheet
            try:
                self._worksheet = spreadsheet.worksheet(self.GSHEETS_WORKSHEET_NAME)
            except gspread.WorksheetNotFound:
                # Create the worksheet with headers
                self._worksheet = spreadsheet.add_worksheet(
                    title=self.GSHEETS_WORKSHEET_NAME,
                    rows=100, cols=4
                )
                self._worksheet.update('A1:D1', [['id', 'first_name', 'last_name', 'variations']])
                self._worksheet.format('A1:D1', {'textFormat': {'bold': True}})

            self._use_gsheets = True
        except Exception as e:
            print(f"Warning: Could not initialize Google Sheets client: {e}")
            self._use_gsheets = False

    @property
    def storage_backend(self) -> str:
        """Return the current storage backend being used."""
        return "google_sheets" if self._use_gsheets else "none"

    @property
    def is_configured(self) -> bool:
        """Return True if Google Sheets is properly configured."""
        return self._use_gsheets

    @property
    def configuration_error(self) -> Optional[str]:
        """Return the configuration error message, if any."""
        return self._gsheets_error

    # Common encoding issues (UTF-8 mojibake)
    ENCODING_FIXES = {
        'Ã§': 'ç', 'Ã©': 'é', 'Ã¨': 'è', 'Ãª': 'ê', 'Ã«': 'ë',
        'Ã ': 'à', 'Ã¢': 'â', 'Ã®': 'î', 'Ã¯': 'ï', 'Ã´': 'ô',
        'Ã¹': 'ù', 'Ã»': 'û', 'Ã¼': 'ü', 'Ã¿': 'ÿ', 'Å"': 'œ',
        'Ã‰': 'É', 'Ã€': 'À', 'Ã‡': 'Ç', 'Ãˆ': 'È', 'Ãœ': 'Ü',
        'â€™': "'", 'â€"': '-', 'Â ': ' ', '\xa0': ' ',
    }

    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues (mojibake) in text."""
        if not text:
            return text

        result = str(text)

        # Apply known fixes
        for corrupted, correct in self.ENCODING_FIXES.items():
            result = result.replace(corrupted, correct)

        # Try to fix remaining mojibake
        try:
            if 'Ã' in result or 'Â' in result:
                fixed = result.encode('latin-1').decode('utf-8')
                if 'Ã' not in fixed and 'Â' not in fixed:
                    result = fixed
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass

        return result

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        - Fixes encoding issues
        - Removes accents/diacritics
        - Converts to lowercase
        - Normalizes whitespace
        """
        if not text:
            return ""

        # First, fix encoding issues
        normalized = self._fix_encoding(str(text))

        # Normalize unicode (decompose accents)
        normalized = unicodedata.normalize('NFD', normalized)
        # Remove diacritical marks
        normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        # Lowercase
        normalized = normalized.lower()
        # Remove punctuation except spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        # Normalize whitespace
        normalized = ' '.join(normalized.split())

        return normalized

    def _build_patterns(self):
        """Build compiled regex patterns for all advisors."""
        high_priority = []
        medium_priority = []
        low_priority = []

        for advisor in self.advisors:
            first = re.escape(self._normalize_text(advisor.first_name))
            last = re.escape(self._normalize_text(advisor.last_name))

            # HIGH PRIORITY: Full name patterns
            high_patterns = [
                rf'\b{first}\s+{last}\b',
                rf'\b{last}\s*,?\s*{first}\b',
            ]

            # Add variations as high priority
            for variation in advisor.variations:
                var_normalized = re.escape(self._normalize_text(variation))
                if var_normalized:
                    high_patterns.append(rf'\b{var_normalized}\b')

            # Add first name only as high priority
            high_patterns.append(rf'\b{first}\b')

            # MEDIUM PRIORITY: Initial + Last name
            medium_patterns = [
                rf'\b{first[0]}\.?\s+{last}\b',
                rf'\b{last}\s*,?\s*{first[0]}\.?\b',
            ]

            # LOW PRIORITY: Last name only
            low_patterns = [rf'\b{last}\b']

            # Compile patterns
            for pattern_str in high_patterns:
                try:
                    pattern = re.compile(pattern_str, re.IGNORECASE)
                    high_priority.append((pattern, advisor))
                except re.error:
                    pass

            for pattern_str in medium_patterns:
                try:
                    pattern = re.compile(pattern_str, re.IGNORECASE)
                    medium_priority.append((pattern, advisor))
                except re.error:
                    pass

            for pattern_str in low_patterns:
                try:
                    pattern = re.compile(pattern_str, re.IGNORECASE)
                    low_priority.append((pattern, advisor))
                except re.error:
                    pass

        # Combine patterns in priority order
        self._compiled_patterns = high_priority + medium_priority + low_priority

    def _fuzzy_match(self, text: str, target: str) -> float:
        """Calculate similarity ratio between two strings."""
        return SequenceMatcher(None, text, target).ratio()

    def _load(self):
        """Load advisors from Google Sheets."""
        self._load_from_gsheets()

    def _load_from_gsheets(self):
        """Load advisors from Google Sheets."""
        try:
            records = self._worksheet.get_all_records()
            self.advisors = []
            for row in records:
                variations_str = row.get('variations', '')
                if isinstance(variations_str, str) and variations_str.strip():
                    variations = [v.strip() for v in variations_str.split(',') if v.strip()]
                else:
                    variations = []

                advisor = Advisor(
                    first_name=str(row.get('first_name', '')),
                    last_name=str(row.get('last_name', '')),
                    variations=variations
                )
                advisor._row_id = row.get('id')
                self.advisors.append(advisor)
            self._build_patterns()
        except Exception as e:
            raise RuntimeError(f"Could not load advisors from Google Sheets: {e}")

    def _save(self):
        """Rebuild patterns after changes (data is already saved to Google Sheets)."""
        self._build_patterns()

    def add_advisor(self, first_name: str, last_name: str,
                    variations: Optional[List[str]] = None) -> Advisor:
        """Add a new advisor to Google Sheets."""
        if not self._use_gsheets:
            raise RuntimeError(
                "Cannot add advisor: Google Sheets is not configured. "
                "Please configure GOOGLE_SHEETS_SPREADSHEET_ID and GCP credentials."
            )

        advisor = Advisor(
            first_name=first_name.strip(),
            last_name=last_name.strip(),
            variations=variations or []
        )

        try:
            all_values = self._worksheet.col_values(1)
            ids = [int(v) for v in all_values[1:] if v.isdigit()]
            new_id = max(ids) + 1 if ids else 1
            variations_str = ', '.join(advisor.variations)
            self._worksheet.append_row([
                new_id, advisor.first_name, advisor.last_name, variations_str
            ])
            advisor._row_id = new_id
        except Exception as e:
            raise RuntimeError(f"Could not save advisor to Google Sheets: {e}")

        self.advisors.append(advisor)
        self._save()
        return advisor

    def match(self, name: str, use_fuzzy: bool = True) -> Optional[str]:
        """
        Match a name to an advisor and return the standardized display name.

        Args:
            name: The name to match
            use_fuzzy: Whether to use fuzzy matching as fallback

        Returns:
            Standardized display name (e.g., "Thomas"), or None if no match
        """
        if not name or not isinstance(name, str):
            return None

        name = str(name).strip()
        if not name or name.lower() in ['none', 'nan', 'null']:
            return None

        # Normalize the input
        normalized_name = self._normalize_text(name)

        # Try regex pattern matching first
        for pattern, advisor in self._compiled_patterns:
            if pattern.search(normalized_name):
                return advisor.display_name

        # If no pattern match and fuzzy matching is enabled
        if use_fuzzy:
            best_match = None
            best_score = 0.0

            for advisor in self.advisors:
                for term in advisor.get_all_searchable_terms():
                    normalized_term = self._normalize_text(term)
                    score = self._fuzzy_match(normalized_name, normalized_term)

                    if score > best_score and score >= self.fuzzy_threshold:
                        best_score = score
                        best_match = advisor

            if best_match:
                return best_match.display_name

        return None

    def match_compact(self, name: str, use_fuzzy: bool = True) -> Optional[str]:
        """
        Match a name to an advisor and return the compact display name.

        Args:
            name: The name to match
            use_fuzzy: Whether to use fuzzy matching as fallback

        Returns:
            Compact display name (e.g., "Guillaume, S"), or None if no match
        """
        if not name or not isinstance(name, str):
            return None

        name = str(name).strip()
        if not name or name.lower() in ['none', 'nan', 'null']:
            return None

        # Normalize the input
        normalized_name = self._normalize_text(name)

        # Try regex pattern matching first
        for pattern, advisor in self._compiled_patterns:
            if pattern.search(normalized_name):
                return advisor.display_name_compact

        # If no pattern match and fuzzy matching is enabled
        if use_fuzzy:
            best_match = None
            best_score = 0.0

            for advisor in self.advisors:
                for term in advisor.get_all_searchable_terms():
                    normalized_term = self._normalize_text(term)
                    score = self._fuzzy_match(normalized_name, normalized_term)

                    if score > best_score and score >= self.fuzzy_threshold:
                        best_score = score
                        best_match = advisor

            if best_match:
                return best_match.display_name_compact

        return None

    def match_compact_or_original(self, name: str, use_fuzzy: bool = True) -> str:
        """Match a name and return the compact name, or the original if no match."""
        result = self.match_compact(name, use_fuzzy)
        return result if result else name

    def match_or_original(self, name: str, use_fuzzy: bool = True) -> str:
        """Match a name and return the standardized name, or the original if no match."""
        result = self.match(name, use_fuzzy)
        return result if result else name

    def get_all_advisors(self) -> List[Advisor]:
        """Get all advisors."""
        return self.advisors.copy()

    def reload(self):
        """Force reload advisors from storage."""
        self._load()


# Module-level singleton for easy access
_matcher_instance: Optional[AdvisorMatcher] = None


def get_advisor_matcher() -> AdvisorMatcher:
    """Get the global AdvisorMatcher instance."""
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = AdvisorMatcher()
    return _matcher_instance


def normalize_advisor_name(name: str) -> Optional[str]:
    """
    Normalize an advisor name using the AdvisorMatcher.

    This is a drop-in replacement for the old hardcoded function.

    Args:
        name: The advisor name to normalize

    Returns:
        Normalized advisor name, or None if no match
    """
    if not name:
        return None

    matcher = get_advisor_matcher()
    return matcher.match(name, use_fuzzy=True)


def normalize_advisor_name_or_original(name: str) -> str:
    """
    Normalize an advisor name, returning original if no match.

    Args:
        name: The advisor name to normalize

    Returns:
        Normalized advisor name, or original if no match
    """
    if not name:
        return name

    matcher = get_advisor_matcher()
    return matcher.match_or_original(name, use_fuzzy=True)


def normalize_advisor_name_compact(name: str) -> Optional[str]:
    """
    Normalize an advisor name to compact format (e.g., "Guillaume, S").

    Args:
        name: The advisor name to normalize

    Returns:
        Compact normalized name, or None if no match
    """
    if not name:
        return None

    matcher = get_advisor_matcher()
    return matcher.match_compact(name, use_fuzzy=True)


def normalize_advisor_name_compact_or_original(name: str) -> str:
    """
    Normalize an advisor name to compact format, returning original if no match.

    Args:
        name: The advisor name to normalize

    Returns:
        Compact normalized name, or original if no match
    """
    if not name:
        return name

    matcher = get_advisor_matcher()
    return matcher.match_compact_or_original(name, use_fuzzy=True)
