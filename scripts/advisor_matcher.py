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

Author: Thomas
Date: 2025
"""

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
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
                print(f"   ✓ Found gcp_service_account in Streamlit secrets")
                return Credentials.from_service_account_info(creds_dict, scopes=scopes)
        except Exception as e:
            print(f"   ⚠️ Error reading Streamlit secrets: {e}")

    # Fallback to file-based credentials
    credentials_file = get_secret('GOOGLE_SHEETS_CREDENTIALS_FILE')
    if credentials_file:
        # Resolve relative path from scripts directory
        if not os.path.isabs(credentials_file):
            credentials_file = Path(__file__).parent / credentials_file

        if Path(credentials_file).exists():
            return Credentials.from_service_account_file(str(credentials_file), scopes=scopes)

    return None


@dataclass
class Advisor:
    """Represents an advisor with their name variations."""
    first_name: str  # Prénom principal
    last_name: str   # Nom de famille
    variations: List[str] = field(default_factory=list)  # Variations connues (prénoms, surnoms, etc.)

    @property
    def display_name(self) -> str:
        """Return the standardized display name: 'Prénom + First letter of last name'."""
        if self.last_name:
            return f"{self.first_name} {self.last_name[0].upper()}."
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
    Robust advisor name matcher with fuzzy matching and persistent storage.

    Supports two storage backends:
    1. Google Sheets (primary, cloud-based) - uses GOOGLE_SHEETS_* env vars
    2. Local JSON file (fallback)

    Usage:
        matcher = AdvisorMatcher()

        # Add an advisor
        matcher.add_advisor("Thomas", "Lussier", ["Tom", "T. Lussier"])

        # Match a name
        result = matcher.match("Thomas Lussier")  # Returns "Thomas L."
        result = matcher.match("Lussier, Thomas")  # Returns "Thomas L."
        result = matcher.match("Tom")  # Returns "Thomas L."
    """

    DEFAULT_DATA_FILE = Path(__file__).parent / "advisors_data.json"
    GSHEETS_WORKSHEET_NAME = "Advisors"

    def __init__(self, data_file: Optional[Path] = None, fuzzy_threshold: float = 0.85,
                 use_gsheets: Optional[bool] = None):
        """
        Initialize the advisor matcher.

        Args:
            data_file: Path to JSON file for fallback storage. Uses default if None.
            fuzzy_threshold: Minimum similarity ratio for fuzzy matching (0.0 to 1.0).
                            Higher = stricter matching. Default 0.85.
            use_gsheets: Force Google Sheets on/off. If None, auto-detect from env vars.
        """
        self.data_file = data_file or self.DEFAULT_DATA_FILE
        self.fuzzy_threshold = fuzzy_threshold
        self.advisors: List[Advisor] = []
        self._compiled_patterns: List[Tuple[re.Pattern, Advisor]] = []

        # Initialize Google Sheets client if available
        self._gsheets_client: Optional[Any] = None
        self._worksheet: Optional[Any] = None
        self._use_gsheets = False

        if use_gsheets is None:
            # Auto-detect: use Google Sheets if env vars are set and library is available
            self._init_gsheets()
        elif use_gsheets and GSHEETS_AVAILABLE:
            self._init_gsheets()

        # Load existing data
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
        return "google_sheets" if self._use_gsheets else "local"

    # Common encoding issues (UTF-8 mojibake) - maps corrupted chars to correct ones
    ENCODING_FIXES = {
        # UTF-8 bytes interpreted as Latin-1/Windows-1252
        'Ã§': 'ç',  # ç
        'Ã©': 'é',  # é
        'Ã¨': 'è',  # è
        'Ãª': 'ê',  # ê
        'Ã«': 'ë',  # ë
        'Ã ': 'à',  # à
        'Ã¢': 'â',  # â
        'Ã®': 'î',  # î
        'Ã¯': 'ï',  # ï
        'Ã´': 'ô',  # ô
        'Ã¹': 'ù',  # ù
        'Ã»': 'û',  # û
        'Ã¼': 'ü',  # ü
        'Ã¿': 'ÿ',  # ÿ
        'Å"': 'œ',  # œ
        'Ã‰': 'É',  # É
        'Ã€': 'À',  # À
        'Ã‡': 'Ç',  # Ç
        'Ãˆ': 'È',  # È
        'Ãœ': 'Ü',  # Ü
        'â€™': "'",  # '
        'â€"': '-',  # –
        'â€"': '-',  # —
        'Â ': ' ',  # Non-breaking space
        '\xa0': ' ',  # Non-breaking space (raw)
    }

    def _fix_encoding(self, text: str) -> str:
        """
        Fix common encoding issues (mojibake) in text.

        Handles cases like "FayÃ§al" → "Fayçal" where UTF-8 text
        was incorrectly decoded as Latin-1 or Windows-1252.
        """
        if not text:
            return text

        result = str(text)

        # Apply known fixes
        for corrupted, correct in self.ENCODING_FIXES.items():
            result = result.replace(corrupted, correct)

        # Try to fix remaining mojibake by re-encoding
        try:
            # If text looks like mojibake, try to fix it
            if 'Ã' in result or 'Â' in result:
                # Try encoding as Latin-1 and decoding as UTF-8
                fixed = result.encode('latin-1').decode('utf-8')
                # Only use the fix if it doesn't create new issues
                if 'Ã' not in fixed and 'Â' not in fixed:
                    result = fixed
        except (UnicodeDecodeError, UnicodeEncodeError):
            # If re-encoding fails, keep the partially fixed version
            pass

        return result

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.

        - Fixes encoding issues (mojibake like "FayÃ§al" → "Fayçal")
        - Removes accents/diacritics
        - Converts to lowercase
        - Normalizes whitespace
        - Removes punctuation
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
        """Build compiled regex patterns for all advisors.

        Patterns are organized by priority:
        1. High priority: Full name patterns, variations with first name
        2. Medium priority: Initial + last name patterns
        3. Low priority: Last name only patterns (can cause ambiguity)

        This ordering ensures that more specific patterns are checked first.
        """
        high_priority = []
        medium_priority = []
        low_priority = []

        for advisor in self.advisors:
            # Escape special regex characters in names
            first = re.escape(self._normalize_text(advisor.first_name))
            last = re.escape(self._normalize_text(advisor.last_name))

            # HIGH PRIORITY: Full name patterns (most specific)
            high_patterns = [
                # Full name: "Thomas Lussier"
                rf'\b{first}\s+{last}\b',
                # Reversed: "Lussier Thomas" or "Lussier, Thomas"
                rf'\b{last}\s*,?\s*{first}\b',
            ]

            # Add variations as high priority (user-defined, so should match first)
            for variation in advisor.variations:
                var_normalized = re.escape(self._normalize_text(variation))
                if var_normalized:
                    high_patterns.append(rf'\b{var_normalized}\b')

            # Add first name only as high priority (for unique first names)
            high_patterns.append(rf'\b{first}\b')

            # MEDIUM PRIORITY: Initial + Last name patterns
            medium_patterns = [
                # Initial + Last: "T. Lussier" or "T Lussier"
                rf'\b{first[0]}\.?\s+{last}\b',
                # Last + Initial: "Lussier T." or "Lussier, T"
                rf'\b{last}\s*,?\s*{first[0]}\.?\b',
            ]

            # LOW PRIORITY: Last name only (can cause ambiguity with shared last names)
            low_patterns = [
                rf'\b{last}\b',
            ]

            # Compile and add to priority lists
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
        """Load advisors from storage (Google Sheets or local JSON)."""
        if self._use_gsheets:
            self._load_from_gsheets()
        else:
            self._load_from_json()

    def _load_from_gsheets(self):
        """Load advisors from Google Sheets."""
        try:
            # Get all records (skipping header row)
            records = self._worksheet.get_all_records()
            self.advisors = []
            for row in records:
                # Parse variations from comma-separated string
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
                # Store the row ID for updates (row number in sheet)
                advisor._row_id = row.get('id')
                self.advisors.append(advisor)
            self._build_patterns()
        except Exception as e:
            print(f"Warning: Could not load advisors from Google Sheets: {e}")
            # Fallback to local JSON
            self._use_gsheets = False
            self._load_from_json()

    def _load_from_json(self):
        """Load advisors from local JSON file."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.advisors = [Advisor.from_dict(a) for a in data.get('advisors', [])]
                    self._build_patterns()
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load advisors data: {e}")
                self.advisors = []
        else:
            self.advisors = []

    def _save(self):
        """Save advisors to storage (Google Sheets or local JSON)."""
        if self._use_gsheets:
            # For Google Sheets, saves are done incrementally in CRUD methods
            pass
        else:
            self._save_to_json()
        # Rebuild patterns after saving
        self._build_patterns()

    def _save_to_json(self):
        """Save advisors to local JSON file."""
        data = {
            'advisors': [a.to_dict() for a in self.advisors]
        }
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _get_next_gsheets_id(self) -> int:
        """Get the next available ID for Google Sheets."""
        try:
            all_values = self._worksheet.col_values(1)  # Column A (id)
            ids = [int(v) for v in all_values[1:] if v.isdigit()]  # Skip header
            return max(ids) + 1 if ids else 1
        except Exception:
            return len(self.advisors) + 1

    def _find_gsheets_row(self, row_id: int) -> Optional[int]:
        """Find the row number for a given ID in Google Sheets."""
        try:
            all_values = self._worksheet.col_values(1)  # Column A (id)
            for i, val in enumerate(all_values):
                if val == str(row_id):
                    return i + 1  # 1-indexed
            return None
        except Exception:
            return None

    def add_advisor(self, first_name: str, last_name: str,
                    variations: Optional[List[str]] = None) -> Advisor:
        """
        Add a new advisor.

        Args:
            first_name: Advisor's first name
            last_name: Advisor's last name
            variations: Optional list of name variations (nicknames, etc.)

        Returns:
            The created Advisor object
        """
        advisor = Advisor(
            first_name=first_name.strip(),
            last_name=last_name.strip(),
            variations=variations or []
        )

        if self._use_gsheets:
            try:
                new_id = self._get_next_gsheets_id()
                variations_str = ', '.join(advisor.variations)
                self._worksheet.append_row([
                    new_id,
                    advisor.first_name,
                    advisor.last_name,
                    variations_str
                ])
                advisor._row_id = new_id
            except Exception as e:
                print(f"Warning: Could not save advisor to Google Sheets: {e}")

        self.advisors.append(advisor)
        self._save()
        return advisor

    def update_advisor(self, index: int, first_name: str, last_name: str,
                       variations: Optional[List[str]] = None) -> Optional[Advisor]:
        """
        Update an existing advisor.

        Args:
            index: Index of the advisor to update
            first_name: New first name
            last_name: New last name
            variations: New list of variations

        Returns:
            Updated Advisor object, or None if index is invalid
        """
        if 0 <= index < len(self.advisors):
            advisor = self.advisors[index]
            advisor.first_name = first_name.strip()
            advisor.last_name = last_name.strip()
            if variations is not None:
                advisor.variations = variations

            if self._use_gsheets and hasattr(advisor, '_row_id'):
                try:
                    row_num = self._find_gsheets_row(advisor._row_id)
                    if row_num:
                        variations_str = ', '.join(advisor.variations)
                        self._worksheet.update(f'B{row_num}:D{row_num}', [[
                            advisor.first_name,
                            advisor.last_name,
                            variations_str
                        ]])
                except Exception as e:
                    print(f"Warning: Could not update advisor in Google Sheets: {e}")

            self._save()
            return advisor
        return None

    def delete_advisor(self, index: int) -> bool:
        """
        Delete an advisor.

        Args:
            index: Index of the advisor to delete

        Returns:
            True if deleted, False if index invalid
        """
        if 0 <= index < len(self.advisors):
            advisor = self.advisors[index]

            if self._use_gsheets and hasattr(advisor, '_row_id'):
                try:
                    row_num = self._find_gsheets_row(advisor._row_id)
                    if row_num:
                        self._worksheet.delete_rows(row_num)
                except Exception as e:
                    print(f"Warning: Could not delete advisor from Google Sheets: {e}")

            del self.advisors[index]
            self._save()
            return True
        return False

    def add_variation(self, index: int, variation: str) -> bool:
        """
        Add a variation to an existing advisor.

        Args:
            index: Index of the advisor
            variation: New variation to add

        Returns:
            True if added, False if index invalid
        """
        if 0 <= index < len(self.advisors):
            variation = variation.strip()
            if variation and variation not in self.advisors[index].variations:
                advisor = self.advisors[index]
                advisor.variations.append(variation)

                if self._use_gsheets and hasattr(advisor, '_row_id'):
                    try:
                        row_num = self._find_gsheets_row(advisor._row_id)
                        if row_num:
                            variations_str = ', '.join(advisor.variations)
                            self._worksheet.update(f'D{row_num}', [[variations_str]])
                    except Exception as e:
                        print(f"Warning: Could not update variations in Google Sheets: {e}")

                self._save()
            return True
        return False

    def remove_variation(self, advisor_index: int, variation_index: int) -> bool:
        """
        Remove a variation from an advisor.

        Args:
            advisor_index: Index of the advisor
            variation_index: Index of the variation to remove

        Returns:
            True if removed, False if indices invalid
        """
        if 0 <= advisor_index < len(self.advisors):
            advisor = self.advisors[advisor_index]
            if 0 <= variation_index < len(advisor.variations):
                del advisor.variations[variation_index]

                if self._use_gsheets and hasattr(advisor, '_row_id'):
                    try:
                        row_num = self._find_gsheets_row(advisor._row_id)
                        if row_num:
                            variations_str = ', '.join(advisor.variations)
                            self._worksheet.update(f'D{row_num}', [[variations_str]])
                    except Exception as e:
                        print(f"Warning: Could not update variations in Google Sheets: {e}")

                self._save()
                return True
        return False

    def match(self, name: str, use_fuzzy: bool = True) -> Optional[str]:
        """
        Match a name to an advisor and return the standardized display name.

        Args:
            name: The name to match
            use_fuzzy: Whether to use fuzzy matching as fallback

        Returns:
            Standardized display name (e.g., "Thomas L."), or None if no match
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

        # No match found
        return None

    def match_or_original(self, name: str, use_fuzzy: bool = True) -> str:
        """
        Match a name and return the standardized name, or the original if no match.

        Args:
            name: The name to match
            use_fuzzy: Whether to use fuzzy matching

        Returns:
            Standardized display name or original name
        """
        result = self.match(name, use_fuzzy)
        return result if result else name

    def get_all_advisors(self) -> List[Advisor]:
        """Get all advisors."""
        return self.advisors.copy()

    def get_advisor(self, index: int) -> Optional[Advisor]:
        """Get an advisor by index."""
        if 0 <= index < len(self.advisors):
            return self.advisors[index]
        return None

    def find_advisor_by_name(self, first_name: str, last_name: str) -> Optional[Tuple[int, Advisor]]:
        """
        Find an advisor by first and last name.

        Returns:
            Tuple of (index, Advisor) if found, None otherwise
        """
        for i, advisor in enumerate(self.advisors):
            if (self._normalize_text(advisor.first_name) == self._normalize_text(first_name) and
                self._normalize_text(advisor.last_name) == self._normalize_text(last_name)):
                return (i, advisor)
        return None

    def import_from_legacy_patterns(self, patterns: List[Tuple[str, str]]) -> int:
        """
        Import advisors from legacy pattern format.

        Args:
            patterns: List of (regex_pattern, simplified_name) tuples

        Returns:
            Number of advisors imported
        """
        # Extract unique simplified names
        unique_names = set(name for _, name in patterns)
        imported = 0

        for name in unique_names:
            # Check if already exists
            existing = self.find_advisor_by_name(name, "")
            if not existing:
                # Add with just first name (last name can be added manually)
                self.add_advisor(name, "", [])
                imported += 1

        return imported

    def export_statistics(self) -> Dict:
        """Export statistics about the advisor database."""
        total_variations = sum(len(a.variations) for a in self.advisors)
        return {
            'total_advisors': len(self.advisors),
            'total_variations': total_variations,
            'storage_backend': self.storage_backend,
            'advisors': [
                {
                    'display_name': a.display_name,
                    'full_name': a.full_name,
                    'num_variations': len(a.variations)
                }
                for a in self.advisors
            ]
        }

    def sync_to_gsheets(self) -> Tuple[int, int]:
        """
        Sync local JSON data to Google Sheets.

        This is useful for initial setup or migrating data to Google Sheets.
        Reads from local JSON and uploads all advisors to Google Sheets.

        Returns:
            Tuple of (advisors_synced, errors)
        """
        if not self._use_gsheets:
            raise RuntimeError("Google Sheets is not configured. Set GOOGLE_SHEETS_* env vars.")

        # Load from local JSON first
        local_advisors = []
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    local_advisors = [Advisor.from_dict(a) for a in data.get('advisors', [])]
            except Exception as e:
                print(f"Error reading local JSON: {e}")
                return (0, 1)

        if not local_advisors:
            print("No advisors found in local JSON file.")
            return (0, 0)

        synced = 0
        errors = 0

        try:
            # Clear existing data (except header)
            all_rows = self._worksheet.get_all_values()
            if len(all_rows) > 1:
                self._worksheet.delete_rows(2, len(all_rows))

            # Upload all advisors
            rows_to_add = []
            for i, advisor in enumerate(local_advisors, start=1):
                variations_str = ', '.join(advisor.variations)
                rows_to_add.append([i, advisor.first_name, advisor.last_name, variations_str])

            if rows_to_add:
                self._worksheet.append_rows(rows_to_add)
                synced = len(rows_to_add)

        except Exception as e:
            print(f"Error syncing to Google Sheets: {e}")
            errors = 1

        # Reload from Google Sheets
        self._load_from_gsheets()

        return (synced, errors)

    def sync_from_gsheets(self) -> Tuple[int, int]:
        """
        Sync Google Sheets data to local JSON.

        This is useful for creating a local backup of the cloud data.

        Returns:
            Tuple of (advisors_synced, errors)
        """
        if not self._use_gsheets:
            raise RuntimeError("Google Sheets is not configured. Set GOOGLE_SHEETS_* env vars.")

        try:
            # Load from Google Sheets
            self._load_from_gsheets()

            # Save to local JSON
            self._save_to_json()

            return (len(self.advisors), 0)
        except Exception as e:
            print(f"Error syncing from Google Sheets: {e}")
            return (0, 1)


# Convenience function for quick matching
def match_advisor_name(name: str, data_file: Optional[Path] = None) -> Optional[str]:
    """
    Quick function to match an advisor name.

    Args:
        name: The name to match
        data_file: Optional path to advisors data file

    Returns:
        Standardized display name or None if no match
    """
    matcher = AdvisorMatcher(data_file)
    return matcher.match(name)


if __name__ == "__main__":
    # Demo/test code
    matcher = AdvisorMatcher()

    print("=== Advisor Matcher Demo ===\n")

    # Show current advisors
    print(f"Current advisors: {len(matcher.advisors)}")
    for i, advisor in enumerate(matcher.advisors):
        print(f"  {i+1}. {advisor.display_name} ({advisor.full_name})")
        if advisor.variations:
            print(f"     Variations: {', '.join(advisor.variations)}")

    # Test matching
    test_names = [
        "Thomas Lussier",
        "Lussier, Thomas",
        "T. Lussier",
        "THOMAS LUSSIER",
        "Unknown Person",
    ]

    print("\nTest matching:")
    for name in test_names:
        result = matcher.match_or_original(name)
        print(f"  '{name}' → '{result}'")
