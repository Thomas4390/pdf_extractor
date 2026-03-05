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

import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

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
        except Exception as e:
            # Store error for debugging
            import logging
            logging.warning(f"Failed to load GCP credentials from Streamlit secrets: {e}")

    # Fallback to file-based credentials
    credentials_file = get_secret('GOOGLE_SHEETS_CREDENTIALS_FILE')
    if credentials_file:
        # Resolve relative path from project root
        if not os.path.isabs(credentials_file):
            credentials_file = Path(__file__).parent.parent.parent / credentials_file

        if Path(credentials_file).exists():
            return Credentials.from_service_account_file(str(credentials_file), scopes=scopes)

    return None


# Valid advisor status values
ADVISOR_STATUSES = ["Active", "New", "Inactive"]


@dataclass
class Advisor:
    """Represents an advisor with their name variations."""
    first_name: str  # Prénom principal
    last_name: str   # Nom de famille
    variations: list[str] = field(default_factory=list)  # Variations connues
    status: str = "Active"  # Statut: Active, New, Inactive
    created_at: Optional[str] = None  # Date de création (YYYY-MM-DD)
    email: Optional[str] = None  # Adresse email du conseiller

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

    def get_all_searchable_terms(self) -> list[str]:
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

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'first_name': self.first_name,
            'last_name': self.last_name,
            'variations': self.variations,
            'status': self.status,
            'created_at': self.created_at,
            'email': self.email,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Advisor':
        """Create from dictionary."""
        return cls(
            first_name=data['first_name'],
            last_name=data['last_name'],
            variations=data.get('variations', []),
            status=data.get('status', 'Active'),
            created_at=data.get('created_at'),
            email=data.get('email'),
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
        self.advisors: list[Advisor] = []
        self._compiled_patterns: list[tuple[re.Pattern, Advisor]] = []
        self._gsheets_error: Optional[str] = None
        self.recently_promoted: list[str] = []  # Names auto-promoted from New → Active

        # Initialize Google Sheets client
        self._gsheets_client: Optional[Any] = None
        self._worksheet: Optional[Any] = None
        self._use_gsheets = False

        # Try to initialize Google Sheets
        self._init_gsheets()

        if not self._use_gsheets:
            # Only set generic error if no specific error was set in _init_gsheets
            if self._gsheets_error is None:
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
            self._gsheets_error = "gspread library not installed"
            return

        spreadsheet_id = get_secret('GOOGLE_SHEETS_SPREADSHEET_ID')
        if not spreadsheet_id:
            self._gsheets_error = "GOOGLE_SHEETS_SPREADSHEET_ID not configured"
            return

        # Get credentials (from Streamlit secrets or file)
        credentials = get_gcp_credentials()
        if not credentials:
            self._gsheets_error = "GCP credentials not found (check gcp_service_account in secrets)"
            return

        try:
            self._gsheets_client = gspread.authorize(credentials)
            spreadsheet = self._gsheets_client.open_by_key(spreadsheet_id)

            # Get or create the Advisors worksheet
            try:
                self._worksheet = spreadsheet.worksheet(self.GSHEETS_WORKSHEET_NAME)
                # Ensure all expected columns exist (non-critical)
                try:
                    headers = self._worksheet.row_values(1)
                    if 'status' not in headers:
                        self._worksheet.update_cell(1, 5, 'status')
                        self._worksheet.format('E1', {'textFormat': {'bold': True}})
                        headers = self._worksheet.row_values(1)
                    if 'created_at' not in headers:
                        col_idx = len(headers) + 1
                        self._worksheet.update_cell(1, col_idx, 'created_at')
                        self._worksheet.format(f'{self._col_index_to_letter(col_idx - 1)}1', {'textFormat': {'bold': True}})
                        headers = self._worksheet.row_values(1)
                    if 'email' not in headers:
                        col_idx = len(headers) + 1
                        self._worksheet.update_cell(1, col_idx, 'email')
                        self._worksheet.format(f'{self._col_index_to_letter(col_idx - 1)}1', {'textFormat': {'bold': True}})
                except Exception:
                    pass
            except gspread.WorksheetNotFound:
                # Create the worksheet with headers
                self._worksheet = spreadsheet.add_worksheet(
                    title=self.GSHEETS_WORKSHEET_NAME,
                    rows=100, cols=7
                )
                self._worksheet.update('A1:G1', [['id', 'first_name', 'last_name', 'variations', 'status', 'created_at', 'email']])
                self._worksheet.format('A1:G1', {'textFormat': {'bold': True}})

            self._use_gsheets = True
            self._gsheets_error = None  # Clear any previous error
        except Exception as e:
            import logging
            logging.warning(f"Could not initialize Google Sheets client: {e}")
            self._gsheets_error = f"Google Sheets connection failed: {e}"
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

    @staticmethod
    def _col_index_to_letter(idx: int) -> str:
        """Convert a 0-based column index to Excel-style letter(s) (A, B, ..., Z, AA, AB, ...)."""
        result = ""
        idx += 1  # 1-based
        while idx:
            idx, rem = divmod(idx - 1, 26)
            result = chr(65 + rem) + result
        return result

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

            # Skip advisors with empty first or last name
            if not first or not last:
                continue

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
        """Load advisors from Google Sheets and auto-promote New → Active after 1 month."""
        try:
            records = self._worksheet.get_all_records()
            self.advisors = []
            advisors_to_promote = []

            for row in records:
                variations_str = row.get('variations', '')
                if isinstance(variations_str, str) and variations_str.strip():
                    variations = [v.strip() for v in variations_str.split(',') if v.strip()]
                else:
                    variations = []

                # Get status with default "Active"
                status = str(row.get('status', 'Active')).strip()
                if status not in ADVISOR_STATUSES:
                    status = 'Active'

                created_at = str(row.get('created_at', '')).strip() or None
                email = str(row.get('email', '')).strip() or None

                advisor = Advisor(
                    first_name=str(row.get('first_name', '')),
                    last_name=str(row.get('last_name', '')),
                    variations=variations,
                    status=status,
                    created_at=created_at,
                    email=email,
                )
                advisor._row_id = row.get('id')

                # Auto-promote New → Active after 1 month
                if advisor.status == "New" and advisor.created_at:
                    if self._should_promote_to_active(advisor.created_at):
                        advisor.status = "Active"
                        advisors_to_promote.append(advisor)

                self.advisors.append(advisor)

            # Batch-update promoted advisors in Google Sheets
            self.recently_promoted = [f"{a.first_name} {a.last_name}" for a in advisors_to_promote]
            if advisors_to_promote:
                self._promote_advisors_in_sheet(advisors_to_promote)

            self._build_patterns()
        except Exception as e:
            raise RuntimeError(f"Could not load advisors from Google Sheets: {e}") from e

    @staticmethod
    def _should_promote_to_active(created_at_str: str) -> bool:
        """Check if a New advisor should be promoted to Active (created > 1 month ago)."""
        try:
            created = datetime.strptime(created_at_str, "%Y-%m-%d").date()
            today = date.today()
            # Compare year/month: promote if we're at least 1 full month later
            months_diff = (today.year - created.year) * 12 + (today.month - created.month)
            return months_diff >= 1
        except (ValueError, TypeError):
            return False

    def _promote_advisors_in_sheet(self, advisors: list['Advisor']) -> None:
        """Update promoted advisors' status to Active in Google Sheets."""
        try:
            all_values = self._worksheet.get_all_values()
            updates = []
            # Find the status column index
            headers = all_values[0] if all_values else []
            status_col_idx = headers.index('status') if 'status' in headers else 4

            for advisor in advisors:
                for idx, row in enumerate(all_values):
                    if idx == 0:
                        continue
                    if row and str(row[0]) == str(advisor._row_id):
                        col_letter = self._col_index_to_letter(status_col_idx)
                        cell_ref = f'{col_letter}{idx + 1}'
                        updates.append({'range': cell_ref, 'values': [['Active']]})
                        break

            if updates:
                self._worksheet.batch_update(updates)
                import logging
                names = [f"{a.first_name} {a.last_name}" for a in advisors]
                logging.info(f"Auto-promoted {len(advisors)} advisor(s) from New to Active: {names}")
        except Exception as e:
            import logging
            logging.warning(f"Failed to promote advisors in Google Sheets: {e}")

    def _save(self):
        """Rebuild patterns after changes (data is already saved to Google Sheets)."""
        self._build_patterns()

    def add_advisor(self, first_name: str, last_name: str,
                    variations: Optional[list[str]] = None,
                    status: str = "Active",
                    email: Optional[str] = None) -> Advisor:
        """Add a new advisor to Google Sheets."""
        if not self._use_gsheets:
            raise RuntimeError(
                "Cannot add advisor: Google Sheets is not configured. "
                "Please configure GOOGLE_SHEETS_SPREADSHEET_ID and GCP credentials."
            )

        # Validate status
        if status not in ADVISOR_STATUSES:
            status = "Active"

        created_at = date.today().strftime("%Y-%m-%d")
        advisor = Advisor(
            first_name=first_name.strip(),
            last_name=last_name.strip(),
            variations=variations or [],
            status=status,
            created_at=created_at,
            email=email.strip() if email else None,
        )

        try:
            all_values = self._worksheet.col_values(1)
            ids = [int(v) for v in all_values[1:] if v.isdigit()]
            new_id = max(ids) + 1 if ids else 1
            variations_str = ', '.join(advisor.variations)
            self._worksheet.append_row([
                new_id, advisor.first_name, advisor.last_name, variations_str, advisor.status, created_at, advisor.email or ''
            ])
            advisor._row_id = new_id
        except Exception as e:
            raise RuntimeError(f"Could not save advisor to Google Sheets: {e}") from e

        self.advisors.append(advisor)
        self._save()
        return advisor

    def update_advisor(self, advisor: Advisor, first_name: str = None,
                       last_name: str = None, variations: list[str] = None,
                       status: str = None, email: str = None) -> Advisor:
        """
        Update an existing advisor in Google Sheets.

        Args:
            advisor: The advisor to update (must have _row_id)
            first_name: New first name (optional)
            last_name: New last name (optional)
            variations: New variations list (optional)
            status: New status (optional) - must be Active, New, or Inactive
            email: New email address (optional)

        Returns:
            The updated Advisor
        """
        if not self._use_gsheets:
            raise RuntimeError(
                "Cannot update advisor: Google Sheets is not configured."
            )

        if not hasattr(advisor, '_row_id') or advisor._row_id is None:
            raise ValueError("Advisor does not have a valid row ID")

        # Update the advisor object
        if first_name is not None:
            advisor.first_name = first_name.strip()
        if last_name is not None:
            advisor.last_name = last_name.strip()
        if variations is not None:
            advisor.variations = variations
        if status is not None and status in ADVISOR_STATUSES:
            advisor.status = status
        if email is not None:
            advisor.email = email.strip() if email.strip() else None

        try:
            # Find the row with matching id
            all_values = self._worksheet.get_all_values()
            row_index = None
            for idx, row in enumerate(all_values):
                if idx == 0:  # Skip header
                    continue
                if row and str(row[0]) == str(advisor._row_id):
                    row_index = idx + 1  # gspread uses 1-based indexing
                    break

            if row_index is None:
                raise ValueError(f"Advisor with id {advisor._row_id} not found in sheet")

            # Update the row (includes status, created_at, and email columns)
            variations_str = ', '.join(advisor.variations)
            self._worksheet.update(f'A{row_index}:G{row_index}', [[
                advisor._row_id, advisor.first_name, advisor.last_name, variations_str, advisor.status, advisor.created_at or '', advisor.email or ''
            ]])

        except Exception as e:
            raise RuntimeError(f"Could not update advisor in Google Sheets: {e}") from e

        self._save()
        return advisor

    def delete_advisor(self, advisor: Advisor) -> bool:
        """
        Delete an advisor from Google Sheets.

        Args:
            advisor: The advisor to delete (must have _row_id)

        Returns:
            True if deleted successfully
        """
        if not self._use_gsheets:
            raise RuntimeError(
                "Cannot delete advisor: Google Sheets is not configured."
            )

        if not hasattr(advisor, '_row_id') or advisor._row_id is None:
            raise ValueError("Advisor does not have a valid row ID")

        try:
            # Find the row with matching id
            all_values = self._worksheet.get_all_values()
            row_index = None
            for idx, row in enumerate(all_values):
                if idx == 0:  # Skip header
                    continue
                if row and str(row[0]) == str(advisor._row_id):
                    row_index = idx + 1  # gspread uses 1-based indexing
                    break

            if row_index is None:
                raise ValueError(f"Advisor with id {advisor._row_id} not found in sheet")

            # Delete the row
            self._worksheet.delete_rows(row_index)

            # Remove from local list
            self.advisors = [a for a in self.advisors
                           if not (hasattr(a, '_row_id') and a._row_id == advisor._row_id)]

        except Exception as e:
            raise RuntimeError(f"Could not delete advisor from Google Sheets: {e}") from e

        self._save()
        return True

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

    def match_full_name(self, name: str, use_fuzzy: bool = True) -> Optional[str]:
        """
        Match a name to an advisor and return the full name (first + last).

        Args:
            name: The name to match
            use_fuzzy: Whether to use fuzzy matching as fallback

        Returns:
            Full name (e.g., "Thomas Lussier"), or None if no match
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
                return advisor.full_name

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
                return best_match.full_name

        return None

    def match_full_name_or_original(self, name: str, use_fuzzy: bool = True) -> str:
        """Match a name and return the full name, or the original if no match."""
        result = self.match_full_name(name, use_fuzzy)
        return result if result else name

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

    def get_all_advisors(self) -> list[Advisor]:
        """Get all advisors."""
        return self.advisors.copy()

    def find_advisor(self, first_name: str, last_name: str) -> Optional[Advisor]:
        """
        Find an advisor by exact first name and last name match.

        Args:
            first_name: The first name to search for
            last_name: The last name to search for

        Returns:
            The matching Advisor if found, None otherwise
        """
        if not first_name or not last_name:
            return None

        first_normalized = self._normalize_text(first_name)
        last_normalized = self._normalize_text(last_name)

        for advisor in self.advisors:
            if (self._normalize_text(advisor.first_name) == first_normalized and
                self._normalize_text(advisor.last_name) == last_normalized):
                return advisor

        return None

    def reload(self):
        """Force reload advisors from storage."""
        self._load()


def get_advisor_matcher() -> AdvisorMatcher:
    """Get the global AdvisorMatcher singleton instance."""
    from src.utils.config import get_settings
    return AdvisorMatcher.get_instance(
        fuzzy_threshold=get_settings().advisor_fuzzy_threshold,
    )


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


def normalize_advisor_name_full(name: str) -> Optional[str]:
    """
    Normalize an advisor name to full name format (first + last).

    Args:
        name: The advisor name to normalize

    Returns:
        Full name (e.g., "Thomas Lussier"), or None if no match
    """
    if not name:
        return None

    matcher = get_advisor_matcher()
    return matcher.match_full_name(name, use_fuzzy=True)


def normalize_advisor_name_full_or_original(name: str) -> str:
    """
    Normalize an advisor name to full name format, returning original if no match.

    Args:
        name: The advisor name to normalize

    Returns:
        Full name (e.g., "Thomas Lussier"), or original if no match
    """
    if not name:
        return name

    matcher = get_advisor_matcher()
    return matcher.match_full_name_or_original(name, use_fuzzy=True)
