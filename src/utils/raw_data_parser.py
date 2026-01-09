"""
Rule-based parser for raw_client_data from IDC Statement extraction.

Uses the AdvisorMatcher class for dynamic advisor name matching based on
a database (Google Sheets or local JSON) rather than hardcoded patterns.

Parsing rules:
- Company: known list (UV, Assomption, Beneva, IA, RBC, Manuvie, SSQ, Desjardins, Empire, Ivari)
- Date: YYYY-MM-DD format (can be after company or after advisor with -EZ suffix)
- Policy number: after '#' OR digits followed by -ClientName
- Commission rate: "boni X%", "_X%", "recu p X%", or "recup X%"
- Advisor name: after 'crt' (with possible date suffix _YYYY-MM-DD-EZ) - matched via AdvisorMatcher
- Client name: after 'clt' OR between policy_number and '_crt'
"""

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

# Import the AdvisorMatcher for dynamic advisor name matching
from .advisor_matcher import normalize_advisor_name, normalize_advisor_name_or_original


# =============================================================================
# COMPANY NAME NORMALIZATION (copied from scripts/idc_statements_extractor.py)
# =============================================================================

_COMPANY_NAME_MAPPING = {
    'uv insurance': 'UV',
    'uv assurance': 'UV',
    'uv': 'UV',
    'assomption': 'Assomption',
    'asomption': 'Assomption',
    'assomption vie': 'Assomption',
    'industrielle alliance': 'IA',
    'industrielle': 'IA',
    'ia': 'IA',
    'beneva': 'Beneva',
    'manuvie': 'Manuvie',
    'manulife': 'Manuvie',
    'rbc': 'RBC',
    'rbc assurance': 'RBC',
    'rbc insurance': 'RBC',
    'ssq': 'SSQ',
    'ssq assurance': 'SSQ',
    'desjardins': 'Desjardins',
    'desjardins assurance': 'Desjardins',
    'empire': 'Empire',
    'empire vie': 'Empire',
    'empire life': 'Empire',
    'ivari': 'Ivari',
}


def normalize_company_name(company_name: str) -> Optional[str]:
    """Normalize company name to simplified standard name."""
    if pd.isna(company_name) or company_name is None:
        return None

    name_str = str(company_name).strip()
    if not name_str or name_str in ['None', 'nan', 'NaN']:
        return None

    name_lower = name_str.lower()
    if name_lower in _COMPANY_NAME_MAPPING:
        return _COMPANY_NAME_MAPPING[name_lower]

    for pattern, normalized in _COMPANY_NAME_MAPPING.items():
        if pattern in name_lower:
            return normalized

    return name_str


# =============================================================================
# RAW DATA PARSER
# =============================================================================

@dataclass
class ParsedClientData:
    """Parsed data from raw_client_data string."""

    company_code: Optional[str] = None
    company_number: Optional[str] = None
    policy_date: Optional[str] = None
    commission_rate: Optional[float] = None
    policy_number: Optional[str] = None
    advisor_name: Optional[str] = None
    client_first_name: Optional[str] = None
    client_last_name: Optional[str] = None
    raw_data: str = ""


def parse_raw_client_data(raw_data: str) -> ParsedClientData:
    """
    Parse raw_client_data using fixed rules.

    Handles multiple formats:
    FORMAT 1: "Â UV 7782 2025-11-17\\nboni 75% #111011722 crt\\nBourassa A clt Jeanny\\nBreault"
    FORMAT 2: "Â Assomption_8055_75%\\n#1014289-Mifoubdou_crt\\nBourassa_2025-12-01-EZ"
    FORMAT 3: "Â Assomption_8055_recu\\np 115% 1014494-\\nShabani_crt Ayoub_2025-\\n12-01-EZ"

    Rules:
    - Company: known list from _COMPANY_NAME_MAPPING
    - Date: YYYY-MM-DD format (can be after company or after advisor with -EZ suffix)
    - Policy number: after '#' OR digits followed by -ClientName
    - Commission rate: "boni X%", "_X%", "recu p X%", or "recup X%"
    - Advisor name: after 'crt' (with possible date suffix _YYYY-MM-DD-EZ)
    - Client name: after 'clt' OR between policy_number and '_crt'
    """
    result = ParsedClientData(raw_data=raw_data)

    if not raw_data:
        return result

    # Normalize newlines and whitespace for matching
    normalized = raw_data.replace("\n", " ").replace("  ", " ")
    tokens = normalized.split()

    # 1. Extract company (using existing mapping)
    for pattern, normalized_name in _COMPANY_NAME_MAPPING.items():
        if pattern in raw_data.lower():
            result.company_code = normalized_name
            break

    # Also check for company patterns like "Â UV" or "Assomption_8055"
    if not result.company_code:
        for token in tokens:
            clean_token = token.replace("Â", "").replace("â", "").strip()
            if clean_token.lower() in _COMPANY_NAME_MAPPING:
                result.company_code = _COMPANY_NAME_MAPPING[clean_token.lower()]
                break
            # Check underscore patterns like "Assomption_8055"
            if "_" in clean_token:
                company_part = clean_token.split("_")[0].lower()
                if company_part in _COMPANY_NAME_MAPPING:
                    result.company_code = _COMPANY_NAME_MAPPING[company_part]
                    break

    # 2. Extract company number (4-6 digits after company or with underscore)
    match = re.search(r"[A-Za-z]+[_\s](\d{4,6})", raw_data)
    if match:
        result.company_number = match.group(1)
    else:
        # Try finding standalone 4-6 digit number
        for token in tokens:
            if re.match(r"^\d{4,6}$", token):
                result.company_number = token
                break

    # 3. Extract date (YYYY-MM-DD) - can appear after advisor with -EZ suffix
    # First try standard date
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", raw_data)
    if match:
        result.policy_date = match.group(1)
    else:
        # Try date split across lines like "2025-\n12-01"
        match = re.search(r"(\d{4})-\s*(\d{2})-\s*(\d{2})", normalized)
        if match:
            result.policy_date = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    # 4. Extract commission rate - multiple patterns
    # Pattern: "boni X%", "_X%", "recu p X%", "recup X%"
    match = re.search(r"(?:boni|recu\s*p|recup|_)\s*(\d+)\s*%", raw_data, re.IGNORECASE)
    if match:
        result.commission_rate = float(match.group(1)) / 100
    else:
        # Simple pattern: just X%
        match = re.search(r"(\d+)\s*%", raw_data)
        if match:
            result.commission_rate = float(match.group(1)) / 100

    # 5. Extract policy number - multiple patterns
    # Pattern 1: after # (ex: #111011722 or #1014289-Mifoubdou)
    match = re.search(r"#\s*(\d{6,10})", raw_data, re.IGNORECASE)
    if match:
        result.policy_number = match.group(1)
    else:
        # Pattern 2: digits followed by -ClientName (ex: 1014289-Shabani)
        match = re.search(r"\b(\d{6,10})-[A-Za-zÀ-ÿ]", raw_data)
        if match:
            result.policy_number = match.group(1)
        else:
            # Pattern 3: standalone 7-10 digit number
            for token in tokens:
                clean = token.replace("#", "").split("-")[0]
                if re.match(r"^\d{7,10}$", clean):
                    result.policy_number = clean
                    break

    # 6. Extract client name - multiple patterns
    # Pattern 1: after 'clt' (traditional format)
    match = re.search(r"\bclt\s+(.+?)(?:\s*$|\s+Unknown)", raw_data, re.IGNORECASE | re.DOTALL)
    if match:
        full_name = match.group(1).strip()
        full_name = " ".join(full_name.split())
        parts = full_name.split()
        if len(parts) >= 2:
            result.client_first_name = parts[0]
            result.client_last_name = " ".join(parts[1:])
        elif len(parts) == 1:
            result.client_first_name = parts[0]
    else:
        # Pattern 2: between policy_number- and _crt (ex: 1014289-Mifoubdou_crt)
        match = re.search(r"\d{6,10}-([A-Za-zÀ-ÿ\-]+?)_crt", raw_data, re.IGNORECASE)
        if match:
            client_name = match.group(1).strip()
            result.client_first_name = client_name
        else:
            # Pattern 3: between policy_number- and _crt with space (ex: 1014494- Shabani_crt)
            match = re.search(r"\d{6,10}-\s*([A-Za-zÀ-ÿ\-]+?)[\s_]crt", normalized, re.IGNORECASE)
            if match:
                client_name = match.group(1).strip()
                result.client_first_name = client_name

    # 7. Extract advisor name - multiple patterns
    # Pattern 1: between 'crt' and 'clt'
    match = re.search(r"\bcrt\s+(.+?)\s+clt\b", raw_data, re.IGNORECASE | re.DOTALL)
    if match:
        advisor = match.group(1).strip()
        # Remove date suffix like _2025-12-01-EZ
        advisor = re.sub(r"_?\d{4}-\d{2}-\d{2}(?:-[A-Z]+)?", "", advisor)
        advisor = advisor.replace("_", " ").replace("\n", " ").strip()
        if advisor:
            result.advisor_name = normalize_advisor_name(advisor)
    else:
        # Pattern 2: after '_crt ' or 'crt ' with date suffix
        match = re.search(r"[_\s]crt\s+([A-Za-zÀ-ÿ]+?)(?:_\d{4}|-\d{4}|\s*$)", normalized, re.IGNORECASE)
        if match:
            advisor = match.group(1).strip()
            if advisor:
                result.advisor_name = normalize_advisor_name(advisor)
        else:
            # Pattern 3: just after _crt or crt until end or date
            match = re.search(r"crt\s*([A-Za-zÀ-ÿ\s]+?)(?:_?\d{4}|$)", normalized, re.IGNORECASE)
            if match:
                advisor = match.group(1).strip().replace("_", " ")
                if advisor:
                    result.advisor_name = normalize_advisor_name(advisor)

    return result


def parse_raw_entries_batch(entries: list[dict]) -> list[dict]:
    """
    Parse a batch of raw entries using fixed rules.

    Args:
        entries: List of dicts with 'raw_client_data' field

    Returns:
        List of dicts with parsed fields added
    """
    results = []

    for entry in entries:
        raw_data = entry.get("raw_client_data", "")
        parsed = parse_raw_client_data(raw_data)

        result = {
            # Original fields
            "raw_client_data": entry.get("raw_client_data", ""),
            "account_number": entry.get("account_number", "Unknown"),
            "company": entry.get("company", "Unknown"),
            "product": entry.get("product", "Unknown"),
            "date": entry.get("date", "Unknown"),
            "gross_trailing_fee": entry.get("gross_trailing_fee", "0,00 $"),
            "net_trailing_fee": entry.get("net_trailing_fee", "0,00 $"),
            # Parsed fields
            "company_code": parsed.company_code,
            "company_number": parsed.company_number,
            "policy_date": parsed.policy_date,
            "commission_rate": parsed.commission_rate,
            "policy_number": parsed.policy_number,
            "advisor_name": parsed.advisor_name,
            "client_first_name": parsed.client_first_name,
            "client_last_name": parsed.client_last_name,
        }
        results.append(result)

    return results
