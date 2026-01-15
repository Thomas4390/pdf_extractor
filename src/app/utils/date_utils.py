"""
Date and group detection utilities.

Provides functions for converting dates to French month/year groups
and detecting dates from filenames and data.
"""

import re
from datetime import datetime
from typing import Optional

import pandas as pd


def get_months_fr() -> dict:
    """Retourne le dictionnaire des mois en français."""
    return {
        1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
        5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
        9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
    }


def date_to_group(date_val, fallback_group: str = None) -> str:
    """
    Convertit une date en nom de groupe "Mois YYYY".

    Args:
        date_val: Date (string YYYY-MM-DD, YYYY/MM/DD, datetime, ou Timestamp)
        fallback_group: Groupe à utiliser si la date n'est pas parsable

    Returns:
        str: Nom du groupe (ex: "Octobre 2025")
    """
    months_fr = get_months_fr()

    # Si None ou NaN, utiliser fallback ou date du jour
    if date_val is None or pd.isna(date_val):
        if fallback_group:
            return fallback_group
        now = datetime.now()
        return f"{months_fr[now.month]} {now.year}"

    # Gérer les Timestamp pandas directement
    if isinstance(date_val, pd.Timestamp):
        return f"{months_fr[date_val.month]} {date_val.year}"

    # Gérer les datetime
    if isinstance(date_val, datetime):
        return f"{months_fr[date_val.month]} {date_val.year}"

    date_str = str(date_val).strip()

    # Pattern YYYY-MM-DD ou YYYY/MM/DD
    match = re.match(r'(\d{4})[-/](\d{2})[-/](\d{2})', date_str)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        if 1 <= month <= 12:
            return f"{months_fr[month]} {year}"

    # Pattern DD/MM/YYYY ou DD-MM-YYYY
    match = re.match(r'(\d{2})[-/](\d{2})[-/](\d{4})', date_str)
    if match:
        month = int(match.group(2))
        year = int(match.group(3))
        if 1 <= month <= 12:
            return f"{months_fr[month]} {year}"

    # Essayer de parser avec pandas
    try:
        parsed = pd.to_datetime(date_str)
        if pd.notna(parsed):
            return f"{months_fr[parsed.month]} {parsed.year}"
    except Exception:
        pass

    # Fallback
    if fallback_group:
        return fallback_group
    now = datetime.now()
    return f"{months_fr[now.month]} {now.year}"


def detect_date_from_filename(filename: str) -> Optional[str]:
    """
    Détecte la date/mois à partir du nom de fichier PDF.

    Patterns supportés:
    - rappportremun_21622_2025-10-20.pdf -> "Octobre 2025"
    - Rapport des propositions soumises.20251017_1517.pdf -> "Octobre 2025"
    - 20251017_report.pdf -> "Octobre 2025"

    Returns:
        str: Nom du groupe (ex: "Octobre 2025") ou None si non détecté
    """
    months_fr = get_months_fr()

    # Patterns de date dans le nom de fichier
    patterns = [
        (r'(\d{4})-(\d{2})-(\d{2})', 1, 2),      # 2025-10-20
        (r'\.(\d{4})(\d{2})(\d{2})_', 1, 2),     # .20251017_
        (r'_(\d{4})(\d{2})(\d{2})', 1, 2),       # _20251017
        (r'^(\d{4})(\d{2})(\d{2})', 1, 2),       # 20251017 at start
    ]

    for pattern, year_pos, month_pos in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                year = int(match.group(year_pos))
                month = int(match.group(month_pos))
                if 1 <= month <= 12 and 2020 <= year <= 2030:
                    return f"{months_fr[month]} {year}"
            except (ValueError, IndexError):
                continue

    return None


def detect_groups_from_data(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Analyse le DataFrame extrait et assigne un groupe à chaque ligne basé sur la date.

    Args:
        df: DataFrame avec les données extraites
        source: Type de source (UV, IDC, IDC_STATEMENT, ASSOMPTION)

    Returns:
        DataFrame avec colonne '_target_group' ajoutée par ligne
    """
    df = df.copy()

    # Trouver la colonne de date
    date_column = None
    for col in ['Date', 'date', 'Émission', 'effective_date', 'report_date']:
        if col in df.columns:
            non_null_count = df[col].notna().sum()
            if non_null_count > 0:
                date_column = col
                break

    if date_column is None:
        # Pas de colonne de date - utiliser date du jour
        months_fr = get_months_fr()
        now = datetime.now()
        default_group = f"{months_fr[now.month]} {now.year}"
        df['_target_group'] = default_group
        return df

    # Assigner un groupe à chaque ligne basé sur sa date
    df['_target_group'] = df[date_column].apply(date_to_group)

    return df


def analyze_groups_in_data(df: pd.DataFrame) -> dict:
    """
    Analyse les groupes présents dans un DataFrame.

    Returns:
        {
            'unique_groups': ['Octobre 2025', 'Novembre 2025', ...],
            'spans_multiple_months': True/False,
            'group_counts': {'Octobre 2025': 15, 'Novembre 2025': 3}
        }
    """
    if '_target_group' not in df.columns:
        return {
            'unique_groups': [],
            'spans_multiple_months': False,
            'group_counts': {}
        }

    unique_groups = df['_target_group'].unique().tolist()
    group_counts = df['_target_group'].value_counts().to_dict()

    return {
        'unique_groups': unique_groups,
        'spans_multiple_months': len(unique_groups) > 1,
        'group_counts': group_counts
    }
