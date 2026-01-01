#!/usr/bin/env python3
"""
Assomption Vie PDF Data Extractor - Generalized Version

This script extracts commission and bonus data from Assomption Vie remuneration
PDF reports. It automatically detects the relevant sections based on column
headers rather than fixed page numbers, making it more robust to variations
in document structure.

Key improvements:
- Dynamic section detection based on column headers
- Flexible page location (not hardcoded to pages 3 and 5)
- Better error handling and logging
- More maintainable code structure

Author: Data Extraction System
Version: 2.0.0
"""

from typing import List, Dict, Optional, Tuple
import re
import fitz  # PyMuPDF
import pandas as pd


def extract_all_pages_text(pdf_path: str) -> Dict[int, str]:
    """
    Extract text content from all PDF pages.

    Args:
        pdf_path: Full path to the PDF file

    Returns:
        Dictionary mapping page numbers (1-indexed) to text content

    Raises:
        FileNotFoundError: If PDF file doesn't exist
    """
    try:
        doc = fitz.open(pdf_path)
        pages_text = {}

        for page_num in range(len(doc)):
            page = doc[page_num]
            pages_text[page_num + 1] = page.get_text()  # 1-indexed

        doc.close()
        return pages_text

    except Exception as e:
        raise Exception(f"Error extracting PDF pages: {e}")


def find_section_by_headers(pages_text: Dict[int, str],
                            header_keywords: List[str],
                            section_name: str = "",
                            min_matches: int = None) -> Optional[Tuple[int, str]]:
    """
    Find the page containing a specific section based on header keywords.

    Args:
        pages_text: Dictionary of page numbers to text content
        header_keywords: List of keywords that identify the section header
        section_name: Name of the section for logging purposes
        min_matches: Minimum number of keywords that must match (default: all)

    Returns:
        Tuple of (page_number, page_text) if found, None otherwise
    """
    if min_matches is None:
        min_matches = len(header_keywords)

    best_match = None
    best_match_count = 0

    for page_num, text in pages_text.items():
        text_lower = text.lower()
        # Count how many keywords are present
        match_count = sum(1 for keyword in header_keywords if keyword.lower() in text_lower)

        # If we meet the minimum threshold and it's better than previous matches
        if match_count >= min_matches and match_count > best_match_count:
            best_match = (page_num, text)
            best_match_count = match_count

    if best_match:
        print(
            f"✓ Found '{section_name}' section on page {best_match[0]} ({best_match_count}/{len(header_keywords)} keywords matched)")
        return best_match

    print(f"⚠ Warning: '{section_name}' section not found")
    return None


def parse_commission_data(text: str) -> List[Dict[str, any]]:
    """
    Parse commission data from the commission section of the PDF report.

    This function looks for the commission table which has the following columns:
    - Code (e.g., AOH1)
    - Numéro Police (7-digit policy number)
    - Nom de l'assuré (insured person's name)
    - Produit (product code, e.g., 4T20 B)
    - Émission (issue date in YYYY/MM/DD format)
    - Fréquence paiement (payment frequency)
    - Facturation (billing type)
    - Prime (premium amount)
    - Taux (commission rate as percentage)
    - Commissions (commission amount)

    The function identifies the data section by looking for the column headers
    which are more stable than company names.

    Args:
        text: Raw text extracted from the commission page

    Returns:
        List of dictionaries containing parsed commission records
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    records: List[Dict[str, any]] = []

    # Locate the start of data section by finding the column headers
    # The headers are at the top (Code, Numéro Police, etc.)
    # Then we look for "Commissions de première année"
    # Then the first transaction code
    start_idx: Optional[int] = None

    # First, find where the column headers are
    headers_found = False
    for idx, line in enumerate(lines):
        if 'Code' in line and idx < 20:  # Headers are near the top
            # Check if we have the key headers nearby
            window = ' '.join(lines[idx:min(idx + 15, len(lines))]).lower()
            if 'numéro police' in window and 'commissions' in window:
                headers_found = True
                # Now look for the first transaction code after headers
                for j in range(idx, len(lines)):
                    if re.match(r'^[A-Z]{2,4}\d+$', lines[j]):
                        start_idx = j
                        break
                break

    if start_idx is None:
        print("⚠ Could not find start of commission data section (no column headers or transaction codes found)")
        return records

    # Process lines in fixed blocks of 10
    i = start_idx
    while i < len(lines):
        line = lines[i]

        # Stop when we reach the totals section
        if "Total CPA" in line or "Total des commissions" in line:
            break

        # Check if this line is a transaction code (e.g., AOH1, BCD2)
        if re.match(r'^[A-Z]{2,4}\d+$', line):
            # Ensure we have enough lines for a complete record
            if i + 9 < len(lines):
                try:
                    record = {
                        'Code': lines[i],
                        'Numéro Police': lines[i + 1],
                        'Nom de l\'assuré': lines[i + 2],
                        'Produit': lines[i + 3],
                        'Émission': lines[i + 4],
                        'Fréquence paiement': lines[i + 5],
                        'Facturation': lines[i + 6],
                        'Prime': float(lines[i + 7].replace(',', '.')),
                        'Taux Commission': lines[i + 8],  # Keep as string (e.g., "40,9930%")
                        'Commissions': float(lines[i + 9].replace(',', '.'))
                    }
                    records.append(record)
                    i += 10  # Move to next record
                except (ValueError, IndexError) as e:
                    print(f"⚠ Warning: Error parsing commission record at line {i}: {e}")
                    i += 1
            else:
                break
        else:
            i += 1

    return records


def parse_bonus_data(text: str) -> List[Dict[str, any]]:
    """
    Parse bonus (surcommission) data from the bonus section of the PDF report.

    This function looks for the bonus table with these columns:
    - Polices (policy number)
    - Assurés (insured person's name)
    - Prod. (product code - may span 1-2 lines)
    - Commissions Première Année (first year commission amount)
    - Taux (bonus rate as percentage)
    - Boni (bonus amount)

    The function identifies the section by looking for column headers
    which are more stable than section titles.

    Args:
        text: Raw text extracted from the bonus page

    Returns:
        List of dictionaries containing parsed bonus records
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    records: List[Dict[str, any]] = []

    # Locate the start of data section by finding column headers
    # Look for "Polices", "Assurés", "Prod." appearing together
    start_idx: Optional[int] = None
    for idx, line in enumerate(lines):
        # Check if this area contains the bonus table headers
        if idx + 10 < len(lines):
            window = ' '.join(lines[idx:idx + 15]).lower()
            # Look for key headers that indicate the bonus section data
            if ('polices' in window or 'police' in window) and 'assurés' in window and 'boni' in window:
                # Now find the first policy number (7 digits)
                for j in range(idx, min(idx + 20, len(lines))):
                    if re.match(r'^\d{7}$', lines[j]):
                        start_idx = j
                        break
                if start_idx:
                    break

    if start_idx is None:
        print("⚠ Could not find start of bonus data section (no column headers or policy numbers found)")
        return records

    # Process lines dynamically
    i = start_idx
    while i < len(lines):
        line = lines[i]

        # Stop when we reach the totals line
        if line == "Total" or "Total" in line and i > start_idx + 3:
            break

        # Check if this line is a policy number (7 digits)
        if re.match(r'^\d{7}$', line):
            record: Dict[str, any] = {}
            record['Numéro Police'] = lines[i]
            i += 1

            # Extract insured person's name
            if i < len(lines):
                record['Nom de l\'assuré'] = lines[i]
                i += 1

            # Extract product code (may be on 1 or 2 lines)
            produit_parts = []
            if i < len(lines):
                produit_parts.append(lines[i])
                i += 1

                # Check if the next line is a single letter or short code (part of product code)
                if i < len(lines) and re.match(r'^[A-Z]$', lines[i]):
                    produit_parts.append(lines[i])
                    i += 1

            record['Produit'] = ' '.join(produit_parts)

            # Extract first year commission amount
            if i < len(lines) and re.match(r'^-?\d+,\d+$', lines[i]):
                try:
                    record['Commissions Première Année'] = float(lines[i].replace(',', '.'))
                    i += 1
                except ValueError:
                    pass

            # Extract bonus rate
            if i < len(lines) and '%' in lines[i]:
                record['Taux Boni'] = lines[i]  # Keep as string (e.g., "175,00%")
                i += 1

            # Extract bonus amount
            if i < len(lines) and re.match(r'^-?\d+,\d+$', lines[i]):
                try:
                    record['Boni'] = float(lines[i].replace(',', '.'))
                    i += 1
                except ValueError:
                    pass

            # Only add record if it has the essential fields
            if 'Boni' in record and 'Taux Boni' in record:
                records.append(record)
        else:
            i += 1

    return records


def fuzzy_name_match(name1: str, name2: str) -> bool:
    """
    Check if two names match, allowing for truncation.

    This handles cases where one name is a truncated version of the other,
    such as "PATRICK KUNSEVI BE" vs "PATRICK KUNSEVI BENDA".

    Args:
        name1: First name to compare
        name2: Second name to compare

    Returns:
        True if names match (exact or one starts with the other), False otherwise
    """
    return (name1.startswith(name2) or
            name2.startswith(name1) or
            name1 == name2)


def extract_pdf_data(pdf_path: str) -> pd.DataFrame:
    """
    Extract and merge commission and bonus data from PDF report.

    This is the main orchestration function that:
    1. Extracts all pages from the PDF
    2. Automatically locates the commission and bonus sections
    3. Parses the data from each section
    4. Intelligently merges the two datasets using policy number,
       product code, and fuzzy name matching
    5. Returns a unified DataFrame

    Args:
        pdf_path: Full path to the PDF file

    Returns:
        DataFrame containing merged commission and bonus data

    Raises:
        Exception: If extraction or merging fails
    """
    print("📄 Reading PDF file...")
    pages_text = extract_all_pages_text(pdf_path)
    print(f"✅ Found {len(pages_text)} pages\n")

    # Find commission section
    print("🔍 Searching for commission section...")
    commission_keywords = ["ASSURANCE VIE INDIVIDUELLE", "Commissions de première année", "Numéro Police"]
    commission_section = find_section_by_headers(
        pages_text,
        commission_keywords,
        "Commission Data",
        min_matches=2  # At least 2 keywords must match
    )

    df_commissions = pd.DataFrame()
    if commission_section:
        page_num, page_text = commission_section
        print(f"📄 Extracting commission data from page {page_num}...")
        commission_data = parse_commission_data(page_text)
        df_commissions = pd.DataFrame(commission_data)
        print(f"✅ Found {len(df_commissions)} commission records\n")
    else:
        print("⚠ No commission data found\n")

    # Find bonus section
    print("🔍 Searching for bonus section...")
    bonus_keywords = ["Surcommission sur la production", "Boni", "Polices"]
    bonus_section = find_section_by_headers(
        pages_text,
        bonus_keywords,
        "Bonus Data",
        min_matches=2  # At least 2 keywords must match
    )

    df_bonus = pd.DataFrame()
    if bonus_section:
        page_num, page_text = bonus_section
        print(f"📄 Extracting bonus data from page {page_num}...")
        bonus_data = parse_bonus_data(page_text)
        df_bonus = pd.DataFrame(bonus_data)
        print(f"✅ Found {len(df_bonus)} bonus records\n")
    else:
        print("⚠ No bonus data found\n")

    # Merge the two datasets with intelligent matching
    if not df_commissions.empty and not df_bonus.empty:
        print("🔗 Merging commission and bonus data...")
        df_merged = df_commissions.copy()
        df_merged['Taux Boni'] = None
        df_merged['Boni'] = None

        matched_count = 0

        # For each commission record, find the matching bonus record
        for idx, comm_row in df_commissions.iterrows():
            # Find candidate bonus records with matching policy and product
            candidates = df_bonus[
                (df_bonus['Numéro Police'] == comm_row['Numéro Police']) &
                (df_bonus['Produit'] == comm_row['Produit'])
                ]

            if len(candidates) == 1:
                # Only one match found - use it directly
                bonus_row = candidates.iloc[0]
                df_merged.at[idx, 'Taux Boni'] = bonus_row['Taux Boni']
                df_merged.at[idx, 'Boni'] = bonus_row['Boni']
                matched_count += 1

                # Use the more complete name (longer version)
                if len(bonus_row['Nom de l\'assuré']) > len(comm_row['Nom de l\'assuré']):
                    df_merged.at[idx, 'Nom de l\'assuré'] = bonus_row['Nom de l\'assuré']

            elif len(candidates) > 1:
                # Multiple matches - use fuzzy name matching to find correct one
                comm_name = comm_row['Nom de l\'assuré']

                for _, bonus_row in candidates.iterrows():
                    bonus_name = bonus_row['Nom de l\'assuré']

                    if fuzzy_name_match(comm_name, bonus_name):
                        df_merged.at[idx, 'Taux Boni'] = bonus_row['Taux Boni']
                        df_merged.at[idx, 'Boni'] = bonus_row['Boni']
                        matched_count += 1

                        # Use the more complete name
                        if len(bonus_name) > len(comm_name):
                            df_merged.at[idx, 'Nom de l\'assuré'] = bonus_name
                        break

        print(f"✅ Merged {matched_count}/{len(df_commissions)} records with bonus data")
        return df_merged

    elif not df_commissions.empty:
        print("ℹ Returning commission data only (no bonus data to merge)")
        return df_commissions

    elif not df_bonus.empty:
        print("ℹ Returning bonus data only (no commission data to merge)")
        return df_bonus

    else:
        print("⚠ No data extracted from PDF")
        return pd.DataFrame()


def main() -> None:
    """
    Main execution function.

    Orchestrates the complete workflow:
    1. Extract data from PDF (with automatic section detection)
    2. Display results to console
    3. Save to CSV file
    """
    # Configuration
    pdf_path = "../pdf/assomption/Remuneration (61).pdf"
    output_file = "../results/assomption_data.csv"

    print("=" * 80)
    print("🚀 ASSOMPTION VIE PDF DATA EXTRACTOR v2.0")
    print("=" * 80)
    print("Features: Dynamic section detection, flexible page location")
    print("=" * 80)
    print()

    try:
        # Extract and merge data
        df = extract_pdf_data(pdf_path)

        if df.empty:
            print("\n❌ No data extracted. Please check the PDF format.")
            return

        # Display results to console
        print("\n" + "=" * 80)
        print("EXTRACTED DATA")
        print("=" * 80)
        print(f"\nTotal records: {len(df)}\n")
        print(df.to_string(index=False))

        # Display column information
        print("\n" + "=" * 80)
        print("COLUMN SUMMARY")
        print("=" * 80)
        print(df.info())

        # Calculate summary statistics
        if 'Commissions' in df.columns:
            total_commissions = df['Commissions'].sum()
            print(f"\n💰 Total Commissions: {total_commissions:.2f}")

        if 'Boni' in df.columns:
            total_bonus = df['Boni'].sum()
            print(f"🎁 Total Bonus: {total_bonus:.2f}")

        # Save to CSV with UTF-8 encoding (includes BOM for Excel compatibility)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 Data saved to: {output_file}")
        print("✅ Extraction completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()