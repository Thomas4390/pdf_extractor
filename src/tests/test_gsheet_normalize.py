#!/usr/bin/env python3
"""
Test script for Google Sheets advisor normalization.

This script normalizes advisor names in a Google Sheet to the compact format
"Prénom, Initiale" (e.g., "Guillaume St-Pierre" → "Guillaume, S").

Usage:
    python -m src.tests.test_gsheet_normalize [options]

Options:
    --dry-run           Preview changes without applying them (default)
    --apply             Actually apply the changes to the spreadsheet
    --sheet <name>      Process only the specified worksheet
    --column <name>     Column to normalize (default: 'Conseiller')
    --list-sheets       List all worksheets in the spreadsheet
    --list-advisors     List all advisors in the database

Examples:
    python -m src.tests.test_gsheet_normalize                    # Dry run on all sheets
    python -m src.tests.test_gsheet_normalize --apply            # Apply changes
    python -m src.tests.test_gsheet_normalize --sheet "Data"     # Process specific sheet
    python -m src.tests.test_gsheet_normalize --list-sheets      # List available sheets
    python -m src.tests.test_gsheet_normalize --list-advisors    # List advisor database
"""

import os
import sys
from pathlib import Path

# Add project root to path (src/tests -> src -> project_root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(args: list[str]) -> dict:
    """Parse command line arguments."""
    options = {
        'dry_run': True,
        'sheet': None,
        'column': 'Conseiller',
        'list_sheets': False,
        'list_advisors': False,
    }

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--dry-run':
            options['dry_run'] = True
            i += 1
        elif arg == '--apply':
            options['dry_run'] = False
            i += 1
        elif arg == '--sheet':
            if i + 1 < len(args):
                options['sheet'] = args[i + 1]
                i += 2
            else:
                print("ERROR: --sheet requires a value")
                sys.exit(1)
        elif arg == '--column':
            if i + 1 < len(args):
                options['column'] = args[i + 1]
                i += 2
            else:
                print("ERROR: --column requires a value")
                sys.exit(1)
        elif arg == '--list-sheets':
            options['list_sheets'] = True
            i += 1
        elif arg == '--list-advisors':
            options['list_advisors'] = True
            i += 1
        elif arg in ('--help', '-h'):
            print(__doc__)
            sys.exit(0)
        else:
            print(f"WARNING: Unknown option {arg}")
            i += 1

    return options


def list_sheets():
    """List all worksheets in the spreadsheet."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        print("ERROR: gspread not installed. Run: pip install gspread google-auth")
        return

    from dotenv import load_dotenv
    load_dotenv()

    spreadsheet_id = os.environ.get('GOOGLE_SHEETS_SPREADSHEET_ID')
    if not spreadsheet_id:
        print("ERROR: GOOGLE_SHEETS_SPREADSHEET_ID not set")
        return

    credentials_file = os.environ.get('GOOGLE_SHEETS_CREDENTIALS_FILE')
    if credentials_file and not os.path.isabs(credentials_file):
        credentials_file = PROJECT_ROOT / credentials_file

    if not credentials_file or not Path(credentials_file).exists():
        print(f"ERROR: Credentials file not found: {credentials_file}")
        return

    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    credentials = Credentials.from_service_account_file(str(credentials_file), scopes=scopes)
    client = gspread.authorize(credentials)
    spreadsheet = client.open_by_key(spreadsheet_id)

    print(f"\n{'='*60}")
    print("Google Sheets Worksheets")
    print(f"{'='*60}")
    print(f"Spreadsheet: {spreadsheet.title}")
    print(f"ID: {spreadsheet_id}")
    print(f"\nWorksheets ({len(spreadsheet.worksheets())}):")

    for ws in spreadsheet.worksheets():
        try:
            headers = ws.row_values(1)
            has_conseiller = 'Conseiller' in headers
            print(f"  - {ws.title} ({ws.row_count} rows, {ws.col_count} cols)")
            if headers:
                print(f"    Headers: {', '.join(headers[:8])}{'...' if len(headers) > 8 else ''}")
            if has_conseiller:
                print(f"    ✅ Has 'Conseiller' column")
        except Exception as e:
            print(f"  - {ws.title} (error: {e})")

    print(f"\n{'='*60}\n")


def list_advisors():
    """List all advisors in the database."""
    from src.utils.advisor_matcher import get_advisor_matcher

    matcher = get_advisor_matcher()
    advisors = matcher.get_all_advisors()

    print(f"\n{'='*60}")
    print("Advisor Database")
    print(f"{'='*60}")
    print(f"Storage backend: {matcher.storage_backend}")
    print(f"Total advisors: {len(advisors)}")
    print(f"\nAdvisors (compact format):")

    for advisor in sorted(advisors, key=lambda a: a.first_name):
        variations = ', '.join(advisor.variations[:3]) if advisor.variations else 'none'
        if len(advisor.variations) > 3:
            variations += '...'
        print(f"  {advisor.display_name_compact:20} | Full: {advisor.full_name:25} | Variations: {variations}")

    print(f"\n{'='*60}\n")


def main():
    """Entry point."""
    options = parse_args(sys.argv[1:])

    if options['list_sheets']:
        list_sheets()
        return

    if options['list_advisors']:
        list_advisors()
        return

    # Run normalization
    from src.utils.data_unifier import DataUnifier
    from src.utils.advisor_matcher import get_advisor_matcher

    matcher = get_advisor_matcher()
    unifier = DataUnifier(advisor_matcher=matcher)

    try:
        results = unifier.normalize_gsheet_advisors(
            worksheet_name=options['sheet'],
            column_name=options['column'],
            dry_run=options['dry_run'],
        )

        # Return success/failure based on results
        if results['normalized_count'] > 0 or results['unchanged_count'] > 0:
            sys.exit(0)
        elif results['not_found_count'] > 0:
            print("\nWARNING: Some names could not be matched.")
            print("Consider adding them to the advisor database.")
            sys.exit(0)
        else:
            print("\nNo cells with advisor names found.")
            sys.exit(1)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
