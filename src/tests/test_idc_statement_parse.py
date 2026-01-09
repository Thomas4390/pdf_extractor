#!/usr/bin/env python3
"""
Test script for parsing raw_client_data from IDC Statement extraction.

This script takes the raw extracted data (with raw_client_data field)
and uses VLM to parse it into structured fields.

Usage:
    python -m src.tests.test_idc_statement_parse [pdf_path]

If no PDF path is provided, uses a default test file from cache or extracts first.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path (src/tests -> src -> project_root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


PARSE_SYSTEM_PROMPT = """Tu es un expert en parsing de données d'assurance.
Tu dois parser les données brutes de la colonne "Nom du client" des relevés IDC.

Ces données contiennent plusieurs informations encodées sur plusieurs lignes:
- Code compagnie (ex: UV, Assomption, IA, Beneva, RBC, Manuvie, SSQ, Desjardins, Empire, Ivari)
- Numéro de compagnie (ex: 7782, 8055)
- Date de police (format YYYY-MM-DD)
- Taux de commission/boni (ex: 75%, 80%)
- Numéro de police (souvent précédé de # ou contenant 6-9 chiffres)
- Nom du conseiller (prénom + initiale du nom, ex: "Bourassa A", "Thomas L")
- Nom du client (prénom et nom de famille)

RÈGLES:
1. Parse chaque champ avec précision
2. Le taux de commission doit être converti en décimal (75% → 0.75)
3. Si une information n'est pas trouvable, utilise null
4. Les noms de clients sont souvent à la fin, parfois sur 2 lignes
5. "clt" signifie "client" et précède souvent le nom du client
6. "crt" signifie "courtier" et peut indiquer un type

NORMALISATION DES CONSEILLERS:
Extrais le nom tel quel (prénom + initiale), ex: "Bourassa A", "Thomas L"

Tu dois retourner un JSON valide."""

PARSE_USER_PROMPT_TEMPLATE = """Parse ces {count} entrées raw_client_data en données structurées.

DONNÉES BRUTES À PARSER:
{raw_data_json}

Pour chaque entrée, extrais:
- company_code: Code compagnie (UV, Assomption, IA, Beneva, etc.)
- company_number: Numéro de la compagnie (ex: "7782")
- policy_date: Date de police (YYYY-MM-DD) si présente
- commission_rate: Taux de commission en décimal (0.75 pour 75%)
- policy_number: Numéro de police (sans le #)
- advisor_name: Nom du conseiller (prénom + initiale)
- client_first_name: Prénom du client
- client_last_name: Nom de famille du client

Retourne un JSON avec cette structure:
{{
  "parsed_entries": [
    {{
      "index": 0,
      "company_code": "UV",
      "company_number": "7782",
      "policy_date": "2025-11-17",
      "commission_rate": 0.75,
      "policy_number": "111011722",
      "advisor_name": "Bourassa A",
      "client_first_name": "Jeanny",
      "client_last_name": "Breault-Therrien"
    }}
  ]
}}"""


# DeepSeek model for text parsing (no vision needed)
PARSING_MODEL = "deepseek/deepseek-chat"


async def parse_raw_data(raw_entries: list[dict]) -> dict:
    """Parse raw_client_data entries using DeepSeek (text-only model)."""
    from src.clients.openrouter import OpenRouterClient

    # Use DeepSeek for parsing (text-only, faster and cheaper)
    client = OpenRouterClient(model=PARSING_MODEL)

    # Prepare raw data for parsing
    raw_data_list = []
    for i, entry in enumerate(raw_entries):
        raw_data_list.append({
            "index": i,
            "raw_client_data": entry.get("raw_client_data", ""),
        })

    user_prompt = PARSE_USER_PROMPT_TEMPLATE.format(
        count=len(raw_data_list),
        raw_data_json=json.dumps(raw_data_list, ensure_ascii=False, indent=2),
    )

    # Call LLM for parsing (text-only, no images needed)
    result = await client.extract_with_text(
        system_prompt=PARSE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    return result


def merge_parsed_data(raw_entries: list[dict], parsed_result: dict) -> list[dict]:
    """Merge parsed data back into raw entries."""
    parsed_entries = parsed_result.get("parsed_entries", [])

    # Create index lookup
    parsed_by_index = {p.get("index"): p for p in parsed_entries}

    merged = []
    for i, entry in enumerate(raw_entries):
        parsed = parsed_by_index.get(i, {})
        merged_entry = {
            # Original raw data
            "raw_client_data": entry.get("raw_client_data", ""),
            "account_number": entry.get("account_number", "Unknown"),
            "company": entry.get("company", "Unknown"),
            "product": entry.get("product", "Unknown"),
            "date": entry.get("date", "Unknown"),
            "gross_trailing_fee": entry.get("gross_trailing_fee", "0,00 $"),
            "net_trailing_fee": entry.get("net_trailing_fee", "0,00 $"),
            "dealer": entry.get("dealer"),
            # Parsed fields
            "company_code": parsed.get("company_code"),
            "company_number": parsed.get("company_number"),
            "policy_date": parsed.get("policy_date"),
            "commission_rate": parsed.get("commission_rate"),
            "policy_number": parsed.get("policy_number"),
            "advisor_name": parsed.get("advisor_name"),
            "client_first_name": parsed.get("client_first_name"),
            "client_last_name": parsed.get("client_last_name"),
        }
        merged.append(merged_entry)

    return merged


async def test_parsing(pdf_path: str | None = None):
    """Run parsing test on extracted IDC Statement data."""
    from src.extractors.idc_statement_extractor import IDCStatementExtractor
    from src.utils.model_registry import get_model_config
    from src.utils.pdf import get_pdf_hash

    # Default test file
    if pdf_path is None:
        pdf_dir = PROJECT_ROOT / "pdf/idc_statement"
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print("ERROR: No PDF files found in pdf/idc_statement/")
            return None
        pdf_path = pdf_files[0]
    else:
        pdf_path = Path(pdf_path)

    # Get model configuration
    model_config = get_model_config("IDC_STATEMENT")

    print(f"\n{'='*60}")
    print("IDC Statement PARSING Test (raw_client_data → structured)")
    print(f"{'='*60}")
    print(f"PDF: {pdf_path.name}")
    print(f"\nStep 1 - Extraction (PDF → raw_client_data):")
    print(f"  Model: {model_config.model_id} ({model_config.mode.value})")
    print(f"\nStep 2 - Parsing (raw_client_data → structured fields):")
    print(f"  Model: {PARSING_MODEL} (text-only)")

    # Step 1: Get raw extraction (from cache or new)
    print(f"\n[1/3] Getting raw extraction...")
    extractor = IDCStatementExtractor()

    if extractor.is_cached(pdf_path):
        print("  -> Using cached raw extraction")
    else:
        print("  -> Running new extraction (this may take 10-30 seconds...)")

    raw_result = await extractor.extract(pdf_path)
    trailing_fees = raw_result.get("trailing_fees", [])
    print(f"  Records found: {len(trailing_fees)}")

    if not trailing_fees:
        print("ERROR: No trailing fees found in extraction")
        return None

    # Step 2: Parse raw_client_data
    print(f"\n[2/3] Parsing raw_client_data with VLM...")
    print(f"  Sending {len(trailing_fees)} entries for parsing...")

    start = time.time()
    parsed_result = await parse_raw_data(trailing_fees)
    elapsed = time.time() - start
    print(f"  Parsing completed in {elapsed:.2f}s")

    # Step 3: Merge and display results
    print(f"\n[3/3] Merging parsed data...")
    merged_data = merge_parsed_data(trailing_fees, parsed_result)

    print(f"\n{'='*60}")
    print("PARSED RESULTS")
    print(f"{'='*60}")

    # Count by advisor
    advisors = {}
    for entry in merged_data:
        advisor = entry.get("advisor_name", "Unknown")
        if advisor:
            advisors[advisor] = advisors.get(advisor, 0) + 1

    if advisors:
        print(f"\nPar conseiller:")
        for a, count in sorted(advisors.items()):
            print(f"  {a}: {count}")

    # Show parsed records
    print(f"\n{'='*60}")
    print("FRAIS DE SUIVI (données parsées)")
    print(f"{'='*60}")

    for i, entry in enumerate(merged_data[:10], 1):
        print(f"\n{i}. Client: {entry.get('client_first_name', '?')} {entry.get('client_last_name', '?')}")
        print(f"   Conseiller: {entry.get('advisor_name', 'N/A')}")
        print(f"   Police: #{entry.get('policy_number', 'N/A')}")
        print(f"   Compagnie: {entry.get('company_code', 'N/A')} ({entry.get('company_number', 'N/A')})")
        print(f"   Taux: {entry.get('commission_rate', 'N/A')}")
        print(f"   Date police: {entry.get('policy_date', 'N/A')}")
        print(f"   Compte: {entry.get('account_number', 'N/A')}")
        print(f"   Frais brut: {entry.get('gross_trailing_fee', 'N/A')}")
        print(f"   Frais net: {entry.get('net_trailing_fee', 'N/A')}")
        print(f"   [RAW: {entry.get('raw_client_data', '')[:50]}...]")

    if len(merged_data) > 10:
        print(f"\n... ({len(merged_data) - 10} more records)")

    print(f"\n{'='*60}")
    print("PARSING TEST SUCCESSFUL")
    print(f"{'='*60}\n")

    return {
        "raw_result": raw_result,
        "parsed_result": parsed_result,
        "merged_data": merged_data,
    }


def main():
    """Entry point."""
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(test_parsing(pdf_path))


if __name__ == "__main__":
    main()
