#!/usr/bin/env python3
"""
Test script for IDC Statement DIRECT extraction mode via VLM.

This test validates the direct extraction mode which parses the "Nom du client"
column into structured fields (company_code, advisor_name, client_name, etc.)
in a single extraction step.

Usage:
    python -m src.tests.test_idc_statement_direct [pdf_path]

If no PDF path is provided, uses a default test file.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path (src/tests -> src -> project_root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


async def test_direct_extraction(pdf_path: str | None = None):
    """Run DIRECT extraction test on an IDC Statement PDF."""
    from src.extractors.idc_statement_extractor import IDCStatementExtractor
    from src.utils.model_registry import get_model_config
    from src.utils.pdf import get_pdf_hash, pdf_to_images
    from src.utils.prompt_loader import clear_prompt_cache

    # Clear prompt cache to ensure we use the latest YAML prompts
    clear_prompt_cache()

    # Default test file
    if pdf_path is None:
        pdf_dir = PROJECT_ROOT / "pdf/idc_statement"
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print("ERROR: No PDF files found in pdf/idc_statement/")
            return None
        pdf_path = pdf_files[0]  # Use first PDF
    else:
        pdf_path = Path(pdf_path)

    # Get model configuration
    model_config = get_model_config("IDC_STATEMENT")

    print(f"\n{'='*60}")
    print("IDC Statement DIRECT Extraction Test")
    print(f"{'='*60}")
    print(f"PDF: {pdf_path.name}")
    print(f"Primary: {model_config.model_id} ({model_config.mode.value})")
    if model_config.fallback_model_id:
        fallback_mode = model_config.fallback_mode.value if model_config.fallback_mode else "N/A"
        print(f"Fallback: {model_config.fallback_model_id} ({fallback_mode})")
    if model_config.secondary_fallback_model_id:
        sec_mode = model_config.secondary_fallback_mode.value if model_config.secondary_fallback_mode else "N/A"
        print(f"Secondary: {model_config.secondary_fallback_model_id} ({sec_mode})")

    # Test PDF utilities
    print(f"\n[1/4] Testing PDF utilities...")
    pdf_hash = get_pdf_hash(pdf_path)
    print(f"  Hash: {pdf_hash[:16]}...")

    images = pdf_to_images(pdf_path)
    print(f"  Pages: {len(images)}")
    print(f"  Image sizes: {[len(img)//1024 for img in images]} KB")

    # Test extractor initialization
    print(f"\n[2/4] Initializing extractor...")
    extractor = IDCStatementExtractor()
    print(f"  Source: {extractor.source_name}")
    print(f"  Mode: DIRECT")

    # Verify prompts are loaded from YAML
    print(f"\n[2.5/4] Verifying prompts from YAML...")
    print(f"  system_prompt_direct: {len(extractor.system_prompt_direct)} chars")
    print(f"  user_prompt_direct: {len(extractor.user_prompt_direct)} chars")
    print(f"  Has direct prompts: {extractor._prompt_config.has_direct_prompts()}")

    # Invalidate cache to test with fresh prompts
    print(f"\n[3/4] Cache management...")
    was_cached = extractor.is_direct_cached(pdf_path)
    if was_cached:
        print(f"  Invalidating cache to use fresh prompts...")
        extractor.invalidate_direct_cache(pdf_path)
    print(f"  Was cached: {was_cached} → Now forcing fresh extraction")

    # Run extraction
    print(f"\n[4/4] Running VLM DIRECT extraction...")
    print("  -> Calling VLM API (this may take 10-30 seconds...)")

    start = time.time()

    try:
        result = await extractor.extract_direct(pdf_path)
        elapsed = time.time() - start

        print(f"\n{'='*60}")
        print("DIRECT EXTRACTION RESULTS")
        print(f"{'='*60}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Date rapport: {result.get('date_rapport') or 'N/A'}")
        print(f"Section conseiller: {result.get('advisor_section') or 'N/A'}")

        trailing_fees = result.get('trailing_fees', [])
        print(f"Nombre d'enregistrements: {len(trailing_fees)}")

        # Count by company
        companies = {}
        for f in trailing_fees:
            c = f.get('company') or 'Unknown'
            companies[c] = companies.get(c, 0) + 1
        print(f"\nPar compagnie:")
        for c, count in sorted(companies.items()):
            print(f"  {c}: {count}")

        # Count by advisor (parsed)
        advisors = {}
        for f in trailing_fees:
            a = f.get('advisor_name') or 'Unknown'
            advisors[a] = advisors.get(a, 0) + 1
        print(f"\nPar conseiller (parsé):")
        for a, count in sorted(advisors.items()):
            print(f"  {a}: {count}")

        # Show records with parsed fields
        print(f"\n{'='*60}")
        print("FRAIS DE SUIVI (données parsées)")
        print(f"{'='*60}")
        for i, fee in enumerate(trailing_fees, 1):
            raw_data = fee.get('raw_client_data') or 'N/A'
            # Truncate long raw data for display
            if len(raw_data) > 60:
                raw_display = raw_data[:57] + "..."
            else:
                raw_display = raw_data.replace('\n', ' | ')

            print(f"\n{i}. Raw: {raw_display}")
            print(f"   --- Champs parsés ---")
            print(f"   Company Code: {fee.get('company_code') or 'N/A'}")
            print(f"   Company Number: {fee.get('company_number') or 'N/A'}")
            print(f"   Policy Date: {fee.get('policy_date') or 'N/A'}")
            print(f"   Commission Rate: {fee.get('commission_rate') or 'N/A'}")
            print(f"   Policy Number: {fee.get('policy_number') or 'N/A'}")
            print(f"   Advisor: {fee.get('advisor_name') or 'N/A'}")
            client_first = fee.get('client_first_name') or ''
            client_last = fee.get('client_last_name') or ''
            client_name = f"{client_first} {client_last}".strip() or 'N/A'
            print(f"   Client: {client_name}")
            print(f"   --- Colonnes directes ---")
            print(f"   Compte: {fee.get('account_number') or 'N/A'}")
            print(f"   Compagnie: {fee.get('company') or 'N/A'}")
            print(f"   Produit: {fee.get('product') or 'N/A'}")
            print(f"   Date: {fee.get('date') or 'N/A'}")
            print(f"   Frais brut: {fee.get('gross_trailing_fee') or 'N/A'}")
            print(f"   Frais net: {fee.get('net_trailing_fee') or 'N/A'}")

        # Summary of parsing quality
        print(f"\n{'='*60}")
        print("QUALITE DU PARSING")
        print(f"{'='*60}")
        parsed_fields = [
            'company_code', 'company_number', 'policy_date',
            'commission_rate', 'policy_number', 'advisor_name',
            'client_first_name', 'client_last_name'
        ]
        for field in parsed_fields:
            filled = sum(1 for f in trailing_fees if f.get(field))
            pct = (filled / len(trailing_fees) * 100) if trailing_fees else 0
            print(f"  {field}: {filled}/{len(trailing_fees)} ({pct:.1f}%)")

        print(f"\n{'='*60}")
        print("TEST SUCCESSFUL")
        print(f"{'='*60}\n")
        return result

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_prompt_loading():
    """Test that prompts are correctly loaded from YAML."""
    from src.extractors.idc_statement_extractor import IDCStatementExtractor
    from src.utils.prompt_loader import clear_prompt_cache, load_prompts

    # Clear prompt cache to ensure we load fresh from YAML
    clear_prompt_cache()

    print(f"\n{'='*60}")
    print("PROMPT LOADING TEST")
    print(f"{'='*60}")

    # Test direct loading
    config = load_prompts("idc_statement")
    print(f"\nDirect load from YAML:")
    print(f"  document_type: {config.document_type}")
    print(f"  system_prompt: {len(config.system_prompt)} chars")
    print(f"  user_prompt: {len(config.user_prompt)} chars")
    print(f"  system_prompt_direct: {len(config.system_prompt_direct or '')} chars")
    print(f"  user_prompt_direct: {len(config.user_prompt_direct or '')} chars")
    print(f"  has_direct_prompts: {config.has_direct_prompts()}")

    # Test via extractor
    extractor = IDCStatementExtractor()
    print(f"\nVia extractor:")
    print(f"  system_prompt: {len(extractor.system_prompt)} chars")
    print(f"  user_prompt: {len(extractor.user_prompt)} chars")
    print(f"  system_prompt_direct: {len(extractor.system_prompt_direct)} chars")
    print(f"  user_prompt_direct: {len(extractor.user_prompt_direct)} chars")

    # Verify they match
    assert config.system_prompt == extractor.system_prompt, "system_prompt mismatch!"
    assert config.user_prompt == extractor.user_prompt, "user_prompt mismatch!"
    assert config.system_prompt_direct == extractor.system_prompt_direct, "system_prompt_direct mismatch!"
    assert config.user_prompt_direct == extractor.user_prompt_direct, "user_prompt_direct mismatch!"

    print(f"\n  All prompts match YAML source!")
    print(f"\n{'='*60}")
    print("PROMPT TEST SUCCESSFUL")
    print(f"{'='*60}\n")


def main():
    """Entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--prompts-only":
        # Just test prompt loading without API call
        asyncio.run(test_prompt_loading())
    else:
        pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
        # Run prompt test first
        asyncio.run(test_prompt_loading())
        # Then run extraction test
        asyncio.run(test_direct_extraction(pdf_path))


if __name__ == "__main__":
    main()
