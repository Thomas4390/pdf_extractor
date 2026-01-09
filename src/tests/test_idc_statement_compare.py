#!/usr/bin/env python3
"""
Comparison test: Rule-based parsing vs VLM Direct parsing.

This script compares two methods of extracting structured data from IDC Statements:
1. Raw extraction + Rule-based parsing using regex patterns (fast, deterministic)
2. VLM Direct extraction - VLM parses structured fields directly from PDF (flexible, slower)

Usage:
    python -m src.tests.test_idc_statement_compare [pdf_path]
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def compute_match_score(rule_result: dict, vlm_result: dict) -> dict[str, Any]:
    """
    Compare rule-based and VLM results for a single entry.

    Returns dict with field-by-field comparison.
    """
    fields_to_compare = [
        "company_code",
        "company_number",
        "policy_date",
        "commission_rate",
        "policy_number",
        "advisor_name",
        "client_first_name",
        "client_last_name",
    ]

    matches = {}
    for field in fields_to_compare:
        rule_val = rule_result.get(field)
        vlm_val = vlm_result.get(field)

        # Normalize values for comparison
        if rule_val is not None:
            rule_val = str(rule_val).strip().lower()
        if vlm_val is not None:
            vlm_val = str(vlm_val).strip().lower()

        # Handle None vs empty string
        if rule_val in [None, "", "none"]:
            rule_val = None
        if vlm_val in [None, "", "none"]:
            vlm_val = None

        # Compare
        if rule_val == vlm_val:
            matches[field] = "MATCH"
        elif rule_val is None and vlm_val is not None:
            matches[field] = f"RULE_MISS (VLM: {vlm_result.get(field)})"
        elif rule_val is not None and vlm_val is None:
            matches[field] = f"VLM_MISS (Rule: {rule_result.get(field)})"
        else:
            matches[field] = f"DIFF (Rule: {rule_result.get(field)} | VLM: {vlm_result.get(field)})"

    return matches


async def run_comparison(pdf_path: str | None = None):
    """Run comparison between rule-based and VLM direct parsing."""
    from src.extractors.idc_statement_extractor import IDCStatementExtractor
    from src.utils.model_registry import get_model_config
    from src.utils.raw_data_parser import parse_raw_entries_batch

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

    print(f"\n{'='*70}")
    print("IDC Statement COMPARISON: Rule-Based vs LLM Direct Parsing")
    print(f"{'='*70}")
    print(f"PDF: {pdf_path.name}")
    print(f"Model: {model_config.model_id}")
    print(f"Mode: {model_config.mode.value.upper()}")
    if model_config.fallback_model_id:
        print(f"Fallback: {model_config.fallback_model_id}")
    print()

    extractor = IDCStatementExtractor()
    raw_extraction_ok = False
    vlm_extraction_ok = False

    # Step 1: Get raw extraction for rule-based parsing
    print("[1/4] Getting raw extraction for rule-based parsing...")
    start_raw = time.time()
    try:
        raw_result = await extractor.extract(pdf_path, use_cache=True)
        elapsed_raw = time.time() - start_raw
        trailing_fees_raw = raw_result.get("trailing_fees", [])
        print(f"  Records found: {len(trailing_fees_raw)} (in {elapsed_raw:.2f}s)")
        raw_extraction_ok = len(trailing_fees_raw) > 0
    except Exception as e:
        print(f"  ❌ ERROR: Raw extraction failed - {e}")
        elapsed_raw = time.time() - start_raw
        trailing_fees_raw = []

    if not trailing_fees_raw:
        print("  ⚠️  No trailing fees found in raw extraction")

    # Step 2: Parse with rules
    print("\n[2/4] Parsing with RULE-BASED method...")
    start_rule = time.time()
    rule_results = parse_raw_entries_batch(trailing_fees_raw)
    elapsed_rule = time.time() - start_rule
    print(f"  Completed in {elapsed_rule:.3f}s")

    # Step 3: Get VLM direct extraction
    print("\n[3/4] Running VLM DIRECT extraction...")
    print("  This extracts AND parses fields directly from PDF images...")
    start_vlm = time.time()
    try:
        # Force no cache to get fresh extraction for comparison
        direct_result = await extractor.extract_direct(pdf_path, use_cache=True)
        trailing_fees_direct = direct_result.get("trailing_fees", [])
        elapsed_vlm = time.time() - start_vlm
        print(f"  Records found: {len(trailing_fees_direct)} (in {elapsed_vlm:.2f}s)")
        vlm_extraction_ok = len(trailing_fees_direct) > 0
    except Exception as e:
        print(f"  ❌ ERROR: VLM direct extraction failed - {e}")
        trailing_fees_direct = []
        elapsed_vlm = time.time() - start_vlm

    # Check if we can continue
    if not raw_extraction_ok and not vlm_extraction_ok:
        print(f"\n{'='*70}")
        print("❌ BOTH EXTRACTIONS FAILED - Cannot compare")
        print(f"{'='*70}")
        print("Check the debug files in cache/debug/ for details")
        return None

    # Step 4: Compare results
    print(f"\n[4/4] Comparing results...")

    # Show warning if partial data
    if not raw_extraction_ok:
        print("  ⚠️  Raw extraction failed - showing VLM results only")
    elif not vlm_extraction_ok:
        print("  ⚠️  VLM extraction failed - showing Rule-based results only")

    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")

    # Align records by matching raw_client_data or index
    vlm_by_raw = {}
    for entry in trailing_fees_direct:
        raw = entry.get("raw_client_data", "")
        if raw:
            # Use first 50 chars as key for matching
            vlm_by_raw[raw[:50]] = entry

    # Statistics
    total_fields = 0
    total_matches = 0
    field_stats = {
        "company_code": {"match": 0, "total": 0},
        "company_number": {"match": 0, "total": 0},
        "policy_date": {"match": 0, "total": 0},
        "commission_rate": {"match": 0, "total": 0},
        "policy_number": {"match": 0, "total": 0},
        "advisor_name": {"match": 0, "total": 0},
        "client_first_name": {"match": 0, "total": 0},
        "client_last_name": {"match": 0, "total": 0},
    }

    differences = []
    matched_count = 0

    for i, rule_res in enumerate(rule_results):
        # Try to find matching VLM result
        raw = rule_res.get("raw_client_data", "")
        vlm_res = vlm_by_raw.get(raw[:50], {})

        # Fallback to index if no match by raw_client_data
        if not vlm_res and i < len(trailing_fees_direct):
            vlm_res = trailing_fees_direct[i]

        if vlm_res:
            matched_count += 1

        comparison = compute_match_score(rule_res, vlm_res)

        has_diff = False
        for field, result in comparison.items():
            field_stats[field]["total"] += 1
            total_fields += 1

            if result == "MATCH":
                field_stats[field]["match"] += 1
                total_matches += 1
            else:
                has_diff = True

        if has_diff:
            differences.append({
                "index": i,
                "raw_data": rule_res.get("raw_client_data", "")[:60],
                "comparison": comparison,
            })

    # Print summary statistics
    print(f"\nTotal records - Rules: {len(rule_results)} | VLM Direct: {len(trailing_fees_direct)}")
    print(f"Matched pairs for comparison: {matched_count}")
    print(f"\nTime - Raw+Rules: {elapsed_raw + elapsed_rule:.2f}s | VLM Direct: {elapsed_vlm:.2f}s")
    total_rule_time = elapsed_raw + elapsed_rule
    if total_rule_time > 0 and elapsed_vlm > 0:
        print(f"Speed ratio: VLM is {elapsed_vlm / total_rule_time:.1f}x slower")

    print(f"\n{'='*70}")
    print("FIELD-BY-FIELD ACCURACY")
    print(f"{'='*70}")

    for field, stats in field_stats.items():
        if stats["total"] > 0:
            pct = (stats["match"] / stats["total"]) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"{field:20s}: {bar} {pct:5.1f}% ({stats['match']}/{stats['total']})")

    overall_pct = (total_matches / total_fields) * 100 if total_fields > 0 else 0
    print(f"\n{'OVERALL':20s}: {overall_pct:.1f}% match rate")

    # Show differences
    if differences:
        print(f"\n{'='*70}")
        print(f"DIFFERENCES (showing first 10 of {len(differences)})")
        print(f"{'='*70}")

        for diff in differences[:10]:
            print(f"\n[{diff['index']}] {diff['raw_data']}...")
            for field, result in diff["comparison"].items():
                if result != "MATCH":
                    print(f"   {field}: {result}")

    # Detailed view for ALL records
    print(f"\n{'='*70}")
    print(f"DETAILED COMPARISON (all {len(rule_results)} records)")
    print(f"{'='*70}")

    for i in range(len(rule_results)):
        rule_res = rule_results[i]
        raw = rule_res.get("raw_client_data", "")
        vlm_res = vlm_by_raw.get(raw[:50], {})
        if not vlm_res and i < len(trailing_fees_direct):
            vlm_res = trailing_fees_direct[i]

        # Check if there are any differences for this record
        has_differences = False
        fields = ["company_code", "commission_rate", "policy_number",
                  "advisor_name", "client_first_name", "client_last_name"]
        for field in fields:
            rule_val = str(rule_res.get(field, "")).lower().strip() or ""
            vlm_val = str(vlm_res.get(field, "")).lower().strip() or ""
            if rule_val != vlm_val:
                has_differences = True
                break

        # Print record header with indicator
        status = "⚠️" if has_differences else "✅"
        print(f"\n--- Record {i+1}/{len(rule_results)} {status} ---")
        print(f"RAW: {raw[:70]}...")
        print(f"\n{'Field':<20} {'Rule-Based':<25} {'VLM Direct':<25}")
        print("-" * 70)

        for field in fields:
            rule_val = str(rule_res.get(field, "")) or "-"
            vlm_val = str(vlm_res.get(field, "")) or "-"
            match_mark = "✓" if rule_val.lower() == vlm_val.lower() else "✗"
            print(f"{field:<20} {rule_val:<25} {vlm_val:<25} {match_mark}")

    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}\n")

    return {
        "rule_results": rule_results,
        "vlm_results": trailing_fees_direct,
        "elapsed_rule": elapsed_rule,
        "elapsed_vlm": elapsed_vlm,
        "field_stats": field_stats,
        "differences": differences,
    }


def main():
    """Entry point."""
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_comparison(pdf_path))


if __name__ == "__main__":
    main()
