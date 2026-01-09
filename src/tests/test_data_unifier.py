#!/usr/bin/env python3
"""
Test script for DataUnifier.

Usage:
    python -m src.tests.test_data_unifier [source]

Sources: UV, IDC, IDC_STATEMENT, ASSOMPTION, ALL (default: ALL)

This script tests the DataUnifier by:
1. Loading cached extraction results (if available)
2. Converting them to standardized DataFrames
3. Displaying the results
"""

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_mock_uv_report():
    """Create a mock UVReport for testing without API calls."""
    from src.models.uv import UVReport, UVActivity

    return UVReport(
        date_rapport="2025-10-13",
        nom_conseiller="9491-1377 QUEBEC INC",
        numero_conseiller="21621",
        sous_conseiller="21622 - ACHRAF EL HAJJI",
        activites=[
            UVActivity(
                contrat="110970886",
                assure="BALDWIN RAYMOND",
                protection="Vie entière Valeurs Élevées",
                montant_base=Decimal("1196.00"),
                taux_partage=Decimal("100.0"),
                taux_commission=Decimal("55.0"),
                resultat=Decimal("657.80"),
                type_commission="Boni 1ère année vie",
                taux_boni=Decimal("175.0"),
                remuneration=Decimal("1151.15"),
            ),
            UVActivity(
                contrat="110971504",
                assure="MADJIGUENE SOW",
                protection="Temporaire 10 ans",
                montant_base=Decimal("699.00"),
                taux_partage=Decimal("40.0"),
                taux_commission=Decimal("55.0"),
                resultat=Decimal("153.78"),
                type_commission="Commission 1ère année vie",
                taux_boni=Decimal("175.0"),
                remuneration=Decimal("269.12"),
            ),
        ],
    )


def create_mock_idc_report():
    """Create a mock IDCReport for testing."""
    from src.models.idc import IDCReport, IDCProposition

    return IDCReport(
        titre="Rapport des propositions soumises",
        date_rapport="2025-11-24",
        vendeur="Greenberg, Thomas",
        propositions=[
            IDCProposition(
                assureur="RBC INSURANCE",
                client="SMITH, JOHN",
                type_regime="Permanent",
                police="1014157",
                statut="Approved",
                date="2025-10-15",
                nombre=Decimal("1.00"),
                taux_cpa=Decimal("100.0"),
                couverture="100 000,00 $",
                prime_police="1 234,56 $",
                prime_commissionnable="1 234,56 $",
                commission="617,28 $",
            ),
            IDCProposition(
                assureur="CANADA LIFE",
                client="DOE, JANE",
                type_regime="Term",
                police="ABC789",
                statut="Pending",
                date="2025-11-01",
                nombre=Decimal("0.40"),
                taux_cpa=Decimal("50.0"),
                couverture="250 000,00 $",
                prime_police="500,00 $",
                prime_commissionnable="500,00 $",
                commission="100,00 $",
            ),
        ],
    )


def create_mock_assomption_report():
    """Create a mock AssomptionReport for testing."""
    from src.models.assomption import AssomptionReport, AssomptionCommission

    return AssomptionReport(
        periode_debut="2025/10/02",
        periode_fin="2025/10/06",
        date_paie="2025/10/09",
        numero_courtier="35552",
        nom_courtier="9491-1377 Québec Inc.",
        commissions=[
            AssomptionCommission(
                code="AOH1",
                numero_police="1011221",
                nom_assure="MUADI MUNYA TSHIMANGA",
                produit="4T20 B",
                date_emission="2025/09/26",
                frequence_paiement="Mensuel",
                facturation="COM/PAC",
                prime=Decimal("499.05"),
                taux_commission=Decimal("40.993"),
                commission=Decimal("224.58"),
                taux_boni=Decimal("175.0"),
                boni=Decimal("393.02"),
            ),
            AssomptionCommission(
                code="BCD2",
                numero_police="1011452",
                nom_assure="DAOUYA TARABET",
                produit="5L A",
                date_emission="2025/10/01",
                frequence_paiement="Annuel",
                facturation="COM/PAC",
                prime=Decimal("-142.56"),
                taux_commission=Decimal("45.0"),
                commission=Decimal("-58.44"),
                taux_boni=Decimal("175.0"),
                boni=Decimal("-134.24"),
            ),
        ],
    )


def create_mock_idc_statement_report():
    """Create a mock IDCStatementReport for testing."""
    from src.models.idc_statement import IDCStatementReport, IDCTrailingFeeRaw

    return IDCStatementReport(
        titre="Détails des frais de suivi",
        date_rapport="2025-10-17",
        advisor_section="Achraf El Hajji - 3449L3138",
        trailing_fees=[
            IDCTrailingFeeRaw(
                raw_client_data="Â UV 7782 2025-11-17\nboni 75% #111011722 crt\nBourassa A clt Jeanny\nBreault-Therrien",
                account_number="N894713",
                company="UV",
                product="RRSP",
                date="2025-10-15",
                gross_trailing_fee="123,45 $",
                net_trailing_fee="98,76 $",
            ),
            IDCTrailingFeeRaw(
                raw_client_data="Assomption 8055 2025-10-10\n80% #123456\nThomas L clt John Doe",
                account_number="1234567",
                company="Assomption",
                product="TFSA",
                date="2025-11-01",
                gross_trailing_fee="-50,00 $",
                net_trailing_fee="-40,00 $",
            ),
        ],
    )


async def test_unifier_with_mocks():
    """Test DataUnifier with mock data."""
    from src.utils.data_unifier import DataUnifier, BoardType

    unifier = DataUnifier()

    print("\n" + "=" * 70)
    print("DataUnifier Test with Mock Data")
    print("=" * 70)

    # Test UV
    print("\n[1/4] Testing UV conversion...")
    uv_report = create_mock_uv_report()
    df_uv, board_type_uv = unifier.unify(uv_report, "UV")
    print(f"  Board Type: {board_type_uv.value}")
    print(f"  Rows: {len(df_uv)}")
    print(f"  Columns: {list(df_uv.columns)}")
    print(f"\n  Sample data:")
    print(df_uv[['# de Police', 'Nom Client', 'PA', 'Com', 'Total']].to_string(index=False))

    # Test IDC
    print("\n[2/4] Testing IDC conversion...")
    idc_report = create_mock_idc_report()
    df_idc, board_type_idc = unifier.unify(idc_report, "IDC")
    print(f"  Board Type: {board_type_idc.value}")
    print(f"  Rows: {len(df_idc)}")
    print(f"\n  Sample data:")
    print(df_idc[['# de Police', 'Nom Client', 'Compagnie', 'Statut', 'PA']].to_string(index=False))

    # Test Assomption
    print("\n[3/4] Testing Assomption conversion...")
    assomption_report = create_mock_assomption_report()
    df_assomption, board_type_assomption = unifier.unify(assomption_report, "ASSOMPTION")
    print(f"  Board Type: {board_type_assomption.value}")
    print(f"  Rows: {len(df_assomption)}")
    print(f"\n  Sample data:")
    print(df_assomption[['# de Police', 'Nom Client', 'PA', 'Com', 'Boni']].to_string(index=False))

    # Test IDC Statement
    print("\n[4/4] Testing IDC Statement conversion...")
    idc_statement_report = create_mock_idc_statement_report()
    df_statement, board_type_statement = unifier.unify(idc_statement_report, "IDC_STATEMENT")
    print(f"  Board Type: {board_type_statement.value}")
    print(f"  Rows: {len(df_statement)}")
    print(f"  Columns: {list(df_statement.columns)}")
    print(f"\n  Sample data:")
    print(df_statement[['# de Police', 'Nom Client', 'Compagnie', 'Sur-Com', 'Statut']].to_string(index=False))

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)

    return True


async def test_unifier_with_real_data(source: str):
    """Test DataUnifier with real extracted data from cache."""
    from src.utils.data_unifier import DataUnifier, BoardType
    from src.extractors import (
        UVExtractor,
        IDCExtractor,
        IDCStatementExtractor,
        AssomptionExtractor,
    )

    # Define test PDFs
    test_pdfs = {
        "UV": PROJECT_ROOT / "pdf/uv/rappportremun_21621_2025-10-13.pdf",
        "IDC": PROJECT_ROOT / "pdf/idc/Rapport des propositions soumises.20251124_1638.pdf",
        "ASSOMPTION": PROJECT_ROOT / "pdf/assomption/Remuneration (61).pdf",
        "IDC_STATEMENT": PROJECT_ROOT / "pdf/idc_statement/Détails des frais de suivi.20251105_1113.pdf",
    }

    extractors = {
        "UV": UVExtractor(),
        "IDC": IDCExtractor(),
        "ASSOMPTION": AssomptionExtractor(),
        "IDC_STATEMENT": IDCStatementExtractor(),
    }

    unifier = DataUnifier()

    sources_to_test = [source] if source != "ALL" else list(test_pdfs.keys())

    print("\n" + "=" * 70)
    print("DataUnifier Test with Real Data")
    print("=" * 70)

    for src in sources_to_test:
        pdf_path = test_pdfs.get(src)
        if not pdf_path or not pdf_path.exists():
            print(f"\n[{src}] Skipping - PDF not found: {pdf_path}")
            continue

        extractor = extractors[src]

        print(f"\n[{src}] Testing conversion...")
        print(f"  PDF: {pdf_path.name}")

        # Check if cached
        is_cached = extractor.is_cached(pdf_path)
        if not is_cached:
            print(f"  Status: Not cached - running extraction (may take 10-30s)...")
        else:
            print(f"  Status: Using cached result")

        try:
            # Extract
            report = await extractor.extract(pdf_path)

            # Convert
            df, board_type = unifier.unify(report, src)

            print(f"  Board Type: {board_type.value}")
            print(f"  Rows: {len(df)}")
            print(f"  Columns ({len(df.columns)}): {list(df.columns)}")

            # Show sample
            print(f"\n  DataFrame preview:")
            with pd.option_context('display.max_columns', 10, 'display.width', 120):
                print(df.head(3).to_string())

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)


async def main():
    """Main entry point."""
    import pandas as pd

    # Parse args
    source = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"

    if source == "MOCK":
        await test_unifier_with_mocks()
    else:
        valid_sources = ["UV", "IDC", "ASSOMPTION", "IDC_STATEMENT", "ALL"]
        if source not in valid_sources:
            print(f"Invalid source: {source}")
            print(f"Valid sources: {', '.join(valid_sources)}, MOCK")
            sys.exit(1)

        # First run mock tests
        print("\n>>> Running mock data tests first...\n")
        await test_unifier_with_mocks()

        # Then run real data tests if requested
        if source != "MOCK":
            print("\n>>> Running real data tests...\n")
            await test_unifier_with_real_data(source)


if __name__ == "__main__":
    asyncio.run(main())
