#!/usr/bin/env python3
"""
Tests for the reconciliation engine.

Usage:
    python -m src.tests.test_reconciler
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.utils.reconciler import (
    Reconciler,
    ReconciliationStatus,
    RecuClassification,
)


def test_classify_commission():
    """Test classification of commission lines."""
    r = Reconciler()

    # UV-style commission
    c = r.classify_row("Protection A | Commission 1ère année (Partage: 40%, Com: 50%)")
    assert c is not None
    assert c.recu_field == "Reçu 1"
    assert c.compare_column == "Com"

    # Assomption-style commission
    c = r.classify_row("Commission | ABC123 | Produit X | Freq: Mensuel")
    assert c is not None
    assert c.recu_field == "Reçu 1"
    assert c.compare_column == "Com"


def test_classify_boni():
    """Test classification of bonus lines."""
    r = Reconciler()

    # UV-style boni
    c = r.classify_row("Protection A | Boni 1ère année vie (Partage: 40%, Com: 50%)")
    assert c is not None
    assert c.recu_field == "Reçu 2"
    assert c.compare_column == "Boni"

    # Assomption-style boni
    c = r.classify_row("Boni | Produit Y | Taux Boni: 25%")
    assert c is not None
    assert c.recu_field == "Reçu 2"
    assert c.compare_column == "Boni"


def test_classify_surcom():
    """Test classification of sur-commission lines."""
    r = Reconciler()

    # UV Inc sur-com
    c = r.classify_row("Protection A | Commission 1ère année (Partage: 40%, Com: 50%) [Sur-Com]")
    assert c is not None
    assert c.recu_field == "Reçu 3"
    assert c.compare_column == "Sur-Com"

    # Variations
    c = r.classify_row("some text surcom stuff")
    assert c is not None
    assert c.recu_field == "Reçu 3"


def test_classify_priority_order():
    """Test that sur-com takes priority over boni, which takes priority over commission."""
    r = Reconciler()

    # Sur-Com should win even if "commission" and "boni" are present
    c = r.classify_row("Commission Boni [Sur-Com]")
    assert c.recu_field == "Reçu 3"

    # Boni should win over commission
    c = r.classify_row("Commission Boni stuff")
    assert c.recu_field == "Reçu 2"


def test_classify_no_match():
    """Test that unrecognized text returns None."""
    r = Reconciler()

    assert r.classify_row("") is None
    assert r.classify_row("something else entirely") is None
    assert r.classify_row(None) is None


def test_threshold_high_pa():
    """Test threshold is 10% for PA > 500."""
    r = Reconciler()
    assert r.determine_threshold(1000.0) == 10.0
    assert r.determine_threshold(500.01) == 10.0


def test_threshold_low_pa():
    """Test threshold is 20% for PA <= 500 or None."""
    r = Reconciler()
    assert r.determine_threshold(500.0) == 20.0
    assert r.determine_threshold(100.0) == 20.0
    assert r.determine_threshold(None) == 20.0
    assert r.determine_threshold(0) == 20.0


def test_reconcile_basic():
    """Test basic reconciliation with matching records."""
    r = Reconciler()

    hist_df = pd.DataFrame([
        {
            "# de Police": "POL001",
            "Compagnie": "UV Inc",
            "Texte": "Protection | Commission 1ère année (Partage: 40%, Com: 50%)",
            "Reçu": 100.0,
            "Statut": "Payé",
        },
        {
            "# de Police": "POL001",
            "Compagnie": "UV Inc",
            "Texte": "Protection | Boni 1ère année vie (Partage: 40%, Com: 50%)",
            "Reçu": 50.0,
            "Statut": "Payé",
        },
    ])

    sales_df = pd.DataFrame([
        {
            "# de Police": "POL001",
            "Com": 100.0,
            "Boni": 50.0,
            "Sur-Com": None,
            "PA": 1000.0,
            "item_id": "12345",
            "Conseiller": "Jean Dupont",
        },
    ])

    result = r.reconcile(hist_df, sales_df)

    assert result.total_hist_lines == 2
    assert result.total_groups == 2  # Com group + Boni group
    assert result.found == 2
    assert result.passed == 2
    assert result.flagged == 0

    # Check sales updates group by item_id
    updates = result.get_sales_updates()
    assert "12345" in updates
    assert updates["12345"]["Reçu 1"] == 100.0
    assert updates["12345"]["Reçu 2"] == 50.0


def test_reconcile_aggregation():
    """Test that multiple lines with same police+classification are summed."""
    r = Reconciler()

    hist_df = pd.DataFrame([
        {
            "# de Police": "POL001",
            "Compagnie": "UV Inc",
            "Texte": "Protection A | Commission 1ère année",
            "Reçu": 60.0,
            "Statut": "Payé",
        },
        {
            "# de Police": "POL001",
            "Compagnie": "UV Inc",
            "Texte": "Protection B | Commission 1ère année",
            "Reçu": 40.0,
            "Statut": "Payé",
        },
        {
            "# de Police": "POL001",
            "Compagnie": "UV Inc",
            "Texte": "Protection A | Boni 1ère année vie",
            "Reçu": 20.0,
            "Statut": "Payé",
        },
        {
            "# de Police": "POL001",
            "Compagnie": "UV Inc",
            "Texte": "Protection B | Boni 1ère année vie",
            "Reçu": 30.0,
            "Statut": "Payé",
        },
    ])

    sales_df = pd.DataFrame([
        {
            "# de Police": "POL001",
            "Com": 100.0,
            "Boni": 50.0,
            "Sur-Com": None,
            "PA": 1000.0,
            "item_id": "12345",
            "Conseiller": "Jean Dupont",
        },
    ])

    result = r.reconcile(hist_df, sales_df)

    # 4 Payé lines → aggregated into 2 groups (Commission, Boni)
    assert result.total_hist_lines == 4
    assert result.total_groups == 2
    assert result.passed == 2
    assert result.flagged == 0

    # Commission group: 60 + 40 = 100 → matches Com=100
    # Boni group: 20 + 30 = 50 → matches Boni=50
    updates = result.get_sales_updates()
    assert updates["12345"]["Reçu 1"] == 100.0
    assert updates["12345"]["Reçu 2"] == 50.0

    # hist_indices should contain all original indices
    com_match = [m for m in result.matches if m.classification and m.classification.recu_field == "Reçu 1"][0]
    assert len(com_match.hist_indices) == 2
    assert com_match.line_count == 2


def test_reconcile_surcom_aggregation():
    """Test aggregation works for Sur-Com lines too."""
    r = Reconciler()

    hist_df = pd.DataFrame([
        {
            "# de Police": "POL001",
            "Compagnie": "UV Inc",
            "Texte": "Protection A | Commission (Partage: 100%, Com: 50%) [Sur-Com]",
            "Reçu": 30.0,
            "Statut": "Payé",
        },
        {
            "# de Police": "POL001",
            "Compagnie": "UV Inc",
            "Texte": "Protection B | Commission (Partage: 100%, Com: 50%) [Sur-Com]",
            "Reçu": 20.0,
            "Statut": "Payé",
        },
    ])

    sales_df = pd.DataFrame([
        {
            "# de Police": "POL001",
            "Com": 200.0,
            "Boni": None,
            "Sur-Com": 50.0,
            "PA": 1000.0,
            "item_id": "99",
            "Conseiller": "Marie",
        },
    ])

    result = r.reconcile(hist_df, sales_df)

    assert result.total_groups == 1
    assert result.passed == 1
    assert result.matches[0].recu_amount == 50.0  # 30 + 20
    assert result.matches[0].classification.recu_field == "Reçu 3"


def test_reconcile_not_found():
    """Test reconciliation when police not found in sales."""
    r = Reconciler()

    hist_df = pd.DataFrame([
        {
            "# de Police": "POL999",
            "Compagnie": "UV Inc",
            "Texte": "Commission stuff",
            "Reçu": 100.0,
            "Statut": "Payé",
        },
    ])

    sales_df = pd.DataFrame([
        {
            "# de Police": "POL001",
            "Com": 100.0,
            "Boni": None,
            "Sur-Com": None,
            "PA": 1000.0,
            "item_id": "12345",
            "Conseiller": "Jean Dupont",
        },
    ])

    result = r.reconcile(hist_df, sales_df)
    assert result.not_found == 1
    assert result.passed == 0


def test_reconcile_flagged():
    """Test reconciliation with deviation above threshold."""
    r = Reconciler()

    hist_df = pd.DataFrame([
        {
            "# de Police": "POL001",
            "Compagnie": "UV Inc",
            "Texte": "Commission test",
            "Reçu": 200.0,  # 100% deviation from expected 100
            "Statut": "Payé",
        },
    ])

    sales_df = pd.DataFrame([
        {
            "# de Police": "POL001",
            "Com": 100.0,
            "Boni": None,
            "Sur-Com": None,
            "PA": 1000.0,
            "item_id": "12345",
            "Conseiller": "Jean Dupont",
        },
    ])

    result = r.reconcile(hist_df, sales_df)
    assert result.flagged == 1
    assert result.passed == 0


def test_reconcile_filters_paye_only():
    """Test that only Payé rows are processed."""
    r = Reconciler()

    hist_df = pd.DataFrame([
        {
            "# de Police": "POL001",
            "Compagnie": "UV Inc",
            "Texte": "Commission test",
            "Reçu": 100.0,
            "Statut": "Payé",
        },
        {
            "# de Police": "POL002",
            "Compagnie": "UV Inc",
            "Texte": "Commission test",
            "Reçu": -50.0,
            "Statut": "Charge back",
        },
    ])

    sales_df = pd.DataFrame([
        {
            "# de Police": "POL001",
            "Com": 100.0,
            "Boni": None,
            "Sur-Com": None,
            "PA": 1000.0,
            "item_id": "12345",
            "Conseiller": "Jean Dupont",
        },
    ])

    result = r.reconcile(hist_df, sales_df)
    assert result.total_hist_lines == 1  # Only the Payé row


def test_reconcile_hist_updates_with_aggregation():
    """Test get_passed_hist_updates returns all indices from aggregated groups."""
    r = Reconciler()

    hist_df = pd.DataFrame([
        {
            "# de Police": "POL001",
            "Compagnie": "UV Inc",
            "Texte": "Commission A",
            "Reçu": 60.0,
            "Statut": "Payé",
        },
        {
            "# de Police": "POL001",
            "Compagnie": "UV Inc",
            "Texte": "Commission B",
            "Reçu": 40.0,
            "Statut": "Payé",
        },
    ])

    sales_df = pd.DataFrame([
        {
            "# de Police": "POL001",
            "Com": 100.0,
            "Boni": None,
            "Sur-Com": None,
            "PA": 1000.0,
            "item_id": "12345",
            "Conseiller": "Marie Tremblay",
        },
    ])

    result = r.reconcile(hist_df, sales_df)
    hist_updates = result.get_passed_hist_updates()
    # Both indices should be returned, each with the same conseiller
    assert len(hist_updates) == 2
    assert hist_updates[0] == (0, "Marie Tremblay")
    assert hist_updates[1] == (1, "Marie Tremblay")


def test_reconcile_to_display_dataframe():
    """Test display DataFrame generation."""
    r = Reconciler()

    hist_df = pd.DataFrame([
        {
            "# de Police": "POL001",
            "Compagnie": "UV Inc",
            "Texte": "Commission test",
            "Reçu": 100.0,
            "Statut": "Payé",
        },
    ])

    sales_df = pd.DataFrame([
        {
            "# de Police": "POL001",
            "Com": 100.0,
            "Boni": None,
            "Sur-Com": None,
            "PA": 1000.0,
            "item_id": "12345",
            "Conseiller": "Jean Dupont",
        },
    ])

    result = r.reconcile(hist_df, sales_df)
    display = result.to_display_dataframe()

    assert len(display) == 1
    assert "# Police" in display.columns
    assert "Statut" in display.columns
    assert "Lignes" in display.columns
    assert display.iloc[0]["Statut"] == "Vérifié"


def test_reconcile_empty_hist():
    """Test reconciliation with no Payé rows."""
    r = Reconciler()

    hist_df = pd.DataFrame([
        {
            "# de Police": "POL001",
            "Compagnie": "UV Inc",
            "Texte": "Commission test",
            "Reçu": -100.0,
            "Statut": "Charge back",
        },
    ])

    sales_df = pd.DataFrame(columns=["# de Police", "Com", "Boni", "Sur-Com", "PA", "item_id", "Conseiller"])

    result = r.reconcile(hist_df, sales_df)
    assert result.total_hist_lines == 0
    assert result.matches == []


def test_reconcile_within_threshold():
    """Test that small deviations within threshold pass."""
    r = Reconciler()

    # PA > 500 → threshold = 10%
    hist_df = pd.DataFrame([
        {
            "# de Police": "POL001",
            "Compagnie": "UV Inc",
            "Texte": "Commission test",
            "Reçu": 95.0,  # 5% deviation from 100 → within 10%
            "Statut": "Payé",
        },
    ])

    sales_df = pd.DataFrame([
        {
            "# de Police": "POL001",
            "Com": 100.0,
            "Boni": None,
            "Sur-Com": None,
            "PA": 1000.0,
            "item_id": "12345",
            "Conseiller": "Jean Dupont",
        },
    ])

    result = r.reconcile(hist_df, sales_df)
    assert result.passed == 1
    assert result.flagged == 0


def test_reconcile_mixed_police_aggregation():
    """Test aggregation with multiple police numbers."""
    r = Reconciler()

    hist_df = pd.DataFrame([
        {"# de Police": "POL001", "Compagnie": "UV", "Texte": "Commission A", "Reçu": 50.0, "Statut": "Payé"},
        {"# de Police": "POL001", "Compagnie": "UV", "Texte": "Commission B", "Reçu": 50.0, "Statut": "Payé"},
        {"# de Police": "POL002", "Compagnie": "UV", "Texte": "Boni X", "Reçu": 25.0, "Statut": "Payé"},
        {"# de Police": "POL002", "Compagnie": "UV", "Texte": "Boni Y", "Reçu": 25.0, "Statut": "Payé"},
    ])

    sales_df = pd.DataFrame([
        {"# de Police": "POL001", "Com": 100.0, "Boni": None, "Sur-Com": None, "PA": 800.0, "item_id": "A1", "Conseiller": "Alice"},
        {"# de Police": "POL002", "Com": None, "Boni": 50.0, "Sur-Com": None, "PA": 300.0, "item_id": "B2", "Conseiller": "Bob"},
    ])

    result = r.reconcile(hist_df, sales_df)

    assert result.total_hist_lines == 4
    assert result.total_groups == 2
    assert result.passed == 2

    updates = result.get_sales_updates()
    assert updates["A1"] == {"Reçu 1": 100.0}
    assert updates["B2"] == {"Reçu 2": 50.0}


def main():
    """Run all tests."""
    tests = [
        test_classify_commission,
        test_classify_boni,
        test_classify_surcom,
        test_classify_priority_order,
        test_classify_no_match,
        test_threshold_high_pa,
        test_threshold_low_pa,
        test_reconcile_basic,
        test_reconcile_aggregation,
        test_reconcile_surcom_aggregation,
        test_reconcile_not_found,
        test_reconcile_flagged,
        test_reconcile_filters_paye_only,
        test_reconcile_hist_updates_with_aggregation,
        test_reconcile_to_display_dataframe,
        test_reconcile_empty_hist,
        test_reconcile_within_threshold,
        test_reconcile_mixed_police_aggregation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"  ✅ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
