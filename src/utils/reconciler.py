"""
Reconciliation Engine
=====================

Cross-board reconciliation between Paiement Historique and Ventes/Production.

Matches payment records by `# de Police`, classifies each into the correct
Reçu field (1, 2, or 3), compares amounts against a dynamic threshold,
and prepares updates for both Monday.com boards.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


class ReconciliationStatus(str, Enum):
    """Status of a reconciliation match."""
    PASSED = "Vérifié"
    FLAGGED = "Écart"
    NOT_FOUND = "Non trouvé"
    UNCLASSIFIED = "Non classifié"


@dataclass
class RecuClassification:
    """Which Reçu field (1/2/3) and which comparison column (Com/Boni/Sur-Com)."""
    recu_field: str       # "Reçu 1", "Reçu 2", or "Reçu 3"
    compare_column: str   # "Com", "Boni", or "Sur-Com"
    label: str            # Human-readable label


@dataclass
class ReconciliationMatch:
    """Result for a single historical payment line."""
    hist_index: int                          # Index in the historical DataFrame
    police_number: str                       # # de Police
    compagnie: str                           # Compagnie
    texte: str                               # Texte from historical
    recu_amount: Optional[float]             # Reçu from historical
    classification: Optional[RecuClassification]  # Reçu 1/2/3
    reference_amount: Optional[float]        # Com/Boni/Sur-Com from sales
    threshold_pct: Optional[float]           # Threshold percentage used
    ecart_pct: Optional[float]              # Actual deviation percentage
    status: ReconciliationStatus             # Final status
    sales_item_id: Optional[str] = None      # Monday.com item_id in sales board
    conseiller: Optional[str] = None         # Advisor from sales board
    pa_amount: Optional[float] = None        # PA from sales board


@dataclass
class ReconciliationResult:
    """Global reconciliation result with metrics and helper methods."""
    matches: list[ReconciliationMatch] = field(default_factory=list)

    # --- Metrics ---

    @property
    def total_paye(self) -> int:
        return len(self.matches)

    @property
    def found(self) -> int:
        return sum(1 for m in self.matches if m.status != ReconciliationStatus.NOT_FOUND)

    @property
    def not_found(self) -> int:
        return sum(1 for m in self.matches if m.status == ReconciliationStatus.NOT_FOUND)

    @property
    def passed(self) -> int:
        return sum(1 for m in self.matches if m.status == ReconciliationStatus.PASSED)

    @property
    def flagged(self) -> int:
        return sum(1 for m in self.matches if m.status == ReconciliationStatus.FLAGGED)

    @property
    def unclassified(self) -> int:
        return sum(1 for m in self.matches if m.status == ReconciliationStatus.UNCLASSIFIED)

    # --- Helper methods ---

    def get_sales_updates(self) -> dict[str, dict[str, float]]:
        """Get updates grouped by sales item_id.

        Returns:
            Dict mapping item_id to {recu_field: amount}, e.g.:
            {"12345": {"Reçu 1": 100.0, "Reçu 2": 50.0}}
        """
        updates: dict[str, dict[str, float]] = {}
        for m in self.matches:
            if (
                m.status == ReconciliationStatus.PASSED
                and m.sales_item_id
                and m.classification
                and m.recu_amount is not None
            ):
                if m.sales_item_id not in updates:
                    updates[m.sales_item_id] = {}
                updates[m.sales_item_id][m.classification.recu_field] = m.recu_amount
        return updates

    def get_passed_hist_updates(self) -> list[tuple[int, Optional[str]]]:
        """Get historical indices and advisors for lines that passed.

        Returns:
            List of (hist_index, conseiller) tuples.
        """
        return [
            (m.hist_index, m.conseiller)
            for m in self.matches
            if m.status == ReconciliationStatus.PASSED
        ]

    def to_display_dataframe(self) -> pd.DataFrame:
        """Convert matches to a DataFrame for UI display."""
        rows = []
        for m in self.matches:
            rows.append({
                "# Police": m.police_number,
                "Compagnie": m.compagnie,
                "Texte": m.texte,
                "Reçu →": m.classification.recu_field if m.classification else "—",
                "Montant": m.recu_amount,
                "Référence": m.reference_amount,
                "Seuil": f"{m.threshold_pct:.0f}%" if m.threshold_pct is not None else "—",
                "Écart": f"{m.ecart_pct:.1f}%" if m.ecart_pct is not None else "—",
                "Statut": m.status.value,
                "Conseiller": m.conseiller or "—",
            })
        return pd.DataFrame(rows)


class Reconciler:
    """Reconciliation engine for cross-board matching."""

    @staticmethod
    def classify_row(texte: str) -> Optional[RecuClassification]:
        """Classify a historical row into Reçu 1/2/3 based on Texte content.

        Priority order (most specific first):
        1. Sur-Com → Reçu 3
        2. Boni → Reçu 2
        3. Commission → Reçu 1

        Args:
            texte: The Texte field from the historical payment row.

        Returns:
            RecuClassification or None if no match.
        """
        if not texte:
            return None

        texte_lower = texte.lower()

        # Rule 1: Sur-Com (most specific, UV Inc only)
        if "sur-com" in texte_lower or "surcom" in texte_lower:
            return RecuClassification(
                recu_field="Reçu 3",
                compare_column="Sur-Com",
                label="Sur-Commission",
            )

        # Rule 2: Boni
        if "boni" in texte_lower:
            return RecuClassification(
                recu_field="Reçu 2",
                compare_column="Boni",
                label="Boni",
            )

        # Rule 3: Commission (includes "commission 1ère année", etc.)
        if "commission" in texte_lower:
            return RecuClassification(
                recu_field="Reçu 1",
                compare_column="Com",
                label="Commission",
            )

        return None

    @staticmethod
    def determine_threshold(pa: Optional[float]) -> float:
        """Determine the comparison threshold based on PA amount.

        Args:
            pa: Prime annualisée (annual premium).

        Returns:
            Threshold as a percentage (10.0 or 20.0).
        """
        if pa is not None and pa > 500:
            return 10.0
        return 20.0

    def reconcile(
        self,
        hist_df: pd.DataFrame,
        sales_df: pd.DataFrame,
    ) -> ReconciliationResult:
        """Run reconciliation between historical payments and sales/production.

        Args:
            hist_df: Historical payments DataFrame (with Statut, # de Police,
                     Compagnie, Texte, Reçu columns).
            sales_df: Sales/production DataFrame (with # de Police, Com, Boni,
                      Sur-Com, PA, item_id, Conseiller columns).

        Returns:
            ReconciliationResult with all matches and metrics.
        """
        result = ReconciliationResult()

        # Filter historical to only "Payé" rows
        paye_mask = hist_df.get("Statut", pd.Series(dtype=str)).astype(str).str.strip() == "Payé"
        hist_paye = hist_df[paye_mask]

        if hist_paye.empty:
            return result

        # Build sales lookup by # de Police
        sales_lookup: dict[str, pd.Series] = {}
        if "# de Police" in sales_df.columns:
            for _, row in sales_df.iterrows():
                police = str(row.get("# de Police", "")).strip()
                if police:
                    sales_lookup[police] = row

        # Process each Payé line
        for idx, hist_row in hist_paye.iterrows():
            police = str(hist_row.get("# de Police", "")).strip()
            compagnie = str(hist_row.get("Compagnie", ""))
            texte = str(hist_row.get("Texte", ""))
            recu_amount = self._to_float(hist_row.get("Reçu"))

            # Try to find in sales
            sales_row = sales_lookup.get(police)

            if sales_row is None:
                result.matches.append(ReconciliationMatch(
                    hist_index=idx,
                    police_number=police,
                    compagnie=compagnie,
                    texte=texte,
                    recu_amount=recu_amount,
                    classification=None,
                    reference_amount=None,
                    threshold_pct=None,
                    ecart_pct=None,
                    status=ReconciliationStatus.NOT_FOUND,
                ))
                continue

            # Classify
            classification = self.classify_row(texte)
            if classification is None:
                result.matches.append(ReconciliationMatch(
                    hist_index=idx,
                    police_number=police,
                    compagnie=compagnie,
                    texte=texte,
                    recu_amount=recu_amount,
                    classification=None,
                    reference_amount=None,
                    threshold_pct=None,
                    ecart_pct=None,
                    status=ReconciliationStatus.UNCLASSIFIED,
                    sales_item_id=str(sales_row.get("item_id", "")),
                    conseiller=self._get_conseiller(sales_row),
                    pa_amount=self._to_float(sales_row.get("PA")),
                ))
                continue

            # Get reference amount and PA
            reference = self._to_float(sales_row.get(classification.compare_column))
            pa = self._to_float(sales_row.get("PA"))
            threshold = self.determine_threshold(pa)
            conseiller = self._get_conseiller(sales_row)
            sales_item_id = str(sales_row.get("item_id", ""))

            # Compare
            ecart_pct = self._compute_ecart(recu_amount, reference)
            if ecart_pct is not None and ecart_pct <= threshold:
                status = ReconciliationStatus.PASSED
            else:
                status = ReconciliationStatus.FLAGGED

            result.matches.append(ReconciliationMatch(
                hist_index=idx,
                police_number=police,
                compagnie=compagnie,
                texte=texte,
                recu_amount=recu_amount,
                classification=classification,
                reference_amount=reference,
                threshold_pct=threshold,
                ecart_pct=ecart_pct,
                status=status,
                sales_item_id=sales_item_id,
                conseiller=conseiller,
                pa_amount=pa,
            ))

        return result

    @staticmethod
    def _to_float(value) -> Optional[float]:
        """Safely convert a value to float."""
        if value is None:
            return None
        try:
            f = float(value)
            return f if f == f else None  # NaN check
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _compute_ecart(
        actual: Optional[float],
        reference: Optional[float],
    ) -> Optional[float]:
        """Compute deviation percentage between actual and reference.

        Returns:
            Absolute deviation percentage, or None if comparison not possible.
        """
        if actual is None or reference is None:
            return None
        if reference == 0:
            return 100.0 if actual != 0 else 0.0
        return abs((actual - reference) / reference) * 100

    @staticmethod
    def _get_conseiller(sales_row: pd.Series) -> Optional[str]:
        """Extract advisor name from sales row."""
        conseiller = sales_row.get("Conseiller")
        if conseiller is None or pd.isna(conseiller):
            return None
        name = str(conseiller).strip()
        return name if name else None
