"""
Reconciliation Engine
=====================

Cross-board reconciliation between Paiement Historique and Ventes/Production.

Matches payment records by `# de Police`, classifies each into the correct
Reçu field (1, 2, or 3), compares summed amounts against a dynamic threshold,
and prepares updates for both Monday.com boards.

Key behavior:
- Multiple historical lines with the same (# de Police, classification)
  are aggregated: their Reçu amounts are summed before comparison.
- Each aggregated group produces one ReconciliationMatch.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional

import pandas as pd


class ReconciliationStatus(StrEnum):
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
    """Result for an aggregated group of historical payment lines."""
    hist_indices: list[int]                  # All DF indices in this group
    police_number: str                       # # de Police
    compagnie: str                           # Compagnie
    texte: str                               # Representative texte
    recu_amount: Optional[float]             # SUMMED Reçu from historical
    classification: Optional[RecuClassification]  # Reçu 1/2/3
    reference_amount: Optional[float]        # Com/Boni/Sur-Com from sales
    threshold_pct: Optional[float]           # Threshold percentage used
    ecart_pct: Optional[float]              # Actual deviation percentage
    status: ReconciliationStatus             # Final status
    sales_item_id: Optional[str] = None      # Monday.com item_id in sales board
    conseiller: Optional[str] = None         # Advisor from sales board
    pa_amount: Optional[float] = None        # PA from sales board
    line_count: int = 1                      # Number of lines aggregated


@dataclass
class ReconciliationResult:
    """Global reconciliation result with metrics and helper methods."""
    matches: list[ReconciliationMatch] = field(default_factory=list)
    total_hist_lines: int = 0  # Total Payé lines before aggregation

    # --- Metrics ---

    @property
    def total_groups(self) -> int:
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
            List of (hist_index, conseiller) tuples — one per original row.
        """
        results = []
        for m in self.matches:
            if m.status == ReconciliationStatus.PASSED:
                for idx in m.hist_indices:
                    results.append((idx, m.conseiller))
        return results

    def to_display_dataframe(self) -> pd.DataFrame:
        """Convert matches to a DataFrame for UI display."""
        rows = []
        for m in self.matches:
            rows.append({
                "# Police": m.police_number,
                "Compagnie": m.compagnie,
                "Type": m.classification.label if m.classification else "—",
                "Reçu →": m.classification.recu_field if m.classification else "—",
                "Montant": m.recu_amount,
                "Lignes": m.line_count,
                "Référence": m.reference_amount,
                "Seuil": f"{m.threshold_pct:.0f}%" if m.threshold_pct is not None else "—",
                "Écart": f"{m.ecart_pct:.1f}%" if m.ecart_pct is not None else "—",
                "Statut": m.status.value,
                "Conseiller": m.conseiller or "—",
            })
        return pd.DataFrame(rows)

    # Status priority: worst first (higher index = worse)
    _STATUS_PRIORITY = {
        ReconciliationStatus.PASSED: 0,
        ReconciliationStatus.UNCLASSIFIED: 1,
        ReconciliationStatus.NOT_FOUND: 2,
        ReconciliationStatus.FLAGGED: 3,
    }

    def to_sales_view_dataframe(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """Board-format sales view: one row per police with Reçu 1/2/3 filled.

        Columns produced:
        # de Police | Compagnie | Conseiller | PA | Com | Reçu 1 |
        Boni | Reçu 2 | Sur-Com | Reçu 3 | Statut Rapp.

        Args:
            sales_df: Sales/production DataFrame from Monday.com.

        Returns:
            DataFrame with one row per police number (only matched polices).
        """
        # Build sales lookup
        sales_lookup: dict[str, pd.Series] = {}
        if "# de Police" in sales_df.columns:
            for _, row in sales_df.iterrows():
                police = str(row.get("# de Police", "")).strip()
                if police:
                    sales_lookup[police] = row

        # Group matches by police_number
        by_police: dict[str, list[ReconciliationMatch]] = defaultdict(list)
        for m in self.matches:
            by_police[m.police_number].append(m)

        rows = []
        for police, matches in by_police.items():
            sales_row = sales_lookup.get(police)

            # Base info from first match or sales
            compagnie = matches[0].compagnie
            conseiller = matches[0].conseiller or "—"
            pa = matches[0].pa_amount

            # Sales reference values
            com_ref = None
            boni_ref = None
            surcom_ref = None
            if sales_row is not None:
                com_ref = Reconciler._to_float(sales_row.get("Com"))
                boni_ref = Reconciler._to_float(sales_row.get("Boni"))
                surcom_ref = Reconciler._to_float(sales_row.get("Sur-Com"))
                if pa is None:
                    pa = Reconciler._to_float(sales_row.get("PA"))

            # Find match for each Reçu type
            recu_1 = recu_2 = recu_3 = None
            worst_status = ReconciliationStatus.PASSED

            for m in matches:
                if self._STATUS_PRIORITY.get(m.status, 0) > self._STATUS_PRIORITY.get(worst_status, 0):
                    worst_status = m.status

                if m.classification is None:
                    continue

                if m.classification.recu_field == "Reçu 1":
                    recu_1 = m.recu_amount
                elif m.classification.recu_field == "Reçu 2":
                    recu_2 = m.recu_amount
                elif m.classification.recu_field == "Reçu 3":
                    recu_3 = m.recu_amount

            rows.append({
                "# de Police": police,
                "Compagnie": compagnie,
                "Conseiller": conseiller,
                "PA": pa,
                "Com": com_ref,
                "Reçu 1": recu_1,
                "Boni": boni_ref,
                "Reçu 2": recu_2,
                "Sur-Com": surcom_ref,
                "Reçu 3": recu_3,
                "Statut Rapp.": worst_status.value,
            })

        return pd.DataFrame(rows)

    def to_hist_view_dataframe(self, hist_df: pd.DataFrame) -> pd.DataFrame:
        """Board-format historical view: Payé rows with Conseiller/Vérifié updated.

        For PASSED matches, fills in Conseiller from sales and sets Vérifié = "✓".

        Columns produced:
        # de Police | Nom Client | Compagnie | Conseiller | Vérifié |
        Reçu | Texte | Statut

        Args:
            hist_df: Historical payments DataFrame.

        Returns:
            DataFrame with Payé rows, Conseiller and Vérifié updated for matches.
        """
        # Filter to Payé only
        paye_mask = hist_df.get("Statut", pd.Series(dtype=str)).astype(str).str.strip() == "Payé"
        hist_paye = hist_df[paye_mask].copy()

        if hist_paye.empty:
            return pd.DataFrame()

        # Build lookup: hist_index → (conseiller, status)
        passed_lookup: dict[int, str] = {}  # index → conseiller
        for m in self.matches:
            if m.status == ReconciliationStatus.PASSED:
                for idx in m.hist_indices:
                    passed_lookup[idx] = m.conseiller or ""

        # Add Verifié and update Conseiller
        hist_paye["Verifié"] = False
        for idx in hist_paye.index:
            if idx in passed_lookup:
                hist_paye.at[idx, "Verifié"] = True
                if passed_lookup[idx]:
                    hist_paye.at[idx, "Conseiller"] = passed_lookup[idx]

        # Ensure Conseiller column exists
        if "Conseiller" not in hist_paye.columns:
            hist_paye["Conseiller"] = ""

        # Select and order columns
        output_cols = [
            "# de Police", "Nom Client", "Compagnie", "Conseiller",
            "Verifié", "Reçu", "Texte", "Statut",
        ]
        # Only include columns that exist
        cols = [c for c in output_cols if c in hist_paye.columns]
        return hist_paye[cols].reset_index(drop=True)


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

        Lines with the same (# de Police, classification) are aggregated:
        their Reçu amounts are summed before comparing against the sales
        reference value.

        Args:
            hist_df: Historical payments DataFrame.
            sales_df: Sales/production DataFrame.

        Returns:
            ReconciliationResult with all matches and metrics.
        """
        result = ReconciliationResult()

        # Filter historical to only "Payé" rows
        paye_mask = hist_df.get("Statut", pd.Series(dtype=str)).astype(str).str.strip() == "Payé"
        hist_paye = hist_df[paye_mask]

        if hist_paye.empty:
            return result

        result.total_hist_lines = len(hist_paye)

        # Build sales lookup by # de Police
        sales_lookup: dict[str, pd.Series] = {}
        if "# de Police" in sales_df.columns:
            for _, row in sales_df.iterrows():
                police = str(row.get("# de Police", "")).strip()
                if police:
                    sales_lookup[police] = row

        # --- Phase 1: Classify each row and group by (police, classification) ---
        # Key: (police_number, recu_field or "NOT_FOUND" or "UNCLASSIFIED")
        GroupKey = tuple[str, str]
        groups: dict[GroupKey, list[tuple[int, float, str, str]]] = defaultdict(list)
        # Each entry: (df_index, recu_amount, compagnie, texte)
        classification_map: dict[GroupKey, Optional[RecuClassification]] = {}

        for idx, hist_row in hist_paye.iterrows():
            police = str(hist_row.get("# de Police", "")).strip()
            compagnie = str(hist_row.get("Compagnie", ""))
            texte = str(hist_row.get("Texte", ""))
            recu_amount = self._to_float(hist_row.get("Reçu")) or 0.0

            sales_row = sales_lookup.get(police)

            if sales_row is None:
                key = (police, "_NOT_FOUND")
                groups[key].append((idx, recu_amount, compagnie, texte))
                classification_map[key] = None
                continue

            classification = self.classify_row(texte)
            if classification is None:
                key = (police, "_UNCLASSIFIED")
                groups[key].append((idx, recu_amount, compagnie, texte))
                classification_map[key] = None
                continue

            key = (police, classification.recu_field)
            groups[key].append((idx, recu_amount, compagnie, texte))
            classification_map[key] = classification

        # --- Phase 2: For each group, sum amounts and compare ---
        for group_key, entries in groups.items():
            police, recu_field_or_status = group_key
            indices = [e[0] for e in entries]
            total_recu = sum(e[1] for e in entries)
            compagnie = entries[0][2]
            # Use first texte as representative
            texte = entries[0][3]
            classification = classification_map[group_key]

            sales_row = sales_lookup.get(police)

            if recu_field_or_status == "_NOT_FOUND":
                result.matches.append(ReconciliationMatch(
                    hist_indices=indices,
                    police_number=police,
                    compagnie=compagnie,
                    texte=texte,
                    recu_amount=total_recu if total_recu else None,
                    classification=None,
                    reference_amount=None,
                    threshold_pct=None,
                    ecart_pct=None,
                    status=ReconciliationStatus.NOT_FOUND,
                    line_count=len(entries),
                ))
                continue

            if recu_field_or_status == "_UNCLASSIFIED":
                result.matches.append(ReconciliationMatch(
                    hist_indices=indices,
                    police_number=police,
                    compagnie=compagnie,
                    texte=texte,
                    recu_amount=total_recu if total_recu else None,
                    classification=None,
                    reference_amount=None,
                    threshold_pct=None,
                    ecart_pct=None,
                    status=ReconciliationStatus.UNCLASSIFIED,
                    sales_item_id=str(sales_row.get("item_id", "")) if sales_row is not None else None,
                    conseiller=self._get_conseiller(sales_row) if sales_row is not None else None,
                    pa_amount=self._to_float(sales_row.get("PA")) if sales_row is not None else None,
                    line_count=len(entries),
                ))
                continue

            # Normal classified group — compare summed amount against reference
            reference = self._to_float(sales_row.get(classification.compare_column))
            pa = self._to_float(sales_row.get("PA"))
            threshold = self.determine_threshold(pa)
            conseiller = self._get_conseiller(sales_row)
            sales_item_id = str(sales_row.get("item_id", ""))

            ecart_pct = self._compute_ecart(total_recu, reference)
            if ecart_pct is not None and ecart_pct <= threshold:
                status = ReconciliationStatus.PASSED
            else:
                status = ReconciliationStatus.FLAGGED

            result.matches.append(ReconciliationMatch(
                hist_indices=indices,
                police_number=police,
                compagnie=compagnie,
                texte=texte,
                recu_amount=round(total_recu, 2),
                classification=classification,
                reference_amount=reference,
                threshold_pct=threshold,
                ecart_pct=ecart_pct,
                status=status,
                sales_item_id=sales_item_id,
                conseiller=conseiller,
                pa_amount=pa,
                line_count=len(entries),
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
