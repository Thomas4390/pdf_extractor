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

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ReconciliationStatus(StrEnum):
    """Status of a reconciliation match."""
    PASSED = "Vérifié"
    FLAGGED = "Écart"
    NOT_FOUND = "Non trouvé"
    UNCLASSIFIED = "Non classifié"
    CB_VERIFIED = "CB Vérifié"
    CB_FLAGGED = "CB Écart"


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
    is_chargeback: bool = False              # True if this group contains Charge back lines


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

    @property
    def cb_verified(self) -> int:
        return sum(1 for m in self.matches if m.status == ReconciliationStatus.CB_VERIFIED)

    @property
    def cb_flagged(self) -> int:
        return sum(1 for m in self.matches if m.status == ReconciliationStatus.CB_FLAGGED)

    @property
    def total_chargebacks(self) -> int:
        """Total number of chargeback lines."""
        return sum(m.line_count for m in self.matches if m.is_chargeback)

    # --- Helper methods ---

    def get_sales_updates(self) -> dict[str, dict[str, float]]:
        """Get updates grouped by sales item_id with net amounts (Payé - CB).

        Returns:
            Dict mapping item_id to {recu_field: amount}, e.g.:
            {"12345": {"Reçu 1": 30.0}}  (net of Payé $510 - CB $480)
        """
        updates: dict[str, dict[str, float]] = {}

        # 1) Collect PASSED amounts
        for m in self.matches:
            if (
                m.status == ReconciliationStatus.PASSED
                and m.sales_item_id
                and m.classification
                and m.recu_amount is not None
            ):
                updates.setdefault(m.sales_item_id, {})[m.classification.recu_field] = m.recu_amount

        # 2) Subtract CB_VERIFIED amounts (recu_amount is already negative)
        for m in self.matches:
            if (
                m.status == ReconciliationStatus.CB_VERIFIED
                and m.sales_item_id
                and m.classification
                and m.recu_amount is not None
            ):
                field = m.classification.recu_field
                if m.sales_item_id in updates and field in updates[m.sales_item_id]:
                    updates[m.sales_item_id][field] += m.recu_amount  # negative → subtraction
                else:
                    # CB without matching Payé in this batch — write negative as-is
                    updates.setdefault(m.sales_item_id, {})[field] = m.recu_amount

        return updates

    def get_passed_hist_updates(self) -> list[tuple[int, Optional[str]]]:
        """Get historical indices and advisors for lines that passed.

        Note: For display/preview counts only. The actual writeback uses
        get_all_hist_updates() which includes FLAGGED and UNCLASSIFIED lines.

        Returns:
            List of (hist_index, conseiller) tuples — one per original row.
        """
        results = []
        for m in self.matches:
            if m.status == ReconciliationStatus.PASSED:
                for idx in m.hist_indices:
                    results.append((idx, m.conseiller))
        return results

    def get_all_hist_updates(self) -> list[tuple[int, Optional[str], bool]]:
        """Get historical indices and advisors for all matched items.

        Returns all matches (PASSED, FLAGGED, UNCLASSIFIED, CB_VERIFIED) —
        not NOT_FOUND or CB_FLAGGED — so the writeback can write Conseiller
        for all found items and Verifié/Pas Verifié labels.

        Returns:
            List of (hist_index, conseiller, is_passed) tuples.
        """
        results = []
        for m in self.matches:
            if m.status in (ReconciliationStatus.NOT_FOUND, ReconciliationStatus.CB_FLAGGED):
                continue
            is_passed = m.status in (ReconciliationStatus.PASSED, ReconciliationStatus.CB_VERIFIED)
            for idx in m.hist_indices:
                results.append((idx, m.conseiller, is_passed))
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
                "CB": "CB" if m.is_chargeback else "",
            })
        return pd.DataFrame(rows)

    # Status priority: worst first (higher index = worse)
    _STATUS_PRIORITY = {
        ReconciliationStatus.PASSED: 0,
        ReconciliationStatus.CB_VERIFIED: 1,
        ReconciliationStatus.UNCLASSIFIED: 2,
        ReconciliationStatus.NOT_FOUND: 3,
        ReconciliationStatus.CB_FLAGGED: 4,
        ReconciliationStatus.FLAGGED: 5,
    }

    def to_sales_view_dataframe(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """Board-format sales view: one row per police with Reçu 1/2/3 and écarts.

        Columns produced:
        # de Police | Compagnie | Conseiller | PA |
        Com | Reçu 1 | Écart 1 | Boni | Reçu 2 | Écart 2 |
        Sur-Com | Reçu 3 | Écart 3 | Total | Total Reçu | Écart Total |
        Statut Rapp.

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
                    if police in sales_lookup:
                        logger.warning(
                            "Reconciler: duplicate police '%s' in sales view — using last row",
                            police,
                        )
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
            total_ref = None
            total_recu_ref = None
            if sales_row is not None:
                com_ref = Reconciler._to_float(sales_row.get("Com"))
                boni_ref = Reconciler._to_float(sales_row.get("Boni"))
                surcom_ref = Reconciler._to_float(sales_row.get("Sur-Com"))
                total_ref = Reconciler._to_float(sales_row.get("Total"))
                total_recu_ref = Reconciler._to_float(sales_row.get("Total Reçu"))
                if pa is None:
                    pa = Reconciler._to_float(sales_row.get("PA"))

            # Find match for each Reçu type (Payé and CB separately)
            recu_paye = {"Reçu 1": None, "Reçu 2": None, "Reçu 3": None}
            recu_cb = {"Reçu 1": None, "Reçu 2": None, "Reçu 3": None}
            ecart_1 = ecart_2 = ecart_3 = None
            worst_status = ReconciliationStatus.PASSED
            has_cb = False

            for m in matches:
                if self._STATUS_PRIORITY.get(m.status, 0) > self._STATUS_PRIORITY.get(worst_status, 0):
                    worst_status = m.status

                if m.classification is None:
                    continue

                field = m.classification.recu_field
                if m.is_chargeback:
                    has_cb = True
                    recu_cb[field] = m.recu_amount  # negative
                else:
                    recu_paye[field] = m.recu_amount

                # Store ecart_pct for both Payé and CB matches
                if field == "Reçu 1" and ecart_1 is None:
                    ecart_1 = m.ecart_pct
                elif field == "Reçu 2" and ecart_2 is None:
                    ecart_2 = m.ecart_pct
                elif field == "Reçu 3" and ecart_3 is None:
                    ecart_3 = m.ecart_pct

            # Compute net amounts (Payé + CB where CB is negative)
            def _net(paye_val, cb_val):
                if paye_val is None and cb_val is None:
                    return None
                return (paye_val or 0.0) + (cb_val or 0.0)

            recu_1 = _net(recu_paye["Reçu 1"], recu_cb["Reçu 1"])
            recu_2 = _net(recu_paye["Reçu 2"], recu_cb["Reçu 2"])
            recu_3 = _net(recu_paye["Reçu 3"], recu_cb["Reçu 3"])

            # Compute Total Reçu from actual matched amounts (not formula)
            computed_total_recu = sum(v for v in (recu_1, recu_2, recu_3) if v is not None)
            # Use computed total when formula-based total is unavailable
            display_total_recu = total_recu_ref if total_recu_ref is not None else computed_total_recu

            # Compute Total écart using computed total when formula unavailable
            if total_ref is not None:
                ecart_total = Reconciler._compute_ecart(display_total_recu, total_ref)
            else:
                ecart_total = None

            row_data = {
                "# de Police": police,
                "Compagnie": compagnie,
                "Conseiller": conseiller,
                "PA": pa,
                "Com": com_ref,
                "Reçu 1": recu_1,
                "Écart 1": f"{ecart_1:.1f}%" if ecart_1 is not None else "—",
                "Boni": boni_ref,
                "Reçu 2": recu_2,
                "Écart 2": f"{ecart_2:.1f}%" if ecart_2 is not None else "—",
                "Sur-Com": surcom_ref,
                "Reçu 3": recu_3,
                "Écart 3": f"{ecart_3:.1f}%" if ecart_3 is not None else "—",
                "Total": total_ref,
                "Total Reçu": display_total_recu,
                "Écart Total": f"{ecart_total:.1f}%" if ecart_total is not None else "—",
                "Statut Rapp.": worst_status.value,
            }
            if has_cb:
                row_data["CB"] = "Oui"
            rows.append(row_data)

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
        # Filter to Payé and Charge back
        statut_col = hist_df.get("Statut", pd.Series(dtype=str)).astype(str).str.strip()
        active_mask = statut_col.isin(["Payé", "Charge back"])
        hist_active = hist_df[active_mask].copy()

        if hist_active.empty:
            return pd.DataFrame()

        # Build lookups: index → conseiller for all found matches,
        # and set of passed indices for Verifié
        conseiller_lookup: dict[int, str] = {}  # index → conseiller
        passed_indices: set[int] = set()
        for m in self.matches:
            if m.status in (ReconciliationStatus.NOT_FOUND, ReconciliationStatus.CB_FLAGGED):
                continue
            for idx in m.hist_indices:
                conseiller_lookup[idx] = m.conseiller or ""
            if m.status in (ReconciliationStatus.PASSED, ReconciliationStatus.CB_VERIFIED):
                passed_indices.update(m.hist_indices)

        # Ensure Conseiller column exists before mutations
        if "Conseiller" not in hist_active.columns:
            hist_active["Conseiller"] = ""

        # Add Verifié labels and update Conseiller
        hist_active["Verifié"] = "Pas Verifié"
        for idx in hist_active.index:
            if idx in conseiller_lookup and conseiller_lookup[idx]:
                hist_active.at[idx, "Conseiller"] = conseiller_lookup[idx]
            if idx in passed_indices:
                hist_active.at[idx, "Verifié"] = "Verifié"

        # Select and order columns
        output_cols = [
            "# de Police", "Nom Client", "Compagnie", "Conseiller",
            "Verifié", "Reçu", "Texte", "Statut",
        ]
        # Only include columns that exist
        cols = [c for c in output_cols if c in hist_active.columns]
        return hist_active[cols].reset_index(drop=True)


class Reconciler:
    """Reconciliation engine for cross-board matching."""

    # Regex to extract the category label between "|" and "(" in structured Texte.
    # Example: "Protection A | Commission (Partage: 40%, Com: 50%, TB: 0%)"
    #   → captures "Commission"
    _STRUCTURED_LABEL_RE = re.compile(r"\|\s*([^(|]+?)\s*\(")

    @staticmethod
    def classify_row(texte: str) -> Optional[RecuClassification]:
        """Classify a historical row into Reçu 1/2/3 based on Texte content.

        For **structured** Texte (contains "|" separator):
        1. Check for "[sur-com]" suffix → Reçu 3 (strongest signal)
        2. Extract category label between "|" and "("
        3. Match label: sur-com → Reçu 3, boni → Reçu 2, commission → Reçu 1

        For **unstructured** Texte (no "|"): fall back to substring matching.

        Args:
            texte: The Texte field from the historical payment row.

        Returns:
            RecuClassification or None if no match.
        """
        if not texte:
            return None

        texte_lower = texte.lower()

        # --- Structured Texte path (contains "|") ---
        if "|" in texte:
            # Rule 1: [Sur-Com] suffix → always Reçu 3
            if texte_lower.rstrip().endswith("[sur-com]"):
                return RecuClassification(
                    recu_field="Reçu 3",
                    compare_column="Sur-Com",
                    label="Sur-Commission",
                )

            # Rule 2: Extract category label between "|" and "("
            m = Reconciler._STRUCTURED_LABEL_RE.search(texte)
            if m:
                label = m.group(1).strip().lower()
                if "sur-com" in label or "surcom" in label:
                    return RecuClassification(
                        recu_field="Reçu 3",
                        compare_column="Sur-Com",
                        label="Sur-Commission",
                    )
                if "boni" in label:
                    return RecuClassification(
                        recu_field="Reçu 2",
                        compare_column="Boni",
                        label="Boni",
                    )
                if "commission" in label:
                    return RecuClassification(
                        recu_field="Reçu 1",
                        compare_column="Com",
                        label="Commission",
                    )

            # Structured but no parenthesized params — check each segment
            # (handles both "label | ..." Assomption format and "... | label" UV format)
            segments = [s.strip().lower() for s in texte.split("|")]
            for seg in segments:
                if "sur-com" in seg or "surcom" in seg:
                    return RecuClassification(
                        recu_field="Reçu 3",
                        compare_column="Sur-Com",
                        label="Sur-Commission",
                    )
            for seg in segments:
                if "boni" in seg:
                    return RecuClassification(
                        recu_field="Reçu 2",
                        compare_column="Boni",
                        label="Boni",
                    )
            for seg in segments:
                if "commission" in seg:
                    return RecuClassification(
                        recu_field="Reçu 1",
                        compare_column="Com",
                        label="Commission",
                    )

            return None

        # --- Unstructured Texte fallback (no "|") ---
        # Rule 1: Sur-Com (most specific)
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

        # Rule 3: Commission
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

        When the sales reference (Com/Boni/Sur-Com) is None — typically
        because Boni and Sur-Com are formula columns whose values weren't
        enriched — the match is marked PASSED (or CB_VERIFIED for
        chargebacks).  This ensures Reçu 2/3 are always written even when
        Monday.com formula enrichment fails.

        Args:
            hist_df: Historical payments DataFrame.
            sales_df: Sales/production DataFrame.

        Returns:
            ReconciliationResult with all matches and metrics.
        """
        result = ReconciliationResult()

        # Filter historical to "Payé" + "Charge back" rows
        statut_col = hist_df.get("Statut", pd.Series(dtype=str)).astype(str).str.strip()
        active_mask = statut_col.isin(["Payé", "Charge back"])
        hist_active = hist_df[active_mask]

        if hist_active.empty:
            return result

        result.total_hist_lines = len(hist_active)

        # Build sales lookup by # de Police
        sales_lookup: dict[str, pd.Series] = {}
        if "# de Police" in sales_df.columns:
            for _, row in sales_df.iterrows():
                police = str(row.get("# de Police", "")).strip()
                if police:
                    if police in sales_lookup:
                        logger.warning(
                            "Reconciler: duplicate police '%s' in sales board — using last row",
                            police,
                        )
                    sales_lookup[police] = row

        # Warn if key columns are entirely null in the sales DataFrame
        # Com is a regular numbers column; Boni, Sur-Com, Total, Total Reçu
        # are formula columns that require FormulaValue enrichment.
        for _fc in ("Com", "Boni", "Sur-Com", "Total", "Total Reçu"):
            if _fc in sales_df.columns and sales_df[_fc].dropna().empty:
                logger.warning(
                    "Reconciler: sales column '%s' is entirely null — "
                    "formula extraction from Monday.com likely failed. "
                    "All related écarts will be None.",
                    _fc,
                )

        # --- Phase 1: Classify each row and group by (police, classification) ---
        # Key: (police_number, recu_field or "NOT_FOUND" or "UNCLASSIFIED")
        # CB lines get a "CB_" prefix, e.g. "CB_Reçu 1"
        GroupKey = tuple[str, str]
        groups: dict[GroupKey, list[tuple[int, float, str, str]]] = defaultdict(list)
        # Each entry: (df_index, recu_amount, compagnie, texte)
        classification_map: dict[GroupKey, Optional[RecuClassification]] = {}

        for idx, hist_row in hist_active.iterrows():
            police = str(hist_row.get("# de Police", "")).strip()
            compagnie = str(hist_row.get("Compagnie", ""))
            texte = str(hist_row.get("Texte", ""))
            recu_amount = self._to_float(hist_row.get("Reçu")) or 0.0
            statut = str(hist_row.get("Statut", "")).strip()
            is_cb = (statut == "Charge back")

            sales_row = sales_lookup.get(police)

            if sales_row is None:
                key = (police, "CB_NOT_FOUND" if is_cb else "_NOT_FOUND")
                groups[key].append((idx, recu_amount, compagnie, texte))
                classification_map[key] = None
                continue

            classification = self.classify_row(texte)
            if classification is None:
                key = (police, "CB_UNCLASSIFIED" if is_cb else "_UNCLASSIFIED")
                groups[key].append((idx, recu_amount, compagnie, texte))
                classification_map[key] = None
                continue

            if is_cb:
                key = (police, f"CB_{classification.recu_field}")
            else:
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

            is_cb_group = recu_field_or_status.startswith("CB_")

            sales_row = sales_lookup.get(police)

            # --- NOT_FOUND groups (both Payé and CB) ---
            if recu_field_or_status in ("_NOT_FOUND", "CB_NOT_FOUND"):
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
                    is_chargeback=is_cb_group,
                ))
                continue

            # --- UNCLASSIFIED groups (both Payé and CB) ---
            if recu_field_or_status in ("_UNCLASSIFIED", "CB_UNCLASSIFIED"):
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
                    is_chargeback=is_cb_group,
                ))
                continue

            # --- Classified group — compare summed amount against reference ---
            reference = self._to_float(sales_row.get(classification.compare_column))
            pa = self._to_float(sales_row.get("PA"))
            threshold = self.determine_threshold(pa)
            conseiller = self._get_conseiller(sales_row)
            sales_item_id = str(sales_row.get("item_id", ""))

            if is_cb_group:
                # Chargebacks: compare abs(total_recu) vs reference
                compare_amount = abs(total_recu)
                ecart_pct = self._compute_ecart(compare_amount, reference)
                if ecart_pct is not None and ecart_pct <= threshold:
                    status = ReconciliationStatus.CB_VERIFIED
                elif reference is None:
                    # Formula columns (Boni, Sur-Com) often return None
                    status = ReconciliationStatus.CB_VERIFIED
                else:
                    status = ReconciliationStatus.CB_FLAGGED
            else:
                # Payé: compare summed Reçu against sales reference
                ecart_pct = self._compute_ecart(total_recu, reference)
                if ecart_pct is not None and ecart_pct <= threshold:
                    status = ReconciliationStatus.PASSED
                elif reference is None:
                    # Formula columns (Boni, Sur-Com) often return None —
                    # trust the extracted amount and write it
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
                is_chargeback=is_cb_group,
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

    # Advisor names to blank out (normalized, lower-case)
    _EXCLUDED_ADVISORS = {'achraf el hajji'}

    @staticmethod
    def _get_conseiller(sales_row: pd.Series) -> Optional[str]:
        """Extract advisor name from sales row."""
        conseiller = sales_row.get("Conseiller")
        if conseiller is None or pd.isna(conseiller):
            return None
        name = str(conseiller).strip()
        if not name:
            return None
        if name.lower() in Reconciler._EXCLUDED_ADVISORS:
            return None
        return name
