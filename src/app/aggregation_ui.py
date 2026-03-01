"""
UI components for the Data Aggregation mode.

Provides Streamlit components for:
- Data preview
- Aggregation execution
"""

import pandas as pd
import streamlit as st

from ..utils.aggregator import SourceBoardConfig

# =============================================================================
# DATA PREVIEW
# =============================================================================

def render_source_data_preview(
    source_key: str,
    config: SourceBoardConfig,
    df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
) -> None:
    """
    Render preview for a single source's data.

    Args:
        source_key: Source identifier
        config: Source configuration
        df: Raw DataFrame from Monday.com
        filtered_df: Date-filtered DataFrame
        aggregated_df: Aggregated by advisor DataFrame
    """
    with st.expander(
        f"📊 {config.display_name} ({len(aggregated_df)} conseillers)",
        expanded=False,
    ):
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Lignes brutes", len(df))
        with col2:
            st.metric("Après filtrage", len(filtered_df))
        with col3:
            st.metric("Conseillers", len(aggregated_df))
        with col4:
            # Total
            value_col = config.aggregate_column
            total = aggregated_df[value_col].sum() if value_col in aggregated_df.columns else 0
            st.metric(f"Total {value_col}", f"{total:,.2f}")

        # Preview table
        if not aggregated_df.empty:
            # Format numbers for display
            display_df = aggregated_df.copy()
            if value_col in display_df.columns:
                display_df[value_col] = display_df[value_col].apply(
                    lambda x: f"{x:,.2f}" if pd.notna(x) else "-"
                )
            st.dataframe(display_df, width="stretch", hide_index=True)
        else:
            st.warning("Aucune donnée après filtrage.")


def render_combined_preview(combined_df: pd.DataFrame) -> None:
    """
    Render preview of combined aggregated data (non-editable) with summary stats.

    Shows:
    - Key metrics cards (AE CA, Collected, PA Vendues)
    - Profitability summary (if available)
    - Data table

    Args:
        combined_df: DataFrame with all sources combined by advisor
    """
    if combined_df.empty:
        st.warning("Aucune donnée à afficher.")
        return

    # Summary stats
    advisor_count = len(combined_df)
    categorical_cols = {"Conseiller", "Profitable", "Advisor_Status"}
    numeric_cols = [col for col in combined_df.columns if col not in categorical_cols]

    # Priority metrics to show first
    priority_metrics = ["AE CA", "Collected", "PA Vendues"]
    display_metrics = [m for m in priority_metrics if m in numeric_cols]
    # Add remaining up to 4 total
    for m in numeric_cols:
        if m not in display_metrics and len(display_metrics) < 4:
            display_metrics.append(m)

    # Display main KPI cards
    st.markdown("##### 📊 Indicateurs clés")
    kpi_cols = st.columns(min(len(display_metrics) + 1, 5))

    with kpi_cols[0]:
        st.metric("👥 Conseillers", advisor_count)

    for idx, col in enumerate(display_metrics):
        with kpi_cols[idx + 1]:
            total = combined_df[col].sum()
            if isinstance(total, (int, float)):
                # Format with $ for monetary values
                if col in ["AE CA", "Collected", "Profit", "Total Dépenses"]:
                    st.metric(col, f"${total:,.0f}")
                else:
                    st.metric(col, f"{total:,.0f}")

    # Profitability summary (if available)
    if "Profitable" in combined_df.columns:
        st.markdown("##### 💰 Répartition par profitabilité")
        status_counts = combined_df["Profitable"].value_counts()

        # Define colors and labels
        status_info = {
            "Win": ("🟢", "Win (>100%)"),
            "Middle": ("🟡", "Middle (20-100%)"),
            "Loss": ("🔴", "Loss (<20%)"),
            "N/A": ("⚪", "Sans données"),
        }

        prof_cols = st.columns(4)
        for idx, (status, (emoji, label)) in enumerate(status_info.items()):
            count = status_counts.get(status, 0)
            pct = (count / advisor_count * 100) if advisor_count > 0 else 0
            with prof_cols[idx]:
                st.markdown(f"""
                <div style="text-align: center; padding: 8px; background: #f8f9fa; border-radius: 8px;">
                    <div style="font-size: 24px;">{emoji}</div>
                    <div style="font-size: 18px; font-weight: 600;">{count}</div>
                    <div style="font-size: 12px; color: #6B7280;">{label}</div>
                    <div style="font-size: 11px; color: #9CA3AF;">{pct:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # Display data table
    st.markdown("##### 📋 Données détaillées")
    display_df = combined_df.copy()

    # Format numeric columns for display, skip categorical columns
    for col in display_df.columns:
        if col not in categorical_cols:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:,.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x
            )

    st.dataframe(display_df, width="stretch", hide_index=True, height=400)


def render_editable_preview(
    combined_df: pd.DataFrame,
    target_board_name: str,
    group_name: str,
) -> pd.DataFrame:
    """
    Render an editable preview of the data to be uploaded.

    Args:
        combined_df: DataFrame with all sources combined by advisor
        target_board_name: Name of target board
        group_name: Name of target group

    Returns:
        Edited DataFrame (may have been modified by user)
    """
    if combined_df.empty:
        st.warning("Aucune donnée à afficher.")
        return combined_df

    # Target info header
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📤 Aperçu de l'upload</div>
        <p class="section-description">
            Vérifiez et modifiez les données avant l'envoi vers Monday.com.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Target details
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-icon">📋</div>
            <div class="info-content">
                <div class="info-label">Board cible</div>
                <div class="info-value">{target_board_name}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-icon">📁</div>
            <div class="info-content">
                <div class="info-label">Groupe cible</div>
                <div class="info-value">{group_name}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Editable dataframe
    st.markdown("**✏️ Éditez les valeurs si nécessaire :**")

    # Configure column display
    # Exclude categorical columns from numeric treatment
    categorical_cols = {"Conseiller", "Profitable", "Advisor_Status"}
    numeric_cols = [col for col in combined_df.columns if col not in categorical_cols]

    column_config = {
        "Conseiller": st.column_config.TextColumn(
            "Conseiller",
            help="Nom du conseiller",
            disabled=True,  # Don't allow editing advisor names
        ),
    }

    # Add Profitable as a text column if present
    if "Profitable" in combined_df.columns:
        column_config["Profitable"] = st.column_config.TextColumn(
            "Profitable",
            help="Statut de profitabilité",
            disabled=True,
        )

    # Add Advisor_Status as a text column if present
    if "Advisor_Status" in combined_df.columns:
        column_config["Advisor_Status"] = st.column_config.TextColumn(
            "Advisor_Status",
            help="Statut du conseiller (Active/New/Inactive)",
            disabled=True,
        )

    # Contextual help per column
    COLUMN_HELP = {
        "PA Vendues": "Primes annualisées vendues pour la période",
        "Collected": "Montants collectés via AE Tracker",
        "AE CA": "Chiffre d'affaires AE (paiement historique)",
        "Coût": "Coût total du conseiller (négatif)",
        "Dépenses par Conseiller": "Dépenses directes attribuées au conseiller (négatif)",
        "Leads": "Nombre de leads générés",
        "Bonus": "Bonus attribué (négatif = charge)",
        "Récompenses": "Récompenses attribuées",
        "Total Dépenses": "Somme des coûts, dépenses et bonus",
        "Profit": "AE CA + Récompenses + charges",
        "CA/Lead": "Chiffre d'affaires par lead",
        "Profit/Lead": "Profit par lead",
        "Ratio Brut": "Ratio brut de profitabilité (%)",
        "Ratio Net": "Ratio net de profitabilité (%)",
    }

    for col in numeric_cols:
        column_config[col] = st.column_config.NumberColumn(
            col,
            help=COLUMN_HELP.get(col, f"Valeur agrégée pour {col}"),
            format="%.2f",
        )

    # Use data_editor for interactive editing
    edited_df = st.data_editor(
        combined_df,
        column_config=column_config,
        width="stretch",
        hide_index=True,
        num_rows="fixed",  # Don't allow adding/removing rows
        key="agg_data_editor",
    )

    # Show totals after editing
    st.markdown("---")
    st.markdown("**📊 Totaux après modification :**")

    total_cols = st.columns(len(numeric_cols) + 1)
    with total_cols[0]:
        st.metric("Conseillers", len(edited_df))

    for idx, col in enumerate(numeric_cols):
        with total_cols[idx + 1]:
            total = edited_df[col].sum()
            st.metric(f"Total {col}", f"{total:,.2f}")

    return edited_df


def render_execution_summary(
    target_board_name: str,
    group_name: str,
    advisor_count: int,
    sources_count: int,
) -> None:
    """
    Render execution summary before running upsert.

    Args:
        target_board_name: Name of target board
        group_name: Target group name
        advisor_count: Number of advisors to upsert
        sources_count: Number of active sources
    """
    st.markdown("""
    <div class="section-card">
        <div class="section-title">🚀 Prêt à exécuter</div>
        <p class="section-description">Vérifiez les informations ci-dessous avant de lancer l'upsert.</p>
    </div>
    """, unsafe_allow_html=True)

    # Summary cards
    st.markdown(f"""
    <div class="summary-stats">
        <div class="summary-stat">
            <div class="stat-value">{advisor_count}</div>
            <div class="stat-label">Conseillers</div>
        </div>
        <div class="summary-stat">
            <div class="stat-value">{sources_count}</div>
            <div class="stat-label">Sources</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Target info
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-icon">📋</div>
            <div class="info-content">
                <div class="info-label">Board cible</div>
                <div class="info-value">{target_board_name}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-icon">📁</div>
            <div class="info-content">
                <div class="info-label">Groupe cible</div>
                <div class="info-value">{group_name}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_execution_result(result: dict) -> None:
    """
    Render the result of an upsert operation.

    Args:
        result: Dict with {updated: int, created: int, moved: int, errors: list}
    """
    if result["errors"]:
        st.error(f"⚠️ {len(result['errors'])} erreur(s) rencontrée(s)")
        with st.expander("Voir les erreurs"):
            for err in result["errors"]:
                st.text(err)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mis à jour", result.get("updated", 0), delta_color="normal")
    with col2:
        st.metric("Créés", result.get("created", 0), delta_color="normal")
    with col3:
        st.metric("Déplacés", result.get("moved", 0), delta_color="off")

    if result.get("updated", 0) > 0 or result.get("created", 0) > 0:
        st.success("✅ Upsert terminé avec succès!")


# =============================================================================
# NAVIGATION BUTTONS
# =============================================================================

def render_navigation_buttons(
    current_step: int,
    max_step: int = 4,
    can_proceed: bool = True,
    next_label: str = "Suivant →",
    back_label: str = "← Retour",
    execute_label: str = "🚀 Exécuter l'upsert",
) -> tuple[bool, bool]:
    """
    Render navigation buttons for wizard steps.

    Args:
        current_step: Current step number (1-4)
        max_step: Maximum step number
        can_proceed: Whether the next button should be enabled
        next_label: Label for next button
        back_label: Label for back button
        execute_label: Label for final execute button

    Returns:
        Tuple of (go_back, go_next) booleans
    """
    col1, col2, col3 = st.columns([1, 2, 1])

    go_back = False
    go_next = False

    with col1:
        if current_step > 1:
            if st.button(back_label, key="agg_nav_back"):
                go_back = True

    with col3:
        if current_step < max_step:
            if st.button(
                next_label,
                type="primary",
                disabled=not can_proceed,
                key="agg_nav_next",
            ):
                go_next = True
        elif current_step == max_step:
            if st.button(
                execute_label,
                type="primary",
                disabled=not can_proceed,
                key="agg_nav_execute",
            ):
                go_next = True

    return go_back, go_next
