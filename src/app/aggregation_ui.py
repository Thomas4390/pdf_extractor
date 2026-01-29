"""
UI components for the Data Aggregation mode.

Provides Streamlit components for:
- Source board selection
- Date period selection
- Data preview
- Aggregation execution
"""

from typing import Optional

import pandas as pd
import streamlit as st

from ..utils.aggregator import (
    DatePeriod,
    SOURCE_BOARDS,
    SourceBoardConfig,
    get_group_name_for_period,
    get_period_date_range,
)


# =============================================================================
# STEPPER UI
# =============================================================================

def render_aggregation_stepper(current_step: int) -> None:
    """
    Render a visual progress stepper for the 4-step wizard.

    Uses the same CSS-based design as the extraction stepper.

    Args:
        current_step: Current step (1-4)
    """
    steps = [
        ("1", "Sources", "📋"),
        ("2", "Période", "📅"),
        ("3", "Aperçu", "🔍"),
        ("4", "Exécution", "🚀"),
    ]

    cols = st.columns(len(steps))
    for idx, (num, label, icon) in enumerate(steps):
        step_num = idx + 1
        with cols[idx]:
            is_current = step_num == current_step
            is_completed = step_num < current_step
            is_future = step_num > current_step

            # Determine CSS class
            if is_current:
                css_class = "current"
            elif is_completed:
                css_class = "completed"
            else:
                css_class = "future"

            # Render step visual
            display_icon = "✅" if is_completed else icon
            st.markdown(f"""
            <div class="stepper-step {css_class}">
                <div class="step-icon">{display_icon}</div>
                <div class="step-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

            # Add clickable button for completed stages
            if is_completed:
                if st.button(f"← Retour", key=f"agg_stepper_nav_{step_num}", width="stretch"):
                    st.session_state.agg_step = step_num
                    st.rerun()

    st.markdown("---")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _filter_boards(boards: list[dict], search_query: str) -> list[dict]:
    """
    Filter boards by search query (case-insensitive).

    Args:
        boards: List of board dicts with 'name' and 'id'
        search_query: Search string to filter by

    Returns:
        Filtered list of boards
    """
    if not search_query or not search_query.strip():
        return boards

    query = search_query.lower().strip()
    # Split query into words for multi-word search
    words = query.split()

    filtered = []
    for board in boards:
        board_name_lower = board["name"].lower()
        # Board matches if ALL words are found in the name
        if all(word in board_name_lower for word in words):
            filtered.append(board)

    return filtered


# =============================================================================
# STEP 1: SOURCE SELECTION
# =============================================================================

def render_source_selection(
    boards: list[dict],
    selected_sources: dict[str, Optional[int]],
) -> dict[str, Optional[int]]:
    """
    Render source board selection UI with search filtering.

    Args:
        boards: List of available boards from Monday.com
        selected_sources: Current selection {source_key: board_id}

    Returns:
        Updated selection dict
    """
    st.subheader("📋 Sélection des sources")
    st.markdown(
        "Sélectionnez les boards sources et la colonne à agréger pour chaque source."
    )

    # Search input for filtering boards
    search_query = st.text_input(
        "🔍 Rechercher un board",
        placeholder="Tapez pour filtrer les boards...",
        key="agg_source_search",
        help="Filtrez les boards par nom (ex: 'paiement' ou 'vente production')"
    )

    # Filter boards based on search
    filtered_boards = _filter_boards(boards, search_query)

    # Show filter status
    if search_query:
        st.caption(f"📋 {len(filtered_boards)} board(s) trouvé(s) sur {len(boards)}")

    # Build board options from filtered list
    board_options = {b["name"]: int(b["id"]) for b in filtered_boards}
    # Keep full options for preserving existing selections
    all_board_options = {b["name"]: int(b["id"]) for b in boards}
    board_names = ["-- Sélectionner --"] + list(board_options.keys())

    updated_selection = {}

    for source_key, config in SOURCE_BOARDS.items():
        col1, col2 = st.columns([3, 1])

        with col1:
            # Checkbox + dropdown
            enabled = st.checkbox(
                config.display_name,
                value=source_key in selected_sources and selected_sources[source_key] is not None,
                key=f"agg_source_enabled_{source_key}",
            )

            if enabled:
                # Find current selection (from all boards, not just filtered)
                current_board_id = selected_sources.get(source_key)
                current_board_name = None
                if current_board_id:
                    for name, bid in all_board_options.items():
                        if bid == current_board_id:
                            current_board_name = name
                            break

                # Build display options - include current selection even if not in filter
                display_names = list(board_names)  # Copy
                if current_board_name and current_board_name not in display_names:
                    display_names.insert(1, current_board_name)  # After "-- Sélectionner --"

                default_idx = 0
                if current_board_name and current_board_name in display_names:
                    default_idx = display_names.index(current_board_name)

                selected_name = st.selectbox(
                    "Board source",
                    options=display_names,
                    index=default_idx,
                    key=f"agg_board_select_{source_key}",
                    label_visibility="collapsed",
                )

                if selected_name != "-- Sélectionner --":
                    # Get ID from all_board_options to support selections outside filter
                    updated_selection[source_key] = all_board_options.get(selected_name)

        with col2:
            st.markdown(
                f"<div style='padding-top:35px;color:#666;'>"
                f"→ <strong>{config.aggregate_column}</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )

    return updated_selection


def render_target_board_selection(
    boards: list[dict],
    current_selection: Optional[int],
) -> Optional[int]:
    """
    Render target board selection with search filtering.

    Args:
        boards: List of available boards
        current_selection: Currently selected board ID

    Returns:
        Selected board ID or None
    """
    st.subheader("🎯 Board cible")

    # Search input for filtering target board
    search_query = st.text_input(
        "🔍 Rechercher le board cible",
        placeholder="Tapez pour filtrer...",
        key="agg_target_search",
        help="Filtrez les boards par nom"
    )

    # Filter boards based on search
    filtered_boards = _filter_boards(boards, search_query)

    # Show filter status
    if search_query:
        st.caption(f"📋 {len(filtered_boards)} board(s) trouvé(s) sur {len(boards)}")

    # Build options
    board_options = {b["name"]: int(b["id"]) for b in filtered_boards}
    all_board_options = {b["name"]: int(b["id"]) for b in boards}
    board_names = ["-- Sélectionner --"] + list(board_options.keys())

    # Find current selection (from all boards)
    current_name = None
    if current_selection:
        for name, bid in all_board_options.items():
            if bid == current_selection:
                current_name = name
                break

    # Include current selection even if not in filter
    display_names = list(board_names)
    if current_name and current_name not in display_names:
        display_names.insert(1, current_name)

    default_idx = 0
    if current_name and current_name in display_names:
        default_idx = display_names.index(current_name)

    selected_name = st.selectbox(
        "Board de destination",
        options=display_names,
        index=default_idx,
        key="agg_target_board",
    )

    if selected_name != "-- Sélectionner --":
        return all_board_options.get(selected_name)
    return None


# =============================================================================
# STEP 2: PERIOD SELECTION
# =============================================================================

def render_period_selection(
    current_period: DatePeriod,
    use_custom_group: bool = False,
    custom_group_name: str = "",
) -> tuple[DatePeriod, bool, str]:
    """
    Render date period selection UI with optional custom group name.

    Args:
        current_period: Currently selected period
        use_custom_group: Whether to use custom group name
        custom_group_name: Current custom group name

    Returns:
        Tuple of (selected_period, use_custom_group, custom_group_name)
    """
    # Period selection with horizontal segmented buttons
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📅 Période de filtrage</div>
        <p class="section-description">Sélectionnez la période pour filtrer les données à agréger.</p>
    </div>
    """, unsafe_allow_html=True)

    # Period options as pill buttons
    period_options = list(DatePeriod)
    period_cols = st.columns(len(period_options))

    selected_period = current_period
    for idx, period in enumerate(period_options):
        with period_cols[idx]:
            is_selected = period == current_period
            btn_type = "primary" if is_selected else "secondary"
            if st.button(
                period.display_name,
                key=f"period_btn_{period.value}",
                type=btn_type,
                width="stretch",
            ):
                selected_period = period

    # Show date range preview
    start_date, end_date = get_period_date_range(selected_period)
    auto_group_name = get_group_name_for_period(selected_period)

    st.markdown("---")

    # Date range display
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-icon">📆</div>
            <div class="info-content">
                <div class="info-label">Plage de dates</div>
                <div class="info-value">{start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-icon">📁</div>
            <div class="info-content">
                <div class="info-label">Groupe auto-généré</div>
                <div class="info-value">{auto_group_name}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Target group selection
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📁 Groupe de destination</div>
        <p class="section-description">Le groupe Monday.com où les données seront insérées/mises à jour.</p>
    </div>
    """, unsafe_allow_html=True)

    # Toggle between auto and manual group
    group_mode = st.radio(
        "Mode de sélection du groupe",
        options=["auto", "manual"],
        format_func=lambda x: "🔄 Automatique (basé sur la période)" if x == "auto" else "✏️ Manuel (nom personnalisé)",
        index=1 if use_custom_group else 0,
        key="agg_group_mode",
        horizontal=True,
    )

    use_custom = group_mode == "manual"
    final_group_name = custom_group_name

    if use_custom:
        # Manual group input
        final_group_name = st.text_input(
            "Nom du groupe cible",
            value=custom_group_name if custom_group_name else auto_group_name,
            placeholder="Ex: Décembre 2025, Q4 2025...",
            key="agg_custom_group_input",
            help="Entrez le nom exact du groupe tel qu'il apparaît (ou apparaîtra) dans Monday.com"
        )

        if final_group_name:
            st.success(f"✅ Groupe cible : **{final_group_name}**")
        else:
            st.warning("⚠️ Veuillez entrer un nom de groupe")
    else:
        final_group_name = auto_group_name
        st.info(f"📁 Groupe cible automatique : **{auto_group_name}**")

    return selected_period, use_custom, final_group_name


# =============================================================================
# STEP 3: DATA PREVIEW
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
    categorical_cols = ["Conseiller", "Profitable"]
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
    st.markdown(f"""
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
    categorical_cols = ["Conseiller", "Profitable"]
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

    for col in numeric_cols:
        column_config[col] = st.column_config.NumberColumn(
            col,
            help=f"Valeur agrégée pour {col}",
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


def render_upload_summary(
    combined_df: pd.DataFrame,
    target_board_name: str,
    group_name: str,
    sources_count: int,
) -> None:
    """
    Render a summary of what will be uploaded.

    Args:
        combined_df: DataFrame to be uploaded
        target_board_name: Target board name
        group_name: Target group name
        sources_count: Number of data sources
    """
    if combined_df.empty:
        return

    numeric_cols = [col for col in combined_df.columns if col != "Conseiller"]

    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">📋 Récapitulatif de l'upload</div>
    </div>
    """, unsafe_allow_html=True)

    # Main stats
    st.markdown(f"""
    <div class="summary-stats">
        <div class="summary-stat">
            <div class="stat-value">{len(combined_df)}</div>
            <div class="stat-label">Conseillers</div>
        </div>
        <div class="summary-stat">
            <div class="stat-value">{sources_count}</div>
            <div class="stat-label">Sources</div>
        </div>
        <div class="summary-stat">
            <div class="stat-value">{len(numeric_cols)}</div>
            <div class="stat-label">Colonnes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Column totals
    st.markdown("**Valeurs totales par colonne :**")
    for col in numeric_cols:
        total = combined_df[col].sum()
        st.markdown(f"- **{col}** : {total:,.2f}")


# =============================================================================
# STEP 4: EXECUTION
# =============================================================================

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
