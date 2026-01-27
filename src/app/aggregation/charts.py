"""
Charts components for the Aggregation mode.

Provides interactive Plotly charts for visualizing aggregated data by advisor.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# =============================================================================
# COLOR PALETTE - Professional gradient palette
# =============================================================================

CHART_COLORS = {
    # Metric-specific colors
    "PA Vendues": "#3B82F6",      # Blue
    "Collected": "#10B981",        # Emerald green
    "AE CA": "#F59E0B",            # Amber/Orange
    # UI colors
    "primary": "#6366F1",          # Indigo
    "secondary": "#8B5CF6",        # Violet
    "accent": "#EC4899",           # Pink
    "success": "#10B981",          # Green
    "warning": "#F59E0B",          # Amber
    "danger": "#EF4444",           # Red
}

# Gradient color scales for heatmaps and continuous data
COLOR_SCALES = {
    # Darker, more saturated colors for better visibility
    "performance": [[0, "#DC2626"], [0.5, "#F59E0B"], [1, "#059669"]],  # Dark Red -> Amber -> Dark Green
    "blue": [[0, "#DBEAFE"], [1, "#1E40AF"]],
    "green": [[0, "#D1FAE5"], [1, "#047857"]],
    "purple": [[0, "#EDE9FE"], [1, "#5B21B6"]],
}

# Default chart height
CHART_HEIGHT = 400


# =============================================================================
# KPI CARDS
# =============================================================================

def render_kpi_cards(
    df: pd.DataFrame,
    metric_columns: list[str],
) -> None:
    """
    Render styled KPI cards showing totals and key statistics.

    Args:
        df: DataFrame with metric columns
        metric_columns: List of metric column names
    """
    if df.empty or not metric_columns:
        return

    # Calculate stats for each metric
    stats = []
    for col in metric_columns:
        if col in df.columns:
            total = df[col].sum()
            avg = df[col].mean()
            top_advisor = df.loc[df[col].idxmax(), "Conseiller"] if not df[col].isna().all() else "N/A"
            top_value = df[col].max()
            stats.append({
                "name": col,
                "total": total,
                "avg": avg,
                "top_advisor": top_advisor,
                "top_value": top_value,
                "color": CHART_COLORS.get(col, CHART_COLORS["primary"]),
            })

    # Render cards
    cols = st.columns(len(stats))
    for idx, stat in enumerate(stats):
        with cols[idx]:
            # Card with colored border
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {stat['color']}15, {stat['color']}05);
                border-left: 4px solid {stat['color']};
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 8px;
            ">
                <div style="font-size: 12px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;">
                    {stat['name']}
                </div>
                <div style="font-size: 28px; font-weight: 700; color: {stat['color']}; margin: 4px 0;">
                    {stat['total']:,.0f}
                </div>
                <div style="font-size: 11px; color: #9CA3AF;">
                    Moyenne: {stat['avg']:,.0f} | Top: {stat['top_advisor'][:15]}...
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# BAR CHARTS
# =============================================================================

def render_advisor_bar_chart(
    df: pd.DataFrame,
    value_column: str,
    title: str | None = None,
    color: str | None = None,
    horizontal: bool = True,
) -> None:
    """
    Render a bar chart showing values by advisor.

    Args:
        df: DataFrame with 'Conseiller' and value columns
        value_column: Name of the column to plot
        title: Chart title (defaults to column name)
        color: Bar color (defaults to column-specific color)
        horizontal: If True, render horizontal bars (better for many advisors)
    """
    if df.empty or value_column not in df.columns:
        st.warning(f"Aucune donnée pour {value_column}")
        return

    # Sort by value descending
    plot_df = df.copy()
    plot_df = plot_df.sort_values(value_column, ascending=horizontal)

    # Get color
    bar_color = color or CHART_COLORS.get(value_column, CHART_COLORS["primary"])

    # Create chart
    if horizontal:
        fig = px.bar(
            plot_df,
            x=value_column,
            y="Conseiller",
            orientation="h",
            title=title or f"{value_column} par conseiller",
            color_discrete_sequence=[bar_color],
        )
    else:
        fig = px.bar(
            plot_df,
            x="Conseiller",
            y=value_column,
            title=title or f"{value_column} par conseiller",
            color_discrete_sequence=[bar_color],
        )

    # Style the chart
    fig.update_layout(
        height=max(CHART_HEIGHT, len(plot_df) * 25) if horizontal else CHART_HEIGHT,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title=value_column if horizontal else "Conseiller",
        yaxis_title="Conseiller" if horizontal else value_column,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )

    # Enhanced tooltips
    fig.update_traces(
        texttemplate="%{x:,.0f}" if horizontal else "%{y:,.0f}",
        textposition="outside",
        textfont_size=10,
        hovertemplate="<b>%{y}</b><br>" + f"{value_column}: " + "%{x:,.0f}<extra></extra>" if horizontal else
                      "<b>%{x}</b><br>" + f"{value_column}: " + "%{y:,.0f}<extra></extra>",
    )

    st.plotly_chart(fig, width="stretch")


def render_stacked_comparison_chart(
    df: pd.DataFrame,
    value_columns: list[str],
    title: str = "Comparaison par conseiller",
) -> None:
    """
    Render a grouped bar chart comparing all metrics per advisor.

    Args:
        df: DataFrame with 'Conseiller' and value columns
        value_columns: List of column names to compare
        title: Chart title
    """
    if df.empty:
        st.warning("Aucune donnée à afficher")
        return

    # Filter to existing columns
    existing_cols = [col for col in value_columns if col in df.columns]
    if not existing_cols:
        st.warning("Aucune colonne de valeur trouvée")
        return

    # Sort by total value (sum of all columns)
    plot_df = df.copy()
    plot_df["_total"] = plot_df[existing_cols].sum(axis=1)
    plot_df = plot_df.sort_values("_total", ascending=True)
    plot_df = plot_df.drop(columns=["_total"])

    # Create grouped bar chart
    fig = go.Figure()

    for col in existing_cols:
        color = CHART_COLORS.get(col, None)
        fig.add_trace(go.Bar(
            name=col,
            y=plot_df["Conseiller"],
            x=plot_df[col],
            orientation="h",
            marker_color=color,
            text=plot_df[col].apply(lambda x: f"{x:,.0f}"),
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>" + f"{col}: " + "%{x:,.0f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        barmode="group",
        height=max(500, len(plot_df) * 45),
        margin=dict(l=20, r=40, t=60, b=20),
        xaxis_title="Montant",
        yaxis_title="",
        xaxis=dict(tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=13)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
    )

    # Larger text on bars
    fig.update_traces(textfont_size=12)

    st.plotly_chart(fig, width="stretch")


def render_top_advisors_chart(
    df: pd.DataFrame,
    value_column: str,
    top_n: int = 5,
    title: str | None = None,
) -> None:
    """
    Render a chart showing the top N advisors for a specific metric.

    Args:
        df: DataFrame with 'Conseiller' and value columns
        value_column: Column to rank by
        top_n: Number of top advisors to show
        title: Chart title
    """
    if df.empty or value_column not in df.columns:
        st.warning(f"Aucune donnée pour {value_column}")
        return

    # Get top N
    top_df = df.nlargest(top_n, value_column)[["Conseiller", value_column]]

    # Create chart with gradient colors
    color = CHART_COLORS.get(value_column, CHART_COLORS["primary"])

    fig = px.bar(
        top_df,
        x=value_column,
        y="Conseiller",
        orientation="h",
        title=title or f"Top {top_n} - {value_column}",
        color=value_column,
        color_continuous_scale=[[0, "#f0f0f0"], [1, color]],
    )

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=50, t=50, b=20),
        xaxis_title="",
        yaxis_title="",
        showlegend=False,
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )

    # Add value labels with better formatting
    fig.update_traces(
        texttemplate="%{x:,.0f}",
        textposition="outside",
        textfont_size=12,
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>" + f"{value_column}: " + "%{x:,.0f}<extra></extra>",
    )

    st.plotly_chart(fig, width="stretch")


# =============================================================================
# PARETO CHART (80/20 Analysis)
# =============================================================================

def render_pareto_chart(
    df: pd.DataFrame,
    value_column: str,
    title: str | None = None,
) -> None:
    """
    Render a Pareto chart (80/20 analysis) for a metric.

    Shows bars for individual values and a cumulative line to identify
    which advisors contribute to 80% of the total.

    Args:
        df: DataFrame with 'Conseiller' and value columns
        value_column: Column to analyze
        title: Chart title
    """
    if df.empty or value_column not in df.columns:
        st.warning(f"Aucune donnée pour {value_column}")
        return

    # Sort by value descending
    plot_df = df[["Conseiller", value_column]].copy()
    plot_df = plot_df.sort_values(value_column, ascending=False).reset_index(drop=True)

    # Calculate cumulative percentage
    total = plot_df[value_column].sum()
    plot_df["cumulative"] = plot_df[value_column].cumsum()
    plot_df["cumulative_pct"] = (plot_df["cumulative"] / total * 100) if total > 0 else 0

    # Find 80% threshold
    threshold_idx = (plot_df["cumulative_pct"] >= 80).idxmax() if (plot_df["cumulative_pct"] >= 80).any() else len(plot_df) - 1

    # Color bars: those contributing to 80% in primary color, others in gray
    colors = [CHART_COLORS.get(value_column, CHART_COLORS["primary"]) if i <= threshold_idx else "#D1D5DB"
              for i in range(len(plot_df))]

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar chart
    fig.add_trace(
        go.Bar(
            x=plot_df["Conseiller"],
            y=plot_df[value_column],
            name=value_column,
            marker_color=colors,
            text=plot_df[value_column].apply(lambda x: f"{x:,.0f}"),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>" + f"{value_column}: " + "%{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Cumulative line
    fig.add_trace(
        go.Scatter(
            x=plot_df["Conseiller"],
            y=plot_df["cumulative_pct"],
            name="Cumul %",
            mode="lines+markers",
            line=dict(color=CHART_COLORS["accent"], width=2),
            marker=dict(size=6),
            hovertemplate="<b>%{x}</b><br>Cumul: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    # Add 80% threshold line
    fig.add_hline(
        y=80, line_dash="dash", line_color=CHART_COLORS["danger"],
        annotation_text="80%", secondary_y=True,
    )

    fig.update_layout(
        title=title or f"Analyse Pareto - {value_column}",
        height=500,
        margin=dict(l=20, r=20, t=60, b=120),
        xaxis_tickangle=-45,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )

    fig.update_yaxes(title_text=value_column, secondary_y=False)
    fig.update_yaxes(title_text="Cumul (%)", secondary_y=True, range=[0, 105])

    st.plotly_chart(fig, width="stretch")

    # Show insight
    advisors_for_80 = threshold_idx + 1
    total_advisors = len(plot_df)
    pct_advisors = (advisors_for_80 / total_advisors * 100) if total_advisors > 0 else 0
    st.caption(f"**{advisors_for_80} conseiller(s)** ({pct_advisors:.0f}%) contribuent à 80% du total de {value_column}")


# =============================================================================
# HEATMAP
# =============================================================================

def render_performance_heatmap(
    df: pd.DataFrame,
    value_columns: list[str],
    title: str = "Classement par métrique",
) -> None:
    """
    Render a heatmap showing absolute ranking across metrics.

    Each cell shows the advisor's position (1 = best) for that metric.

    Args:
        df: DataFrame with 'Conseiller' and value columns
        value_columns: List of metric columns to include
        title: Chart title
    """
    if df.empty:
        st.warning("Aucune donnée à afficher")
        return

    # Filter to existing columns
    existing_cols = [col for col in value_columns if col in df.columns]
    if not existing_cols:
        st.warning("Aucune colonne de valeur trouvée")
        return

    # Create ranking matrix (1 = best, N = worst)
    plot_df = df.copy()
    total_advisors = len(plot_df)

    for col in existing_cols:
        # Calculate absolute rank (1 = highest value = best)
        plot_df[f"{col}_rank"] = plot_df[col].rank(ascending=False, method="min").astype(int)

    # Sort by average rank (lower is better)
    rank_cols = [f"{col}_rank" for col in existing_cols]
    plot_df["_avg_rank"] = plot_df[rank_cols].mean(axis=1)
    plot_df = plot_df.sort_values("_avg_rank", ascending=True)

    # Extract matrix for heatmap (invert for color scale: low rank = high value = green)
    z_data = plot_df[rank_cols].values
    # Normalize for color scale (1 -> 100, N -> 0) so best ranks get green
    z_data_normalized = ((total_advisors - z_data + 1) / total_advisors) * 100
    y_labels = plot_df["Conseiller"].tolist()
    x_labels = existing_cols

    # Create heatmap with normalized colors but display actual ranks
    fig = go.Figure(data=go.Heatmap(
        z=z_data_normalized,
        x=x_labels,
        y=y_labels,
        colorscale=COLOR_SCALES["performance"],
        showscale=True,
        colorbar=dict(
            title="Position",
            tickvals=[10, 50, 90],
            ticktext=[f"#{total_advisors}", f"#{total_advisors//2}", "#1"],
        ),
        hovertemplate="<b>%{y}</b><br>%{x}: Position %{customdata}<extra></extra>",
        customdata=z_data,
    ))

    # Add text annotations with actual rank positions
    annotations = []
    for i, row in enumerate(z_data):
        for j, val in enumerate(row):
            # Determine text color based on normalized value
            norm_val = z_data_normalized[i][j]
            annotations.append(dict(
                x=x_labels[j],
                y=y_labels[i],
                text=f"#{int(val)}",
                showarrow=False,
                font=dict(
                    color="white",
                    size=11,
                    family="Inter, sans-serif",
                ),
            ))

    fig.update_layout(
        title=title,
        annotations=annotations,
        height=max(400, len(y_labels) * 35),
        margin=dict(l=150, r=20, t=50, b=50),
        xaxis=dict(side="top"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )

    st.plotly_chart(fig, width="stretch")


# =============================================================================
# CONTRIBUTION CHART
# =============================================================================

def render_contribution_chart(
    df: pd.DataFrame,
    value_column: str,
    title: str | None = None,
) -> None:
    """
    Render a horizontal bar chart showing each advisor's contribution percentage.

    Args:
        df: DataFrame with 'Conseiller' and value columns
        value_column: Column to analyze
        title: Chart title
    """
    if df.empty or value_column not in df.columns:
        st.warning(f"Aucune donnée pour {value_column}")
        return

    # Calculate contributions
    plot_df = df[["Conseiller", value_column]].copy()
    total = plot_df[value_column].sum()
    plot_df["contribution_pct"] = (plot_df[value_column] / total * 100) if total > 0 else 0
    plot_df = plot_df.sort_values("contribution_pct", ascending=True)

    # Get color
    bar_color = CHART_COLORS.get(value_column, CHART_COLORS["primary"])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=plot_df["Conseiller"],
        x=plot_df["contribution_pct"],
        orientation="h",
        marker=dict(
            color=plot_df["contribution_pct"],
            colorscale=[[0, "#E5E7EB"], [1, bar_color]],
        ),
        text=plot_df.apply(lambda r: f"{r['contribution_pct']:.1f}% ({r[value_column]:,.0f})", axis=1),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title=title or f"Contribution au {value_column}",
        height=max(CHART_HEIGHT, len(plot_df) * 25),
        margin=dict(l=20, r=100, t=50, b=20),
        xaxis_title="Contribution (%)",
        yaxis_title="",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )

    fig.update_traces(cliponaxis=False)

    st.plotly_chart(fig, width="stretch")


# =============================================================================
# RADAR CHART
# =============================================================================

def render_advisor_radar_chart(
    df: pd.DataFrame,
    advisor_name: str,
    value_columns: list[str],
    title: str | None = None,
) -> None:
    """
    Render a radar chart showing an advisor's performance across all metrics.

    Args:
        df: DataFrame with 'Conseiller' and value columns
        advisor_name: Name of the advisor to highlight
        value_columns: List of metric columns
        title: Chart title
    """
    if df.empty:
        st.warning("Aucune donnée à afficher")
        return

    # Filter to existing columns
    existing_cols = [col for col in value_columns if col in df.columns]
    if not existing_cols:
        return

    # Get advisor data
    advisor_data = df[df["Conseiller"] == advisor_name]
    if advisor_data.empty:
        st.warning(f"Conseiller '{advisor_name}' non trouvé")
        return

    # Normalize values (0-100 scale based on max in each column)
    normalized_values = []
    actual_values = []
    for col in existing_cols:
        max_val = df[col].max()
        val = advisor_data[col].iloc[0]
        actual_values.append(val)
        normalized_values.append((val / max_val * 100) if max_val > 0 else 0)

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=normalized_values + [normalized_values[0]],  # Close the polygon
        theta=existing_cols + [existing_cols[0]],
        fill="toself",
        name=advisor_name,
        line_color=CHART_COLORS["primary"],
        fillcolor=f"rgba(99, 102, 241, 0.3)",
        hovertemplate="<b>%{theta}</b><br>Valeur: %{text:,.0f}<br>Score: %{r:.0f}%<extra></extra>",
        text=actual_values + [actual_values[0]],
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix="%",
            ),
        ),
        title=title or f"Profil de {advisor_name}",
        height=350,
        margin=dict(l=60, r=60, t=60, b=40),
        showlegend=False,
        font=dict(family="Inter, sans-serif"),
    )

    st.plotly_chart(fig, width="stretch")


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_charts_tab(
    combined_df: pd.DataFrame,
    period_name: str,
) -> None:
    """
    Render the complete charts tab content.

    Args:
        combined_df: Combined aggregated DataFrame
        period_name: Name of the selected period for display
    """
    if combined_df is None or combined_df.empty:
        st.warning("Aucune donnée à visualiser pour cette période.")
        return

    # Get numeric columns (metrics)
    metric_columns = [col for col in combined_df.columns if col != "Conseiller"]

    if not metric_columns:
        st.warning("Aucune métrique à visualiser.")
        return

    st.markdown(f"### Visualisations pour **{period_name}**")

    # =========================================================================
    # KPI Cards Section
    # =========================================================================
    st.markdown("#### Indicateurs clés")
    render_kpi_cards(combined_df, metric_columns)

    st.markdown("---")

    # =========================================================================
    # Top Performers Section (full width for better readability)
    # =========================================================================
    st.markdown("#### Top 3 par métrique")

    if len(metric_columns) >= 1:
        top_cols = st.columns(min(len(metric_columns), 3))
        for idx, col in enumerate(metric_columns[:3]):
            with top_cols[idx]:
                render_top_advisors_chart(
                    combined_df,
                    col,
                    top_n=3,
                    title=f"{col}",
                )

    st.markdown("---")

    # =========================================================================
    # Performance Comparison Section (full width for better readability)
    # =========================================================================
    st.markdown("#### Performance par conseiller")

    render_stacked_comparison_chart(
        combined_df,
        metric_columns,
        title="Comparaison des métriques",
    )

    st.markdown("---")

    # =========================================================================
    # Heatmap Section
    # =========================================================================
    st.markdown("#### Classement par métrique")
    st.caption("Position de chaque conseiller pour chaque métrique (vert = meilleur, rouge = moins bon)")

    render_performance_heatmap(
        combined_df,
        metric_columns,
        title="",
    )

    st.markdown("---")

    # =========================================================================
    # Pareto Analysis Section
    # =========================================================================
    st.markdown("#### Analyse Pareto (80/20)")
    st.caption("Identifiez quels conseillers contribuent à 80% des résultats")

    pareto_tabs = st.tabs([f"{col}" for col in metric_columns])
    for idx, col in enumerate(metric_columns):
        with pareto_tabs[idx]:
            render_pareto_chart(combined_df, col)

    st.markdown("---")

    # =========================================================================
    # Contribution Analysis Section
    # =========================================================================
    st.markdown("#### Contribution individuelle")

    contrib_tabs = st.tabs([f"{col}" for col in metric_columns])
    for idx, col in enumerate(metric_columns):
        with contrib_tabs[idx]:
            render_contribution_chart(
                combined_df,
                col,
                title=f"Part de chaque conseiller dans {col}",
            )

    st.markdown("---")

    # =========================================================================
    # Individual Advisor Analysis
    # =========================================================================
    st.markdown("#### Analyse individuelle")

    advisors = sorted(combined_df["Conseiller"].unique().tolist())

    selected_advisor = st.selectbox(
        "Sélectionner un conseiller",
        options=advisors,
        key="chart_advisor_selector",
    )

    if selected_advisor:
        col1, col2 = st.columns([1, 1])

        with col1:
            render_advisor_radar_chart(
                combined_df,
                selected_advisor,
                metric_columns,
                title=f"Profil de {selected_advisor}",
            )

        with col2:
            # Show advisor stats
            advisor_data = combined_df[combined_df["Conseiller"] == selected_advisor]
            if not advisor_data.empty:
                st.markdown(f"**Statistiques de {selected_advisor}**")

                for col in metric_columns:
                    val = advisor_data[col].iloc[0]
                    total = combined_df[col].sum()
                    pct = (val / total * 100) if total > 0 else 0
                    rank = (combined_df[col] > val).sum() + 1
                    color = CHART_COLORS.get(col, CHART_COLORS["primary"])

                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {color}10, transparent);
                        border-left: 3px solid {color};
                        padding: 12px;
                        margin: 8px 0;
                        border-radius: 4px;
                    ">
                        <div style="font-weight: 600; color: {color};">{col}</div>
                        <div style="font-size: 20px; font-weight: 700;">{val:,.2f}</div>
                        <div style="font-size: 12px; color: #6B7280;">
                            {pct:.1f}% du total | Rang #{rank} sur {len(combined_df)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
