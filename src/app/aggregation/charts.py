"""
Charts components for the Aggregation mode.

Provides interactive Plotly charts for visualizing aggregated data by advisor.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# Color palette for consistent styling
CHART_COLORS = {
    "PA Vendues": "#1f77b4",      # Blue
    "Collected": "#2ca02c",        # Green
    "AE CA": "#ff7f0e",            # Orange
    "primary": "#667eea",          # Purple gradient start
    "secondary": "#764ba2",        # Purple gradient end
}

# Default chart height
CHART_HEIGHT = 400


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
            title=title or f"📊 {value_column} par conseiller",
            color_discrete_sequence=[bar_color],
        )
    else:
        fig = px.bar(
            plot_df,
            x="Conseiller",
            y=value_column,
            title=title or f"📊 {value_column} par conseiller",
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
    )

    # Add value labels
    fig.update_traces(
        texttemplate="%{x:,.0f}" if horizontal else "%{y:,.0f}",
        textposition="outside",
        textfont_size=10,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_stacked_comparison_chart(
    df: pd.DataFrame,
    value_columns: list[str],
    title: str = "📊 Comparaison par conseiller",
) -> None:
    """
    Render a grouped/stacked bar chart comparing all metrics per advisor.

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
        ))

    fig.update_layout(
        title=title,
        barmode="group",
        height=max(CHART_HEIGHT, len(plot_df) * 40),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Montant",
        yaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_totals_pie_chart(
    df: pd.DataFrame,
    value_columns: list[str],
    title: str = "📊 Répartition des totaux",
) -> None:
    """
    Render a pie chart showing the distribution of totals across metrics.

    Args:
        df: DataFrame with value columns
        value_columns: List of column names to include
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

    # Calculate totals
    totals = {col: df[col].sum() for col in existing_cols}

    # Create pie chart data
    labels = list(totals.keys())
    values = list(totals.values())
    colors = [CHART_COLORS.get(col, CHART_COLORS["primary"]) for col in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.4,  # Donut chart
        textinfo="label+percent",
        textposition="outside",
        hovertemplate="<b>%{label}</b><br>Montant: %{value:,.0f}<br>Part: %{percent}<extra></extra>",
    )])

    fig.update_layout(
        title=title,
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)


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
        title=title or f"🏆 Top {top_n} - {value_column}",
        color=value_column,
        color_continuous_scale=[[0, "#f0f0f0"], [1, color]],
    )

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title=value_column,
        yaxis_title="",
        showlegend=False,
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # Add value labels
    fig.update_traces(
        texttemplate="%{x:,.0f}",
        textposition="outside",
        textfont_size=11,
    )

    st.plotly_chart(fig, use_container_width=True)


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
        fillcolor=f"rgba(102, 126, 234, 0.3)",
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
        title=title or f"📊 Profil de {advisor_name}",
        height=350,
        margin=dict(l=60, r=60, t=60, b=40),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


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

    st.markdown(f"### 📈 Visualisations pour **{period_name}**")

    # Summary section with totals pie chart and top performers
    st.markdown("#### 🎯 Vue d'ensemble")

    col1, col2 = st.columns([1, 2])

    with col1:
        render_totals_pie_chart(
            combined_df,
            metric_columns,
            title="Répartition des totaux",
        )

    with col2:
        # Top performers for each metric in columns
        if len(metric_columns) >= 1:
            sub_cols = st.columns(min(len(metric_columns), 3))
            for idx, col in enumerate(metric_columns[:3]):
                with sub_cols[idx]:
                    render_top_advisors_chart(
                        combined_df,
                        col,
                        top_n=3,
                        title=f"🏆 Top 3 - {col}",
                    )

    st.markdown("---")

    # Comparison chart
    st.markdown("#### 📊 Comparaison globale")
    render_stacked_comparison_chart(
        combined_df,
        metric_columns,
        title="Performance par conseiller",
    )

    st.markdown("---")

    # Individual metric charts
    st.markdown("#### 📈 Détail par métrique")

    # Create tabs for each metric
    metric_tabs = st.tabs([f"📊 {col}" for col in metric_columns])

    for idx, col in enumerate(metric_columns):
        with metric_tabs[idx]:
            render_advisor_bar_chart(
                combined_df,
                col,
                title=f"{col} par conseiller",
                horizontal=True,
            )

    st.markdown("---")

    # Individual advisor analysis
    st.markdown("#### 👤 Analyse individuelle")

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
                st.markdown(f"**📋 Statistiques de {selected_advisor}**")

                for col in metric_columns:
                    val = advisor_data[col].iloc[0]
                    total = combined_df[col].sum()
                    pct = (val / total * 100) if total > 0 else 0
                    rank = (combined_df[col] > val).sum() + 1

                    st.markdown(f"""
                    **{col}**
                    - Valeur: **{val:,.2f}**
                    - Part du total: **{pct:.1f}%**
                    - Rang: **#{rank}** sur {len(combined_df)}
                    """)
