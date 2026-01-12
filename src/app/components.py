"""
Reusable UI components for the Commission Pipeline Streamlit application.

Contains:
- Stepper navigation
- Breadcrumb navigation
- Metrics dashboard
- Verification utilities
"""

import streamlit as st
import pandas as pd
from typing import Optional


# =============================================================================
# NAVIGATION COMPONENTS
# =============================================================================

def render_breadcrumb(
    selected_source: Optional[str] = None,
    uploaded_files: Optional[list] = None,
    board_name: Optional[str] = None
) -> None:
    """
    Render breadcrumb navigation showing current context.

    Args:
        selected_source: The selected PDF source type
        uploaded_files: List of uploaded files
        board_name: Name of the selected Monday.com board
    """
    parts = ["Accueil"]

    if selected_source:
        parts.append(selected_source)

    if uploaded_files:
        file_count = len(uploaded_files)
        parts.append(f"{file_count} fichier{'s' if file_count > 1 else ''}")

    if board_name:
        display_name = board_name[:22] + "..." if len(board_name) > 25 else board_name
        parts.append(f'Board "{display_name}"')

    breadcrumb_html = '<div class="breadcrumb">'
    for i, part in enumerate(parts):
        is_active = i == len(parts) - 1
        breadcrumb_html += f'<span class="breadcrumb-item{"" if not is_active else " active"}">{part}</span>'
        if i < len(parts) - 1:
            breadcrumb_html += '<span class="breadcrumb-separator">›</span>'
    breadcrumb_html += '</div>'

    st.markdown(breadcrumb_html, unsafe_allow_html=True)


def render_stepper(current_stage: int) -> Optional[int]:
    """
    Render the clickable progress stepper.

    Args:
        current_stage: Current stage number (1-3)

    Returns:
        New stage number if user clicked a completed stage, None otherwise
    """
    stages = [
        ("1", "Configuration", "folder"),
        ("2", "Prévisualisation", "search"),
        ("3", "Upload", "cloud")
    ]

    icons = {
        "folder": "\U0001F4C1",  # Folder emoji
        "search": "\U0001F50D",  # Magnifying glass
        "cloud": "\u2601\ufe0f"  # Cloud
    }

    new_stage = None
    cols = st.columns(3)

    for i, (num, name, icon_key) in enumerate(stages):
        stage_num = i + 1
        with cols[i]:
            is_current = stage_num == current_stage
            is_completed = stage_num < current_stage

            # Determine CSS class
            if is_current:
                css_class = "current"
            elif is_completed:
                css_class = "completed"
            else:
                css_class = "future"

            # Display icon
            display_icon = "\u2705" if is_completed else icons.get(icon_key, "")

            st.markdown(f"""
            <div class="stepper-step {css_class}">
                <div class="step-icon">{display_icon}</div>
                <div class="step-label">{name}</div>
            </div>
            """, unsafe_allow_html=True)

            # Add clickable button for completed stages
            if is_completed:
                if st.button("Retour", key=f"stepper_nav_{stage_num}", width="stretch"):
                    new_stage = stage_num

    return new_stage


# =============================================================================
# METRICS COMPONENTS
# =============================================================================

def render_metrics_dashboard(
    row_count: int,
    cost: str,
    model: str,
    status: str
) -> None:
    """
    Render the metrics dashboard header.

    Args:
        row_count: Number of extracted rows
        cost: Cost display string (e.g., "$0.0234" or "Cache")
        model: Model name (will be truncated if > 20 chars)
        status: Status display (e.g., "OK" or "3 Ecarts")
    """
    model_display = model[:20] + "..." if len(model) > 20 else model

    st.markdown(f"""
    <div class="metrics-dashboard animate-fade-in">
        <div style="display: flex; justify-content: space-around; align-items: center;">
            <div class="metric-item">
                <div class="metric-value">{row_count}</div>
                <div class="metric-label">Lignes</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{cost}</div>
                <div class="metric-label">Coût</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{model_display}</div>
                <div class="metric-label">Modèle</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{status}</div>
                <div class="metric-label">Statut</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_upload_dashboard(
    item_count: int,
    board_name: str,
    group_count: int,
    file_count: int
) -> None:
    """
    Render the upload stage metrics dashboard.

    Args:
        item_count: Number of items to upload
        board_name: Name of the target board
        group_count: Number of groups
        file_count: Number of source files
    """
    board_display = board_name[:18] + "..." if len(board_name) > 18 else board_name

    st.markdown(f"""
    <div class="metrics-dashboard animate-fade-in">
        <div style="display: flex; justify-content: space-around; align-items: center;">
            <div class="metric-item">
                <div class="metric-value">{item_count}</div>
                <div class="metric-label">Items</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{board_display}</div>
                <div class="metric-label">Board</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{group_count}</div>
                <div class="metric-label">Groupes</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{file_count}</div>
                <div class="metric-label">Fichiers</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_success_box(title: str, message: str) -> None:
    """Render a success message box."""
    st.markdown(f"""
    <div class="success-box animate-fade-in">
        <strong>{title}</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# DATA VERIFICATION COMPONENTS
# =============================================================================

def verify_recu_vs_com(df: pd.DataFrame, tolerance_pct: float = 10.0) -> pd.DataFrame:
    """
    Verify that Reçu is within tolerance range of calculated Com for each row.

    The comparison uses a CALCULATED commission value based on the formula:
        Com_calculée = ROUND((PA * 0.4) * 0.5, 2)

    Args:
        df: DataFrame with 'Reçu' and 'PA' columns
        tolerance_pct: Tolerance percentage (default 10%)

    Returns:
        DataFrame with added columns:
        - 'Com Calculée': The calculated commission for comparison
        - 'Vérification (±X%)': Status flag
    """
    result_df = df.copy()

    # Check if required columns exist
    if 'Reçu' not in result_df.columns or 'PA' not in result_df.columns:
        return result_df

    # Convert to numeric
    recu = pd.to_numeric(result_df['Reçu'], errors='coerce')
    pa = pd.to_numeric(result_df['PA'], errors='coerce')

    # Calculate expected commission: ROUND((PA * 0.4) * 0.5, 2)
    com_calculee = (pa * 0.4 * 0.5).round(2)

    # Add calculated commission column
    result_df['Com Calculée'] = com_calculee

    # Calculate tolerance bounds
    tolerance = tolerance_pct / 100.0
    lower_bound = com_calculee * (1 - tolerance)
    upper_bound = com_calculee * (1 + tolerance)

    # Calculate percentage difference
    pct_diff = ((recu - com_calculee) / com_calculee * 100).round(1)

    # Create verification column
    verification = []
    for i in range(len(result_df)):
        r = recu.iloc[i]
        c = com_calculee.iloc[i]
        diff = pct_diff.iloc[i]

        if pd.isna(r) or pd.isna(c) or c == 0:
            verification.append('-')
        elif r > upper_bound.iloc[i]:
            verification.append(f'✅ +{diff}%')  # Bonus
        elif r < lower_bound.iloc[i]:
            verification.append(f'⚠️ {diff}%')  # Problem
        else:
            verification.append('✓ OK')

    result_df[f'Vérification (±{tolerance_pct:.0f}%)'] = verification

    return result_df


def get_verification_stats(df: pd.DataFrame) -> dict:
    """Get statistics about verification results."""
    verif_cols = [col for col in df.columns if col.startswith('Vérification')]
    if not verif_cols:
        return {'ok': 0, 'bonus': 0, 'ecart': 0, 'na': 0}

    verif = df[verif_cols[0]].astype(str)

    return {
        'ok': verif.str.contains('OK', na=False).sum(),
        'bonus': verif.str.contains('✅', na=False).sum(),
        'ecart': verif.str.contains('⚠️', na=False).sum(),
        'na': (verif == '-').sum()
    }


def reorder_columns_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Réordonne les colonnes pour l'affichage:
    1. Colonnes normales (sans underscore)
    2. Colonnes de calcul/vérification
    3. Colonnes avec underscore (_source_file, _target_group, etc.)
    """
    cols = df.columns.tolist()

    # Séparer les colonnes
    underscore_cols = [c for c in cols if c.startswith('_')]
    calc_verify_cols = [c for c in cols if 'Vérification' in c or c == 'Com Calculée']
    normal_cols = [c for c in cols if c not in underscore_cols and c not in calc_verify_cols]

    # Nouvel ordre
    new_order = normal_cols + calc_verify_cols + underscore_cols

    return df[new_order]
