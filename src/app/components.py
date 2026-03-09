"""
Reusable UI components for the Commission Pipeline Streamlit application.

Contains:
- Metrics dashboard
- Verification utilities
"""

import pandas as pd
import streamlit as st

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

    When _Taux Partage and _Taux Boni columns are present (UV data):
        If _Taux Boni != 0: Com_calculée = ROUND((PA * _Taux Partage) * 0.5 * _Taux Boni, 2)
        If _Taux Boni == 0: Com_calculée = ROUND((PA * _Taux Partage) * 0.5, 2)

    Fallback (no taux columns): Com_calculée = ROUND((PA * 0.4) * 0.5, 2)

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

    # Use extracted taux columns when available, otherwise fallback to hardcoded 0.4
    if '_Taux Partage' in result_df.columns:
        taux_partage = pd.to_numeric(result_df['_Taux Partage'], errors='coerce').fillna(0.4)
    else:
        taux_partage = 0.4

    if '_Taux Boni' in result_df.columns:
        taux_boni = pd.to_numeric(result_df['_Taux Boni'], errors='coerce').fillna(0.0)
    else:
        taux_boni = 0.0

    # Calculate: base = (PA * Taux Partage) * 0.5
    # If Taux Boni != 0: multiply by Taux Boni
    base = pa * taux_partage * 0.5
    boni_multiplier = taux_boni.where(taux_boni != 0, 1.0) if isinstance(taux_boni, pd.Series) else (taux_boni if taux_boni != 0 else 1.0)
    com_calculee = (base * boni_multiplier)

    # If _police_count is present (Assomption aggregated rows), multiply by count
    if '_police_count' in result_df.columns:
        police_count = pd.to_numeric(result_df['_police_count'], errors='coerce').fillna(1.0)
        com_calculee = com_calculee * police_count

    com_calculee = com_calculee.round(2)

    # Add calculated commission column
    result_df['Com Calculée'] = com_calculee

    # Calculate tolerance bounds
    tolerance = tolerance_pct / 100.0
    lower_bound = com_calculee * (1 - tolerance)
    upper_bound = com_calculee * (1 + tolerance)

    # Calculate percentage difference
    pct_diff = ((recu - com_calculee) / com_calculee * 100).round(1)

    # Calculate actual monetary difference (Reçu - Com Calculée)
    ecart = (recu - com_calculee).round(2)

    # Create verification column
    verification = []
    for i in range(len(result_df)):
        r = recu.iloc[i]
        c = com_calculee.iloc[i]
        e = ecart.iloc[i]

        if pd.isna(r) or pd.isna(c) or c == 0:
            verification.append('-')
        else:
            # Format the difference with sign
            ecart_str = f"+{e}" if e >= 0 else f"{e}"
            if r > upper_bound.iloc[i]:
                verification.append(f'✅ {ecart_str}$')
            elif r < lower_bound.iloc[i]:
                verification.append(f'⚠️ {ecart_str}$')
            else:
                verification.append(f'✓ {ecart_str}$')

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
