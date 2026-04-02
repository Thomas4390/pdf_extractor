"""
Sidebar rendering module.

Provides the main sidebar UI with connection status, session info,
mode toggle, and help documentation.
"""

import streamlit as st

from src.app.state import get_secret, reset_pipeline
from src.app.utils.board_utils import load_boards_async
from src.utils.aggregator import DatePeriod


def render_sidebar() -> None:
    """Render enhanced sidebar with better design and comprehensive help."""
    with st.sidebar:
        # Header with app branding
        st.markdown("""
        <div class="sidebar-header">
            <h2>📊 Commission Pipeline</h2>
            <div class="version">v1.0 • Extraction & Upload</div>
        </div>
        """, unsafe_allow_html=True)

        # Mode toggle with styled buttons
        st.markdown('<div class="sidebar-section-title">🎯 Mode</div>', unsafe_allow_html=True)
        current_mode = st.session_state.app_mode

        # Mode toggle buttons - 2 modes
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Extraction", key="mode_extraction", width="stretch",
                        type="primary" if current_mode == "extraction" else "secondary"):
                if current_mode != "extraction":
                    st.session_state.app_mode = "extraction"
                    st.rerun()
        with col2:
            if st.button("📊 Agrégation", key="mode_aggregation", width="stretch",
                        type="primary" if current_mode == "aggregation" else "secondary"):
                if current_mode != "aggregation":
                    st.session_state.app_mode = "aggregation"
                    st.session_state.agg_step = 1
                    if st.session_state.agg_period is None:
                        st.session_state.agg_period = DatePeriod.MONTH_1
                    st.rerun()

        st.markdown("---")

        # Connection status section
        _render_connection_status()

        st.markdown("---")

        # Session info
        _render_session_info()

        st.markdown("---")

        # Help section
        _render_help_section()

        # Footer
        st.markdown("---")
        st.caption("🛠️ Commission Pipeline v1.0")
        st.caption("Powered by OpenRouter & Monday.com")


def _render_connection_status() -> None:
    """Render the connection status section."""
    st.markdown('<div class="sidebar-section-title">🔗 Connexions</div>', unsafe_allow_html=True)

    api_from_secrets = get_secret('MONDAY_API_KEY') is not None

    # Monday.com API status
    if st.session_state.monday_api_key:
        status_text = "API Secrets" if api_from_secrets else "API Connectée"
        st.markdown(f"""
        <div class="status-indicator connected">
            <span>✓</span> <span>Monday.com: {status_text}</span>
        </div>
        """, unsafe_allow_html=True)

        if not api_from_secrets:
            if st.button("Déconnecter", key="disconnect_api", width="stretch"):
                st.session_state.monday_api_key = None
                st.session_state.monday_boards = None
                st.rerun()
    else:
        st.markdown("""
        <div class="status-indicator disconnected">
            <span>✗</span> <span>Monday.com: Non connecté</span>
        </div>
        """, unsafe_allow_html=True)

        api_key = st.text_input(
            "Clé API Monday.com",
            type="password",
            placeholder="Entrez votre clé API...",
            key="sidebar_api_key",
            label_visibility="collapsed"
        )
        if api_key:
            if st.button("🔌 Connecter", type="primary", width="stretch"):
                st.session_state.monday_api_key = api_key
                st.rerun()

        st.caption("💡 Ou configurez `MONDAY_API_KEY` dans secrets.toml")

    # Board status with stats
    if st.session_state.monday_api_key:
        st.markdown("---")
        if st.session_state.boards_loading:
            st.markdown("""
            <div class="status-indicator loading">
                <span>⏳</span> <span>Chargement des boards...</span>
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state.get('boards_error'):
            st.error(f"Erreur: {st.session_state.boards_error}")
            if st.button("🔄 Réessayer", width="stretch", type="primary"):
                st.session_state.boards_error = None
                st.session_state.monday_boards = None
                load_boards_async(force_rerun=True)
        elif st.session_state.monday_boards:
            board_count = len(st.session_state.monday_boards)
            st.markdown(f"""
            <div class="sidebar-info-card">
                <div class="label">Boards disponibles</div>
                <div class="value">📋 {board_count} boards</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔄 Rafraîchir", width="stretch"):
                st.session_state.monday_boards = None
                load_boards_async(force_rerun=True)
        else:
            if st.button("📥 Charger les boards", width="stretch", type="primary"):
                load_boards_async(force_rerun=True)


def _render_session_info() -> None:
    """Render the session info section."""
    st.markdown('<div class="sidebar-section-title">📈 Session actuelle</div>', unsafe_allow_html=True)

    files_count = len(st.session_state.uploaded_files) if st.session_state.uploaded_files else 0
    rows_count = len(st.session_state.combined_data) if st.session_state.combined_data is not None else 0

    if st.session_state.get("app_mode") == "aggregation":
        step_display = f"{st.session_state.get('agg_step', 1)}/3"
    else:
        step_display = f"{st.session_state.stage}/3"

    st.markdown(f"""
    <div class="sidebar-stats">
        <div class="sidebar-stat">
            <div class="number">{step_display}</div>
            <div class="label">Étape</div>
        </div>
        <div class="sidebar-stat">
            <div class="number">{files_count}</div>
            <div class="label">Fichiers</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if rows_count > 0:
        st.markdown(f"""
        <div class="sidebar-info-card" style="margin-top: 0.5rem;">
            <div class="label">Lignes extraites</div>
            <div class="value">{rows_count} lignes</div>
        </div>
        """, unsafe_allow_html=True)

    # Quick actions — two-click confirmation
    if st.session_state.stage > 1:
        if st.session_state.get("_confirm_reset"):
            st.warning("Êtes-vous sûr ? Les données extraites seront perdues.")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Confirmer", type="primary"):
                    st.session_state._confirm_reset = False
                    reset_pipeline()
                    st.rerun()
            with col_no:
                if st.button("Annuler"):
                    st.session_state._confirm_reset = False
                    st.rerun()
        else:
            if st.button("⬅️ Recommencer", width="stretch"):
                st.session_state._confirm_reset = True
                st.rerun()


def _render_help_section() -> None:
    """Render the comprehensive help section."""
    with st.expander("📖 Guide d'utilisation", expanded=False):
        st.markdown("""
        <div class="help-section">

        <h4>📄 Sources PDF supportées</h4>
        <ul>
            <li><strong>UV Assurance</strong> - Relevés de commissions UV</li>
            <li><strong>IDC</strong> - Relevés Industrial Alliance</li>
            <li><strong>IDC Statement</strong> - Statements détaillés IDC</li>
            <li><strong>Assomption Vie</strong> - Relevés Assomption</li>
        </ul>

        <h4>🔄 Workflow en 3 étapes</h4>
        <ul>
            <li><strong>Étape 1:</strong> Sélectionner la source et uploader les PDFs</li>
            <li><strong>Étape 2:</strong> Vérifier et modifier les données extraites</li>
            <li><strong>Étape 3:</strong> Exporter vers Monday.com</li>
        </ul>

        <h4>✨ Fonctionnalités</h4>
        <ul>
            <li>Extraction automatique via IA (VLM)</li>
            <li>Vérification des commissions calculées</li>
            <li>Normalisation des noms de conseillers</li>
            <li>Support multi-fichiers et multi-mois</li>
            <li>Cache intelligent pour éviter les re-extractions</li>
        </ul>

        <div class="help-tip">
            <strong>💡 Astuce:</strong> Les fichiers déjà extraits sont mis en cache.
            Réuploadez le même PDF pour utiliser le cache et économiser du temps.
        </div>

        <h4>⚙️ Configuration requise</h4>
        <ul>
            <li><code>MONDAY_API_KEY</code> - Clé API Monday.com</li>
            <li><code>OPENROUTER_API_KEY</code> - Pour l'extraction IA</li>
            <li><code>GOOGLE_SHEETS_*</code> - Pour la base conseillers</li>
        </ul>

        <h4>🔍 Vérification des données</h4>
        <p>Le système vérifie automatiquement que:</p>
        <ul>
            <li><strong>✓ OK</strong> - Commission dans la tolérance (±10%)</li>
            <li><strong>✅ Bonus</strong> - Commission supérieure au calcul</li>
            <li><strong>⚠️ Écart</strong> - Commission inférieure au calcul</li>
        </ul>

        <h4>❓ Support</h4>
        <p>En cas de problème, vérifiez:</p>
        <ul>
            <li>La qualité du PDF (scan lisible)</li>
            <li>La connexion API Monday.com</li>
            <li>Les logs dans la console</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)
