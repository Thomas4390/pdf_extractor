"""
Advisor management UI components.

Provides the interface for managing advisors and their name variations
which are used for automatic name normalization during PDF extraction.
"""

import streamlit as st


def render_advisor_management_tab() -> None:
    """Render advisor management interface."""
    # Check for pending toast message
    if st.session_state.get("_advisor_toast_message"):
        st.toast(st.session_state._advisor_toast_message, icon="✅")
        st.session_state._advisor_toast_message = None

    st.markdown("### 👥 Gestion des Conseillers")

    try:
        from src.utils.advisor_matcher import get_advisor_matcher, Advisor
    except ImportError:
        st.error("Module advisor_matcher non disponible")
        return

    # Initialize matcher (or reset if outdated version missing required methods)
    required_methods = ['find_advisor', 'update_advisor', 'delete_advisor']
    if (st.session_state.advisor_matcher is None or
        not all(hasattr(st.session_state.advisor_matcher, m) for m in required_methods)):
        # Reset both the module-level singleton and class-level singleton
        from src.utils import advisor_matcher as am_module
        from src.utils.advisor_matcher import AdvisorMatcher
        am_module._matcher_instance = None
        AdvisorMatcher._instance = None
        st.session_state.advisor_matcher = get_advisor_matcher()

    matcher = st.session_state.advisor_matcher

    # Check if Google Sheets is configured
    if not matcher.is_configured:
        error_detail = matcher.configuration_error or "Unknown error"
        st.warning(f"""
        ⚠️ **Google Sheets non configuré**

        La gestion des conseillers nécessite une connexion à Google Sheets.

        **Erreur:** `{error_detail}`

        **Pour configurer:**
        1. Créez un projet Google Cloud et activez l'API Sheets
        2. Créez un compte de service et téléchargez le fichier JSON
        3. Configurez les variables d'environnement:
           - `GOOGLE_SHEETS_SPREADSHEET_ID` - ID de votre spreadsheet
           - `GOOGLE_SHEETS_CREDENTIALS_FILE` - Chemin vers le fichier JSON

        **Ou dans Streamlit secrets:**
        ```toml
        [gcp_service_account]
        type = "service_account"
        project_id = "..."
        # ... autres champs du service account
        ```

        *La normalisation des noms de conseillers sera désactivée.*
        """)

        # Retry button
        if st.button("🔄 Réessayer la connexion", type="primary"):
            # Force reset all singletons
            from src.utils import advisor_matcher as am_module
            from src.utils.advisor_matcher import AdvisorMatcher
            am_module._matcher_instance = None
            AdvisorMatcher._instance = None
            st.session_state.advisor_matcher = None
            st.rerun()
        return

    st.info("""
    **Gestion des noms de conseillers**

    Cette section permet de gérer les conseillers et leurs variations de noms.
    Le système utilise ces données pour normaliser automatiquement les noms
    lors de l'extraction des données PDF.

    **Format de sortie:** Prénom, Initiale (ex: "Thomas, L")
    """)

    st.divider()

    # Statistics
    advisors = matcher.get_all_advisors()
    total_variations = sum(len(a.variations) for a in advisors)

    # Count by status
    active_count = sum(1 for a in advisors if a.status == "Active")
    new_count = sum(1 for a in advisors if a.status == "New")
    inactive_count = sum(1 for a in advisors if a.status == "Inactive")

    cols = st.columns(4)
    cols[0].metric("Total Conseillers", len(advisors))
    cols[1].metric("Actifs", active_count, help="Statut: Active")
    cols[2].metric("Nouveaux", new_count, help="Statut: New")
    cols[3].metric("Inactifs", inactive_count, help="Statut: Inactive")

    st.divider()

    # Add new advisor
    _render_add_advisor_form(matcher)

    st.divider()

    # List existing advisors
    _render_advisors_list(advisors, matcher)

    st.divider()

    # Test matching
    _render_matching_test(matcher)


def _render_add_advisor_form(matcher) -> None:
    """Render the add advisor form."""
    from src.utils.advisor_matcher import get_advisor_matcher, ADVISOR_STATUSES

    st.markdown("#### ➕ Ajouter un conseiller")

    with st.form("add_advisor_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            new_first_name = st.text_input(
                "Prénom",
                placeholder="Ex: Thomas",
                key="new_advisor_first_name"
            )

        with col2:
            new_last_name = st.text_input(
                "Nom de famille",
                placeholder="Ex: Lussier",
                key="new_advisor_last_name"
            )

        with col3:
            new_status = st.selectbox(
                "Statut",
                options=ADVISOR_STATUSES,
                index=0,  # Default to "Active"
                key="new_advisor_status"
            )

        new_variations = st.text_input(
            "Variations (séparées par des virgules)",
            placeholder="Ex: Tom, T. Lussier, Tommy",
            help="Entrez les différentes façons dont ce nom peut apparaître",
            key="new_advisor_variations"
        )

        submitted = st.form_submit_button("➕ Ajouter le conseiller", type="primary")

        if submitted:
            if new_first_name and new_last_name:
                variations = []
                if new_variations:
                    variations = [v.strip() for v in new_variations.split(',') if v.strip()]

                # Check if exists
                existing = matcher.find_advisor(new_first_name, new_last_name)
                if existing:
                    st.error(f"❌ Ce conseiller existe déjà")
                else:
                    try:
                        advisor = matcher.add_advisor(new_first_name, new_last_name, variations, new_status)
                        # Reset matcher to get fresh data
                        from src.utils import advisor_matcher as am_module
                        am_module._matcher_instance = None
                        st.session_state.advisor_matcher = None
                        st.session_state._advisor_toast_message = f"✅ Conseiller ajouté: {advisor.display_name_compact}"
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Erreur: {e}")
            else:
                st.error("❌ Veuillez entrer le prénom et le nom de famille")


def _get_status_badge(status: str) -> str:
    """Return HTML badge for advisor status."""
    colors = {
        "Active": ("#10B981", "#D1FAE5"),   # Green
        "New": ("#3B82F6", "#DBEAFE"),       # Blue
        "Inactive": ("#9CA3AF", "#F3F4F6"),  # Gray
    }
    fg, bg = colors.get(status, ("#6B7280", "#F3F4F6"))
    return f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600;">{status}</span>'


def _render_advisors_list(advisors: list, matcher) -> None:
    """Render the list of existing advisors with edit/delete options."""
    from src.utils.advisor_matcher import get_advisor_matcher, ADVISOR_STATUSES

    st.markdown("#### 📋 Conseillers existants")

    if not advisors:
        st.info("Aucun conseiller enregistré. Ajoutez-en un ci-dessus.")
    else:
        for idx, advisor in enumerate(advisors):
            advisor_id = getattr(advisor, '_row_id', idx)
            status_badge = _get_status_badge(advisor.status)
            with st.expander(f"**{advisor.first_name} {advisor.last_name}**", expanded=False):
                # Show status and compact format
                st.markdown(f"**Format compact:** {advisor.display_name_compact} &nbsp; {status_badge}", unsafe_allow_html=True)

                st.markdown("**Variations:**")
                if advisor.variations:
                    for var in advisor.variations:
                        st.text(f"  • {var}")
                else:
                    st.caption("Aucune variation définie")

                st.divider()

                # Edit/Delete buttons
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("✏️ Modifier", key=f"edit_btn_{advisor_id}", width="stretch"):
                        st.session_state[f'editing_advisor_{advisor_id}'] = True

                with col2:
                    if st.button("🗑️ Supprimer", key=f"delete_btn_{advisor_id}", type="secondary", width="stretch"):
                        try:
                            advisor_name = f"{advisor.first_name} {advisor.last_name}"
                            matcher.delete_advisor(advisor)
                            # Reset singleton and session state
                            from src.utils import advisor_matcher as am_module
                            am_module._matcher_instance = None
                            st.session_state.advisor_matcher = None
                            st.session_state._advisor_toast_message = f"✅ Conseiller supprimé: {advisor_name}"
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Erreur: {e}")

                # Edit form
                if st.session_state.get(f'editing_advisor_{advisor_id}', False):
                    st.markdown("---")
                    st.markdown("**Modifier le conseiller:**")

                    with st.form(key=f"edit_form_{advisor_id}"):
                        edit_col1, edit_col2 = st.columns(2)
                        with edit_col1:
                            new_first = st.text_input(
                                "Prénom",
                                value=advisor.first_name,
                                key=f"edit_first_{advisor_id}"
                            )
                        with edit_col2:
                            new_last = st.text_input(
                                "Nom",
                                value=advisor.last_name,
                                key=f"edit_last_{advisor_id}"
                            )

                        edit_col3, edit_col4 = st.columns([3, 1])
                        with edit_col3:
                            new_vars = st.text_input(
                                "Variations (séparées par des virgules)",
                                value=', '.join(advisor.variations),
                                key=f"edit_vars_{advisor_id}"
                            )
                        with edit_col4:
                            current_status_idx = ADVISOR_STATUSES.index(advisor.status) if advisor.status in ADVISOR_STATUSES else 0
                            new_status = st.selectbox(
                                "Statut",
                                options=ADVISOR_STATUSES,
                                index=current_status_idx,
                                key=f"edit_status_{advisor_id}"
                            )

                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            submitted = st.form_submit_button("💾 Enregistrer", type="primary")
                        with col_cancel:
                            cancelled = st.form_submit_button("❌ Annuler")

                        if submitted:
                            try:
                                variations = [v.strip() for v in new_vars.split(',') if v.strip()] if new_vars else []
                                matcher.update_advisor(
                                    advisor,
                                    first_name=new_first,
                                    last_name=new_last,
                                    variations=variations,
                                    status=new_status
                                )
                                # Reset singleton and session state
                                from src.utils import advisor_matcher as am_module
                                am_module._matcher_instance = None
                                st.session_state.advisor_matcher = None
                                del st.session_state[f'editing_advisor_{advisor_id}']
                                st.session_state._advisor_toast_message = f"✅ Conseiller mis à jour: {new_first} {new_last}"
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Erreur: {e}")

                        if cancelled:
                            del st.session_state[f'editing_advisor_{advisor_id}']
                            st.rerun()


def _render_matching_test(matcher) -> None:
    """Render the matching test section."""
    st.markdown("#### 🔍 Tester la correspondance")

    test_name = st.text_input(
        "Entrez un nom à tester",
        placeholder="Ex: Thomas Lussier, Lussier Thomas, T. Lussier...",
        key="test_name_input"
    )

    if test_name:
        result = matcher.match_compact(test_name)
        if result:
            st.success(f"✅ Correspondance trouvée: **{result}**")
        else:
            st.warning(f"⚠️ Aucune correspondance pour: \"{test_name}\"")
