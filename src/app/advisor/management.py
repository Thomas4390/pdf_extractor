"""
Advisor management UI components.

Provides the interface for managing advisors and their name variations
which are used for automatic name normalization during PDF extraction.
"""

import streamlit as st


def render_advisor_management_tab() -> None:
    """Render advisor management interface."""
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
        st.warning(f"""
        ⚠️ **Google Sheets non configuré**

        La gestion des conseillers nécessite une connexion à Google Sheets.

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

    cols = st.columns(3)
    cols[0].metric("Conseillers", len(advisors))
    cols[1].metric("Variations totales", total_variations)
    cols[2].metric("Stockage", f"{'☁️ Cloud' if matcher.storage_backend == 'google_sheets' else '💾 Local'}")

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
    from src.utils.advisor_matcher import get_advisor_matcher

    st.markdown("#### ➕ Ajouter un conseiller")

    with st.form("add_advisor_form", clear_on_submit=True):
        col1, col2 = st.columns(2)

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
                        advisor = matcher.add_advisor(new_first_name, new_last_name, variations)
                        st.success(f"✅ Conseiller ajouté: {advisor.display_name_compact}")
                        st.session_state.advisor_matcher = get_advisor_matcher()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Erreur: {e}")
            else:
                st.error("❌ Veuillez entrer le prénom et le nom de famille")


def _render_advisors_list(advisors: list, matcher) -> None:
    """Render the list of existing advisors with edit/delete options."""
    from src.utils.advisor_matcher import get_advisor_matcher

    st.markdown("#### 📋 Conseillers existants")

    if not advisors:
        st.info("Aucun conseiller enregistré. Ajoutez-en un ci-dessus.")
    else:
        for idx, advisor in enumerate(advisors):
            advisor_id = getattr(advisor, '_row_id', idx)
            with st.expander(f"**{advisor.first_name} {advisor.last_name}**", expanded=False):
                st.markdown(f"**Format compact:** {advisor.display_name_compact}")

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
                            matcher.delete_advisor(advisor)
                            st.success(f"✅ Conseiller supprimé")
                            # Reset singleton and session state
                            from src.utils import advisor_matcher as am_module
                            am_module._matcher_instance = None
                            st.session_state.advisor_matcher = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Erreur: {e}")

                # Edit form
                if st.session_state.get(f'editing_advisor_{advisor_id}', False):
                    st.markdown("---")
                    st.markdown("**Modifier le conseiller:**")

                    with st.form(key=f"edit_form_{advisor_id}"):
                        new_first = st.text_input(
                            "Prénom",
                            value=advisor.first_name,
                            key=f"edit_first_{advisor_id}"
                        )
                        new_last = st.text_input(
                            "Nom",
                            value=advisor.last_name,
                            key=f"edit_last_{advisor_id}"
                        )
                        new_vars = st.text_input(
                            "Variations (séparées par des virgules)",
                            value=', '.join(advisor.variations),
                            key=f"edit_vars_{advisor_id}"
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
                                    variations=variations
                                )
                                st.success(f"✅ Conseiller mis à jour")
                                # Reset singleton and session state
                                from src.utils import advisor_matcher as am_module
                                am_module._matcher_instance = None
                                st.session_state.advisor_matcher = None
                                del st.session_state[f'editing_advisor_{advisor_id}']
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
