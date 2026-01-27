"""
Stage 2: Preview - Data preview and editing.

Provides the data preview interface with verification,
configuration options, and data editing capabilities.
"""

import io
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.data_unifier import BoardType
from src.utils.model_registry import get_available_models, get_model_config

from src.app.state import get_pipeline, reset_pipeline, sanitize_filename, cleanup_temp_files
from src.app.utils.async_helpers import run_async
from src.app.utils.navigation import render_stepper, render_breadcrumb
from src.app.utils.date_utils import get_months_fr, detect_groups_from_data, analyze_groups_in_data
from src.app.components import (
    render_metrics_dashboard,
    verify_recu_vs_com,
    get_verification_stats,
    reorder_columns_for_display,
)


def run_extraction() -> None:
    """Run the extraction process with detailed progress display."""
    uploaded_files = st.session_state.uploaded_files
    if not uploaded_files:
        return

    st.session_state.is_processing = True
    st.session_state.extraction_results = {}
    st.session_state.combined_data = None

    # Get model configuration
    source = st.session_state.selected_source
    model_config = get_model_config(source) if source else None

    # Check if a custom model was selected
    selected_model = st.session_state.get('selected_model')
    if selected_model:
        # Temporarily update the model registry for this extraction
        from src.utils.model_registry import register_model, ModelConfig, ExtractionMode
        if source:
            original_config = model_config
            new_config = ModelConfig(
                model_id=selected_model,
                mode=original_config.mode,
                fallback_model_id=original_config.fallback_model_id,
                fallback_mode=original_config.fallback_mode,
                secondary_fallback_model_id=original_config.secondary_fallback_model_id,
                secondary_fallback_mode=original_config.secondary_fallback_mode,
                temperature=original_config.temperature,
                max_tokens=original_config.max_tokens,
                page_config=original_config.page_config,
                ocr_engine=original_config.ocr_engine,
                text_analysis_model=original_config.text_analysis_model,
            )
            register_model(source, new_config)
            model_config = new_config

    pipeline = get_pipeline()

    # Reset extractor clients if a custom model was selected
    # This ensures new clients are created with the updated model config
    if selected_model:
        pipeline.reset_extractor_clients()

    # ===== ENHANCED PROGRESS UI =====
    st.markdown("---")

    # Model info - simple elegant display
    if model_config:
        primary_model = model_config.model_id.split("/")[-1] if "/" in model_config.model_id else model_config.model_id
        st.markdown(f"""
        <div style="display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 20px; margin-bottom: 16px;">
            <span style="font-size: 16px;">🤖</span>
            <span style="color: white; font-weight: 500; font-size: 14px;">Modèle: {primary_model}</span>
        </div>
        """, unsafe_allow_html=True)

    # File list with status
    st.markdown("### 📁 Fichiers en cours de traitement")
    file_status_container = st.container()

    # Create placeholders for each file
    file_placeholders = {}
    with file_status_container:
        for i, file in enumerate(uploaded_files):
            file_placeholders[file.name] = {
                'container': st.empty(),
                'status': 'pending',
                'index': i
            }
            file_placeholders[file.name]['container'].markdown(
                f"⏳ **{i+1}/{len(uploaded_files)}** - `{file.name}` - *En attente...*"
            )

    # Overall progress
    progress_bar = st.progress(0, text="Démarrage de l'extraction...")
    status_text = st.empty()
    fallback_alert = st.empty()

    # Clean up previous temp files before creating new ones
    cleanup_temp_files()

    # Save files to temp directory with sanitized names
    temp_dir = tempfile.mkdtemp()
    temp_paths = []

    for file in uploaded_files:
        # Sanitize filename to prevent path traversal attacks
        safe_filename = sanitize_filename(file.name)
        temp_path = Path(temp_dir) / safe_filename
        temp_path.write_bytes(file.read())
        temp_paths.append(temp_path)
        file.seek(0)

    st.session_state.temp_pdf_paths = temp_paths

    # Track current file and results
    file_results = {}

    # Progress callback with enhanced display
    def on_progress(current: int, total: int, filename: str) -> None:
        progress = current / total if total > 0 else 0
        progress_bar.progress(progress, text=f"Traitement: {filename} ({current}/{total})")

        # Update file status
        for fname, placeholder in file_placeholders.items():
            idx = placeholder['index']
            if idx < current - 1:
                # Completed
                if fname in file_results:
                    result = file_results[fname]
                    if result.success:
                        rows = result.row_count
                        placeholder['container'].markdown(
                            f"✅ **{idx+1}/{total}** - `{fname}` - **{rows} lignes extraites**"
                        )
                    else:
                        placeholder['container'].markdown(
                            f"❌ **{idx+1}/{total}** - `{fname}` - *Erreur: {result.error[:50]}...*" if result.error and len(result.error) > 50 else f"❌ **{idx+1}/{total}** - `{fname}` - *Erreur: {result.error}*"
                        )
            elif idx == current - 1:
                # Currently processing (just finished)
                placeholder['container'].markdown(
                    f"🔄 **{idx+1}/{total}** - `{fname}` - *Finalisation...*"
                )
            elif fname == filename:
                # Currently being processed
                placeholder['container'].markdown(
                    f"🔄 **{idx+1}/{total}** - `{fname}` - *Extraction en cours...*"
                )

    try:
        # Run extraction
        batch_result = run_async(
            pipeline.process_batch(
                pdf_paths=temp_paths,
                source=st.session_state.selected_source,
                force_refresh=st.session_state.force_refresh,
                progress_callback=on_progress
            )
        )

        st.session_state.batch_result = batch_result

        # Store results and update file status
        results = {}
        fallback_used_files = []

        for result in batch_result.results:
            filename = Path(result.pdf_path).name
            results[filename] = result
            file_results[filename] = result

            # Update final status for each file
            if filename in file_placeholders:
                idx = file_placeholders[filename]['index']
                total = len(uploaded_files)
                if result.success:
                    rows = result.row_count
                    model_used = ""
                    fallback_indicator = ""

                    if result.usage and result.usage.model:
                        actual_model = result.usage.model.split("/")[-1] if "/" in result.usage.model else result.usage.model
                        model_used = f" ({actual_model})"

                        # Check if fallback was used
                        if model_config:
                            primary = model_config.model_id.split("/")[-1] if "/" in model_config.model_id else model_config.model_id
                            if actual_model != primary:
                                fallback_indicator = " 🔄"
                                fallback_used_files.append((filename, actual_model))

                    file_placeholders[filename]['container'].markdown(
                        f"✅ **{idx+1}/{total}** - `{filename}` - **{rows} lignes**{model_used}{fallback_indicator}"
                    )
                else:
                    error_msg = result.error[:60] + "..." if result.error and len(result.error) > 60 else (result.error or "Erreur inconnue")
                    file_placeholders[filename]['container'].markdown(
                        f"❌ **{idx+1}/{total}** - `{filename}` - *{error_msg}*"
                    )

        # Show fallback alerts if any
        if fallback_used_files:
            with fallback_alert.container():
                st.warning(f"⚠️ **Modèle de secours utilisé** pour {len(fallback_used_files)} fichier(s)")
                for fname, model in fallback_used_files:
                    st.caption(f"  • `{fname}` → {model}")

        st.session_state.extraction_results = results

        # Store usage stats
        st.session_state.extraction_usage = batch_result.total_usage

        # Get combined data and detect groups
        combined_df = batch_result.get_combined_dataframe()
        if combined_df is not None and not combined_df.empty:
            combined_df = detect_groups_from_data(combined_df, st.session_state.selected_source)
        st.session_state.combined_data = combined_df

        # Show success with model info
        progress_bar.progress(1.0, text="Extraction terminée!")
        if batch_result.failed == 0:
            status_text.success(f"✅ **{batch_result.total_rows} lignes** extraites de **{batch_result.successful} fichier(s)**")
        else:
            status_text.warning(f"⚠️ **{batch_result.total_rows} lignes** extraites. **{batch_result.failed} fichier(s)** en erreur.")

        # Show cost summary if available
        if batch_result.total_usage and batch_result.total_usage.cost > 0:
            st.info(f"💰 Coût total: **${batch_result.total_usage.cost:.4f}** | Tokens: {batch_result.total_usage.total_tokens:,}")

    except Exception as e:
        status_text.error(f"Échec de l'extraction: {e}")
        # Store the error for display in render_stage_2
        st.session_state.extraction_error = str(e)
        import traceback
        st.session_state.extraction_traceback = traceback.format_exc()

    finally:
        st.session_state.is_processing = False
        st.session_state.selected_model = None  # Reset custom model after extraction
        st.session_state.force_refresh = False  # Reset force refresh flag


def render_stage_2() -> None:
    """Render data preview stage with modern tabs layout."""
    st.markdown("## 📊 Pipeline de Commissions")
    render_stepper()
    render_breadcrumb()

    # Extract data if not done
    if st.session_state.combined_data is None:
        if not st.session_state.uploaded_files:
            st.error("Aucun fichier à traiter")
            if st.button("Recommencer"):
                reset_pipeline()
                st.rerun()
            return

        with st.spinner("Extraction en cours..."):
            run_extraction()
            st.rerun()
        return

    df = st.session_state.combined_data

    if df is None or df.empty:
        _render_empty_data_error()
        return

    # ===========================================
    # METRICS DASHBOARD HEADER (Always Visible)
    # ===========================================
    usage = st.session_state.get('extraction_usage')
    model_name = ""
    cost_display = "Cache"
    if usage:
        model_name = usage.model.split("/")[-1] if usage.model and "/" in usage.model else (usage.model or "N/A")
        cost_display = f"${usage.cost:.4f}" if usage.cost > 0 else "Cache"

    # Determine status
    has_verification_cols = 'Reçu' in df.columns and 'PA' in df.columns
    if has_verification_cols:
        df_verified = verify_recu_vs_com(df, tolerance_pct=st.session_state.get('verification_tolerance', 10.0))
        stats = get_verification_stats(df_verified)
        status_icon = "OK" if stats['ecart'] == 0 else f"{stats['ecart']} Ecarts"
    else:
        df_verified = df
        stats = None
        status_icon = "OK"

    # Dashboard metrics
    render_metrics_dashboard(
        row_count=len(df),
        cost=cost_display,
        model=model_name,
        status=status_icon
    )

    # ===========================================
    # TABBED INTERFACE (Replaces Expanders)
    # ===========================================
    tab_data, tab_verify, tab_config, tab_actions = st.tabs([
        "Données",
        "Vérification",
        "Configuration",
        "Actions"
    ])

    # ----- TAB 1: DONNÉES -----
    with tab_data:
        _render_data_tab(df, df_verified, has_verification_cols)

    # ----- TAB 2: VÉRIFICATION -----
    with tab_verify:
        _render_verification_tab(df, has_verification_cols)

    # ----- TAB 3: CONFIGURATION -----
    with tab_config:
        _render_configuration_tab(df, model_name, cost_display)

    # ----- TAB 4: ACTIONS -----
    with tab_actions:
        _render_actions_tab(df)

    # ===========================================
    # STICKY ACTION FOOTER
    # ===========================================
    st.markdown("---")

    footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

    with footer_col1:
        if st.button("Retour", width="stretch"):
            reset_pipeline()
            st.rerun()

    with footer_col3:
        if st.button("Uploader vers Monday.com", type="primary", width="stretch"):
            st.session_state.stage = 3
            st.rerun()


def _render_empty_data_error() -> None:
    """Render error message when no data is extracted."""
    st.error("Aucune donnée extraite")

    # Show detailed error information if available
    batch_result = st.session_state.get('batch_result')
    if batch_result:
        if batch_result.failed > 0:
            st.warning(f"⚠️ {batch_result.failed} fichier(s) en erreur sur {batch_result.total_pdfs}")
            for result in batch_result.results:
                if not result.success and result.error:
                    st.error(f"**{Path(result.pdf_path).name}**: {result.error}")
        elif batch_result.total_pdfs == 0:
            st.info("Aucun fichier n'a été traité. Vérifiez que les fichiers sont bien des PDFs valides.")
        else:
            st.info("L'extraction a réussi mais n'a retourné aucune donnée. Le PDF ne contient peut-être pas les informations attendues.")
    else:
        # Check for stored extraction error
        extraction_error = st.session_state.get('extraction_error')
        if extraction_error:
            st.error(f"**Erreur d'extraction**: {extraction_error}")
            with st.expander("Voir le traceback complet"):
                st.code(st.session_state.get('extraction_traceback', 'No traceback available'))
        else:
            # Check if API key is configured
            from src.utils.config import _get_secret, settings
            api_key = _get_secret("OPENROUTER_API_KEY") or settings.openrouter_api_key
            if not api_key:
                st.error("⚠️ **OPENROUTER_API_KEY** n'est pas configurée. Ajoutez-la dans les secrets Streamlit ou le fichier .env")

    if st.button("Recommencer"):
        reset_pipeline()
        st.rerun()


def _render_data_tab(df: pd.DataFrame, df_verified: pd.DataFrame, has_verification_cols: bool) -> None:
    """Render the data tab."""
    # Quick stats
    cols = st.columns(4)
    cols[0].metric("Lignes", len(df))
    cols[1].metric("Colonnes", len(df.columns))
    if '# de Police' in df.columns:
        cols[2].metric("Contrats", df['# de Police'].notna().sum())
    elif 'Contrat' in df.columns:
        cols[2].metric("Contrats", df['Contrat'].notna().sum())
    else:
        cols[2].metric("Contrats", "-")
    cols[3].metric("Doublons", df.duplicated().sum())

    st.markdown("---")

    # Display the dataframe
    if has_verification_cols:
        st.dataframe(reorder_columns_for_display(df_verified), width="stretch", height=400)
    else:
        st.dataframe(reorder_columns_for_display(df), width="stretch", height=400)

    # Extraction details
    results = st.session_state.extraction_results
    if results:
        with st.expander("Détails de l'extraction", expanded=False):
            for filename, result in results.items():
                if result.success:
                    st.markdown(f"**{filename}**: {result.row_count} lignes ({result.extraction_time_ms}ms)")
                else:
                    st.markdown(f"**{filename}**: {result.error}")


def _render_verification_tab(df: pd.DataFrame, has_verification_cols: bool) -> None:
    """Render the verification tab."""
    if has_verification_cols:
        st.markdown("### Vérification Reçu vs Commission")
        st.caption("Formule: `Com Calculée = ROUND((PA x 0.4) x 0.5, 2)`")

        tolerance = st.slider(
            "Tolérance (%)",
            min_value=1.0,
            max_value=50.0,
            value=st.session_state.get('verification_tolerance', 10.0),
            step=1.0,
            key="verification_tolerance_slider"
        )
        st.session_state.verification_tolerance = tolerance

        df_verified = verify_recu_vs_com(df, tolerance_pct=tolerance)
        stats = get_verification_stats(df_verified)

        # Stats display
        stat_cols = st.columns(4)
        stat_cols[0].metric("OK", stats['ok'])
        stat_cols[1].metric("Bonus", stats['bonus'])
        stat_cols[2].metric("Ecart", stats['ecart'])
        stat_cols[3].metric("N/A", stats['na'])

        if stats['ecart'] > 0:
            st.warning(f"**{stats['ecart']} ligne(s)** ont un écart négatif")
        if stats['bonus'] > 0:
            st.success(f"**{stats['bonus']} ligne(s)** ont un bonus")

        st.markdown("---")

        # Find verification column
        verif_col = [col for col in df_verified.columns if col.startswith('Vérification')]
        if verif_col:
            verif_col = verif_col[0]

            # Show table with ecarts (negative differences)
            if stats['ecart'] > 0:
                st.markdown("#### ⚠️ Lignes avec écarts négatifs")
                ecart_df = df_verified[df_verified[verif_col].astype(str).str.contains('⚠️', na=False)]

                # Select relevant columns for display
                display_cols = ['# de Police', 'Nom Client', 'Conseiller', 'PA', 'Reçu', 'Com Calculée', verif_col]
                display_cols = [c for c in display_cols if c in ecart_df.columns]

                st.dataframe(
                    ecart_df[display_cols],
                    width="stretch",
                    hide_index=True
                )

            # Show table with bonus (positive differences)
            if stats['bonus'] > 0:
                st.markdown("#### ✅ Lignes avec bonus")
                bonus_df = df_verified[df_verified[verif_col].astype(str).str.contains('✅', na=False)]

                display_cols = ['# de Police', 'Nom Client', 'Conseiller', 'PA', 'Reçu', 'Com Calculée', verif_col]
                display_cols = [c for c in display_cols if c in bonus_df.columns]

                st.dataframe(
                    bonus_df[display_cols],
                    width="stretch",
                    hide_index=True
                )

            # Show all data with verification
            with st.expander("📊 Voir toutes les données avec vérification", expanded=False):
                st.dataframe(
                    df_verified,
                    width="stretch",
                    hide_index=True
                )
    else:
        st.info("La vérification n'est pas disponible pour ce type de données (colonnes 'Reçu' et 'PA' requises).")


def _render_configuration_tab(df: pd.DataFrame, model_name: str, cost_display: str) -> None:
    """Render the configuration tab."""
    st.markdown("### Résumé de la Configuration")

    config_cols = st.columns(2)
    with config_cols[0]:
        st.markdown("**Source**")
        st.info(st.session_state.selected_source or "N/A")

        st.markdown("**Fichiers**")
        st.info(f"{len(st.session_state.uploaded_files)} fichier(s)")

    with config_cols[1]:
        board_name = st.session_state._current_board_name or "N/A"
        st.markdown("**Board de destination**")
        st.info(board_name[:40] + "..." if len(board_name) > 40 else board_name)

        st.markdown("**Type de table**")
        st.info("Ventes" if st.session_state.selected_board_type == BoardType.SALES_PRODUCTION else "Paiements")

    st.markdown("---")

    # ===========================================
    # MODEL SELECTION & RE-EXTRACTION
    # ===========================================
    st.markdown("### 🤖 Modèle d'extraction")

    # Show current model info
    current_model = model_name if model_name else "Modèle par défaut"
    st.markdown(f"**Modèle actuel:** `{current_model}`")
    st.markdown(f"**Coût de l'extraction:** {cost_display}")

    st.caption("Sélectionnez un autre modèle pour ré-extraire les données.")

    available_models = get_available_models()
    model_options = list(available_models.keys())
    model_labels = [f"{v}" for v in available_models.values()]

    # Create a mapping for display
    model_display_map = dict(zip(model_labels, model_options))

    col_model, col_btn = st.columns([3, 1])
    with col_model:
        selected_label = st.selectbox(
            "Choisir un modèle",
            options=model_labels,
            index=0,
            key="model_selector",
            help="Sélectionnez un modèle VLM pour l'extraction"
        )
        selected_model_id = model_display_map[selected_label]

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Ré-extraire", type="primary", width="stretch"):
            st.session_state.selected_model = selected_model_id
            st.session_state.combined_data = None
            st.session_state.extraction_results = {}
            st.session_state.extraction_usage = None
            st.session_state.force_refresh = True
            # IMPORTANT: Reset pipeline to force new extractors with new model
            st.session_state.pipeline = None
            st.rerun()

    st.markdown("---")

    # ===========================================
    # PER-FILE GROUP ASSIGNMENT
    # ===========================================
    _render_group_assignment(df)


def _render_group_assignment(df: pd.DataFrame) -> None:
    """Render the per-file group assignment section."""
    st.markdown("### 📁 Groupes de destination par fichier")

    if '_source_file' in df.columns:
        unique_files = df['_source_file'].unique().tolist()

        if len(unique_files) > 1:
            st.info(f"🎯 **{len(unique_files)} fichiers** - Chaque fichier peut être envoyé vers un groupe Monday.com différent.")

            # Show current group summary
            with st.expander("📊 Résumé actuel des groupes", expanded=True):
                group_summary = {}
                for filename in unique_files:
                    file_df = df[df['_source_file'] == filename]
                    # Check both column existence AND non-empty DataFrame
                    if '_target_group' in file_df.columns and not file_df.empty:
                        group = file_df['_target_group'].iloc[0]
                    else:
                        group = "Auto"
                    if group not in group_summary:
                        group_summary[group] = []
                    group_summary[group].append((filename, len(file_df)))

                for group, files in group_summary.items():
                    st.markdown(f"**📁 {group}**")
                    for fname, count in files:
                        st.caption(f"  └─ `{fname}` ({count} lignes)")

        # Generate group options
        months_fr = get_months_fr()
        now = datetime.now()
        group_options_list = []
        for offset in range(-6, 7):
            month = now.month + offset
            year = now.year
            if month < 1:
                month += 12
                year -= 1
            elif month > 12:
                month -= 12
                year += 1
            group_options_list.append(f"{months_fr[month]} {year}")

        # Initialize file_group_overrides if not exists
        if 'file_group_overrides' not in st.session_state:
            st.session_state.file_group_overrides = {}

        if len(unique_files) > 1:
            st.markdown("#### ✏️ Modifier les groupes")
            st.caption("Sélectionnez un groupe prédéfini ou entrez un nom personnalisé pour chaque fichier.")

            for filename in unique_files:
                file_df = df[df['_source_file'] == filename]
                # Check both column existence AND non-empty DataFrame
                if '_target_group' in file_df.columns and not file_df.empty:
                    current_group = file_df['_target_group'].iloc[0]
                else:
                    current_group = "Auto"
                row_count = len(file_df)

                # Use container with border for better visibility
                with st.container():
                    col_file, col_group, col_manual = st.columns([2, 2, 2])

                    with col_file:
                        # Truncate long filenames
                        display_name = filename[:25] + "..." if len(filename) > 25 else filename
                        st.markdown(f"**📄 {display_name}**")
                        st.caption(f"{row_count} lignes • Actuel: `{current_group}`")

                    with col_group:
                        # Get current override or detected group
                        current_override = st.session_state.file_group_overrides.get(filename, current_group)

                        # Find index of current group in options
                        try:
                            default_idx = group_options_list.index(current_override)
                        except ValueError:
                            default_idx = 6  # Middle of the list (current month)

                        selected_group = st.selectbox(
                            f"Groupe pour {filename}",
                            options=group_options_list,
                            index=default_idx,
                            key=f"group_select_{filename}",
                            label_visibility="collapsed"
                        )

                        # Always track the selection
                        if selected_group != current_group:
                            st.session_state.file_group_overrides[filename] = selected_group

                    with col_manual:
                        manual_input = st.text_input(
                            f"Groupe personnalisé pour {filename}",
                            placeholder="Ex: Janvier 2026",
                            key=f"manual_group_{filename}",
                            label_visibility="collapsed"
                        )
                        if manual_input and manual_input.strip():
                            st.session_state.file_group_overrides[filename] = manual_input.strip()

                    st.markdown("---")

            # Always show apply button when multiple files
            col_apply, col_reset = st.columns([3, 1])
            with col_apply:
                if st.button("✅ Appliquer les modifications de groupes", type="primary", width="stretch"):
                    # Update the dataframe with new groups
                    changes_made = False
                    for filename in unique_files:
                        # Check selectbox value
                        select_key = f"group_select_{filename}"
                        manual_key = f"manual_group_{filename}"

                        new_group = None
                        # Manual input takes priority
                        if manual_key in st.session_state and st.session_state[manual_key]:
                            new_group = st.session_state[manual_key].strip()
                        elif filename in st.session_state.file_group_overrides:
                            new_group = st.session_state.file_group_overrides[filename]
                        elif select_key in st.session_state:
                            new_group = st.session_state[select_key]

                        if new_group:
                            df.loc[df['_source_file'] == filename, '_target_group'] = new_group
                            changes_made = True

                    if changes_made:
                        st.session_state.combined_data = df
                        st.session_state.file_group_overrides = {}  # Reset overrides
                        st.toast("✅ Groupes mis à jour!", icon="✅")
                        st.rerun()
                    else:
                        st.info("Aucune modification détectée.")

            with col_reset:
                if st.button("🔄 Reset", width="stretch"):
                    st.session_state.file_group_overrides = {}
                    st.rerun()

        else:
            # Single file mode within _source_file block
            if '_target_group' in df.columns:
                groups_info = analyze_groups_in_data(df)

                st.markdown("### Groupes Détectés")
                if groups_info['spans_multiple_months']:
                    st.warning(f"Les données couvrent **{len(groups_info['unique_groups'])} mois différents**.")

                for group, count in groups_info['group_counts'].items():
                    st.markdown(f"- **{group}**: {count} lignes")

                st.markdown("---")

                # Manual override with text input option
                st.markdown("### Modifier le groupe")
                st.caption("Sélectionnez ou entrez manuellement un groupe.")

                group_options = ["(Garder auto-détection)"] + group_options_list

                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    manual_group = st.selectbox("Groupe prédéfini", group_options, key="manual_group_override_single")
                with col2:
                    custom_group = st.text_input(
                        "Ou entrez un groupe personnalisé",
                        placeholder="Ex: Janvier 2026",
                        key="custom_group_input_single"
                    )
                with col3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    final_group = custom_group.strip() if custom_group and custom_group.strip() else (
                        manual_group if manual_group != "(Garder auto-détection)" else None
                    )
                    if final_group:
                        if st.button("Appliquer", width="stretch", type="primary", key="apply_single_group"):
                            df['_target_group'] = final_group
                            st.session_state.combined_data = df
                            st.toast(f"✅ Groupe modifié: {final_group}", icon="✅")
                            st.rerun()

    elif '_target_group' in df.columns:
        # Single file mode - show simple group override
        groups_info = analyze_groups_in_data(df)

        st.markdown("### Groupes Détectés")
        if groups_info['spans_multiple_months']:
            st.warning(f"Les données couvrent **{len(groups_info['unique_groups'])} mois différents**.")

        for group, count in groups_info['group_counts'].items():
            st.markdown(f"- **{group}**: {count} lignes")

        st.markdown("---")

        # Manual override with text input option
        st.markdown("### Modifier le groupe")
        st.caption("Sélectionnez ou entrez manuellement un groupe.")

        months_fr = get_months_fr()
        now = datetime.now()
        group_options = ["(Garder auto-détection)"]

        for offset in range(-3, 4):
            month = now.month + offset
            year = now.year
            if month < 1:
                month += 12
                year -= 1
            elif month > 12:
                month -= 12
                year += 1
            group_options.append(f"{months_fr[month]} {year}")

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            manual_group = st.selectbox("Groupe prédéfini", group_options, key="manual_group_override")
        with col2:
            custom_group = st.text_input(
                "Ou entrez un groupe personnalisé",
                placeholder="Ex: Janvier 2026",
                key="custom_group_input"
            )
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            final_group = custom_group.strip() if custom_group and custom_group.strip() else (
                manual_group if manual_group != "(Garder auto-détection)" else None
            )
            if final_group:
                if st.button("Appliquer", width="stretch", type="primary"):
                    df['_target_group'] = final_group
                    st.session_state.combined_data = df
                    st.toast(f"✅ Groupe modifié: {final_group}", icon="✅")
                    st.rerun()

    # Column info
    st.markdown("---")
    st.markdown("### Informations Colonnes")
    col_info = pd.DataFrame({
        'Colonne': df.columns,
        'Type': df.dtypes.astype(str),
        'Non-Null': df.notna().sum().values,
        'Null': df.isna().sum().values
    })
    st.dataframe(col_info, width="stretch", height=200)


def _render_actions_tab(df: pd.DataFrame) -> None:
    """Render the actions tab."""
    st.markdown("### Exporter les données")

    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Télécharger CSV",
            data=csv,
            file_name=f"commissions_{st.session_state.selected_source}.csv",
            mime="text/csv",
            width="stretch"
        )
    with col2:
        # Excel export
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Commissions')
        excel_data = output.getvalue()
        st.download_button(
            "Télécharger Excel",
            data=excel_data,
            file_name=f"commissions_{st.session_state.selected_source}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch"
        )

    st.markdown("---")
    st.markdown("### Remplacer les données")
    st.caption("Uploader un fichier Excel/CSV modifié pour remplacer les données extraites.")

    excel_file = st.file_uploader(
        "Fichier Excel/CSV modifié",
        type=['xlsx', 'xls', 'csv'],
        key="excel_upload"
    )

    if excel_file:
        try:
            if excel_file.name.endswith('.csv'):
                uploaded_df = pd.read_csv(excel_file)
            else:
                uploaded_df = pd.read_excel(excel_file)

            st.success(f"{excel_file.name} chargé ({len(uploaded_df)} lignes)")

            if st.button("Utiliser ce fichier", type="primary"):
                st.session_state.combined_data = uploaded_df
                st.session_state.data_modified = True
                st.rerun()

        except Exception as e:
            st.error(f"Erreur: {e}")
