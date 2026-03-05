"""
Stage 2: Preview - Data preview and editing.

Provides the data preview interface with verification,
configuration options, and data editing capabilities.
"""

import io
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.app.components import (
    get_verification_stats,
    render_metrics_dashboard,
    reorder_columns_for_display,
    verify_recu_vs_com,
)
from src.app.state import cleanup_temp_files, get_pipeline, reset_pipeline, sanitize_filename
from src.app.utils.async_helpers import run_async
from src.app.utils.date_utils import analyze_groups_in_data, detect_groups_from_data, get_months_fr
from src.app.utils.navigation import render_breadcrumb, render_stepper
from src.utils.config import get_settings
from src.utils.data_unifier import BoardType
from src.utils.model_registry import get_available_models, get_model_config


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
        from src.utils.model_registry import ModelConfig, register_model
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
    timeout_warning = st.empty()
    extraction_start_time = time.time()

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

        # Timeout warning if extraction is taking long
        elapsed = time.time() - extraction_start_time
        remaining_ratio = (total - current) / total if total > 0 else 0
        est_remaining = (elapsed / max(current, 1)) * (total - current)
        if elapsed > 60 and est_remaining > 30:
            elapsed_min = int(elapsed // 60)
            timeout_warning.warning(
                f"L'extraction prend du temps ({elapsed_min}min écoulées, "
                f"~{int(est_remaining)}s restantes estimées)..."
            )

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


def _render_extraction_alerts() -> None:
    """Show transparency alerts for cache hits, fallback usage, JSON repairs, and warnings."""
    batch_result = st.session_state.get("batch_result")
    if not batch_result:
        return

    cached_files = []
    fallback_files = []
    repaired_files = []

    for result in batch_result.results:
        fname = Path(result.pdf_path).name
        if result.was_cached:
            cached_files.append(fname)
        if result.usage and result.usage.was_fallback:
            fallback_files.append(fname)
        if result.usage and result.usage.json_repaired:
            repaired_files.append(fname)

    if cached_files:
        st.info(f"Résultats en cache pour : {', '.join(cached_files)}")
    if fallback_files:
        st.warning(f"Modèle de secours utilisé pour : {', '.join(fallback_files)}")
    if repaired_files:
        st.warning(f"Réparation JSON appliquée pour : {', '.join(repaired_files)}")

    # Show pipeline warnings if any
    all_warnings = batch_result.all_warnings
    if all_warnings:
        with st.expander(f"Avertissements ({len(all_warnings)})", expanded=False):
            for w in all_warnings:
                st.caption(f"• {Path(w.pdf_path).name}: {w.message}")


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
        df_verified = verify_recu_vs_com(df, tolerance_pct=st.session_state.get('verification_tolerance', getattr(get_settings(), 'verification_tolerance_pct', 10.0)))
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

    # Extraction transparency alerts
    _render_extraction_alerts()

    # ===========================================
    # TABBED INTERFACE (Replaces Expanders)
    # ===========================================
    is_historical = st.session_state.selected_board_type == BoardType.HISTORICAL_PAYMENTS

    if is_historical:
        tab_data, tab_verify, tab_recon, tab_config, tab_actions = st.tabs([
            "Données",
            "Vérification",
            "Rapprochement",
            "Configuration",
            "Actions"
        ])
    else:
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

    # ----- TAB 3: RAPPROCHEMENT (Historical only) -----
    if is_historical:
        with tab_recon:
            _render_reconciliation_tab(df)

    # ----- TAB: CONFIGURATION -----
    with tab_config:
        _render_configuration_tab(df, model_name, cost_display)

    # ----- TAB: ACTIONS -----
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

    col_retry, col_reset = st.columns(2)
    with col_retry:
        if st.button("Réessayer l'extraction"):
            # Reset extraction state but keep files and config
            st.session_state.combined_data = None
            st.session_state.extraction_results = {}
            st.session_state.batch_result = None
            st.session_state.extraction_usage = None
            st.session_state.extraction_error = None
            st.session_state.extraction_traceback = None
            st.rerun()
    with col_reset:
        if st.button("Recommencer"):
            reset_pipeline()
            st.rerun()


def _check_duplicates_against_board(df: pd.DataFrame) -> pd.DataFrame:
    """Check for duplicate policy numbers against the Monday.com board.

    Only runs for SALES_PRODUCTION board type. Results are cached in session state.

    Args:
        df: DataFrame with '# de Police' column

    Returns:
        DataFrame with '_is_duplicate' column added
    """
    if st.session_state.selected_board_type != BoardType.SALES_PRODUCTION:
        return df

    if '# de Police' not in df.columns:
        return df

    board_id = st.session_state.selected_board_id
    if not board_id:
        return df

    # Only fetch once per pipeline run
    if not st.session_state.duplicate_check_done:
        try:
            with st.spinner("Vérification des doublons sur Monday.com..."):
                pipeline = get_pipeline()
                existing = run_async(
                    pipeline.monday.get_existing_policy_numbers(int(board_id))
                )
                st.session_state.existing_policy_numbers = existing
                st.session_state.duplicate_check_done = True
        except Exception as e:
            st.warning(f"Impossible de vérifier les doublons: {e}")
            st.session_state.duplicate_check_done = True
            st.session_state.existing_policy_numbers = set()

    existing = st.session_state.existing_policy_numbers or set()

    # Mark duplicates
    df = df.copy()
    df['_is_duplicate'] = df['# de Police'].apply(
        lambda x: str(x).strip() in existing if pd.notna(x) else False
    )
    dup_count = df['_is_duplicate'].sum()
    st.session_state.duplicate_count = int(dup_count)
    st.session_state.combined_data = df

    return df


def _render_data_tab(df: pd.DataFrame, df_verified: pd.DataFrame, has_verification_cols: bool) -> None:
    """Render the data tab."""
    # Check for duplicates against Monday.com board
    df = _check_duplicates_against_board(df)

    dup_count = st.session_state.duplicate_count

    # Quick stats
    n_stats = 5 if dup_count > 0 else 4
    cols = st.columns(n_stats)
    cols[0].metric("Lignes", len(df))
    cols[1].metric("Colonnes", len(df.columns))
    if '# de Police' in df.columns:
        cols[2].metric("Contrats", df['# de Police'].notna().sum())
    elif 'Contrat' in df.columns:
        cols[2].metric("Contrats", df['Contrat'].notna().sum())
    else:
        cols[2].metric("Contrats", "-")
    cols[3].metric("Doublons internes", df.drop(columns=[c for c in df.columns if c.startswith('_')], errors='ignore').duplicated().sum())
    if dup_count > 0:
        cols[4].metric("Doublons Monday", dup_count)

    # Duplicate warning
    if dup_count > 0:
        st.warning(
            f"⚠️ **{dup_count} ligne(s)** ont un # de Police déjà présent sur le board Monday.com. "
            "Ces lignes seront exclues lors de l'upload."
        )

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

    # Upload button — only in Données tab
    st.markdown("---")
    _, _, btn_col = st.columns([1, 2, 1])
    with btn_col:
        if st.button("Uploader vers Monday.com", type="primary", width="stretch", key="upload_from_data_tab"):
            st.session_state.stage = 3
            st.rerun()


def _render_verification_tab(df: pd.DataFrame, has_verification_cols: bool) -> None:
    """Render the verification tab."""
    # Advisor normalization display
    normalizations = df.attrs.get('advisor_normalizations', [])
    if normalizations:
        with st.expander(f"Normalisation des noms ({len(normalizations)} modification(s))", expanded=False):
            for orig, norm in normalizations:
                st.caption(f"• {orig} → {norm}")

    if has_verification_cols:
        st.markdown("### Vérification Reçu vs Commission")
        st.caption("Formule: `Com Calculée = ROUND((PA × Taux Partage) × 0.5 [× Taux Boni si ≠ 0], 2)`")

        tolerance = st.slider(
            "Tolérance (%)",
            min_value=1.0,
            max_value=50.0,
            value=st.session_state.get('verification_tolerance', getattr(get_settings(), 'verification_tolerance_pct', 10.0)),
            step=1.0,
            key="verification_tolerance_slider"
        )
        st.session_state.verification_tolerance = tolerance

        df_verified = verify_recu_vs_com(df, tolerance_pct=tolerance)
        stats = get_verification_stats(df_verified)

        # Summary one-liner
        if stats['ecart'] == 0:
            st.success(f"Vérification OK — {stats['ok']} lignes conformes, {stats['bonus']} bonus")
        else:
            st.warning(f"{stats['ecart']} écart(s) détecté(s) sur {stats['ok'] + stats['ecart'] + stats['bonus']} lignes")

        # Stats display
        stat_cols = st.columns(4)
        stat_cols[0].metric("OK", stats['ok'])
        stat_cols[1].metric("Bonus", stats['bonus'])
        stat_cols[2].metric("Ecart", stats['ecart'])
        stat_cols[3].metric("N/A", stats['na'])

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


def _render_sort_options(df: pd.DataFrame) -> None:
    """Render sort options for the extracted data."""
    # Build available sort choices based on columns present
    sort_choices: dict[str, str | None] = {"Ordre original": None}
    if 'Nom Client' in df.columns:
        sort_choices["Nom Client (A → Z)"] = "Nom Client"
    if '# de Police' in df.columns:
        sort_choices["# de Police (croissant)"] = "# de Police"
    if 'Conseiller' in df.columns:
        sort_choices["Conseiller (A → Z)"] = "Conseiller"

    if len(sort_choices) <= 1:
        return

    st.markdown("### Tri du tableau")

    col_sort, col_btn = st.columns([3, 1])
    with col_sort:
        selected_label = st.selectbox(
            "Trier par",
            options=list(sort_choices.keys()),
            index=0,
            key="sort_select",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        sort_col = sort_choices[selected_label]
        if sort_col is not None:
            if st.button("Trier", key="apply_sort", type="primary", width="stretch"):
                df_sorted = df.sort_values(by=sort_col, key=lambda s: s.str.lower() if s.dtype == object else s, na_position='last').reset_index(drop=True)
                st.session_state.combined_data = df_sorted
                st.toast(f"Tableau trié par {sort_col}")
                st.rerun()
        else:
            st.button("Trier", key="apply_sort_disabled", disabled=True, width="stretch")

    st.markdown("---")


def _render_compagnie_override(df: pd.DataFrame) -> None:
    """Render Compagnie name override for UV and Assomption sources."""
    source = st.session_state.get('selected_source', '')
    if source not in ('UV', 'ASSOMPTION') or 'Compagnie' not in df.columns:
        return

    unique_compagnies = sorted(df['Compagnie'].dropna().unique().tolist())
    if not unique_compagnies:
        return

    # Preset options per source
    if source == 'UV':
        presets = ['UV Inc', 'UV Perso']
    else:
        presets = ['Assomption']

    # Build full options list: presets first, then any detected values not already in presets
    all_options = list(presets)
    for c in unique_compagnies:
        if c not in all_options:
            all_options.append(c)

    st.markdown("### Nom de compagnie")

    if len(unique_compagnies) == 1:
        # Single value — simple override
        current = unique_compagnies[0]
        current_idx = all_options.index(current) if current in all_options else 0

        col_sel, col_custom, col_btn = st.columns([2, 2, 1])
        with col_sel:
            selected = st.selectbox(
                "Compagnie",
                options=all_options,
                index=current_idx,
                key="compagnie_override_select",
            )
        with col_custom:
            custom = st.text_input(
                "Ou entrez un nom personnalisé",
                placeholder="Ex: UV Perso",
                key="compagnie_override_custom",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            new_value = custom.strip() if custom and custom.strip() else selected
            if new_value != current:
                if st.button("Appliquer", key="apply_compagnie", type="primary", width="stretch"):
                    df['Compagnie'] = new_value
                    st.session_state.combined_data = df
                    st.toast(f"Compagnie modifiée: {current} → {new_value}")
                    st.rerun()
            else:
                st.button("Appliquer", key="apply_compagnie_disabled", disabled=True, width="stretch")

    else:
        # Multiple values — per-value override
        st.caption(f"{len(unique_compagnies)} valeur(s) détectée(s) dans les données.")

        changes = {}
        for i, current in enumerate(unique_compagnies):
            current_idx = all_options.index(current) if current in all_options else 0
            count = int((df['Compagnie'] == current).sum())

            col_label, col_sel, col_custom = st.columns([2, 2, 2])
            with col_label:
                st.markdown(f"**{current}**")
                st.caption(f"{count} ligne(s)")
            with col_sel:
                selected = st.selectbox(
                    f"Nouvelle valeur pour {current}",
                    options=all_options,
                    index=current_idx,
                    key=f"compagnie_override_{i}",
                    label_visibility="collapsed",
                )
            with col_custom:
                custom = st.text_input(
                    f"Personnalisé pour {current}",
                    placeholder="Nom personnalisé",
                    key=f"compagnie_custom_{i}",
                    label_visibility="collapsed",
                )

            new_value = custom.strip() if custom and custom.strip() else selected
            if new_value != current:
                changes[current] = new_value

        if changes:
            if st.button("Appliquer les modifications", key="apply_compagnie_multi", type="primary"):
                for old_val, new_val in changes.items():
                    df.loc[df['Compagnie'] == old_val, 'Compagnie'] = new_val
                st.session_state.combined_data = df
                summary = ", ".join(f"{k} → {v}" for k, v in changes.items())
                st.toast(f"Compagnie modifiée: {summary}")
                st.rerun()

    st.markdown("---")


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
    # COMPAGNIE OVERRIDE (UV & Assomption only)
    # ===========================================
    _render_compagnie_override(df)

    # ===========================================
    # SORT ORDER
    # ===========================================
    _render_sort_options(df)

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
    model_display_map = dict(zip(model_labels, model_options, strict=False))

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
    # Check for pending toast message (shown after rerun)
    if st.session_state.get("_group_toast_message"):
        st.toast(st.session_state._group_toast_message, icon="✅")
        st.session_state._group_toast_message = None

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
                        st.session_state._group_toast_message = "✅ Groupes mis à jour!"
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
                            st.session_state._group_toast_message = f"✅ Groupe modifié: {final_group}"
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
                    st.session_state._group_toast_message = f"✅ Groupe modifié: {final_group}"
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


def _render_reconciliation_tab(df: pd.DataFrame) -> None:
    """Render the reconciliation tab for cross-board matching.

    Auto-loads Ventes/Production data on first render, then runs
    reconciliation automatically.  Lines with the same police number
    and classification (Com/Boni/Sur-Com) are aggregated before comparison.
    """
    from src.utils.aggregator import SOURCE_BOARDS
    from src.utils.reconciler import Reconciler, ReconciliationStatus

    # Resolve board ID (always use vente_production)
    vp_config = SOURCE_BOARDS.get("vente_production")
    board_id = vp_config.board_id if vp_config else None

    if not board_id:
        st.error("Board Ventes/Production non configuré dans SOURCE_BOARDS.")
        return

    st.session_state.reconciliation_board_id = board_id

    # Resolve real board name from loaded boards list
    _boards = st.session_state.get("monday_boards") or []
    _real_name = None
    for _b in _boards:
        if str(_b.get("id")) == str(board_id):
            _real_name = _b["name"]
            break
    st.session_state.reconciliation_board_name = _real_name or vp_config.display_name

    # --- Auto-load sales data on first visit ---
    # Two-step load: fetch board data fast (no formula enrichment), then
    # enrich formula columns only for items whose # de Police appears in
    # the historical data.  This avoids the slow full-board FormulaValue
    # pass which is rate-limited to 10k values/min.
    if not st.session_state.reconciliation_sales_loaded:
        with st.spinner("Chargement des données Ventes/Production..."):
            try:
                pipeline = get_pipeline()
                bid = int(board_id)

                # Step 1: fast load without formula enrichment
                all_items = run_async(
                    pipeline.monday.extract_board_data(
                        bid, skip_formula_enrichment=True,
                    )
                )

                # Step 2: find which items match historical police numbers
                hist_police = set(
                    str(v).strip()
                    for v in df.get("# de Police", pd.Series(dtype=str))
                    if pd.notna(v) and str(v).strip()
                )

                matched_items = []
                for item in all_items:
                    for cv in item.get("column_values", []):
                        if (
                            cv.get("column", {}).get("title") == "# de Police"
                            and str(cv.get("text", "")).strip() in hist_police
                        ):
                            matched_items.append(item)
                            break

                # Step 3: enrich formulas only for matched items
                # Formula columns (Boni, Sur-Com, Total, Total Reçu, etc.)
                # return empty text/value in the standard query — only the
                # FormulaValue display_value (via enrichment) has actual data.
                enrichment_ok = False
                if matched_items:
                    run_async(
                        pipeline.monday.enrich_formula_columns(matched_items)
                    )
                    remaining = pipeline.monday._count_missing_formula_display_values(
                        matched_items
                    )
                    enrichment_ok = remaining == 0
                    if remaining > 0:
                        st.warning(
                            f"Enrichissement formules partiel : {remaining} "
                            f"valeur(s) manquante(s) sur {len(matched_items)} items. "
                            "Les colonnes formule (Boni, Sur-Com, Total, Total Reçu) "
                            "pourraient être incomplètes."
                        )

                sales_df = pipeline.monday.board_items_to_dataframe(
                    all_items, include_item_id=True,
                )
                st.session_state.reconciliation_sales_df = sales_df
                st.session_state.reconciliation_sales_loaded = True
                st.session_state.reconciliation_enabled = True
            except Exception as e:
                st.error(f"Impossible de charger Ventes/Production: {e}")
                return

    sales_df = st.session_state.reconciliation_sales_df

    # --- Warn if formula columns are all null ---
    # Note: Com is a regular numbers column (always has text).
    # Boni, Sur-Com, Total, Total Reçu are formula columns that require
    # FormulaValue enrichment to return data.
    _formula_cols = ["Boni", "Sur-Com", "Total", "Total Reçu"]
    _empty_formula_cols = [
        c for c in _formula_cols
        if c in sales_df.columns and sales_df[c].dropna().empty
    ]
    if _empty_formula_cols:
        st.warning(
            f"⚠️ Les colonnes formule **{', '.join(_empty_formula_cols)}** sont "
            "entièrement vides dans Ventes/Production. L'extraction des formules "
            "Monday.com a probablement échoué — les écarts seront marqués « — »."
        )

    # --- Run reconciliation ---
    reconciler = Reconciler()
    result = reconciler.reconcile(
        df, sales_df,
        allow_none_reference=bool(_empty_formula_cols),
    )
    st.session_state.reconciliation_result = result

    if result.total_hist_lines == 0:
        st.warning("Aucune ligne avec Statut « Payé » dans les données historiques.")
        return

    # --- Header metrics ---
    m_cols = st.columns(5)
    m_cols[0].metric("Lignes Payé", result.total_hist_lines)
    m_cols[1].metric("Groupes", result.total_groups)
    m_cols[2].metric("Vérifiées", result.passed)
    m_cols[3].metric("Écarts", result.flagged)
    m_cols[4].metric("Non trouvées", result.not_found)

    # --- Status highlight helper ---
    def _highlight_status(row, status_col="Statut Rapp."):
        status = row.get(status_col, "")
        if status == ReconciliationStatus.PASSED.value:
            return ["background-color: rgba(0, 200, 83, 0.1)"] * len(row)
        elif status == ReconciliationStatus.FLAGGED.value:
            return ["background-color: rgba(255, 152, 0, 0.1)"] * len(row)
        elif status == ReconciliationStatus.NOT_FOUND.value:
            return ["background-color: rgba(244, 67, 54, 0.1)"] * len(row)
        return [""] * len(row)

    # --- Sales / Production table ---
    sales_view_df = result.to_sales_view_dataframe(sales_df)

    if not sales_view_df.empty:
        st.subheader("Ventes / Production — Mise à jour Reçu")
        styled_sales = sales_view_df.style.apply(_highlight_status, axis=1)
        st.dataframe(styled_sales, width="stretch", height=400, hide_index=True)

    # --- Historical payments table ---
    hist_view_df = result.to_hist_view_dataframe(df)

    if not hist_view_df.empty:
        st.subheader("Paiement Historique — Mise à jour Conseiller / Vérifié")

        def _highlight_verified(row):
            verified = row.get("Verifié", "")
            if verified == "Verifié":
                return ["background-color: rgba(0, 200, 83, 0.1)"] * len(row)
            return [""] * len(row)

        styled_hist = hist_view_df.style.apply(_highlight_verified, axis=1)
        st.dataframe(styled_hist, width="stretch", height=400, hide_index=True)

    # --- Detail sections ---
    if result.not_found > 0:
        with st.expander(f"Polices non trouvées ({result.not_found})", expanded=False):
            for m in result.matches:
                if m.status == ReconciliationStatus.NOT_FOUND:
                    st.markdown(f"- `{m.police_number}` — {m.compagnie}")

    if result.flagged > 0:
        with st.expander(f"Écarts ({result.flagged})", expanded=False):
            for m in result.matches:
                if m.status == ReconciliationStatus.FLAGGED:
                    ecart_str = f"{m.ecart_pct:.1f}%" if m.ecart_pct is not None else "N/A"
                    label = m.classification.label if m.classification else "?"
                    lines = f" ({m.line_count} lignes)" if m.line_count > 1 else ""
                    st.markdown(
                        f"- `{m.police_number}` — {label}{lines}: "
                        f"Reçu={m.recu_amount}, Réf={m.reference_amount}, "
                        f"Écart={ecart_str} (seuil {m.threshold_pct:.0f}%)"
                    )

    # Overwrite warnings
    if result.passed > 0:
        sales_updates = result.get_sales_updates()
        overwrite_warnings = []
        for item_id, fields in sales_updates.items():
            for field_name, _ in fields.items():
                item_rows = sales_df[sales_df["item_id"].astype(str) == str(item_id)]
                if not item_rows.empty and field_name in item_rows.columns:
                    existing = item_rows.iloc[0].get(field_name)
                    if existing is not None and str(existing).strip() not in ("", "0", "0.0", "None"):
                        overwrite_warnings.append(
                            f"`{field_name}` de l'item {item_id} "
                            f"contient déjà `{existing}`"
                        )
        if overwrite_warnings:
            with st.expander(f"Valeurs existantes ({len(overwrite_warnings)})", expanded=True):
                st.warning(
                    "Les champs suivants seront écrasés lors du writeback :"
                )
                for w in overwrite_warnings:
                    st.markdown(f"- {w}")

    # =================================================================
    # UPLOAD SECTION — dual-board upload directly from reconciliation
    # =================================================================
    st.markdown("---")

    recon_upload_result = st.session_state.get("_recon_upload_result")

    if recon_upload_result is None:
        # --- Upload summary ---
        sales_board_name = st.session_state.get("reconciliation_board_name") or "Ventes/Production"
        hist_board_name = st.session_state._current_board_name or "Paiement Historique"

        dup_count = st.session_state.get('duplicate_count', 0)
        has_duplicates = '_is_duplicate' in df.columns and dup_count > 0
        upload_count = len(df) - dup_count if has_duplicates else len(df)

        st.markdown("### Upload vers Monday.com")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Board 1 — {hist_board_name}**")
            st.markdown(f"- {upload_count} nouvelles lignes")
            if has_duplicates:
                st.caption(f"({dup_count} doublon(s) exclus)")
            hist_passed = result.get_passed_hist_updates()
            if hist_passed:
                st.markdown(f"- {len(hist_passed)} lignes → Conseiller + Vérifié")

        with col2:
            st.markdown(f"**Board 2 — {sales_board_name}**")
            sales_updates = result.get_sales_updates()
            recu_fields: set[str] = set()
            for fields in sales_updates.values():
                recu_fields.update(fields.keys())
            st.markdown(
                f"- {len(sales_updates)} polices → "
                f"{', '.join(sorted(recu_fields)) or 'Reçu'}"
            )
            st.markdown(f"- {result.passed} correspondances vérifiées")
            if result.flagged > 0:
                st.markdown(f"- {result.flagged} écarts (non écrits)")

        # --- Buttons ---
        btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
        with btn_col2:
            if st.button(
                "Uploader les 2 boards",
                type="primary",
                key="recon_upload_btn",
                use_container_width=True,
            ):
                _execute_recon_upload(df, result)

    else:
        # --- Display upload result ---
        _render_recon_upload_result(recon_upload_result)

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Nouveau pipeline", key="recon_new_pipeline_btn", use_container_width=True):
                reset_pipeline()
                st.rerun()
        with btn_col2:
            board_id = st.session_state.selected_board_id
            st.link_button(
                "Ouvrir Monday.com",
                f"https://monday.com/boards/{board_id}",
                use_container_width=True,
            )

    # Reload button at the bottom, compact
    st.markdown("---")
    if st.button("Recharger Ventes/Production", key="recon_reload_btn"):
        st.session_state.reconciliation_sales_loaded = False
        st.session_state.reconciliation_sales_df = None
        st.session_state.reconciliation_result = None
        st.session_state._recon_upload_result = None
        st.rerun()


def _execute_recon_upload(df: pd.DataFrame, recon_result) -> None:
    """Execute dual-board upload from the reconciliation tab.

    1. Upload new historical items to Paiement Historique.
    2. Writeback Reçu 1/2/3 on Ventes/Production.
    3. Writeback Conseiller + Vérifié on newly uploaded Paiement Historique items.
    """
    from src.app.extraction.stage_3 import _execute_reconciliation_writeback

    pipeline = get_pipeline()
    board_id = st.session_state.selected_board_id

    # Filter out duplicates
    upload_df = df
    if '_is_duplicate' in df.columns:
        upload_df = df[~df['_is_duplicate']].copy()

    # Get unique groups
    if '_target_group' in upload_df.columns:
        unique_groups = upload_df['_target_group'].unique().tolist()
    else:
        unique_groups = ["Data"]

    total_items = len(upload_df)
    if total_items == 0:
        st.warning("Aucune donnée à uploader.")
        return

    progress_bar = st.progress(0, text="Board 1/2 — Paiement Historique: démarrage...")
    status_text = st.empty()

    items_uploaded = 0
    items_failed = 0
    all_errors: list[str] = []
    all_item_ids: list[str] = []
    all_index_to_item_id: dict[int, str] = {}
    items_processed = 0

    try:
        # --- Board 1: Upload new historical items ---
        for group_idx, group_name in enumerate(unique_groups):
            status_text.markdown(
                f"**Board 1/2** — Groupe {group_idx + 1}/{len(unique_groups)}: {group_name}"
            )

            if '_target_group' in upload_df.columns:
                group_df = upload_df[upload_df['_target_group'] == group_name].copy()
            else:
                group_df = upload_df.copy()

            export_df = group_df.drop(
                columns=[c for c in group_df.columns if c.startswith('_')],
                errors='ignore',
            )

            try:
                group_result = run_async(
                    pipeline.monday.get_or_create_group(board_id, str(group_name))
                )
                group_id = group_result.id if group_result.success else None
                if not group_id:
                    raise Exception(f"Impossible de créer le groupe: {group_result.error}")

                def on_progress(
                    current: int, total: int,
                    _offset=items_processed, _label=group_name,
                ) -> None:
                    overall = (_offset + current) / total_items
                    progress_bar.progress(
                        min(overall, 0.99),
                        text=f"Board 1/2 — {_label} ({_offset + current}/{total_items})",
                    )

                result = run_async(
                    pipeline.monday.upload_dataframe(
                        df=export_df,
                        board_id=board_id,
                        group_id=group_id,
                        progress_callback=on_progress,
                    )
                )

                items_uploaded += result.success
                items_failed += result.failed
                all_errors.extend(result.errors)
                all_item_ids.extend(result.item_ids)
                all_index_to_item_id.update(result.index_to_item_id)
                items_processed += len(export_df)

            except Exception as e:
                items_failed += len(export_df)
                items_processed += len(export_df)
                all_errors.append(f"Groupe {group_name}: {e}")

        progress_bar.progress(1.0, text="Board 1/2 — Paiement Historique: terminé!")

        # Store partial result so stage_3 writeback can append to it
        st.session_state.upload_result = {
            "total": total_items,
            "success": items_uploaded,
            "failed": items_failed,
            "errors": all_errors,
            "groups": len(unique_groups),
            "item_ids": all_item_ids,
        }

        # --- Board 2: Reconciliation writeback ---
        _execute_reconciliation_writeback(
            pipeline, board_id, recon_result, all_index_to_item_id, df,
            progress_bar, status_text,
        )

        st.session_state._recon_upload_result = st.session_state.upload_result

    except Exception as e:
        st.error(f"Échec de l'upload: {e}")
        st.session_state._recon_upload_result = {
            "total": total_items,
            "success": items_uploaded,
            "failed": items_failed + (total_items - items_processed),
            "errors": all_errors + [str(e)],
            "groups": 0,
        }

    finally:
        progress_bar.empty()
        status_text.empty()
        st.rerun()


def _render_recon_upload_result(result: dict) -> None:
    """Display dual-board upload result inside the reconciliation tab."""
    total = result.get("total", 0)
    success = result.get("success", 0)
    failed = result.get("failed", 0)
    groups = result.get("groups", 1)
    recon_wb = result.get("reconciliation_writeback")

    st.markdown("### Résultat de l'upload")

    col1, col2 = st.columns(2)

    with col1:
        board_name = st.session_state._current_board_name or "Paiement Historique"
        st.markdown(f"**Board 1 — {board_name}**")
        if failed == 0:
            st.success(f"{success}/{total} lignes uploadées dans {groups} groupe(s)")
        else:
            st.warning(f"{success}/{total} uploadées. {failed} en erreur.")

        if recon_wb:
            hist_updated = recon_wb.get("hist_updated", 0)
            if hist_updated > 0:
                st.success(f"{hist_updated} lignes → Conseiller + Vérifié")

    with col2:
        sales_name = st.session_state.get("reconciliation_board_name") or "Ventes/Production"

        st.markdown(f"**Board 2 — {sales_name}**")
        if recon_wb:
            sales_updated = recon_wb.get("sales_updated", 0)
            if sales_updated > 0:
                st.success(f"{sales_updated} polices → Reçu mis à jour")
            else:
                st.info("Aucune mise à jour")
        else:
            st.info("Writeback non exécuté")

    # Errors
    all_errors = result.get("errors", [])
    wb_errors = recon_wb.get("wb_error_details", []) if recon_wb else []
    total_errors = all_errors + wb_errors
    if total_errors:
        with st.expander(f"Erreurs ({len(total_errors)})"):
            for error in total_errors[:10]:
                st.error(error)


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
