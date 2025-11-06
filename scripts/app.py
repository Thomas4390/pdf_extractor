"""
Streamlit Application - Insurance Commission Data Pipeline
===========================================================

Application web pour extraire, visualiser et uploader les données
de commissions d'assurance vers Monday.com.

Author: Thomas
Date: 2025-10-30
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path
from io import StringIO
import sys

# Import pipeline components
from main import (
    InsuranceCommissionPipeline,
    PipelineConfig,
    InsuranceSource,
    ColorPrint
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Insurance Commission Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'stage' not in st.session_state:
        st.session_state.stage = 1

    if 'pdf_file' not in st.session_state:
        st.session_state.pdf_file = None

    if 'pdf_path' not in st.session_state:
        st.session_state.pdf_path = None

    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None

    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None

    if 'config' not in st.session_state:
        st.session_state.config = None

    if 'upload_results' not in st.session_state:
        st.session_state.upload_results = None

    if 'data_modified' not in st.session_state:
        st.session_state.data_modified = False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to a temporary location and return the path.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Path to the saved file
    """
    # Create temp directory if it doesn't exist
    temp_dir = Path("./temp")
    temp_dir.mkdir(exist_ok=True)

    # Save file
    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(file_path)


def cleanup_temp_file(file_path: str = None):
    """
    Clean up temporary file.

    Args:
        file_path: Path to file to delete. If None, uses session state pdf_path
    """
    if file_path is None:
        file_path = st.session_state.get('pdf_path')

    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            # Silent fail - don't interrupt user experience
            pass


def reset_pipeline():
    """Reset pipeline state to start over."""
    # Clean up temporary file before reset
    cleanup_temp_file()

    st.session_state.stage = 1
    st.session_state.pdf_file = None
    st.session_state.pdf_path = None
    st.session_state.extracted_data = None
    st.session_state.pipeline = None
    st.session_state.config = None
    st.session_state.upload_results = None
    st.session_state.data_modified = False


# =============================================================================
# STAGE 1: CONFIGURATION AND UPLOAD
# =============================================================================

def render_stage_1():
    """Render configuration and file upload stage."""
    st.title("📊 Insurance Commission Data Pipeline")
    st.markdown("---")

    st.header("📁 Étape 1: Configuration et Upload")

    with st.form("config_form"):
        # PDF Upload
        st.subheader("1️⃣ Upload du PDF")
        uploaded_file = st.file_uploader(
            "Déposez ou sélectionnez votre fichier PDF",
            type=['pdf'],
            help="Fichier PDF contenant les données de commissions"
        )

        st.markdown("---")

        # Source Selection
        st.subheader("2️⃣ Source des Données")
        source = st.selectbox(
            "Sélectionnez la source d'assurance",
            options=["UV", "IDC", "ASSOMPTION"],
            help="Type de document PDF à traiter"
        )

        st.markdown("---")

        # Monday.com Configuration
        st.subheader("3️⃣ Configuration Monday.com")

        col1, col2 = st.columns(2)

        with col1:
            monday_api_key = st.text_input(
                "Clé API Monday.com",
                type="password",
                help="Votre clé API Monday.com pour l'authentification",
                key="monday_api_key_input"
            )

            board_name_input = st.text_input(
                "Nom du Board",
                placeholder=f"Ex: Commissions {source}",
                help="Nom du board Monday.com (sera créé s'il n'existe pas). Laissez vide pour utiliser le nom par défaut.",
                key="board_name_input"
            )

            # Show what will be used
            if board_name_input and board_name_input.strip():
                st.caption(f"📋 Nom du board: **{board_name_input.strip()}**")
            else:
                st.caption(f"📋 Nom par défaut sera utilisé: **Commissions {source}**")

        with col2:
            month_group = st.text_input(
                "Groupe de Mois (optionnel)",
                value="",
                placeholder="Ex: Octobre 2025",
                help="Nom du groupe pour organiser les données (optionnel)"
            )

            col_reuse1, col_reuse2 = st.columns(2)
            with col_reuse1:
                reuse_board = st.checkbox(
                    "Réutiliser board existant",
                    value=True,
                    help="Si coché, utilisera le board existant avec le même nom"
                )
            with col_reuse2:
                reuse_group = st.checkbox(
                    "Réutiliser groupe existant",
                    value=True,
                    help="Si coché, utilisera le groupe existant avec le même nom"
                )

        st.markdown("---")

        # Submit button
        submitted = st.form_submit_button(
            "🚀 Extraire les données",
            use_container_width=True,
            type="primary"
        )

        if submitted:
            # Validation
            errors = []

            if not uploaded_file:
                errors.append("❌ Veuillez uploader un fichier PDF")

            if not monday_api_key:
                errors.append("❌ Veuillez fournir une clé API Monday.com")

            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Save uploaded file
                pdf_path = save_uploaded_file(uploaded_file)

                # Determine final board name - CRITICAL: Use the exact input from user
                # Access via session_state for reliability in forms
                board_name_from_state = st.session_state.get('board_name_input', '')

                # If user entered something, use it exactly as-is
                # If user left it empty, use default
                if board_name_from_state and board_name_from_state.strip():
                    final_board_name = board_name_from_state.strip()
                else:
                    final_board_name = f"Commissions {source}"

                # Debug: Show what was captured (temporary for debugging)
                st.info(f"""
                **Debug Info:**
                - Source sélectionnée: `{source}`
                - Board name (direct var): `'{board_name_input}'`
                - Board name (session_state): `'{board_name_from_state}'`
                - Board name final: `'{final_board_name}'`
                - Month group: `'{month_group}'`
                """)

                # Create configuration
                try:
                    config = PipelineConfig(
                        source=InsuranceSource(source),
                        pdf_path=pdf_path,
                        month_group=month_group if month_group else None,
                        board_name=final_board_name,  # This should be exactly what user entered
                        monday_api_key=monday_api_key,
                        output_dir="./results",
                        reuse_board=reuse_board,
                        reuse_group=reuse_group
                    )


                    # Store in session state
                    st.session_state.pdf_file = uploaded_file
                    st.session_state.pdf_path = pdf_path
                    st.session_state.config = config

                    # Move to next stage
                    st.session_state.stage = 2
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Erreur de configuration: {e}")


# =============================================================================
# STAGE 2: EXTRACTION AND PREVIEW
# =============================================================================

def render_stage_2():
    """Render data extraction and preview stage."""
    st.title("📊 Insurance Commission Data Pipeline")
    st.markdown("---")

    st.header("🔍 Étape 2: Extraction et Prévisualisation")

    # Show configuration summary
    config = st.session_state.config

    with st.expander("📋 Résumé de la Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Source", config.source.value)
            st.metric("Fichier PDF", Path(config.pdf_path).name)
        with col2:
            st.metric("Board Monday.com", config.board_name)
            st.metric("Groupe de Mois", config.month_group or "Aucun")
        with col3:
            st.metric("Réutiliser Board", "✅" if config.reuse_board else "❌")
            st.metric("Réutiliser Groupe", "✅" if config.reuse_group else "❌")

    st.markdown("---")

    # Extract data if not already done
    if st.session_state.extracted_data is None:
        with st.spinner("🔄 Extraction des données en cours..."):
            try:
                # Create pipeline
                pipeline = InsuranceCommissionPipeline(config)

                # Execute Steps 1 and 2
                success_step1 = pipeline._step1_extract_data()
                if not success_step1:
                    st.error("❌ Échec de l'extraction des données")
                    if st.button("🔄 Recommencer"):
                        reset_pipeline()
                        st.rerun()
                    return

                success_step2 = pipeline._step2_process_data()
                if not success_step2:
                    st.error("❌ Échec du traitement des données")
                    if st.button("🔄 Recommencer"):
                        reset_pipeline()
                        st.rerun()
                    return

                # Store results
                st.session_state.extracted_data = pipeline.final_data
                st.session_state.pipeline = pipeline

                st.success("✅ Extraction réussie!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Erreur lors de l'extraction: {e}")
                with st.expander("Détails de l'erreur"):
                    st.exception(e)
                if st.button("🔄 Recommencer"):
                    reset_pipeline()
                    st.rerun()
                return

    # Display extracted data
    df = st.session_state.extracted_data

    if df is not None and not df.empty:
        # Show modification status if data was modified
        if st.session_state.data_modified:
            st.info("ℹ️ **Données modifiées** - Vous utilisez un fichier Excel uploadé au lieu des données extraites du PDF.")

        # Statistics
        st.subheader("📊 Statistiques")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Nombre de lignes", len(df))
        with col2:
            st.metric("Nombre de colonnes", len(df.columns))
        with col3:
            # Count non-null values in key columns
            if 'contract_number' in df.columns:
                non_null = df['contract_number'].notna().sum()
                st.metric("Contrats valides", non_null)
        with col4:
            # Check for duplicates
            duplicates = df.duplicated().sum()
            st.metric("Doublons", duplicates)

        st.markdown("---")

        # Data preview
        st.subheader("📋 Aperçu des Données")

        # Show column info
        with st.expander("ℹ️ Information sur les Colonnes"):
            col_info = pd.DataFrame({
                'Colonne': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.notna().sum(),
                'Null': df.isna().sum()
            })
            st.dataframe(col_info, use_container_width=True)

        # Interactive data viewer
        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )

        # Download option
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Télécharger les données (CSV)",
            data=csv,
            file_name=f"commissions_{config.source.value}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        st.markdown("---")

        # Excel Upload Option for manual corrections
        with st.expander("📤 Uploader un fichier Excel modifié (optionnel)", expanded=False):
            st.info("""
            **Modifier les données avant l'upload**

            Vous pouvez télécharger les données en CSV, les modifier dans Excel,
            puis uploader le fichier modifié ici. Le fichier uploadé remplacera
            les données extraites avant l'upload vers Monday.com.

            ⚠️ **Important**: Le fichier Excel doit contenir toutes les colonnes du tableau ci-dessus.
            """)

            excel_file = st.file_uploader(
                "Sélectionnez votre fichier Excel modifié",
                type=['xlsx', 'xls', 'csv'],
                help="Fichier Excel ou CSV avec les données corrigées",
                key="excel_upload"
            )

            if excel_file is not None:
                try:
                    # Read the uploaded file
                    if excel_file.name.endswith('.csv'):
                        uploaded_df = pd.read_csv(excel_file)
                    else:
                        uploaded_df = pd.read_excel(excel_file)

                    st.success(f"✅ Fichier chargé: {excel_file.name}")

                    # Validate columns
                    required_columns = set(df.columns)
                    uploaded_columns = set(uploaded_df.columns)

                    missing_columns = required_columns - uploaded_columns
                    extra_columns = uploaded_columns - required_columns

                    if missing_columns:
                        st.error(f"❌ Colonnes manquantes: {', '.join(missing_columns)}")
                    elif extra_columns:
                        st.warning(f"⚠️ Colonnes supplémentaires (seront ignorées): {', '.join(extra_columns)}")
                        # Keep only required columns
                        uploaded_df = uploaded_df[list(required_columns)]

                    # Show preview
                    st.subheader("📋 Aperçu du fichier uploadé")
                    st.dataframe(uploaded_df.head(10), use_container_width=True)

                    # Statistics comparison
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Lignes - Original", len(df))
                        st.metric("Lignes - Modifié", len(uploaded_df), delta=len(uploaded_df) - len(df))
                    with col2:
                        st.metric("Colonnes - Original", len(df.columns))
                        st.metric("Colonnes - Modifié", len(uploaded_df.columns))
                    with col3:
                        if not missing_columns:
                            st.success("✅ Structure valide")
                        else:
                            st.error("❌ Structure invalide")

                    # Button to replace data
                    if not missing_columns:
                        if st.button("✅ Utiliser ce fichier pour l'upload", type="primary", use_container_width=True):
                            st.session_state.extracted_data = uploaded_df
                            st.session_state.pipeline.final_data = uploaded_df
                            st.session_state.data_modified = True
                            st.success("✅ Données remplacées par le fichier uploadé!")
                            st.rerun()

                except Exception as e:
                    st.error(f"❌ Erreur lors de la lecture du fichier: {e}")

        st.markdown("---")

        # Action buttons
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.write("")  # Spacer

        with col2:
            if st.button("⬅️ Retour", use_container_width=True):
                reset_pipeline()
                st.rerun()

        with col3:
            if st.button(
                    "➡️ Continuer vers Monday.com",
                    use_container_width=True,
                    type="primary"
            ):
                st.session_state.stage = 3
                st.rerun()

    else:
        st.error("❌ Aucune donnée extraite")
        if st.button("🔄 Recommencer"):
            reset_pipeline()
            st.rerun()


# =============================================================================
# STAGE 3: UPLOAD TO MONDAY.COM
# =============================================================================

def render_stage_3():
    """Render Monday.com upload stage."""
    st.title("📊 Insurance Commission Data Pipeline")
    st.markdown("---")

    st.header("☁️ Étape 3: Upload vers Monday.com")

    # Show data summary
    df = st.session_state.extracted_data
    config = st.session_state.config

    # Show modification status if data was modified
    if st.session_state.data_modified:
        st.warning("⚠️ **Attention** - Vous allez uploader des données modifiées (fichier Excel uploadé)")

    st.subheader("📊 Résumé")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lignes à uploader", len(df))
    with col2:
        st.metric("Board cible", config.board_name)
    with col3:
        st.metric("Groupe", config.month_group or "Défaut")

    st.markdown("---")

    # Upload process
    if st.session_state.upload_results is None:
        st.warning("⚠️ Les données vont être uploadées vers Monday.com. Cette opération est irréversible.")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.stage = 2
                st.rerun()

        with col2:
            if st.button(
                    "🚀 Uploader vers Monday.com",
                    use_container_width=True,
                    type="primary"
            ):
                with st.spinner("☁️ Upload en cours vers Monday.com..."):
                    try:
                        pipeline = st.session_state.pipeline

                        # Execute Steps 3 and 4
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Step 3: Setup Monday board
                        status_text.text("Configuration du board Monday.com...")
                        progress_bar.progress(25)

                        try:
                            success_step3 = pipeline._step3_setup_monday_board()

                            if not success_step3:
                                st.error("❌ Échec de la configuration du board")
                                st.error("Vérifiez que votre clé API Monday.com est valide et que vous avez les permissions nécessaires.")
                                return
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la configuration du board: {e}")
                            with st.expander("Détails de l'erreur"):
                                st.exception(e)
                            return

                        progress_bar.progress(50)

                        # Step 4: Upload data with real-time progress
                        status_text.text("Upload des données vers Monday.com...")

                        # Prepare items for batch creation
                        items_to_create = pipeline._prepare_monday_items(df)
                        total_items = len(items_to_create)

                        # Upload in batches with progress updates
                        batch_size = 10  # Upload 10 items at a time
                        results = []

                        for i in range(0, total_items, batch_size):
                            batch = items_to_create[i:i + batch_size]
                            batch_num = (i // batch_size) + 1
                            total_batches = (total_items + batch_size - 1) // batch_size

                            # Update status
                            status_text.text(f"Upload vers Monday.com... ({i + len(batch)}/{total_items} items)")

                            # Upload batch
                            batch_results = pipeline.monday_client.create_items_batch(
                                board_id=pipeline.board_id,
                                items=batch,
                                group_id=pipeline.group_id
                            )
                            results.extend(batch_results)

                            # Update progress bar: 50% to 100%
                            progress_percent = 50 + int(50 * (i + len(batch)) / total_items)
                            progress_bar.progress(min(progress_percent, 100))

                        progress_bar.progress(100)
                        status_text.empty()

                        # Analyze results
                        successful = sum(1 for r in results if r.success)
                        failed = len(results) - successful

                        if successful > 0:
                            st.session_state.upload_results = {
                                'success': True,
                                'board_id': pipeline.board_id,
                                'group_id': pipeline.group_id,
                                'items_uploaded': successful,
                                'items_failed': failed
                            }
                            st.success(f"✅ Upload réussi! ({successful}/{total_items} items)")
                            if failed > 0:
                                st.warning(f"⚠️ {failed} items ont échoué")
                            st.rerun()
                        else:
                            st.error("❌ Échec de l'upload des données - Aucun item n'a été uploadé")

                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'upload: {e}")
                        with st.expander("Détails de l'erreur"):
                            st.exception(e)

    else:
        # Show results
        results = st.session_state.upload_results

        if results['success']:
            st.success("✅ Upload terminé avec succès!")

            # Results details
            st.subheader("📈 Résultats")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Items uploadés", results['items_uploaded'])
            with col2:
                failed_count = results.get('items_failed', 0)
                st.metric("Items échoués", failed_count, delta=None if failed_count == 0 else -failed_count)
            with col3:
                st.metric("Board ID", results['board_id'])
            with col4:
                st.metric("Group ID", results['group_id'] or "Défaut")

            st.markdown("---")

            # Success message
            st.balloons()

            # Success message with details
            failed_count = results.get('items_failed', 0)
            if failed_count == 0:
                st.info(f"""
                🎉 **Upload réussi!**

                Les données ont été uploadées vers Monday.com avec succès.
                - Board: **{config.board_name}**
                - Groupe: **{config.month_group or 'Groupe par défaut'}**
                - Items créés: **{results['items_uploaded']}**

                Vous pouvez maintenant consulter vos données dans Monday.com.
                """)
            else:
                st.warning(f"""
                ⚠️ **Upload partiellement réussi**

                Les données ont été uploadées vers Monday.com avec quelques erreurs.
                - Board: **{config.board_name}**
                - Groupe: **{config.month_group or 'Groupe par défaut'}**
                - Items créés avec succès: **{results['items_uploaded']}**
                - Items échoués: **{failed_count}**

                Vérifiez vos données et permissions Monday.com.
                """)

            st.markdown("---")

            # Action buttons
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🔄 Nouveau Pipeline", use_container_width=True, type="primary"):
                    reset_pipeline()
                    st.rerun()

            with col2:
                # Link to Monday.com (if board_id is available)
                if results['board_id']:
                    monday_url = f"https://monday.com/boards/{results['board_id']}"
                    st.markdown(f"[🔗 Ouvrir dans Monday.com]({monday_url})")

        else:
            st.error("❌ L'upload a échoué")
            if st.button("🔄 Recommencer"):
                reset_pipeline()
                st.rerun()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        st.markdown("---")

        # Current stage indicator
        stage_names = {
            1: "📁 Configuration",
            2: "🔍 Prévisualisation",
            3: "☁️ Upload"
        }

        st.subheader("Étapes du Pipeline")
        for stage_num, stage_name in stage_names.items():
            if stage_num == st.session_state.stage:
                st.markdown(f"**➡️ {stage_name}**")
            elif stage_num < st.session_state.stage:
                st.markdown(f"✅ {stage_name}")
            else:
                st.markdown(f"⚪ {stage_name}")

        st.markdown("---")

        # Information section
        st.subheader("ℹ️ Informations")
        st.info("""
        **Pipeline de Commissions d'Assurance**

        Cette application permet de:
        1. Extraire les données de PDF
        2. Visualiser les données extraites
        3. Uploader vers Monday.com

        **Sources supportées:**
        - UV Assurance
        - IDC
        - Assomption Vie
        """)

        st.markdown("---")

        # Reset button
        if st.button("🔄 Réinitialiser", use_container_width=True):
            reset_pipeline()
            st.rerun()

    # Render appropriate stage
    if st.session_state.stage == 1:
        render_stage_1()
    elif st.session_state.stage == 2:
        render_stage_2()
    elif st.session_state.stage == 3:
        render_stage_3()


if __name__ == "__main__":
    main()