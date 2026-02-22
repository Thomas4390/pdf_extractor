"""
Stage 3: Upload - Upload data to Monday.com.

Provides the interface for confirming and executing the upload
of extracted data to Monday.com boards.
"""

import pandas as pd
import streamlit as st

from src.app.state import get_pipeline, reset_pipeline
from src.app.utils.async_helpers import run_async
from src.app.utils.navigation import render_stepper, render_breadcrumb
from src.app.components import (
    render_upload_dashboard,
    render_success_box,
)


def render_stage_3() -> None:
    """Render upload stage with modern styling."""
    st.markdown("## Pipeline de Commissions")
    render_stepper()
    render_breadcrumb()

    df = st.session_state.combined_data

    if df is None or df.empty:
        st.error("Aucune donnée à uploader")
        if st.button("Recommencer"):
            reset_pipeline()
            st.rerun()
        return

    if st.session_state.data_modified:
        st.warning("Upload de données modifiées")

    # Count duplicates that will be excluded
    dup_count = st.session_state.get('duplicate_count', 0)
    has_duplicates = '_is_duplicate' in df.columns and dup_count > 0
    upload_count = len(df) - dup_count if has_duplicates else len(df)

    # Summary Dashboard (similar to Stage 2)
    unique_groups = df['_target_group'].unique() if '_target_group' in df.columns else []
    board_name = st.session_state._current_board_name or "N/A"

    render_upload_dashboard(
        item_count=upload_count,
        board_name=board_name,
        group_count=len(unique_groups) if len(unique_groups) > 0 else 1,
        file_count=len(st.session_state.extraction_results)
    )

    if has_duplicates:
        st.warning(f"⚠️ **{dup_count} doublon(s)** seront exclus de l'upload (# de Police déjà sur le board).")

    # Target group display
    if '_target_group' in df.columns and len(unique_groups) > 0:
        if len(unique_groups) == 1:
            st.info(f"📁 **Groupe cible:** {unique_groups[0]}")
        else:
            st.info(f"📁 **Groupes cibles:** {', '.join(str(g) for g in unique_groups)}")
            with st.expander("Détail par groupe", expanded=False):
                for group in unique_groups:
                    group_count = len(df[df['_target_group'] == group])
                    st.markdown(f"**{group}**: {group_count} items")

    # Data preview before upload
    with st.expander("📋 Aperçu des données à envoyer", expanded=True):
        # Remove internal columns for display
        display_df = df.drop(columns=[c for c in df.columns if c.startswith('_')], errors='ignore')
        st.dataframe(display_df, width="stretch", height=300)
        st.caption(f"📊 {len(display_df)} lignes × {len(display_df.columns)} colonnes")

    st.markdown("---")

    # Upload process
    if st.session_state.upload_result is None:
        # Check if upload is in progress
        if st.session_state.is_uploading:
            # Execute the upload (after rerun from button click)
            execute_upload(df)
        else:
            st.info("Les données vont être uploadées vers Monday.com.")

            footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

            with footer_col1:
                if st.button("Retour", width="stretch"):
                    st.session_state.stage = 2
                    st.rerun()

            with footer_col3:
                if st.button(
                    "🚀 Confirmer l'upload",
                    type="primary",
                    width="stretch"
                ):
                    # Set uploading state and rerun to hide the button
                    st.session_state.is_uploading = True
                    st.rerun()
    else:
        render_upload_result(st.session_state.upload_result)

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("Nouveau pipeline", width="stretch"):
                reset_pipeline()
                st.rerun()
        with col3:
            board_id = st.session_state.selected_board_id
            st.link_button("Ouvrir Monday.com", f"https://monday.com/boards/{board_id}", width="stretch")


def execute_upload(df: pd.DataFrame) -> None:
    """Execute the upload to Monday.com."""
    st.session_state.upload_result = None

    # Filter out duplicates before upload
    if '_is_duplicate' in df.columns:
        df = df[~df['_is_duplicate']].copy()

    pipeline = get_pipeline()
    board_id = st.session_state.selected_board_id

    progress_bar = st.progress(0, text="Démarrage de l'upload...")
    status_text = st.empty()

    try:
        # Get unique groups
        if '_target_group' in df.columns:
            unique_groups = df['_target_group'].unique().tolist()
        else:
            unique_groups = ["Data"]

        total_items = len(df)
        if total_items == 0:
            st.warning("Aucune donnée à uploader.")
            return

        items_uploaded = 0
        items_failed = 0
        all_errors = []

        # Track items processed so far for progress display
        items_processed_before_group = 0

        for group_idx, group_name in enumerate(unique_groups):
            status_text.markdown(f"📁 **Groupe {group_idx + 1}/{len(unique_groups)}:** {group_name}")

            # Filter data for this group
            if '_target_group' in df.columns:
                group_df = df[df['_target_group'] == group_name].copy()
            else:
                group_df = df.copy()

            # Remove internal columns
            export_df = group_df.drop(columns=[c for c in group_df.columns if c.startswith('_')], errors='ignore')

            try:
                # Create or get group
                group_result = run_async(
                    pipeline.monday.get_or_create_group(board_id, str(group_name))
                )
                group_id = group_result.id if group_result.success else None

                if not group_id:
                    raise Exception(f"Impossible de créer le groupe: {group_result.error}")

                # Progress callback - show overall progress
                def on_progress(current: int, total: int) -> None:
                    # Calculate overall progress across all groups
                    overall_current = items_processed_before_group + current
                    overall_progress = overall_current / total_items
                    progress_bar.progress(
                        min(overall_progress, 0.99),
                        text=f"Upload: {group_name} ({overall_current}/{total_items})"
                    )

                # Upload
                result = run_async(
                    pipeline.monday.upload_dataframe(
                        df=export_df,
                        board_id=board_id,
                        group_id=group_id,
                        progress_callback=on_progress
                    )
                )

                items_uploaded += result.success
                items_failed += result.failed
                all_errors.extend(result.errors)

                # Update items processed before next group
                items_processed_before_group += len(export_df)

            except Exception as e:
                items_failed += len(export_df)
                items_processed_before_group += len(export_df)
                all_errors.append(f"Groupe {group_name}: {str(e)}")

        progress_bar.progress(1.0, text=f"Upload terminé! ({total_items}/{total_items})")
        status_text.empty()

        # Store result
        st.session_state.upload_result = {
            "total": total_items,
            "success": items_uploaded,
            "failed": items_failed,
            "errors": all_errors,
            "groups": len(unique_groups),
        }

        if items_failed == 0:
            st.success(f"{items_uploaded} éléments uploadés dans {len(unique_groups)} groupe(s)!")
        else:
            st.warning(f"{items_uploaded}/{total_items} uploadés. {items_failed} en erreur.")

    except Exception as e:
        st.error(f"Échec de l'upload: {e}")
        st.session_state.upload_result = {
            "total": len(df),
            "success": 0,
            "failed": len(df),
            "errors": [str(e)],
            "groups": 0,
        }

    finally:
        st.session_state.is_uploading = False
        progress_bar.empty()
        st.rerun()


def render_upload_result(result: dict) -> None:
    """Render upload result summary with modern styling."""
    total = result.get("total", 0)
    success = result.get("success", 0)
    failed = result.get("failed", 0)
    groups = result.get("groups", 1)

    if failed == 0:
        render_success_box(
            title="Upload réussi!",
            message=f"{success}/{total} éléments uploadés dans {groups} groupe(s)."
        )
    else:
        st.warning(f"{success}/{total} éléments uploadés. {failed} en erreur.")

    errors = result.get("errors", [])
    if errors:
        with st.expander("Voir les erreurs"):
            for error in errors[:10]:
                st.error(error)
