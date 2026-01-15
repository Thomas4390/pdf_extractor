"""
Navigation UI components.

Provides breadcrumb and stepper components for the multi-stage wizard.
"""

import streamlit as st


def render_breadcrumb() -> None:
    """Render breadcrumb navigation showing current context."""
    parts = ["Accueil"]

    if st.session_state.selected_source:
        parts.append(st.session_state.selected_source)

    if st.session_state.uploaded_files:
        file_count = len(st.session_state.uploaded_files)
        parts.append(f"{file_count} fichier{'s' if file_count > 1 else ''}")

    if st.session_state._current_board_name:
        board_name = st.session_state._current_board_name
        if len(board_name) > 25:
            board_name = board_name[:22] + "..."
        parts.append(f'Board "{board_name}"')

    breadcrumb_html = '<div class="breadcrumb">'
    for i, part in enumerate(parts):
        is_active = i == len(parts) - 1
        breadcrumb_html += f'<span class="breadcrumb-item{"" if not is_active else " active"}">{part}</span>'
        if i < len(parts) - 1:
            breadcrumb_html += '<span class="breadcrumb-separator">›</span>'
    breadcrumb_html += '</div>'

    st.markdown(breadcrumb_html, unsafe_allow_html=True)


def render_stepper() -> None:
    """Render the clickable progress stepper in main content area."""
    stages = [
        ("1", "Configuration", "📁"),
        ("2", "Prévisualisation", "🔍"),
        ("3", "Upload", "☁️")
    ]

    cols = st.columns(3)
    for i, (num, name, icon) in enumerate(stages):
        stage_num = i + 1
        with cols[i]:
            is_current = stage_num == st.session_state.stage
            is_completed = stage_num < st.session_state.stage
            is_future = stage_num > st.session_state.stage

            # Determine CSS class
            if is_current:
                css_class = "current"
            elif is_completed:
                css_class = "completed"
            else:
                css_class = "future"

            # Render step visual
            display_icon = "✅" if is_completed else icon
            st.markdown(f"""
            <div class="stepper-step {css_class}">
                <div class="step-icon">{display_icon}</div>
                <div class="step-label">{name}</div>
            </div>
            """, unsafe_allow_html=True)

            # Add clickable button for completed stages
            if is_completed:
                if st.button(f"← Retour", key=f"stepper_nav_{stage_num}", width="stretch"):
                    # Reset extraction state when going back to stage 1
                    if stage_num == 1:
                        st.session_state.combined_data = None
                        st.session_state.extraction_results = {}
                        st.session_state.batch_result = None
                        st.session_state.extraction_usage = None
                        st.session_state.upload_result = None
                        st.session_state.file_group_overrides = {}
                    st.session_state.stage = stage_num
                    st.rerun()
