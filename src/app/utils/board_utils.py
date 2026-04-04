"""
Board utility functions for Monday.com integration.

Provides helpers for sorting, filtering, and loading boards.
"""

import re
import threading
from typing import Optional

import pandas as pd
import streamlit as st

from src.app.utils.async_helpers import run_async

# Global storage for background loading results (thread-safe)
_background_agg_data: dict = {}
_background_agg_loading: bool = False
_background_agg_error: Optional[str] = None
_background_agg_progress: dict = {}
# Date range used for the last successful load. Used to invalidate the
# cache when the user switches to a period outside the loaded window.
_background_agg_date_range: Optional[tuple[str, str]] = None


def get_board_name_by_id(boards: list[dict], board_id: int, fallback: str = "Unknown") -> str:
    """Look up a board name by its ID."""
    for b in boards:
        if int(b["id"]) == board_id:
            return b["name"]
    return fallback


def sort_and_filter_boards(boards: list, search_query: str = "") -> list:
    """Sort boards with priority keywords first and filter by search query."""
    if not boards:
        return []

    filtered_boards = boards
    if search_query and search_query.strip():
        search_lower = search_query.lower().strip()
        filtered_boards = [b for b in boards if search_lower in b['name'].lower()]

    priority_1_keywords = ['paiement', 'historique']
    priority_2_keywords = ['vente', 'production']

    def get_priority(board_name: str) -> tuple:
        name_lower = board_name.lower()
        if any(kw in name_lower for kw in priority_1_keywords):
            return (0, name_lower)
        if any(kw in name_lower for kw in priority_2_keywords):
            return (1, name_lower)
        return (2, name_lower)

    return sorted(filtered_boards, key=lambda b: get_priority(b['name']))


def detect_board_type_from_name(board_name: str) -> str:
    """Detect the board type based on regex patterns in the board name."""
    if not board_name:
        return "Paiements Historiques"

    name_lower = board_name.lower()

    sales_patterns = [
        r'vente[s]?', r'production[s]?', r'sales?', r'prod\b',
        r'commercial', r'soumis', r'proposition[s]?',
    ]

    payment_patterns = [
        r'paiement[s]?', r'historique[s]?', r'payment[s]?', r'history',
        r'hist\b', r'reçu[s]?', r'commission[s]?', r'statement[s]?',
    ]

    for pattern in sales_patterns:
        if re.search(pattern, name_lower):
            return "Ventes et Production"

    for pattern in payment_patterns:
        if re.search(pattern, name_lower):
            return "Paiements Historiques"

    return "Paiements Historiques"


def load_boards_async(force_rerun: bool = False) -> None:
    """Load Monday.com boards automatically when API key is set."""
    # Import here to avoid circular imports
    from src.app.state import get_pipeline

    if (st.session_state.monday_api_key and
        st.session_state.monday_boards is None and
        not st.session_state.boards_loading):
        try:
            st.session_state.boards_loading = True
            st.session_state.boards_error = None
            pipeline = get_pipeline()
            if pipeline.monday_configured:
                boards = run_async(pipeline.monday.list_boards())
                st.session_state.monday_boards = boards
            st.session_state.boards_loading = False
            if force_rerun:
                st.rerun()
        except Exception as e:
            st.session_state.boards_loading = False
            st.session_state.boards_error = str(e)


def _load_aggregation_data_thread(
    api_key: str,
    selected_sources: dict,
    board_names: dict[int, str] | None = None,
    date_range: Optional[tuple[str, str]] = None,
) -> None:
    """
    Background thread function to load aggregation source data.

    Loads all source boards and advisor history concurrently using
    asyncio.gather for significantly faster loading. Formula enrichment
    is skipped (not needed for aggregation) to avoid rate limits.

    Args:
        api_key: Monday.com API key
        selected_sources: Dict of {source_key: board_id}
        board_names: Optional dict of {board_id: board_name} for display
        date_range: Optional (start_iso, end_iso) tuple in "YYYY-MM-DD" format
            used to push date filtering server-side via Monday query_params.
            Falls back to client-side filtering if the board's date column
            can't be resolved.
    """
    global _background_agg_data, _background_agg_loading, _background_agg_error
    global _background_agg_progress, _background_agg_date_range

    import asyncio

    from src.clients.monday import MondayClient
    from src.utils.aggregator import METRICS_BOARD_CONFIG, SOURCE_BOARDS

    _background_agg_loading = True
    _background_agg_error = None
    _background_agg_data = {}
    _background_agg_date_range = date_range

    if board_names is None:
        board_names = {}

    # Shared mutable state for progress tracking across async tasks
    progress_state = {
        "current": 0,
        "total": 1,  # Updated once we know total work units
        "logs": [],   # Completed item log entries
    }

    def _update_progress(message: str, increment: bool = True) -> None:
        """Update the global progress dict from any async task."""
        global _background_agg_progress
        if increment:
            progress_state["current"] += 1
        progress_state["logs"].append(message)
        _background_agg_progress = {
            "current": progress_state["current"],
            "total": progress_state["total"],
            "current_source": message,
            "logs": list(progress_state["logs"]),
        }

    _background_agg_progress = {
        "current": 0,
        "total": 1,
        "current_source": "Connexion à Monday.com...",
        "logs": [],
    }

    try:
        client = MondayClient(api_key=api_key)
        history_board_id = METRICS_BOARD_CONFIG.board_id

        # Store raw items per source for deferred formula enrichment
        _source_items: dict[str, list[dict]] = {}

        async def _resolve_date_filter(
            board_id: int, date_column_title: str,
        ) -> Optional[tuple[str, str, str]]:
            """Build server-side date filter for extract_board_data.

            Returns None if no range available or column cannot be resolved;
            caller will fall back to client-side date filtering.
            """
            if not date_range or not date_column_title:
                return None
            col_id = await client._resolve_column_id_by_title(
                board_id, date_column_title,
            )
            if not col_id:
                return None
            start_iso, end_iso = date_range
            return (col_id, start_iso, end_iso)

        async def _load_one_source(source_key: str, board_id: int, config) -> None:
            """Load a single source board (without formulas) and log completion."""
            display_label = board_names.get(board_id, config.display_name)
            try:
                date_filter = await _resolve_date_filter(
                    board_id, getattr(config, "date_column", ""),
                )
                items = await client.extract_board_data(
                    board_id,
                    skip_formula_enrichment=True,
                    date_filter=date_filter,
                )
                _source_items[source_key] = items
                df = client.board_items_to_dataframe(items)
                _background_agg_data[source_key] = df
                _update_progress(f"✅ {display_label} — {len(df)} lignes")
            except Exception as e:
                _background_agg_data[source_key] = pd.DataFrame()
                _update_progress(f"⚠️ {display_label} — erreur: {e}")

        async def _load_advisor_history() -> int:
            """Load advisor history. Returns number of groups loaded."""
            from src.utils.advisor_matcher import normalize_advisor_name_full
            from src.utils.advisor_status import AdvisorStatusCalculator

            try:
                groups = await client.list_groups(history_board_id)

                sorted_groups = []
                for group in groups:
                    title = group.get("title", "")
                    year, month = AdvisorStatusCalculator._parse_month_year(title)
                    if year > 0 and month > 0:
                        sorted_groups.append((year, month, group))
                sorted_groups.sort(key=lambda x: (x[0], x[1]))

                if not sorted_groups:
                    _update_progress("📊 Historique conseillers — aucun groupe")
                    return 0

                # Update total now that we know how many groups there are
                progress_state["total"] += len(sorted_groups)
                _background_agg_progress["total"] = progress_state["total"]

                # Load all group items concurrently
                async def _load_history_group(year_month_group):
                    _year, _month, g = year_month_group
                    items = await client.extract_board_data(
                        history_board_id, group_id=g["id"],
                        skip_formula_enrichment=True,
                    )
                    _update_progress(f"📊 Historique — {g['title']}")
                    return g, items

                tasks = [_load_history_group(g) for g in sorted_groups]
                group_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process in chronological order
                first_appearances: dict[str, str] = {}
                for ymg, result in zip(sorted_groups, group_results):
                    if isinstance(result, Exception):
                        continue
                    group, items = result
                    df = client.board_items_to_dataframe(items)
                    if df.empty:
                        continue

                    if "Conseiller" in df.columns:
                        advisors = df["Conseiller"].dropna().unique()
                    elif "item_name" in df.columns:
                        advisors = df["item_name"].dropna().unique()
                    else:
                        continue

                    for advisor in advisors:
                        normalized = normalize_advisor_name_full(str(advisor))
                        if normalized and normalized not in first_appearances:
                            first_appearances[normalized] = group["title"]

                AdvisorStatusCalculator._first_appearance_cache = first_appearances
                AdvisorStatusCalculator._cache_loaded = True
                return len(sorted_groups)

            except Exception:
                _update_progress("⚠️ Historique conseillers — erreur (statut par défaut)")
                return 0

        async def _load_all_parallel() -> None:
            """Load all source boards + advisor history concurrently,
            then enrich formula columns sequentially to respect rate limits."""
            source_configs = {}
            source_tasks = []
            for source_key, board_id in selected_sources.items():
                config = SOURCE_BOARDS.get(source_key)
                if config:
                    source_configs[source_key] = config
                    source_tasks.append(_load_one_source(source_key, board_id, config))

            # Total starts at num_sources; advisor history will add its groups
            progress_state["total"] = len(source_tasks)
            _background_agg_progress["total"] = progress_state["total"]

            # Phase 1: Load all data in parallel (without formulas)
            await asyncio.gather(*source_tasks, _load_advisor_history())

            # Phase 2: Enrich formula columns sequentially per board
            # (rate limit: 10k formula values/min — sequential avoids collisions)

            # Pre-count boards with formulas to keep progress total coherent
            def _board_has_formulas(items: list[dict]) -> bool:
                if not items:
                    return False
                return any(
                    cv.get("column", {}).get("type") == "formula"
                    for cv in items[0].get("column_values", [])
                )

            boards_with_formulas = sum(
                1 for items in _source_items.values() if _board_has_formulas(items)
            )
            if boards_with_formulas:
                progress_state["total"] += boards_with_formulas
                _background_agg_progress["total"] = progress_state["total"]

            for source_key, items in _source_items.items():
                if not items:
                    continue
                config = source_configs.get(source_key)
                if not config:
                    continue
                board_id = selected_sources[source_key]
                display_label = board_names.get(board_id, config.display_name)

                if not _board_has_formulas(items):
                    continue

                try:
                    _update_progress(
                        f"🔢 Formules — {display_label}...", increment=False,
                    )
                    await client.enrich_formula_columns(items)
                    # Rebuild DataFrame with enriched formula values
                    df = client.board_items_to_dataframe(items)
                    _background_agg_data[source_key] = df
                    _update_progress(f"🔢 Formules — {display_label} ✅")
                except Exception:
                    _update_progress(
                        f"⚠️ Formules — {display_label} (fallback texte)"
                    )

        asyncio.run(_load_all_parallel())

        _background_agg_progress = {
            "current": progress_state["total"],
            "total": progress_state["total"],
            "current_source": "Terminé",
            "logs": progress_state["logs"],
        }

    except Exception as e:
        _background_agg_error = str(e)
    finally:
        _background_agg_loading = False


def start_background_aggregation_load() -> bool:
    """
    Start loading aggregation data in the background.

    Returns:
        True if loading was started, False if already loading, data exists, or no API key
    """
    global _background_agg_loading, _background_agg_data, _background_agg_date_range

    # Don't reload if already loading
    if _background_agg_loading:
        return False

    # Resolve requested period to (start_iso, end_iso).
    requested_range = _resolve_selected_date_range()

    # If the user switched to a period outside the loaded window, invalidate
    # the cache so we reload with the new server-side filter applied.
    if (
        _background_agg_date_range is not None
        and _background_agg_date_range != requested_range
    ):
        _background_agg_data = {}
        st.session_state.agg_source_data = {}
        st.session_state.agg_data_loaded = False
        _background_agg_date_range = None

    # Don't reload if background data already exists (and still covers the
    # current period — checked above)
    if _background_agg_data and not _background_agg_error:
        return False

    # Don't reload if session state already has valid data (cache check)
    existing_data = st.session_state.get("agg_source_data", {})
    data_loaded = st.session_state.get("agg_data_loaded", False)
    loaded_range = st.session_state.get("agg_loaded_date_range")
    if (
        data_loaded
        and existing_data
        and loaded_range == requested_range
        and any(not df.empty for df in existing_data.values() if hasattr(df, 'empty'))
    ):
        return False
    # Range mismatch in session state — force reload by clearing the cache.
    if data_loaded and loaded_range != requested_range:
        st.session_state.agg_source_data = {}
        st.session_state.agg_data_loaded = False

    api_key = st.session_state.get("monday_api_key")
    if not api_key:
        return False

    selected_sources = st.session_state.get("agg_selected_sources", {})
    if not selected_sources:
        return False

    # Build board_id -> board_name mapping from session state
    board_names: dict[int, str] = {}
    boards = st.session_state.get("monday_boards")
    if boards:
        for b in boards:
            board_names[int(b["id"])] = b["name"]

    # Falls back to None (full board fetch) on any error — client-side filter
    # in filter_and_aggregate_data() will still apply correctly.
    thread = threading.Thread(
        target=_load_aggregation_data_thread,
        args=(api_key, selected_sources, board_names, requested_range),
        daemon=True,
    )
    thread.start()

    return True


def _resolve_selected_date_range() -> Optional[tuple[str, str]]:
    """Extract (start_iso, end_iso) from the currently selected period.

    Returns None if no period is selected or on any error. Prefers the
    FlexiblePeriod when available, falling back to the legacy DatePeriod.
    """
    try:
        flexible_period = st.session_state.get("agg_flexible_period")
        if flexible_period is not None:
            start, end = flexible_period.get_date_range()
            return start.isoformat(), end.isoformat()

        legacy_period = st.session_state.get("agg_period")
        if legacy_period is not None:
            from src.utils.aggregator import get_period_date_range
            start, end = get_period_date_range(legacy_period)
            return start.isoformat(), end.isoformat()
    except Exception:
        pass
    return None


def get_background_aggregation_status() -> dict:
    """Get the current status of background aggregation loading."""
    return {
        "loading": _background_agg_loading,
        "error": _background_agg_error,
        "data": _background_agg_data,
        "progress": _background_agg_progress,
    }


def apply_background_aggregation_data() -> bool:
    """Apply the background-loaded data to session state."""
    global _background_agg_data

    if _background_agg_loading:
        return False

    if not _background_agg_data:
        return False

    st.session_state.agg_source_data = _background_agg_data.copy()
    st.session_state.agg_data_loaded = True
    # Remember the date range this data was filtered for, so later calls
    # can detect when the user needs fresh data for a different period.
    st.session_state.agg_loaded_date_range = _background_agg_date_range

    return True


def reset_background_aggregation_data() -> None:
    """Reset the background aggregation data."""
    global _background_agg_data, _background_agg_loading, _background_agg_error
    global _background_agg_progress, _background_agg_date_range

    _background_agg_data = {}
    _background_agg_loading = False
    _background_agg_error = None
    _background_agg_progress = {}
    _background_agg_date_range = None
