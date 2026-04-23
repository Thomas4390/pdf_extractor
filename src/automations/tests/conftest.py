"""Shared fixtures for automations tests."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.automations.config import RankingConfig


@pytest.fixture
def sample_config() -> RankingConfig:
    return RankingConfig(
        source_board_id=18402779315,
        metrics=("Appels faits", "RDV Book", "$$$ Recues"),
        date_column="Date",
        advisor_column="group_title",
        use_group_as_advisor=True,
        months_ago=0,
    )


@pytest.fixture
def reference_date() -> date:
    """Fixed reference date for predictable period filtering in tests."""
    return date(2026, 2, 15)


@pytest.fixture
def raw_board_df() -> pd.DataFrame:
    """
    Mock raw DataFrame as returned by board_items_to_dataframe.

    3 advisors x 4 days in Feb 2026 + 1 day in Jan 2026 (out of scope).
    """
    rows = [
        # Alice — 4 days in Feb 2026
        {"group_title": "Alice", "Date": "2026-02-01", "Appels faits": 10.0, "RDV Book": 2.0, "$$$ Recues": 100.0},
        {"group_title": "Alice", "Date": "2026-02-05", "Appels faits": 15.0, "RDV Book": 3.0, "$$$ Recues": 250.0},
        {"group_title": "Alice", "Date": "2026-02-10", "Appels faits": 20.0, "RDV Book": 1.0, "$$$ Recues": 0.0},
        {"group_title": "Alice", "Date": "2026-02-20", "Appels faits": 5.0, "RDV Book": 0.0, "$$$ Recues": 50.0},
        # Bob — 3 days in Feb 2026
        {"group_title": "Bob", "Date": "2026-02-02", "Appels faits": 30.0, "RDV Book": 5.0, "$$$ Recues": 500.0},
        {"group_title": "Bob", "Date": "2026-02-08", "Appels faits": 25.0, "RDV Book": 4.0, "$$$ Recues": 300.0},
        {"group_title": "Bob", "Date": "2026-02-22", "Appels faits": 10.0, "RDV Book": 1.0, "$$$ Recues": 0.0},
        # Carol — 2 days in Feb 2026 (with one null)
        {"group_title": "Carol", "Date": "2026-02-03", "Appels faits": 40.0, "RDV Book": None, "$$$ Recues": 200.0},
        {"group_title": "Carol", "Date": "2026-02-15", "Appels faits": 50.0, "RDV Book": 2.0, "$$$ Recues": 150.0},
        # Out-of-period row (Jan 2026) — must be filtered out
        {"group_title": "Alice", "Date": "2026-01-30", "Appels faits": 999.0, "RDV Book": 99.0, "$$$ Recues": 9999.0},
    ]
    return pd.DataFrame(rows)
