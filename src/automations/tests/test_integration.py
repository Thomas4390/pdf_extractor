"""
Manual integration test — hits the real Monday API against the test AE Tracker
board (id 18402779315). Skipped by default; run explicitly with:

    pytest src/automations/tests/test_integration.py -v -m integration

Requires MONDAY_API_KEY to be set in the environment or a .env file at the
project root.
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import pytest

from src.automations.config import RankingConfig
from src.automations.monthly_ranking import build_ranking

pytestmark = pytest.mark.integration

TEST_BOARD_ID = 18402779315
KNOWN_METRICS = (
    "Appels faits",
    "Appels répondus",
    "RDV Book",
    "RDV Faits",
    "PA Vendues",
    "$$$ Recues",
)


@pytest.fixture(scope="module")
def monday_client():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("MONDAY_API_KEY")
    if not api_key:
        pytest.skip("MONDAY_API_KEY not set — skipping integration test")

    from src.clients.monday import MondayClient

    return MondayClient(api_key=api_key)


@pytest.fixture(scope="module")
def raw_board_df(monday_client) -> pd.DataFrame:
    items = monday_client.extract_board_data_sync(board_id=TEST_BOARD_ID)
    return monday_client.board_items_to_dataframe(items)


def test_board_has_expected_metrics(raw_board_df: pd.DataFrame) -> None:
    missing = [m for m in KNOWN_METRICS if m not in raw_board_df.columns]
    assert not missing, f"Expected metrics missing from board: {missing}"
    assert "Date" in raw_board_df.columns
    assert "group_title" in raw_board_df.columns


def test_ranking_for_march_2026_produces_non_empty_result(
    raw_board_df: pd.DataFrame,
) -> None:
    """
    Smoke test: the last month known to contain data in the test board is
    March 2026. Targets that month via an explicit reference date so the
    test stays deterministic as real-world time advances.
    """
    config = RankingConfig(
        source_board_id=TEST_BOARD_ID,
        metrics=KNOWN_METRICS,
        date_column="Date",
        advisor_column="group_title",
        use_group_as_advisor=True,
        months_ago=0,
    )
    # Reference in March 2026, months_ago=0 => target month is March 2026.
    df = build_ranking(raw_board_df, config, reference_date=date(2026, 3, 15))

    assert not df.empty, "Expected at least one advisor ranked for March 2026"
    assert "Conseiller" in df.columns
    for metric in KNOWN_METRICS:
        assert metric in df.columns
        assert f"Rang {metric}" in df.columns
    # Ranks should start at 1
    for metric in KNOWN_METRICS:
        assert df[f"Rang {metric}"].min() == 1
