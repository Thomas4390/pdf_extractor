"""Tests for the monthly_ranking orchestrator (pure logic only — no Monday API)."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.automations.config import RankingConfig
from src.automations.monthly_ranking import build_ranking, run


@pytest.mark.unit
class TestBuildRanking:
    def test_filters_to_current_month(
        self,
        sample_config: RankingConfig,
        raw_board_df: pd.DataFrame,
        reference_date: date,
    ) -> None:
        df = build_ranking(raw_board_df, sample_config, reference_date=reference_date)
        # Alice's Jan row (999) must be excluded — if it leaked in, Alice's sum would be 1049
        alice = df[df["Conseiller"] == "Alice"].iloc[0]
        assert alice["Appels faits"] == 10 + 15 + 20 + 5

    def test_sums_per_advisor_per_metric(
        self,
        sample_config: RankingConfig,
        raw_board_df: pd.DataFrame,
        reference_date: date,
    ) -> None:
        df = build_ranking(raw_board_df, sample_config, reference_date=reference_date)
        df_indexed = df.set_index("Conseiller")

        assert df_indexed.loc["Alice", "Appels faits"] == 50
        assert df_indexed.loc["Alice", "RDV Book"] == 6
        assert df_indexed.loc["Alice", "$$$ Recues"] == 400

        assert df_indexed.loc["Bob", "Appels faits"] == 65
        assert df_indexed.loc["Bob", "RDV Book"] == 10
        assert df_indexed.loc["Bob", "$$$ Recues"] == 800

        assert df_indexed.loc["Carol", "Appels faits"] == 90
        # Carol had one None in RDV Book — should be treated as 0
        assert df_indexed.loc["Carol", "RDV Book"] == 2
        assert df_indexed.loc["Carol", "$$$ Recues"] == 350

    def test_adds_rank_column_per_metric(
        self,
        sample_config: RankingConfig,
        raw_board_df: pd.DataFrame,
        reference_date: date,
    ) -> None:
        df = build_ranking(raw_board_df, sample_config, reference_date=reference_date)
        df_indexed = df.set_index("Conseiller")

        # Appels faits: Carol(90) > Bob(65) > Alice(50)
        assert df_indexed.loc["Carol", "Rang Appels faits"] == 1
        assert df_indexed.loc["Bob", "Rang Appels faits"] == 2
        assert df_indexed.loc["Alice", "Rang Appels faits"] == 3

        # RDV Book: Bob(10) > Alice(6) > Carol(2)
        assert df_indexed.loc["Bob", "Rang RDV Book"] == 1
        assert df_indexed.loc["Alice", "Rang RDV Book"] == 2
        assert df_indexed.loc["Carol", "Rang RDV Book"] == 3

        # $$$ Recues: Bob(800) > Alice(400) > Carol(350)
        assert df_indexed.loc["Bob", "Rang $$$ Recues"] == 1
        assert df_indexed.loc["Alice", "Rang $$$ Recues"] == 2
        assert df_indexed.loc["Carol", "Rang $$$ Recues"] == 3

    def test_output_columns_shape(
        self,
        sample_config: RankingConfig,
        raw_board_df: pd.DataFrame,
        reference_date: date,
    ) -> None:
        df = build_ranking(raw_board_df, sample_config, reference_date=reference_date)
        expected_cols = {
            "Conseiller",
            "Appels faits",
            "RDV Book",
            "$$$ Recues",
            "Rang Appels faits",
            "Rang RDV Book",
            "Rang $$$ Recues",
        }
        assert expected_cols.issubset(set(df.columns))
        assert len(df) == 3

    def test_empty_input_returns_empty_df(
        self, sample_config: RankingConfig, reference_date: date
    ) -> None:
        df = build_ranking(pd.DataFrame(), sample_config, reference_date=reference_date)
        assert df.empty

    def test_no_rows_in_period_returns_empty_df(
        self, sample_config: RankingConfig, reference_date: date
    ) -> None:
        # All rows outside Feb 2026
        raw = pd.DataFrame(
            [
                {"group_title": "Alice", "Date": "2025-05-01", "Appels faits": 10.0, "RDV Book": 1.0, "$$$ Recues": 50.0},
            ]
        )
        df = build_ranking(raw, sample_config, reference_date=reference_date)
        assert df.empty

    def test_ties_share_same_rank(
        self, sample_config: RankingConfig, reference_date: date
    ) -> None:
        raw = pd.DataFrame(
            [
                {"group_title": "A", "Date": "2026-02-01", "Appels faits": 10.0, "RDV Book": 1.0, "$$$ Recues": 0.0},
                {"group_title": "B", "Date": "2026-02-01", "Appels faits": 10.0, "RDV Book": 2.0, "$$$ Recues": 0.0},
                {"group_title": "C", "Date": "2026-02-01", "Appels faits": 5.0, "RDV Book": 3.0, "$$$ Recues": 0.0},
            ]
        )
        df = build_ranking(raw, sample_config, reference_date=reference_date)
        df_indexed = df.set_index("Conseiller")
        # A and B both have 10 Appels faits — both should rank 1 (min method)
        assert df_indexed.loc["A", "Rang Appels faits"] == 1
        assert df_indexed.loc["B", "Rang Appels faits"] == 1
        assert df_indexed.loc["C", "Rang Appels faits"] == 3


@pytest.mark.unit
class TestRun:
    def test_run_uses_client_and_returns_ranking(
        self,
        sample_config: RankingConfig,
        raw_board_df: pd.DataFrame,
        reference_date: date,
    ) -> None:
        client = MagicMock()
        client.extract_board_data_sync.return_value = [{"fake": "item"}]
        client.board_items_to_dataframe.return_value = raw_board_df

        df = run(sample_config, client=client, reference_date=reference_date)

        client.extract_board_data_sync.assert_called_once()
        assert len(df) == 3
        assert "Rang Appels faits" in df.columns

    def test_run_honours_source_board_id(
        self,
        sample_config: RankingConfig,
        raw_board_df: pd.DataFrame,
        reference_date: date,
    ) -> None:
        client = MagicMock()
        client.extract_board_data_sync.return_value = [{"fake": "item"}]
        client.board_items_to_dataframe.return_value = raw_board_df

        run(sample_config, client=client, reference_date=reference_date)

        call_kwargs = client.extract_board_data_sync.call_args.kwargs
        assert call_kwargs["board_id"] == sample_config.source_board_id
