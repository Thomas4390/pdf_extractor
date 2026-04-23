"""Tests for RankingConfig env-based loading and validation."""

from __future__ import annotations

import pytest

from src.automations.config import RankingConfig


@pytest.fixture
def base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set all required env vars to valid defaults."""
    monkeypatch.setenv("RANKING_SOURCE_BOARD_ID", "18402779315")
    monkeypatch.setenv("RANKING_METRICS", "Appels faits,RDV Book,$$$ Recues")
    monkeypatch.setenv("RANKING_DATE_COLUMN", "Date")


@pytest.mark.unit
class TestFromEnv:
    def test_basic_loading(self, base_env: None) -> None:
        cfg = RankingConfig.from_env()
        assert cfg.source_board_id == 18402779315
        assert cfg.metrics == ("Appels faits", "RDV Book", "$$$ Recues")
        assert cfg.date_column == "Date"

    def test_defaults_when_optional_missing(self, base_env: None) -> None:
        cfg = RankingConfig.from_env()
        assert cfg.advisor_column == "group_title"
        assert cfg.use_group_as_advisor is True
        assert cfg.months_ago == 0
        assert cfg.target_board_id is None

    def test_metrics_trim_whitespace(
        self, base_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANKING_METRICS", "  Appels faits ,  RDV Book  ,Ventes")
        cfg = RankingConfig.from_env()
        assert cfg.metrics == ("Appels faits", "RDV Book", "Ventes")

    def test_metrics_filters_empty_entries(
        self, base_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANKING_METRICS", "Appels faits,,RDV Book, ")
        cfg = RankingConfig.from_env()
        assert cfg.metrics == ("Appels faits", "RDV Book")

    def test_months_ago_override(
        self, base_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANKING_MONTHS_AGO", "2")
        cfg = RankingConfig.from_env()
        assert cfg.months_ago == 2

    def test_use_group_as_advisor_false(
        self, base_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANKING_USE_GROUP_AS_ADVISOR", "false")
        monkeypatch.setenv("RANKING_ADVISOR_COLUMN", "Conseiller")
        cfg = RankingConfig.from_env()
        assert cfg.use_group_as_advisor is False
        assert cfg.advisor_column == "Conseiller"

    def test_target_board_id_optional(
        self, base_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANKING_TARGET_BOARD_ID", "9999")
        cfg = RankingConfig.from_env()
        assert cfg.target_board_id == 9999

    def test_empty_string_optionals_fall_back_to_defaults(
        self, base_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GitHub Actions injects undefined variables as empty strings rather
        than omitting them — make sure we treat "" the same as unset."""
        monkeypatch.setenv("RANKING_ADVISOR_COLUMN", "")
        monkeypatch.setenv("RANKING_USE_GROUP_AS_ADVISOR", "")
        monkeypatch.setenv("RANKING_MONTHS_AGO", "")
        monkeypatch.setenv("RANKING_TARGET_BOARD_ID", "")
        cfg = RankingConfig.from_env()
        assert cfg.advisor_column == "group_title"
        assert cfg.use_group_as_advisor is True
        assert cfg.months_ago == 0
        assert cfg.target_board_id is None


@pytest.mark.unit
class TestValidation:
    def test_missing_source_board_id_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANKING_METRICS", "A,B")
        monkeypatch.setenv("RANKING_DATE_COLUMN", "Date")
        monkeypatch.delenv("RANKING_SOURCE_BOARD_ID", raising=False)
        with pytest.raises(ValueError, match="RANKING_SOURCE_BOARD_ID"):
            RankingConfig.from_env()

    def test_missing_metrics_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RANKING_SOURCE_BOARD_ID", "1")
        monkeypatch.setenv("RANKING_DATE_COLUMN", "Date")
        monkeypatch.delenv("RANKING_METRICS", raising=False)
        with pytest.raises(ValueError, match="RANKING_METRICS"):
            RankingConfig.from_env()

    def test_empty_metrics_raises(
        self, base_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANKING_METRICS", " , , ")
        with pytest.raises(ValueError, match="at least one metric"):
            RankingConfig.from_env()

    def test_invalid_board_id_raises(
        self, base_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANKING_SOURCE_BOARD_ID", "not-a-number")
        with pytest.raises(ValueError, match="RANKING_SOURCE_BOARD_ID"):
            RankingConfig.from_env()

    def test_invalid_months_ago_raises(
        self, base_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANKING_MONTHS_AGO", "-1")
        with pytest.raises(ValueError, match="months_ago"):
            RankingConfig.from_env()

    def test_frozen_dataclass(self, base_env: None) -> None:
        cfg = RankingConfig.from_env()
        with pytest.raises(Exception):
            cfg.source_board_id = 999  # type: ignore[misc]
