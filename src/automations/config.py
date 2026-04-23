"""Configuration for the monthly ranking automation loaded from env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_metrics(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"Missing required env var: {name}")
    return value


def _parse_int_env(name: str, value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


@dataclass(frozen=True)
class RankingConfig:
    """Immutable configuration for the monthly ranking run."""

    source_board_id: int
    metrics: tuple[str, ...]
    date_column: str
    advisor_column: str = "group_title"
    use_group_as_advisor: bool = True
    months_ago: int = 0
    target_board_id: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.metrics:
            raise ValueError("RANKING_METRICS must contain at least one metric")
        if self.months_ago < 0:
            raise ValueError(
                f"months_ago must be >= 0, got {self.months_ago}"
            )

    @classmethod
    def from_env(cls) -> "RankingConfig":
        source_board_id = _parse_int_env(
            "RANKING_SOURCE_BOARD_ID", _require_env("RANKING_SOURCE_BOARD_ID")
        )
        metrics = _parse_metrics(_require_env("RANKING_METRICS"))
        date_column = _require_env("RANKING_DATE_COLUMN")

        advisor_column = os.environ.get("RANKING_ADVISOR_COLUMN", "group_title")
        use_group_as_advisor = _parse_bool(
            os.environ.get("RANKING_USE_GROUP_AS_ADVISOR", "true")
        )

        months_ago_raw = os.environ.get("RANKING_MONTHS_AGO", "0")
        months_ago = _parse_int_env("RANKING_MONTHS_AGO", months_ago_raw)

        target_board_raw = os.environ.get("RANKING_TARGET_BOARD_ID")
        target_board_id = (
            _parse_int_env("RANKING_TARGET_BOARD_ID", target_board_raw)
            if target_board_raw and target_board_raw.strip()
            else None
        )

        return cls(
            source_board_id=source_board_id,
            metrics=metrics,
            date_column=date_column,
            advisor_column=advisor_column,
            use_group_as_advisor=use_group_as_advisor,
            months_ago=months_ago,
            target_board_id=target_board_id,
        )
