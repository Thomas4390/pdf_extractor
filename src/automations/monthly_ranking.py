"""
Monthly ranking automation — reads the AE tracker board, aggregates per-advisor
totals for the current month, and produces a ranked DataFrame.

Intended to run on a GitHub Actions cron schedule. This iteration does NOT
upsert to a target Monday board — it only produces the DataFrame and logs it
so the ranking logic can be validated before wiring up the writeback.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date
from typing import Any, Optional

import pandas as pd

from src.automations.config import RankingConfig

logger = logging.getLogger(__name__)

ADVISOR_OUTPUT_COLUMN = "Conseiller"


def _target_period(reference_date: date, months_ago: int) -> pd.Period:
    ref = pd.Timestamp(reference_date)
    return ref.to_period("M") - months_ago


def build_ranking(
    raw_df: pd.DataFrame,
    config: RankingConfig,
    reference_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Aggregate a raw Monday board DataFrame into a ranked DataFrame for the
    month targeted by `config.months_ago`.

    The output has one row per advisor, with summed metric columns and a
    "Rang {metric}" column for each metric (min-method, descending).
    """
    if raw_df.empty:
        return pd.DataFrame()

    ref = reference_date or date.today()
    target_period = _target_period(ref, config.months_ago)

    df = raw_df.copy()
    df[config.date_column] = pd.to_datetime(df[config.date_column], errors="coerce")
    df = df[df[config.date_column].dt.to_period("M") == target_period]

    if df.empty:
        return pd.DataFrame()

    for metric in config.metrics:
        if metric not in df.columns:
            logger.warning("Metric %r not found in source data — filling with 0", metric)
            df[metric] = 0
        df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0)

    grouped = (
        df.groupby(config.advisor_column, as_index=False)[list(config.metrics)]
        .sum()
        .rename(columns={config.advisor_column: ADVISOR_OUTPUT_COLUMN})
    )

    for metric in config.metrics:
        grouped[f"Rang {metric}"] = (
            grouped[metric].rank(method="min", ascending=False).astype(int)
        )

    # Stable ordering: sort by first metric descending so the DataFrame is
    # immediately readable in logs.
    primary = config.metrics[0]
    grouped = grouped.sort_values(primary, ascending=False).reset_index(drop=True)

    return grouped


def run(
    config: RankingConfig,
    client: Any,
    reference_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Fetch the source board, build the ranking, and return the DataFrame.
    `client` is duck-typed — any object exposing
    `extract_board_data_sync(board_id=...)` and `board_items_to_dataframe(items)`
    works (used for mocking in tests).
    """
    items = client.extract_board_data_sync(board_id=config.source_board_id)
    raw_df = client.board_items_to_dataframe(items)
    return build_ranking(raw_df, config, reference_date=reference_date)


def _log_dataframe(df: pd.DataFrame, period_label: str) -> None:
    if df.empty:
        logger.warning("No data for %s — empty ranking", period_label)
        return
    logger.info("Ranking for %s (%d advisors):", period_label, len(df))
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 200,
    ):
        logger.info("\n%s", df.to_string(index=False))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    config = RankingConfig.from_env()

    from src.clients.monday import MondayClient

    api_key = os.environ.get("MONDAY_API_KEY")
    if not api_key:
        logger.error("MONDAY_API_KEY is not set")
        return 2

    client = MondayClient(api_key=api_key)

    ref = date.today()
    target_period = _target_period(ref, config.months_ago)
    period_label = target_period.strftime("%B %Y")

    logger.info(
        "Starting ranking run — board=%s, months_ago=%d, target=%s",
        config.source_board_id, config.months_ago, period_label,
    )

    df = run(config, client=client, reference_date=ref)
    _log_dataframe(df, period_label)

    logger.info("Ranking run complete — %d advisors ranked", len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
