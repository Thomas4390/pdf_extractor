"""
Aggregation module for Monday.com data.

Provides the 4-step wizard for aggregating data from multiple
Monday.com boards by advisor and upserting to a target board.
"""

from src.app.aggregation.mode import render_aggregation_mode

__all__ = [
    "render_aggregation_mode",
]
