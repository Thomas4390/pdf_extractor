"""Pydantic models for structured data extraction."""

from .assomption import AssomptionBoni, AssomptionCommission, AssomptionReport
from .coercion import FlexibleDecimal, FlexibleString, FlexibleStringWithDefault, coerce_decimal, coerce_string
from .idc import IDCProposition, IDCReport
from .idc_statement import (
    IDCStatementReport,
    IDCStatementReportParsed,
    IDCTrailingFee,
    IDCTrailingFeeParsed,
    IDCTrailingFeeRaw,
)
from .uv import UVActivity, UVReport

__all__ = [
    "AssomptionBoni",
    "AssomptionCommission",
    "AssomptionReport",
    "FlexibleDecimal",
    "FlexibleString",
    "FlexibleStringWithDefault",
    "IDCProposition",
    "IDCReport",
    "IDCStatementReport",
    "IDCStatementReportParsed",
    "IDCTrailingFee",
    "IDCTrailingFeeParsed",
    "IDCTrailingFeeRaw",
    "UVActivity",
    "UVReport",
    "coerce_decimal",
    "coerce_string",
]
