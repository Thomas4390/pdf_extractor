"""
Shared type coercion utilities for Pydantic models.

Provides flexible type coercion for VLM output, handling common formats
like currency strings, French decimals, and null-like values.
"""

from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Optional

from pydantic import BeforeValidator


def coerce_decimal(v: Any) -> Optional[Decimal]:
    """
    Coerce various input types to Decimal, handling common VLM output formats.

    Handles:
    - None, empty string, "N/A", "n/a" → None
    - Strings with currency symbols: "1 196,00 $" → 1196.00
    - Strings with percentage: "55,000 %" → 55.0
    - French decimal format: "1,5" → 1.5
    - Already Decimal/float/int → Decimal
    """
    if v is None:
        return None

    if isinstance(v, Decimal):
        return v

    if isinstance(v, (int, float)):
        return Decimal(str(v))

    if isinstance(v, str):
        v = v.strip()
        # Handle empty/null-like strings
        if not v or v.lower() in ("", "none", "null", "n/a", "nan", "-"):
            return None

        # Clean currency and percentage symbols
        v = v.replace("$", "").replace("%", "").replace(" ", "")
        # Handle French decimal format (comma as decimal separator)
        v = v.replace(",", ".")
        # Handle multiple dots (thousands separator)
        parts = v.split(".")
        if len(parts) > 2:
            # Assume last part is decimal, rest is thousands
            v = "".join(parts[:-1]) + "." + parts[-1]

        try:
            return Decimal(v) if v else None
        except InvalidOperation:
            return None

    return None


def coerce_string(v: Any) -> Optional[str]:
    """
    Coerce input to string, handling None and empty values gracefully.
    """
    if v is None:
        return None

    v_str = str(v).strip()
    if not v_str or v_str.lower() in ("none", "null", "nan"):
        return None

    return v_str


def coerce_string_with_default(default: str = "Unknown"):
    """
    Create a coercion function that returns a default value instead of None.
    """
    def _coerce(v: Any) -> str:
        if v is None:
            return default

        v_str = str(v).strip()
        if not v_str or v_str.lower() in ("none", "null", "nan", ""):
            return default

        return v_str

    return _coerce


# Type aliases for flexible fields
FlexibleDecimal = Annotated[Optional[Decimal], BeforeValidator(coerce_decimal)]
FlexibleString = Annotated[Optional[str], BeforeValidator(coerce_string)]
FlexibleStringWithDefault = Annotated[str, BeforeValidator(coerce_string_with_default("Unknown"))]
