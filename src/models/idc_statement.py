"""
Pydantic models for IDC Statement (trailing fees) reports.

These models represent the structure of data extracted from IDC
"Détails des frais de suivi" PDF reports.

Strategy:
- First extraction: raw_client_data contains all complex metadata as-is
- Second pass (parsing): VLM parses raw_client_data into structured fields
"""

from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Optional

from pydantic import BaseModel, BeforeValidator, Field, field_validator


# =============================================================================
# FLEXIBLE TYPE COERCION
# =============================================================================

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


# =============================================================================
# RAW MODELS (First extraction pass)
# =============================================================================

class IDCTrailingFeeRaw(BaseModel):
    """
    A single trailing fee record with RAW client data.

    The raw_client_data field contains complex metadata that will be
    parsed in a second step. Example:
    "Â UV 7782 2025-11-17\nboni 75% #111011722 crt\nBourassa A clt Jeanny\nBreault-Therrien"

    This contains: company (UV), company number (7782), date (2025-11-17),
    commission rate (75%), policy number (111011722), advisor (Bourassa A),
    client name (Jeanny Breault-Therrien).
    """

    raw_client_data: FlexibleStringWithDefault = Field(
        default="Unknown",
        description="Raw client data - DO NOT INTERPRET. Copy exactly as displayed.",
        examples=[
            "Â UV 7782 2025-11-17\nboni 75% #111011722 crt\nBourassa A clt Jeanny\nBreault-Therrien",
            "Assomption_8055_2025-10-15_80%_#123456_Thomas L_Client Name",
        ],
    )
    account_number: FlexibleStringWithDefault = Field(
        default="Unknown",
        description="Account number from 'Numéro de compte' column",
        examples=["1234567", "N894713", "Unknown"],
    )
    company: FlexibleStringWithDefault = Field(
        default="Unknown",
        description="Company from 'Compagnie' column (as displayed)",
        examples=["UV", "Assomption", "Beneva"],
    )
    product: FlexibleStringWithDefault = Field(
        default="Unknown",
        description="Product from 'Produit' column",
        examples=["RRSP", "TFSA", "Non-Registered"],
    )
    date: FlexibleStringWithDefault = Field(
        default="Unknown",
        description="Date from 'Date' column (YYYY-MM-DD format)",
        examples=["2025-10-15", "2025-11-01"],
    )
    gross_trailing_fee: FlexibleStringWithDefault = Field(
        default="0,00 $",
        description="Gross trailing fee from 'Frais de suivi brut' column",
        examples=["123,45 $", "50,00 $"],
    )
    net_trailing_fee: FlexibleStringWithDefault = Field(
        default="0,00 $",
        description="Net trailing fee from 'Frais de suivi nets' column",
        examples=["98,76 $", "40,00 $"],
    )


# =============================================================================
# PARSED MODELS (Second extraction pass)
# =============================================================================

class IDCTrailingFeeParsed(BaseModel):
    """
    A trailing fee record with PARSED structured data.

    Created by parsing raw_client_data in a second VLM pass.
    """

    # Parsed from raw_client_data - all optional
    company_code: FlexibleString = Field(
        default=None,
        description="Company code extracted (e.g., 'UV', 'Assomption')",
    )
    company_number: FlexibleString = Field(
        default=None,
        description="Company number (e.g., '7782', '8055')",
    )
    policy_date: FlexibleString = Field(
        default=None,
        description="Policy date from raw data (YYYY-MM-DD)",
    )
    commission_rate: FlexibleDecimal = Field(
        default=None,
        description="Commission rate as percentage (75.0 for 75%)",
    )
    policy_number: FlexibleString = Field(
        default=None,
        description="Policy number (e.g., '111011722')",
    )
    advisor_name: FlexibleString = Field(
        default=None,
        description="Advisor name (e.g., 'Bourassa A', 'Thomas L')",
    )
    client_first_name: FlexibleString = Field(
        default=None,
        description="Client first name",
    )
    client_last_name: FlexibleString = Field(
        default=None,
        description="Client last name",
    )

    # Copied from raw extraction - with defaults
    raw_client_data: FlexibleStringWithDefault = Field(
        default="Unknown",
        description="Original raw data for reference",
    )
    account_number: FlexibleStringWithDefault = Field(default="Unknown")
    company: FlexibleStringWithDefault = Field(default="Unknown")
    product: FlexibleStringWithDefault = Field(default="Unknown")
    date: FlexibleStringWithDefault = Field(default="Unknown")
    gross_trailing_fee: FlexibleStringWithDefault = Field(default="0,00 $")
    net_trailing_fee: FlexibleStringWithDefault = Field(default="0,00 $")

    @property
    def client_full_name(self) -> str:
        """Combine first and last name."""
        parts = [self.client_first_name, self.client_last_name]
        return " ".join(p for p in parts if p) or "Unknown"


# Keep old model for backwards compatibility
class IDCTrailingFee(IDCTrailingFeeRaw):
    """Alias for backwards compatibility."""
    pass


# =============================================================================
# REPORT MODELS
# =============================================================================

class IDCStatementReport(BaseModel):
    """
    Complete IDC Statement (trailing fees) report with RAW data.

    Contains metadata and list of trailing fee records.
    Use IDCStatementReportParsed after second VLM pass for structured data.
    """

    document_type: str = Field(
        default="IDC_STATEMENT",
        description="Document type identifier",
    )

    # Report metadata
    titre: str = Field(
        default="Détails des frais de suivi",
        description="Report title",
    )
    date_rapport: FlexibleString = Field(
        default=None,
        description="Report generation date",
        examples=["2025-10-17"],
    )
    advisor_section: FlexibleString = Field(
        default=None,
        description="Advisor section header (e.g., 'Achraf El Hajji - 3449L3138')",
    )

    # Records (raw data)
    trailing_fees: list[IDCTrailingFeeRaw] = Field(
        default_factory=list,
        description="List of trailing fee records with raw client data",
    )

    @property
    def nombre_enregistrements(self) -> int:
        """Number of trailing fee records."""
        return len(self.trailing_fees)

    @property
    def compagnies_uniques(self) -> list[str]:
        """List of unique companies from Compagnie column."""
        return list(set(f.company for f in self.trailing_fees if f.company and f.company != "Unknown"))

    def frais_par_compagnie(self) -> dict[str, int]:
        """Count records by company."""
        counts: dict[str, int] = {}
        for f in self.trailing_fees:
            if f.company and f.company != "Unknown":
                counts[f.company] = counts.get(f.company, 0) + 1
        return counts


class IDCStatementReportParsed(BaseModel):
    """
    Complete IDC Statement with PARSED structured data.

    Created after second VLM pass that parses raw_client_data.
    """

    document_type: str = Field(
        default="IDC_STATEMENT",
        description="Document type identifier",
    )

    # Report metadata
    titre: str = Field(default="Détails des frais de suivi")
    date_rapport: FlexibleString = Field(default=None)
    advisor_section: FlexibleString = Field(default=None)

    # Parsed records
    trailing_fees: list[IDCTrailingFeeParsed] = Field(
        default_factory=list,
        description="List of parsed trailing fee records",
    )

    @property
    def nombre_enregistrements(self) -> int:
        """Number of trailing fee records."""
        return len(self.trailing_fees)

    @property
    def conseillers_uniques(self) -> list[str]:
        """List of unique advisors."""
        return list(set(
            f.advisor_name for f in self.trailing_fees
            if f.advisor_name
        ))

    def frais_par_conseiller(self) -> dict[str, int]:
        """Count records by advisor."""
        counts: dict[str, int] = {}
        for f in self.trailing_fees:
            if f.advisor_name:
                counts[f.advisor_name] = counts.get(f.advisor_name, 0) + 1
        return counts
