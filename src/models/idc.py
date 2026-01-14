"""
Pydantic models for IDC proposition reports.

These models represent the structure of data extracted from IDC
"Rapport des propositions soumises" PDF reports.
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


# Type aliases for flexible fields
FlexibleDecimal = Annotated[Optional[Decimal], BeforeValidator(coerce_decimal)]
FlexibleString = Annotated[Optional[str], BeforeValidator(coerce_string)]


# =============================================================================
# MODELS
# =============================================================================

class IDCProposition(BaseModel):
    """
    A single proposition record from IDC report.

    Represents one insurance proposition with client, policy,
    and commission information.
    """

    assureur: FlexibleString = Field(
        default=None,
        description="Insurance company name",
        examples=["RBC INSURANCE", "CANADA LIFE"],
    )
    client: FlexibleString = Field(
        default=None,
        description="Client name (format: LASTNAME, F)",
        examples=["SMITH, J", "TREMBLAY, M"],
    )
    type_regime: FlexibleString = Field(
        default=None,
        description="Policy type",
        examples=["Permanent", "Term", "Disability", "Critical Illness"],
    )
    police: FlexibleString = Field(
        default=None,
        description="Policy number",
        examples=["1014157", "ABC123"],
    )
    statut: FlexibleString = Field(
        default=None,
        description="Policy status",
        examples=["Approved", "Inforce", "Pending", "Submitted"],
    )
    date: FlexibleString = Field(
        default=None,
        description="Date (YYYY-MM-DD format)",
        examples=["2025-10-15", "2025-11-01"],
    )
    nombre: FlexibleDecimal = Field(
        default=None,
        description="Quantity/Number between 0.00 and 1.00",
        examples=[1.00, 0.00],
    )
    taux_cpa: FlexibleDecimal = Field(
        default=None,
        description="CPA rate as percentage",
        examples=[100.0, 50.0],
    )
    couverture: FlexibleString = Field(
        default=None,
        description="Coverage amount",
        examples=["100 000,00 $", "250 000,00 $"],
    )
    prime_police: FlexibleString = Field(
        default=None,
        description="Policy premium amount",
        examples=["1 234,56 $", "500,00 $"],
    )
    prime_commissionnable: FlexibleString = Field(
        default=None,
        description="Commissionable premium amount",
        examples=["1 234,56 $", "500,00 $"],
    )
    commission: FlexibleString = Field(
        default=None,
        description="Commission amount",
        examples=["123,45 $", "50,00 $"],
    )

    @field_validator("client", mode="after")
    @classmethod
    def normalize_client(cls, v: Optional[str]) -> Optional[str]:
        """Clean and normalize client name."""
        if v:
            return " ".join(v.upper().split())
        return v


class IDCReport(BaseModel):
    """
    Complete IDC propositions report.

    Contains metadata and list of proposition records.
    """

    # Report metadata
    titre: str = Field(
        default="Rapport des propositions soumises",
        description="Report title",
    )
    date_rapport: FlexibleString = Field(
        default=None,
        description="Report generation date",
        examples=["2025-10-17"],
    )
    vendeur: FlexibleString = Field(
        default=None,
        description="Vendor/Broker name",
        examples=["Greenberg, Thomas"],
    )

    # Document type
    document_type: str = Field(
        default="IDC",
        description="Document type identifier",
    )

    # Records
    propositions: list[IDCProposition] = Field(
        default_factory=list,
        description="List of proposition records",
    )

    @property
    def nombre_propositions(self) -> int:
        """Number of propositions in the report."""
        return len(self.propositions)

    @property
    def assureurs_uniques(self) -> list[str]:
        """List of unique insurers."""
        return list(set(p.assureur for p in self.propositions if p.assureur))

    def propositions_par_type(self) -> dict[str, int]:
        """Count propositions by regime type."""
        counts: dict[str, int] = {}
        for p in self.propositions:
            if p.type_regime:
                counts[p.type_regime] = counts.get(p.type_regime, 0) + 1
        return counts
