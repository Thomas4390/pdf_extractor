"""
Pydantic models for Assomption Vie remuneration reports.

These models represent the structure of data extracted from Assomption Vie
PDF reports, combining commission and bonus data into unified records.
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

class AssomptionCommission(BaseModel):
    """
    A commission record from Assomption Vie report.

    Found on the "Rapport de commissions" page (usually page 3).
    """

    code: FlexibleString = Field(
        default=None,
        description="Transaction code (e.g., 'AOH1')",
        examples=["AOH1", "BCD2"],
    )
    numero_police: FlexibleString = Field(
        default=None,
        description="Policy number (7 digits)",
        examples=["1011221", "1011452"],
    )
    nom_assure: FlexibleString = Field(
        default=None,
        description="Insured person's name",
        examples=["MUADI MUNYA TSHIMANGA", "DAOUYA TARABET"],
    )
    produit: FlexibleString = Field(
        default=None,
        description="Product code",
        examples=["4T20 B", "5L A"],
    )
    date_emission: FlexibleString = Field(
        default=None,
        description="Issue date (YYYY/MM/DD format)",
        examples=["2025/09/26", "2025/10/01"],
    )
    frequence_paiement: FlexibleString = Field(
        default=None,
        description="Payment frequency",
        examples=["Mensuel", "Annuel"],
    )
    facturation: FlexibleString = Field(
        default=None,
        description="Billing type",
        examples=["COM/PAC"],
    )
    prime: FlexibleDecimal = Field(
        default=None,
        description="Premium amount (can be negative for adjustments)",
        examples=[-142.56, 499.05],
    )
    taux_commission: FlexibleDecimal = Field(
        default=None,
        description="Commission rate as percentage",
        examples=[40.993, 45.0],
    )
    commission: FlexibleDecimal = Field(
        default=None,
        description="Commission amount (can be negative)",
        examples=[-58.44, 224.58],
    )
    # Bonus fields (merged from surcommission page)
    taux_boni: FlexibleDecimal = Field(
        default=None,
        description="Bonus rate as percentage (from surcommission page)",
        examples=[175.0],
    )
    boni: FlexibleDecimal = Field(
        default=None,
        description="Bonus amount (from surcommission page)",
        examples=[-134.24, 393.02],
    )

    @field_validator("nom_assure", mode="after")
    @classmethod
    def normalize_nom(cls, v: Optional[str]) -> Optional[str]:
        """Clean and normalize insured name."""
        if v:
            return " ".join(v.upper().split())
        return v


class AssomptionReport(BaseModel):
    """
    Complete Assomption Vie remuneration report.

    Contains metadata and merged commission/bonus records.
    """

    document_type: str = Field(
        default="ASSOMPTION",
        description="Document type identifier",
    )

    # Period information - made flexible
    periode_debut: FlexibleString = Field(
        default=None,
        description="Pay period start date (YYYY/MM/DD)",
        examples=["2025/10/02"],
    )
    periode_fin: FlexibleString = Field(
        default=None,
        description="Pay period end date (YYYY/MM/DD)",
        examples=["2025/10/06"],
    )
    date_paie: FlexibleString = Field(
        default=None,
        description="Payment date (YYYY/MM/DD)",
        examples=["2025/10/09"],
    )

    # Agent/Broker information - made flexible
    numero_courtier: FlexibleString = Field(
        default=None,
        description="Broker number",
        examples=["35552"],
    )
    nom_courtier: FlexibleString = Field(
        default=None,
        description="Broker/Agent name",
        examples=["9491-1377 Québec Inc."],
    )

    # Records
    commissions: list[AssomptionCommission] = Field(
        default_factory=list,
        description="List of commission records with merged bonus data",
    )

    @property
    def nombre_transactions(self) -> int:
        """Number of transactions in the report."""
        return len(self.commissions)

    def calculer_total_commissions(self) -> Decimal:
        """Calculate total commissions from records."""
        return sum(
            (c.commission or Decimal(0)) for c in self.commissions
        )

    def calculer_total_boni(self) -> Decimal:
        """Calculate total bonus from records."""
        return sum(
            (c.boni or Decimal(0)) for c in self.commissions
        )
