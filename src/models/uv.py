"""
Pydantic models for UV Assurance remuneration reports.

These models represent the structure of data extracted from UV Assurance
PDF reports, faithful to the original document format.
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

class UVActivity(BaseModel):
    """
    A single activity line from a UV remuneration report.

    Represents one insurance contract/protection with commission details.
    Each activity is associated with a sub-advisor if present.
    """

    # Sub-advisor for this activity (appears at top of each table section)
    sous_conseiller: FlexibleString = Field(
        default=None,
        description="Sub-advisor for this activity (e.g., '21622 - ACHRAF EL HAJJI')",
        examples=["21622 - ACHRAF EL HAJJI", "21650 - DEREK POIRIER", None],
    )

    # Contract identification - required but flexible
    contrat: FlexibleString = Field(
        default=None,
        description="Contract number (e.g., '110970886')",
        examples=["110970886", "110971504"],
    )
    assure: FlexibleString = Field(
        default=None,
        description="Insured person's name",
        examples=["BALDWIN RAYMOND", "MADJIGUENE SOW"],
    )
    protection: FlexibleString = Field(
        default=None,
        description="Protection type",
        examples=["Vie entière Valeurs Élevées"],
    )

    # Financial data - flexible decimals
    montant_base: FlexibleDecimal = Field(
        default=None,
        description="Base amount in CAD",
        examples=[1196.00, 699.00],
    )
    taux_partage: FlexibleDecimal = Field(
        default=None,
        description="Sharing rate as percentage (e.g., 100.0 for 100%)",
        examples=[100.0, 40.0],
    )
    taux_commission: FlexibleDecimal = Field(
        default=None,
        description="Commission rate as percentage",
        examples=[55.0],
    )
    resultat: FlexibleDecimal = Field(
        default=None,
        description="Result amount in CAD",
        examples=[657.80, 384.45],
    )
    type_commission: FlexibleString = Field(
        default=None,
        description="Commission type",
        examples=["Boni 1ère année vie"],
    )
    taux_boni: FlexibleDecimal = Field(
        default=None,
        description="Bonus rate as percentage",
        examples=[175.0],
    )
    remuneration: FlexibleDecimal = Field(
        default=None,
        description="Final remuneration in CAD",
        examples=[1151.15, 672.79],
    )

    @field_validator("assure", mode="after")
    @classmethod
    def normalize_assure(cls, v: Optional[str]) -> Optional[str]:
        """Clean and normalize insured name."""
        if v:
            return " ".join(v.upper().split())
        return v


class UVReport(BaseModel):
    """
    Complete UV Assurance remuneration report.

    Contains metadata about the report and advisor, plus all activity lines.
    A report can have multiple sub-advisors, each with their own activities.
    """

    document_type: str = Field(
        default="UV",
        description="Document type identifier",
    )
    date_rapport: FlexibleString = Field(
        default=None,
        description="Report date in YYYY-MM-DD format",
        examples=["2025-10-13"],
    )
    nom_conseiller: FlexibleString = Field(
        default=None,
        description="Main advisor name or company name",
        examples=["9491-1377 QUEBEC INC"],
    )
    numero_conseiller: FlexibleString = Field(
        default=None,
        description="Main advisor number",
        examples=["21621"],
    )
    activites: list[UVActivity] = Field(
        default_factory=list,
        description="List of activity lines from the report",
    )

    @field_validator("date_rapport", mode="after")
    @classmethod
    def validate_date(cls, v: Optional[str]) -> Optional[str]:
        """Ensure date format is correct if provided."""
        if not v:
            return v

        v = str(v).strip()
        # Accept YYYY-MM-DD format
        if len(v) == 10 and v[4] == "-" and v[7] == "-":
            return v

        # Try to extract date from string
        import re
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})", v)
        if match:
            return match.group(0)
        return v

    @property
    def nombre_contrats(self) -> int:
        """Number of unique contracts in the report."""
        return len({a.contrat for a in self.activites if a.contrat})

    @property
    def nombre_activites(self) -> int:
        """Total number of activity lines."""
        return len(self.activites)

    @property
    def sous_conseillers_uniques(self) -> list[str]:
        """List of unique sub-advisors in the report."""
        return list(set(
            a.sous_conseiller for a in self.activites
            if a.sous_conseiller
        ))

    @property
    def nombre_sous_conseillers(self) -> int:
        """Number of unique sub-advisors."""
        return len(self.sous_conseillers_uniques)

    def activites_par_sous_conseiller(self) -> dict[str, list[UVActivity]]:
        """Group activities by sub-advisor.

        Returns:
            Dictionary mapping sub-advisor name to list of activities.
            Activities without sub-advisor are grouped under "Principal".
        """
        groups: dict[str, list[UVActivity]] = {}
        for a in self.activites:
            key = a.sous_conseiller or "Principal"
            if key not in groups:
                groups[key] = []
            groups[key].append(a)
        return groups

    def calculer_total(self) -> Decimal:
        """Calculate total remuneration from activities."""
        return sum(
            (a.remuneration or Decimal(0)) for a in self.activites
        )

    def calculer_total_par_sous_conseiller(self) -> dict[str, Decimal]:
        """Calculate total remuneration per sub-advisor.

        Returns:
            Dictionary mapping sub-advisor name to total remuneration.
        """
        totals: dict[str, Decimal] = {}
        for a in self.activites:
            key = a.sous_conseiller or "Principal"
            if key not in totals:
                totals[key] = Decimal(0)
            totals[key] += a.remuneration or Decimal(0)
        return totals
