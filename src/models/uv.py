"""
Pydantic models for UV Assurance remuneration reports.

These models represent the structure of data extracted from UV Assurance
PDF reports, faithful to the original document format.
"""

from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class UVActivity(BaseModel):
    """
    A single activity line from a UV remuneration report.

    Represents one insurance contract/protection with commission details.
    Each activity is associated with a sub-advisor if present.
    """

    # Sub-advisor for this activity (appears at top of each table section)
    sous_conseiller: Optional[str] = Field(
        default=None,
        description="Sub-advisor for this activity (e.g., '21622 - ACHRAF EL HAJJI')",
        examples=["21622 - ACHRAF EL HAJJI", "21650 - DEREK POIRIER", None],
    )

    contrat: str = Field(
        ...,
        description="Contract number (e.g., '110970886')",
        examples=["110970886", "110971504"],
    )
    assure: str = Field(
        ...,
        description="Insured person's name",
        examples=["BALDWIN RAYMOND", "MADJIGUENE SOW"],
    )
    protection: str = Field(
        ...,
        description="Protection type",
        examples=["Vie entière Valeurs Élevées"],
    )
    montant_base: Decimal = Field(
        ...,
        description="Base amount in CAD",
        examples=[1196.00, 699.00],
    )
    taux_partage: Decimal = Field(
        ...,
        description="Sharing rate as percentage (e.g., 100.0 for 100%)",
        examples=[100.0, 40.0],
    )
    taux_commission: Decimal = Field(
        ...,
        description="Commission rate as percentage",
        examples=[55.0],
    )
    resultat: Decimal = Field(
        ...,
        description="Result amount in CAD",
        examples=[657.80, 384.45],
    )
    type_commission: str = Field(
        ...,
        description="Commission type",
        examples=["Boni 1ère année vie"],
    )
    taux_boni: Decimal = Field(
        ...,
        description="Bonus rate as percentage",
        examples=[175.0],
    )
    remuneration: Decimal = Field(
        ...,
        description="Final remuneration in CAD",
        examples=[1151.15, 672.79],
    )

    @field_validator("contrat")
    @classmethod
    def validate_contrat(cls, v: str) -> str:
        """Ensure contract number is clean."""
        return str(v).strip()

    @field_validator("assure")
    @classmethod
    def validate_assure(cls, v: str) -> str:
        """Clean and normalize insured name."""
        return " ".join(str(v).upper().split())


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
    date_rapport: str = Field(
        ...,
        description="Report date in YYYY-MM-DD format",
        examples=["2025-10-13"],
    )
    nom_conseiller: str = Field(
        ...,
        description="Main advisor name or company name",
        examples=["9491-1377 QUEBEC INC"],
    )
    numero_conseiller: str = Field(
        ...,
        description="Main advisor number",
        examples=["21621"],
    )
    activites: list[UVActivity] = Field(
        default_factory=list,
        description="List of activity lines from the report",
    )

    @field_validator("date_rapport")
    @classmethod
    def validate_date(cls, v: str) -> str:
        """Ensure date format is correct."""
        v = str(v).strip()
        # Accept both YYYY-MM-DD and common variations
        if len(v) == 10 and v[4] == "-" and v[7] == "-":
            return v
        # Try to parse other formats
        import re

        match = re.search(r"(\d{4})-(\d{2})-(\d{2})", v)
        if match:
            return match.group(0)
        return v

    @property
    def nombre_contrats(self) -> int:
        """Number of unique contracts in the report."""
        return len({a.contrat for a in self.activites})

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
        return sum(a.remuneration for a in self.activites)

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
            totals[key] += a.remuneration
        return totals
