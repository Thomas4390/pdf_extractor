"""
Pydantic models for Assomption Vie remuneration reports.

These models represent the structure of data extracted from Assomption Vie
PDF reports, combining commission and bonus data into unified records.
"""

from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AssomptionCommission(BaseModel):
    """
    A commission record from Assomption Vie report.

    Found on the "Rapport de commissions" page (usually page 3).
    """

    code: str = Field(
        ...,
        description="Transaction code (e.g., 'AOH1')",
        examples=["AOH1", "BCD2"],
    )
    numero_police: str = Field(
        ...,
        description="Policy number (7 digits)",
        examples=["1011221", "1011452"],
    )
    nom_assure: str = Field(
        ...,
        description="Insured person's name",
        examples=["MUADI MUNYA TSHIMANGA", "DAOUYA TARABET"],
    )
    produit: str = Field(
        ...,
        description="Product code",
        examples=["4T20 B", "5L A"],
    )
    date_emission: str = Field(
        ...,
        description="Issue date (YYYY/MM/DD format)",
        examples=["2025/09/26", "2025/10/01"],
    )
    frequence_paiement: str = Field(
        ...,
        description="Payment frequency",
        examples=["Mensuel", "Annuel"],
    )
    facturation: str = Field(
        ...,
        description="Billing type",
        examples=["COM/PAC"],
    )
    prime: Decimal = Field(
        ...,
        description="Premium amount (can be negative for adjustments)",
        examples=[-142.56, 499.05],
    )
    taux_commission: Decimal = Field(
        ...,
        description="Commission rate as percentage",
        examples=[40.993, 45.0],
    )
    commission: Decimal = Field(
        ...,
        description="Commission amount (can be negative)",
        examples=[-58.44, 224.58],
    )
    # Bonus fields (merged from surcommission page)
    taux_boni: Optional[Decimal] = Field(
        default=None,
        description="Bonus rate as percentage (from surcommission page)",
        examples=[175.0],
    )
    boni: Optional[Decimal] = Field(
        default=None,
        description="Bonus amount (from surcommission page)",
        examples=[-134.24, 393.02],
    )

    @field_validator("numero_police")
    @classmethod
    def validate_police(cls, v: str) -> str:
        """Ensure policy number is clean."""
        return str(v).strip()

    @field_validator("nom_assure")
    @classmethod
    def validate_nom(cls, v: str) -> str:
        """Clean and normalize insured name."""
        return " ".join(str(v).upper().split())


class AssomptionReport(BaseModel):
    """
    Complete Assomption Vie remuneration report.

    Contains metadata and merged commission/bonus records.
    """

    document_type: str = Field(
        default="ASSOMPTION",
        description="Document type identifier",
    )

    # Period information
    periode_debut: str = Field(
        ...,
        description="Pay period start date (YYYY/MM/DD)",
        examples=["2025/10/02"],
    )
    periode_fin: str = Field(
        ...,
        description="Pay period end date (YYYY/MM/DD)",
        examples=["2025/10/06"],
    )
    date_paie: str = Field(
        ...,
        description="Payment date (YYYY/MM/DD)",
        examples=["2025/10/09"],
    )

    # Agent/Broker information
    numero_courtier: str = Field(
        ...,
        description="Broker number",
        examples=["35552"],
    )
    nom_courtier: str = Field(
        ...,
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
        return sum(c.commission for c in self.commissions)

    def calculer_total_boni(self) -> Decimal:
        """Calculate total bonus from records."""
        return sum(c.boni or Decimal(0) for c in self.commissions)
