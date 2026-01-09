"""
Pydantic models for IDC proposition reports.

These models represent the structure of data extracted from IDC
"Rapport des propositions soumises" PDF reports.
"""

from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class IDCProposition(BaseModel):
    """
    A single proposition record from IDC report.

    Represents one insurance proposition with client, policy,
    and commission information.
    """

    assureur: str = Field(
        ...,
        description="Insurance company name",
        examples=["RBC INSURANCE", "CANADA LIFE"],
    )
    client: str = Field(
        ...,
        description="Client name (format: LASTNAME, F)",
        examples=["SMITH, J", "TREMBLAY, M"],
    )
    type_regime: str = Field(
        ...,
        description="Policy type",
        examples=["Permanent", "Term", "Disability", "Critical Illness"],
    )
    police: str = Field(
        ...,
        description="Policy number",
        examples=["1014157", "ABC123"],
    )
    statut: str = Field(
        ...,
        description="Policy status",
        examples=["Approved", "Inforce", "Pending", "Submitted"],
    )
    date: str = Field(
        ...,
        description="Date (YYYY-MM-DD format)",
        examples=["2025-10-15", "2025-11-01"],
    )
    nombre: Decimal = Field(
        ...,
        description="Quantity/Number between 0.00 and 1.00",
        examples=[1.00, 0.00],
    )
    taux_cpa: Decimal = Field(
        ...,
        description="CPA rate as percentage",
        examples=[100.0, 50.0],
    )
    couverture: str = Field(
        ...,
        description="Coverage amount",
        examples=["100 000,00 $", "250 000,00 $"],
    )
    prime_police: str = Field(
        ...,
        description="Policy premium amount",
        examples=["1 234,56 $", "500,00 $"],
    )
    prime_commissionnable: str = Field(
        ...,
        description="Commissionable premium amount",
        examples=["1 234,56 $", "500,00 $"],
    )
    commission: str = Field(
        ...,
        description="Commission amount",
        examples=["123,45 $", "50,00 $"],
    )

    @field_validator("client")
    @classmethod
    def validate_client(cls, v: str) -> str:
        """Clean and normalize client name."""
        return " ".join(str(v).upper().split())

    @field_validator("assureur")
    @classmethod
    def validate_assureur(cls, v: str) -> str:
        """Clean insurer name."""
        return str(v).strip()


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
    date_rapport: Optional[str] = Field(
        default=None,
        description="Report generation date",
        examples=["2025-10-17"],
    )
    vendeur: Optional[str] = Field(
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
        return list(set(p.assureur for p in self.propositions))

    def propositions_par_type(self) -> dict[str, int]:
        """Count propositions by regime type."""
        counts: dict[str, int] = {}
        for p in self.propositions:
            counts[p.type_regime] = counts.get(p.type_regime, 0) + 1
        return counts
