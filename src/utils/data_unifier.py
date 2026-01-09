"""
Data Unification Module
=======================

Converts Pydantic models from VLM extraction into standardized DataFrames
ready for Monday.com upload.

This module handles:
- Conversion from source-specific Pydantic models to pandas DataFrames
- Mapping to French column names
- Commission calculation using universal formula
- Advisor name normalization
- Board type detection based on source
"""

from decimal import Decimal
from enum import Enum
from typing import Optional, Union
import re

import numpy as np
import pandas as pd

from ..models.uv import UVReport
from ..models.idc import IDCReport
from ..models.idc_statement import IDCStatementReport, IDCStatementReportParsed
from ..models.assomption import AssomptionReport


class BoardType(Enum):
    """Type de board Monday.com."""
    HISTORICAL_PAYMENTS = "HISTORICAL_PAYMENTS"  # Paiements historiques (IDC_STATEMENT)
    SALES_PRODUCTION = "SALES_PRODUCTION"        # Ventes et production (UV, IDC, ASSOMPTION)


class DataUnifier:
    """
    Convertit les modèles Pydantic extraits en DataFrames standardisés.

    Responsabilités:
    - Conversion des modèles Pydantic vers DataFrame pandas
    - Mapping vers les colonnes françaises finales
    - Calcul des commissions avec formule universelle
    - Normalisation des noms de conseillers via AdvisorMatcher
    """

    # Colonnes finales pour Paiements Historiques (13 colonnes)
    FINAL_COLUMNS_HISTORICAL = [
        '# de Police',
        'Nom Client',
        'Compagnie',
        'Statut',
        'Conseiller',
        'Verifié',
        'PA',
        'Com',
        'Boni',
        'Sur-Com',
        'Reçu',
        'Date',
        'Texte',
    ]

    # Colonnes finales pour Ventes et Production (19 colonnes)
    FINAL_COLUMNS_SALES = [
        'Date',
        '# de Police',
        'Nom Client',
        'Compagnie',
        'Statut',
        'Conseiller',
        'Complet',
        'PA',
        'Lead/MC',
        'Com',
        'Reçu 1',
        'Boni',
        'Reçu 2',
        'Sur-Com',
        'Reçu 3',
        'Total',
        'Total Reçu',
        'Paie',
        'Texte',
    ]

    # Mapping source → type de board
    # Seul IDC (Propositions) est pour Ventes et Production
    # UV, ASSOMPTION et IDC_STATEMENT sont pour Paiements Historiques
    SOURCE_TO_BOARD_TYPE = {
        'UV': BoardType.HISTORICAL_PAYMENTS,
        'IDC': BoardType.SALES_PRODUCTION,
        'ASSOMPTION': BoardType.HISTORICAL_PAYMENTS,
        'IDC_STATEMENT': BoardType.HISTORICAL_PAYMENTS,
    }

    # Constantes de calcul
    DEFAULT_SHARING_RATE = 0.4       # 40%
    DEFAULT_COMMISSION_RATE = 0.5    # 50%
    DEFAULT_BONUS_RATE = 1.75        # 175%
    DEFAULT_ON_COMMISSION_RATE = 0.75  # 75%

    def __init__(self, advisor_matcher=None):
        """
        Initialise le DataUnifier.

        Args:
            advisor_matcher: Instance optionnelle d'AdvisorMatcher pour
                           normaliser les noms de conseillers
        """
        self.advisor_matcher = advisor_matcher

    def unify(
        self,
        report: Union[UVReport, IDCReport, IDCStatementReport, IDCStatementReportParsed, AssomptionReport],
        source: str
    ) -> tuple[pd.DataFrame, BoardType]:
        """
        Convertit un rapport Pydantic en DataFrame standardisé.

        Args:
            report: Modèle Pydantic extrait par le VLM
            source: Type de source ('UV', 'IDC', 'IDC_STATEMENT', 'ASSOMPTION')

        Returns:
            Tuple (DataFrame avec colonnes françaises, BoardType)

        Raises:
            ValueError: Si la source est inconnue
        """
        source = source.upper()
        if source not in self.SOURCE_TO_BOARD_TYPE:
            raise ValueError(f"Source inconnue: {source}. Sources supportées: {list(self.SOURCE_TO_BOARD_TYPE.keys())}")

        board_type = self.SOURCE_TO_BOARD_TYPE[source]

        # Conversion spécifique par source
        if source == 'UV':
            df = self._convert_uv(report)
        elif source == 'IDC':
            df = self._convert_idc(report)
        elif source == 'IDC_STATEMENT':
            df = self._convert_idc_statement(report)
        elif source == 'ASSOMPTION':
            df = self._convert_assomption(report)

        # Normaliser les noms de conseillers si matcher disponible
        if self.advisor_matcher and 'Conseiller' in df.columns:
            df['Conseiller'] = df['Conseiller'].apply(
                lambda x: self._normalize_advisor(x) if pd.notna(x) else x
            )

        # Appliquer le schéma de colonnes final
        df = self._apply_final_schema(df, board_type)

        # Stocker le board_type dans les attributs du DataFrame
        df.attrs['board_type'] = board_type.value
        df.attrs['source'] = source

        return df, board_type

    def _normalize_advisor(self, name: str) -> str:
        """Normalise un nom de conseiller via l'AdvisorMatcher."""
        if not name or not self.advisor_matcher:
            return name
        result = self.advisor_matcher.match(str(name))
        return result if result else name

    def _calculate_commission(
        self,
        premium: float,
        sharing_rate: float,
        commission_rate: float
    ) -> float:
        """
        Calcule la commission selon la formule universelle.

        commission = prime × taux_partage × taux_commission

        Args:
            premium: Prime annualisée
            sharing_rate: Taux de partage (0.0-1.0)
            commission_rate: Taux de commission (0.0-1.0)

        Returns:
            Montant de la commission
        """
        if pd.isna(premium) or pd.isna(sharing_rate) or pd.isna(commission_rate):
            return None
        return round(premium * sharing_rate * commission_rate, 2)

    def _clean_currency(self, value) -> Optional[float]:
        """
        Nettoie et convertit les valeurs monétaires en float.

        Gère les formats: "1 196,00 $", "348,5 $", "50 000 $"
        """
        if pd.isna(value) or value == '' or value is None:
            return None

        try:
            value_str = str(value).replace('$', '').replace(' ', '').replace(',', '.')
            return float(value_str)
        except (ValueError, AttributeError):
            return None

    def _clean_percentage(self, value) -> Optional[float]:
        """
        Nettoie et convertit les pourcentages en float (0.0-1.0).

        Gère les formats: "55,000 %", "175,00%", "40.5%"
        """
        if pd.isna(value) or value == '' or value is None:
            return None

        try:
            value_str = str(value).replace('%', '').replace(' ', '').replace(',', '.')
            # Diviser par 100 pour obtenir un ratio 0.0-1.0
            return float(value_str) / 100
        except (ValueError, AttributeError):
            return None

    def _format_date(self, value) -> Optional[str]:
        """
        Formate une date en YYYY-MM-DD.

        Gère les formats: YYYY-MM-DD, YYYY/MM/DD, DD/MM/YYYY
        """
        if pd.isna(value) or value == '' or value is None:
            return None

        try:
            if isinstance(value, pd.Timestamp):
                return value.strftime('%Y-%m-%d')

            date_str = str(value).strip()

            # Essayer différents formats
            formats = ['%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%d-%m-%Y']
            for fmt in formats:
                try:
                    return pd.to_datetime(date_str, format=fmt).strftime('%Y-%m-%d')
                except (ValueError, AttributeError):
                    continue

            # Dernier recours: laisser pandas deviner
            return pd.to_datetime(date_str).strftime('%Y-%m-%d')

        except (ValueError, AttributeError):
            return None

    def _is_corporate_advisor(self, advisor_name: str) -> bool:
        """
        Détecte si le conseiller est une corporation (Inc).

        Une corporation contient généralement des chiffres
        (ex: "9491-1377 QUEBEC INC").
        """
        if not advisor_name or pd.isna(advisor_name):
            return False
        return bool(re.search(r'\d', str(advisor_name)))

    def _decimal_to_float(self, value) -> Optional[float]:
        """Convertit une valeur Decimal en float."""
        if pd.isna(value) or value is None:
            return None
        if isinstance(value, Decimal):
            return float(value)
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    # =========================================================================
    # CONVERTISSEURS PAR SOURCE
    # =========================================================================

    def _convert_uv(self, report: UVReport) -> pd.DataFrame:
        """
        Convertit un rapport UV en DataFrame standardisé.

        UV Assurance → Board HISTORICAL_PAYMENTS
        """
        if not report.activites:
            return pd.DataFrame(columns=self.FINAL_COLUMNS_HISTORICAL)

        rows = []
        for act in report.activites:
            # Déterminer le type d'assureur (Inc vs Perso)
            is_corporate = self._is_corporate_advisor(report.nom_conseiller)
            insurer_name = 'UV Inc' if is_corporate else 'UV Perso'

            # Extraire le nom du sous-conseiller si présent
            advisor_name = None
            if report.sous_conseiller:
                # Format: "21622 - ACHRAF EL HAJJI"
                parts = report.sous_conseiller.split(' - ', 1)
                advisor_name = parts[1] if len(parts) > 1 else report.sous_conseiller
            else:
                advisor_name = report.nom_conseiller

            # Convertir les valeurs Decimal
            premium = self._decimal_to_float(act.montant_base)
            sharing_rate = self._decimal_to_float(act.taux_partage) / 100 if act.taux_partage else None
            commission_rate = self._decimal_to_float(act.taux_commission) / 100 if act.taux_commission else None
            bonus_rate = self._decimal_to_float(act.taux_boni) / 100 if act.taux_boni else None
            remuneration = self._decimal_to_float(act.remuneration)

            # Calculer les commissions
            commission = self._calculate_commission(premium, sharing_rate, commission_rate) if sharing_rate and commission_rate else None
            bonus = round(commission * bonus_rate, 2) if commission and bonus_rate else None
            on_commission = self._calculate_commission(premium, (1 - sharing_rate) if sharing_rate else 0, self.DEFAULT_ON_COMMISSION_RATE * commission_rate if commission_rate else 0)

            row = {
                '# de Police': str(act.contrat),
                'Nom Client': act.assure,
                'Compagnie': insurer_name,
                'Statut': 'Approuvé',  # UV = toujours approuvé
                'Conseiller': advisor_name,
                'Verifié': None,
                'PA': premium,
                'Com': commission,
                'Boni': bonus,
                'Sur-Com': on_commission,
                'Reçu': remuneration,  # Rémunération UV = montant reçu
                'Date': self._format_date(report.date_rapport),
                'Texte': act.type_commission,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def _convert_idc(self, report: IDCReport) -> pd.DataFrame:
        """
        Convertit un rapport IDC (Propositions) en DataFrame standardisé.

        IDC Propositions → Board SALES_PRODUCTION
        """
        if not report.propositions:
            return pd.DataFrame(columns=self.FINAL_COLUMNS_SALES)

        rows = []
        for prop in report.propositions:
            # Nettoyer le nom de l'assureur
            insurer_name = prop.assureur
            if 'Assumption Life' in insurer_name or 'ASSUMPTI ON' in insurer_name:
                insurer_name = 'Assomption'

            # Convertir les valeurs
            premium = self._clean_currency(prop.prime_police)
            sharing_rate = self._decimal_to_float(prop.nombre)  # Nombre est déjà 0.0-1.0
            commission_rate = self._clean_percentage(str(prop.taux_cpa) + '%') if prop.taux_cpa else None
            commission_total = self._clean_currency(prop.commission)

            # Statut: Approved → Approuvé, sinon En attente
            status = 'Approuvé' if str(prop.statut).strip().lower() == 'approved' else 'En attente'

            # Calculer les commissions avec formule standard
            commission = self._calculate_commission(premium, sharing_rate, commission_rate) if sharing_rate and commission_rate else None
            bonus = round(commission * self.DEFAULT_BONUS_RATE, 2) if commission else None
            on_commission = self._calculate_commission(
                premium,
                (1 - sharing_rate) if sharing_rate else 0,
                self.DEFAULT_ON_COMMISSION_RATE * commission_rate if commission_rate else 0
            ) if commission_rate else None

            # Total
            total = sum(filter(None, [commission, bonus, on_commission])) or None

            row = {
                'Date': self._format_date(prop.date),
                '# de Police': str(prop.police),
                'Nom Client': prop.client,
                'Compagnie': insurer_name,
                'Statut': status,
                'Conseiller': report.vendeur,
                'Complet': None,
                'PA': premium,
                'Lead/MC': 'Lead',
                'Com': commission,
                'Reçu 1': None,
                'Boni': bonus,
                'Reçu 2': None,
                'Sur-Com': on_commission,
                'Reçu 3': None,
                'Total': total,
                'Total Reçu': None,
                'Paie': None,
                'Texte': f"{prop.type_regime} - {prop.couverture}",
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def _convert_idc_statement(
        self,
        report: Union[IDCStatementReport, IDCStatementReportParsed]
    ) -> pd.DataFrame:
        """
        Convertit un relevé IDC Statement en DataFrame standardisé.

        IDC Statement (Trailing Fees) → Board HISTORICAL_PAYMENTS
        """
        if not report.trailing_fees:
            return pd.DataFrame(columns=self.FINAL_COLUMNS_HISTORICAL)

        rows = []
        for fee in report.trailing_fees:
            # Extraire les données selon le type de modèle (Raw ou Parsed)
            if hasattr(fee, 'client_full_name'):
                # IDCTrailingFeeParsed
                client_name = fee.client_full_name
                advisor_name = fee.advisor_name
                policy_number = fee.policy_number or fee.account_number
            else:
                # IDCTrailingFeeRaw - parser raw_client_data
                client_name = self._parse_client_from_raw(fee.raw_client_data)
                advisor_name = self._parse_advisor_from_raw(fee.raw_client_data)
                policy_number = self._parse_policy_from_raw(fee.raw_client_data) or fee.account_number

            # Convertir le montant des frais de suivi
            trailing_fee = self._clean_currency(fee.net_trailing_fee)

            # Déterminer le statut basé sur le montant
            if trailing_fee is not None:
                if trailing_fee > 0:
                    status = 'Payé'
                elif trailing_fee < 0:
                    status = 'Charge back'
                else:
                    status = None
            else:
                status = None

            row = {
                '# de Police': str(policy_number) if policy_number else 'Unknown',
                'Nom Client': client_name or 'Unknown',
                'Compagnie': fee.company,
                'Statut': status,
                'Conseiller': advisor_name,
                'Verifié': None,  # Sera calculé plus tard si nécessaire
                'PA': None,  # Pas de prime dans les relevés
                'Com': None,
                'Boni': None,
                'Sur-Com': trailing_fee,  # Les frais de suivi vont dans Sur-Com
                'Reçu': trailing_fee,
                'Date': self._format_date(fee.date),
                'Texte': fee.product,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def _parse_client_from_raw(self, raw_data: str) -> Optional[str]:
        """
        Parse le nom du client depuis raw_client_data.

        Format typique: "... clt Jeanny\nBreault-Therrien"
        """
        if not raw_data:
            return None
        # Chercher après "clt " ou à la fin
        match = re.search(r'clt\s+(.+?)(?:\n|$)', raw_data, re.IGNORECASE)
        if match:
            # Prendre le reste jusqu'à la fin
            parts = raw_data[match.start():].split('\n')
            if len(parts) >= 2:
                return f"{parts[0].replace('clt', '').strip()} {parts[1].strip()}"
            return match.group(1).strip()
        return None

    def _parse_advisor_from_raw(self, raw_data: str) -> Optional[str]:
        """
        Parse le nom du conseiller depuis raw_client_data.

        Format typique: "... Bourassa A clt ..."
        """
        if not raw_data:
            return None
        # Chercher un nom avant "clt"
        match = re.search(r'([A-Za-z]+\s+[A-Z])\s+clt', raw_data)
        if match:
            return match.group(1).strip()
        return None

    def _parse_policy_from_raw(self, raw_data: str) -> Optional[str]:
        """
        Parse le numéro de police depuis raw_client_data.

        Format typique: "... #111011722 ..."
        """
        if not raw_data:
            return None
        match = re.search(r'#(\d+)', raw_data)
        if match:
            return match.group(1)
        return None

    def _convert_assomption(self, report: AssomptionReport) -> pd.DataFrame:
        """
        Convertit un rapport Assomption Vie en DataFrame standardisé.

        Assomption Vie → Board HISTORICAL_PAYMENTS
        """
        if not report.commissions:
            return pd.DataFrame(columns=self.FINAL_COLUMNS_HISTORICAL)

        rows = []
        for comm in report.commissions:
            # Convertir les valeurs Decimal
            premium = self._decimal_to_float(comm.prime)
            commission_rate = self._decimal_to_float(comm.taux_commission) / 100 if comm.taux_commission else None
            commission_amount = self._decimal_to_float(comm.commission)
            bonus_rate = self._decimal_to_float(comm.taux_boni) / 100 if comm.taux_boni else None
            bonus_amount = self._decimal_to_float(comm.boni)

            # Assomption a un taux de partage fixe de 40%
            sharing_rate = self.DEFAULT_SHARING_RATE

            # Calculer les commissions avec la formule standard
            calculated_commission = self._calculate_commission(premium, sharing_rate, commission_rate) if commission_rate else None
            calculated_bonus = round(calculated_commission * bonus_rate, 2) if calculated_commission and bonus_rate else None
            on_commission = self._calculate_commission(
                premium,
                (1 - sharing_rate),
                self.DEFAULT_ON_COMMISSION_RATE * commission_rate if commission_rate else 0
            ) if commission_rate else None

            # Montant reçu = commission + bonus de Assomption
            received = sum(filter(None, [commission_amount, bonus_amount])) or None

            row = {
                '# de Police': str(comm.numero_police),
                'Nom Client': comm.nom_assure,
                'Compagnie': 'Assomption',
                'Statut': 'Approuvé',
                'Conseiller': report.nom_courtier,
                'Verifié': None,
                'PA': premium,
                'Com': calculated_commission,
                'Boni': calculated_bonus,
                'Sur-Com': on_commission,
                'Reçu': received,  # Montant total reçu
                'Date': self._format_date(comm.date_emission),
                'Texte': f"{comm.code} - {comm.produit} ({comm.frequence_paiement})",
            }
            rows.append(row)

        return pd.DataFrame(rows)

    # =========================================================================
    # APPLICATION DU SCHÉMA FINAL
    # =========================================================================

    def _apply_final_schema(self, df: pd.DataFrame, board_type: BoardType) -> pd.DataFrame:
        """
        Applique le schéma de colonnes final selon le type de board.

        Args:
            df: DataFrame avec les données converties
            board_type: Type de board (HISTORICAL_PAYMENTS ou SALES_PRODUCTION)

        Returns:
            DataFrame avec uniquement les colonnes finales dans le bon ordre
        """
        if board_type == BoardType.HISTORICAL_PAYMENTS:
            columns = self.FINAL_COLUMNS_HISTORICAL
        else:
            columns = self.FINAL_COLUMNS_SALES

        # Calculer les colonnes dérivées pour SALES_PRODUCTION
        if board_type == BoardType.SALES_PRODUCTION:
            # Total Reçu = Reçu 1 + Reçu 2 + Reçu 3
            if 'Total Reçu' not in df.columns or df['Total Reçu'].isna().all():
                recu1 = pd.to_numeric(df.get('Reçu 1', pd.Series(dtype=float)), errors='coerce').fillna(0)
                recu2 = pd.to_numeric(df.get('Reçu 2', pd.Series(dtype=float)), errors='coerce').fillna(0)
                recu3 = pd.to_numeric(df.get('Reçu 3', pd.Series(dtype=float)), errors='coerce').fillna(0)
                df['Total Reçu'] = recu1 + recu2 + recu3
                # Remplacer 0 par None si tous les reçus sont None
                mask = (
                    df.get('Reçu 1', pd.Series(dtype=float)).isna() &
                    df.get('Reçu 2', pd.Series(dtype=float)).isna() &
                    df.get('Reçu 3', pd.Series(dtype=float)).isna()
                )
                df.loc[mask, 'Total Reçu'] = None

            # Paie = Total - Total Reçu
            if 'Paie' not in df.columns or df['Paie'].isna().all():
                total = pd.to_numeric(df.get('Total', pd.Series(dtype=float)), errors='coerce').fillna(0)
                total_recu = pd.to_numeric(df.get('Total Reçu', pd.Series(dtype=float)), errors='coerce').fillna(0)
                df['Paie'] = total - total_recu

        # Ajouter les colonnes manquantes avec valeurs par défaut
        for col in columns:
            if col not in df.columns:
                df[col] = None

        # Réordonner et filtrer aux colonnes finales uniquement
        return df[columns].copy()

    # =========================================================================
    # UTILITAIRES
    # =========================================================================

    def get_board_type_for_source(self, source: str) -> BoardType:
        """
        Retourne le type de board pour une source donnée.

        Args:
            source: Type de source ('UV', 'IDC', 'IDC_STATEMENT', 'ASSOMPTION')

        Returns:
            BoardType correspondant

        Raises:
            ValueError: Si la source est inconnue
        """
        source = source.upper()
        if source not in self.SOURCE_TO_BOARD_TYPE:
            raise ValueError(f"Source inconnue: {source}")
        return self.SOURCE_TO_BOARD_TYPE[source]

    def get_final_columns(self, board_type: BoardType) -> list[str]:
        """
        Retourne la liste des colonnes finales pour un type de board.

        Args:
            board_type: Type de board

        Returns:
            Liste des noms de colonnes
        """
        if board_type == BoardType.HISTORICAL_PAYMENTS:
            return self.FINAL_COLUMNS_HISTORICAL.copy()
        return self.FINAL_COLUMNS_SALES.copy()
