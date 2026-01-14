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
- Google Sheets advisor normalization
"""

import os
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import re

import pandas as pd

# Try to import Google Sheets client
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to import Streamlit for secrets access
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

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

    def __init__(self, advisor_matcher=None, auto_load_matcher: bool = True):
        """
        Initialise le DataUnifier.

        Args:
            advisor_matcher: Instance optionnelle d'AdvisorMatcher pour
                           normaliser les noms de conseillers
            auto_load_matcher: Si True et advisor_matcher est None, charge
                              automatiquement l'AdvisorMatcher (défaut: True)
        """
        if advisor_matcher is not None:
            self.advisor_matcher = advisor_matcher
        elif auto_load_matcher:
            from .advisor_matcher import get_advisor_matcher
            self.advisor_matcher = get_advisor_matcher()
        else:
            self.advisor_matcher = None

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

        # Agrégation par numéro de police pour Ventes et Production
        if board_type == BoardType.SALES_PRODUCTION and not df.empty:
            df = self._aggregate_by_policy(df)

        # Stocker le board_type dans les attributs du DataFrame
        df.attrs['board_type'] = board_type.value
        df.attrs['source'] = source

        return df, board_type

    def _normalize_advisor(self, name: str) -> str:
        """
        Normalise un nom de conseiller via l'AdvisorMatcher.

        Retourne le format compact "Prénom, Initiale" (ex: "Guillaume, S")
        en matchant avec la base de données des conseillers.
        """
        if not name or not self.advisor_matcher:
            return name
        # Utilise match_compact pour obtenir le format "Prénom, Initiale"
        result = self.advisor_matcher.match_compact(str(name))
        return result if result else name

    def _aggregate_by_policy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrège les données par numéro de police pour le board Ventes et Production.

        Logique d'agrégation:
        - Texte (Client, Compagnie, Statut, Conseiller, etc.): prend la première valeur non-nulle
        - Numériques (PA, Com, Boni, etc.): somme des valeurs
        - Date: prend la plus récente

        Args:
            df: DataFrame avec colonnes SALES_PRODUCTION

        Returns:
            DataFrame agrégé par numéro de police
        """
        if df.empty or '# de Police' not in df.columns:
            return df

        # Colonnes textuelles - prendre la première valeur non-nulle
        text_cols = ['Nom Client', 'Compagnie', 'Statut', 'Conseiller', 'Complet', 'Lead/MC', 'Texte']

        # Colonnes numériques - sommer
        numeric_cols = ['PA', 'Com', 'Reçu 1', 'Boni', 'Reçu 2', 'Sur-Com', 'Reçu 3', 'Total', 'Total Reçu', 'Paie']

        # Colonne date - prendre la plus récente
        date_col = 'Date'

        # Vérifier quelles colonnes existent
        existing_text_cols = [c for c in text_cols if c in df.columns]
        existing_numeric_cols = [c for c in numeric_cols if c in df.columns]

        # Construire le dictionnaire d'agrégation
        agg_dict = {}

        # Fonction helper pour prendre la première valeur non-nulle
        def first_non_null(x):
            non_null = x.dropna()
            return non_null.iloc[0] if len(non_null) > 0 else None

        # Pour les colonnes textuelles: prendre la première valeur non-nulle
        for col in existing_text_cols:
            agg_dict[col] = first_non_null

        # Pour les colonnes numériques: sommer
        for col in existing_numeric_cols:
            agg_dict[col] = 'sum'

        # Pour la date: prendre la plus récente
        if date_col in df.columns:
            agg_dict[date_col] = 'max'

        # Si pas de colonnes à agréger, retourner tel quel
        if not agg_dict:
            return df

        # Nombre de lignes avant agrégation
        rows_before = len(df)

        # Grouper par numéro de police
        df_grouped = df.groupby('# de Police', as_index=False).agg(agg_dict)

        # Réordonner les colonnes selon le schéma final
        final_cols = [c for c in self.FINAL_COLUMNS_SALES if c in df_grouped.columns]
        df_grouped = df_grouped[final_cols]

        # Info si agrégation a eu lieu
        rows_after = len(df_grouped)
        if rows_before > rows_after:
            print(f"  ℹ️  Agrégation: {rows_before} → {rows_after} lignes (par numéro de police)")

        return df_grouped

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

        Mapping JSON → DataFrame:
        - act.resultat → 'Com' (commission extraite du PDF)
        - act.remuneration → 'Reçu' (total reçu incluant boni)
        - act.montant_base → 'PA' (prime annualisée)

        FILTRE: Conserve UNIQUEMENT les activités avec taux_partage != 100%
        (exclut les lignes à 100% de partage)
        """
        if not report.activites:
            return pd.DataFrame(columns=self.FINAL_COLUMNS_HISTORICAL)

        rows = []
        filtered_count = 0

        for act in report.activites:
            # Filtrer les activités avec taux de partage == 100%
            # On ne conserve QUE les lignes avec partage != 100%
            sharing_rate = self._decimal_to_float(act.taux_partage)
            if sharing_rate is None or sharing_rate == 100.0:
                filtered_count += 1
                continue  # Exclure les lignes à 100%

            # Déterminer le type d'assureur (Inc vs Perso)
            is_corporate = self._is_corporate_advisor(report.nom_conseiller)
            insurer_name = 'UV Inc' if is_corporate else 'UV Perso'

            # Extraire le nom du sous-conseiller si présent
            advisor_name = None
            if act.sous_conseiller:
                # Format: "21622 - ACHRAF EL HAJJI"
                parts = act.sous_conseiller.split(' - ', 1)
                advisor_name = parts[1] if len(parts) > 1 else act.sous_conseiller
            else:
                advisor_name = report.nom_conseiller

            # Convertir les valeurs Decimal - utiliser les valeurs EXTRAITES du PDF
            premium = self._decimal_to_float(act.montant_base)
            commission = self._decimal_to_float(act.resultat)  # Commission extraite du PDF
            remuneration = self._decimal_to_float(act.remuneration)  # Total reçu
            bonus_rate = self._decimal_to_float(act.taux_boni) / 100 if act.taux_boni else None
            commission_rate = self._decimal_to_float(act.taux_commission)

            # Calculer le boni basé sur la commission extraite
            bonus = round(commission * bonus_rate, 2) if commission and bonus_rate else None

            # Sur-Com n'est pas directement disponible dans UV, on le met à None
            on_commission = None

            # Construire le texte avec protection, taux de partage et taux de commission
            sharing_rate_str = f"{int(sharing_rate)}%" if sharing_rate else "?"
            commission_rate_str = f"{int(commission_rate)}%" if commission_rate else "?"
            protection_str = act.protection or "N/A"
            texte = f"{protection_str} | {act.type_commission} (Partage: {sharing_rate_str}, Com: {commission_rate_str})"

            # Déterminer le statut basé sur la PA (prime annualisée)
            if premium is not None:
                if premium > 0:
                    status = 'Payé'
                elif premium < 0:
                    status = 'Charge back'
                else:
                    status = None
            else:
                status = None

            row = {
                '# de Police': str(act.contrat),
                'Nom Client': act.assure,
                'Compagnie': insurer_name,
                'Statut': status,
                'Conseiller': advisor_name,
                'Verifié': None,
                'PA': premium,
                'Com': commission,  # Utilise la commission extraite du JSON (act.resultat)
                'Boni': bonus,
                'Sur-Com': on_commission,
                'Reçu': remuneration,  # Utilise la rémunération extraite du JSON
                'Date': self._format_date(report.date_rapport),
                'Texte': texte,
            }
            rows.append(row)

        if filtered_count > 0:
            print(f"  ℹ️  UV: {filtered_count} ligne(s) exclue(s) (taux de partage = 100%)")

        df = pd.DataFrame(rows)

        # Appliquer le ffill conditionnel du numéro de police par nom de client
        df = self._ffill_policy_by_client(df)

        return df

    def _convert_idc(self, report: IDCReport) -> pd.DataFrame:
        """
        Convertit un rapport IDC (Propositions) en DataFrame standardisé.

        IDC Propositions → Board SALES_PRODUCTION

        Mapping JSON → DataFrame:
        - prop.commission → 'Com' (commission extraite du PDF)
        - prop.prime_police → 'PA' (prime annualisée)
        - prop.taux_cpa → utilisé pour calculer Boni et Sur-Com
        """
        if not report.propositions:
            return pd.DataFrame(columns=self.FINAL_COLUMNS_SALES)

        rows = []
        for prop in report.propositions:
            # Normaliser le nom de l'assureur vers les abréviations standardisées
            insurer_name = self._normalize_insurer_name(prop.assureur)

            # Convertir les valeurs extraites du JSON
            premium = self._clean_currency(prop.prime_police)
            commission = self._clean_currency(prop.commission)  # Commission extraite directement du PDF

            # Taux pour calculs dérivés
            commission_rate = self._clean_percentage(str(prop.taux_cpa) + '%') if prop.taux_cpa else None

            # Statut mapping
            statut_lower = str(prop.statut).strip().lower()
            if statut_lower in ['approved', 'inforce', 'issued']:
                status = 'Approuvé'
            elif statut_lower in ['not taken', 'declined']:
                status = 'Refusé'
            else:
                status = 'En attente'

            # Calculer Boni et Sur-Com basés sur la commission extraite
            bonus = round(commission * self.DEFAULT_BONUS_RATE, 2) if commission else None
            # Sur-Com = commission sur la part non-partagée (si applicable)
            on_commission = None  # IDC ne fournit pas cette info directement

            # Total = Com + Boni
            total = sum(filter(None, [commission, bonus])) or None

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
                'Com': commission,  # Utilise la commission extraite du JSON
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
                # Use company_code from parsed data if available (overrides default)
                company_name = fee.company_code if hasattr(fee, 'company_code') and fee.company_code else fee.company
            else:
                # IDCTrailingFeeRaw - parser raw_client_data
                client_name = self._parse_client_from_raw(fee.raw_client_data)
                advisor_name = self._parse_advisor_from_raw(fee.raw_client_data)
                policy_number = self._parse_policy_from_raw(fee.raw_client_data) or fee.account_number
                # Try to extract company from raw_client_data
                company_name = self._parse_company_from_raw(fee.raw_client_data) or fee.company

            # Normalize company name using the standard mapping
            company_name = self._normalize_insurer_name(company_name) if company_name else fee.company

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
                'Compagnie': company_name,
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

        Formats supportés:
        - "... clt FLORA GOMOUE" (avec clt explicite)
        - "... #111065553 crt Legrand DB clt FLORA GOMOUE" (format complet)
        - "... 1014289-Mifoubdou_crt ..." (client entre police et _crt, sans clt)
        - "Jean Dupont" (nom seul sans métadonnées)
        """
        if not raw_data:
            return None

        # Format 1: après "clt" explicite - le plus fiable
        match = re.search(r'clt\s+(.+?)(?:\n|$)', raw_data, re.IGNORECASE)
        if match:
            # Prendre tout après "clt" jusqu'à la fin de la ligne ou fin de texte
            client_part = match.group(1).strip()
            # Si multilignes, prendre les lignes qui suivent aussi
            parts = raw_data[match.end():].split('\n')
            if parts and parts[0].strip() and not any(kw in parts[0].lower() for kw in ['crt', 'boni', '%', 'recu']):
                # La ligne suivante fait partie du nom
                client_part = f"{client_part} {parts[0].strip()}"
            return client_part.upper()

        # Format 2: client entre numéro de police et "_crt" (ex: "1014289-Mifoubdou_crt")
        match = re.search(r'#?\d{6,10}[-_]([A-Za-z]+(?:\s+[A-Za-z]+)*)(?:_crt|crt|\s+crt)', raw_data)
        if match:
            return match.group(1).strip().upper()

        # Format 3: nom seul (pas de métadonnées - pas de %, pas de #, pas de crt)
        if not any(marker in raw_data for marker in ['%', '#', 'crt', 'boni', 'recu', 'Â ']):
            # C'est probablement juste un nom de client
            return raw_data.strip().upper()

        return None

    def _parse_advisor_from_raw(self, raw_data: str) -> Optional[str]:
        """
        Parse le nom du conseiller depuis raw_client_data.

        Le conseiller est TOUJOURS après "crt".

        Formats:
        - "crt Bourassa A clt ..." → "Bourassa A"
        - "crt Legrand DB clt ..." → "Legrand DB"
        - "_crt Lussier,T_2025-12-01-EZ" → "Lussier,T"
        - "crt\nBourassa_2025-12-01-EZ" → "Bourassa"
        """
        if not raw_data:
            return None

        # Patterns pour trouver le conseiller après "crt"
        patterns = [
            # Format: "crt Nom Prénom clt ..."
            r'crt\s+([A-Za-z,]+(?:\s+[A-Za-z])?)\s+clt',
            # Format: "crt Nom Prénom\n" (fin de ligne)
            r'crt\s+([A-Za-z,]+(?:\s+[A-Za-z])?)\s*\n',
            # Format: "_crt Nom_date" ou "_crt Nom,I_date"
            r'_crt\s*([A-Za-z,]+(?:\s*[A-Za-z])?)(?:_\d{4}|$)',
            # Format: "crt Nom" suivi de date ou fin
            r'crt\s+([A-Za-z,]+(?:\s+[A-Za-z]{1,2})?)',
        ]

        for pattern in patterns:
            match = re.search(pattern, raw_data, re.IGNORECASE)
            if match:
                advisor = match.group(1).strip()
                # Nettoyer le suffixe de date si présent
                advisor = re.sub(r'_\d{4}-\d{2}-\d{2}.*$', '', advisor)
                if advisor:
                    return advisor

        return None

    def _parse_policy_from_raw(self, raw_data: str) -> Optional[str]:
        """
        Parse le numéro de police depuis raw_client_data.

        Formats:
        - "#111011722 ..." → "111011722"
        - "1014289-Nom_crt" → "1014289"
        """
        if not raw_data:
            return None

        # Format 1: après #
        match = re.search(r'#(\d{6,10})', raw_data)
        if match:
            return match.group(1)

        # Format 2: numéro suivi de tiret et nom (ex: "1014289-Mifoubdou")
        match = re.search(r'(\d{6,10})[-_][A-Za-z]', raw_data)
        if match:
            return match.group(1)

        return None

    def _parse_company_from_raw(self, raw_data: str) -> Optional[str]:
        """
        Parse le nom de la compagnie depuis raw_client_data.

        L'information de la compagnie se trouve souvent au début du raw_client_data.

        Formats supportés:
        - "Â UV 7782 2025-11-17..." → "UV"
        - "UV 7782 2025-11-17..." → "UV"
        - "Assomption_8055_2025-10-15..." → "Assomption"
        - "IA 1234 2025-10-15..." → "IA"
        - "Beneva_..." → "Beneva"
        """
        if not raw_data:
            return None

        # Clean up the data - remove special characters at start
        clean_data = raw_data.strip()
        if clean_data.startswith('Â '):
            clean_data = clean_data[2:]

        # List of known company names to search for
        known_companies = [
            'UV', 'Assomption', 'IA', 'Industrial Alliance',
            'Beneva', 'RBC', 'ManuVie', 'Manulife', 'Manuvie',
            'Humania', 'Sun Life', 'Canada Life', 'Desjardins',
            'Empire', 'Equitable'
        ]

        # Check at the start of the string (most common pattern)
        for company in known_companies:
            # Pattern: company name followed by space/underscore and numbers
            pattern = rf'^{re.escape(company)}[\s_]\d'
            if re.search(pattern, clean_data, re.IGNORECASE):
                return company

        # Check anywhere in first line
        first_line = clean_data.split('\n')[0] if '\n' in clean_data else clean_data
        for company in known_companies:
            if company.upper() in first_line.upper():
                return company

        return None

    def _convert_assomption(self, report: AssomptionReport) -> pd.DataFrame:
        """
        Convertit un rapport Assomption Vie en DataFrame standardisé.

        Assomption Vie → Board HISTORICAL_PAYMENTS

        Mapping JSON → DataFrame:
        - comm.commission → 'Com' (commission extraite du PDF)
        - comm.boni → 'Boni' (boni extrait du PDF)
        - comm.prime → 'PA' (prime)
        """
        if not report.commissions:
            return pd.DataFrame(columns=self.FINAL_COLUMNS_HISTORICAL)

        rows = []
        for comm in report.commissions:
            # Convertir les valeurs Decimal - utiliser les valeurs EXTRAITES du PDF
            premium = self._decimal_to_float(comm.prime)
            commission = self._decimal_to_float(comm.commission)  # Commission extraite du PDF
            bonus = self._decimal_to_float(comm.boni)  # Boni extrait du PDF

            # Montant reçu = commission + bonus
            received = sum(filter(None, [commission, bonus])) or None

            # Sur-Com n'est pas directement disponible dans Assomption
            on_commission = None

            row = {
                '# de Police': str(comm.numero_police),
                'Nom Client': comm.nom_assure,
                'Compagnie': 'Assomption',
                'Statut': 'Approuvé',
                'Conseiller': report.nom_courtier,
                'Verifié': None,
                'PA': premium,
                'Com': commission,  # Utilise la commission extraite du JSON
                'Boni': bonus,  # Utilise le boni extrait du JSON
                'Sur-Com': on_commission,
                'Reçu': received,  # Montant total reçu (commission + boni)
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
            # Note: On ne calcule Paie que si la colonne n'existe pas déjà.
            # Si la source a explicitement mis Paie à None (comme IDC), on respecte ce choix.
            if 'Paie' not in df.columns:
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

    def _normalize_insurer_name(self, name: str) -> str:
        """
        Normalise le nom de l'assureur vers les abréviations standardisées.

        Mapping:
        - Industrial Alliance / IA Toronto → IA
        - RBC INSURANCE / RBC Life → RBC
        - Assumption Life / Assomption Vie → Assomption
        - Beneva → Beneva
        - UV Assurance → UV
        - Manuvie / Manulife → ManuVie
        - Humania → Humania

        Args:
            name: Nom de l'assureur extrait du PDF

        Returns:
            Nom normalisé
        """
        if not name:
            return name

        name_upper = name.upper().strip()

        # Industrial Alliance → IA
        if 'INDUSTRIAL ALLIANCE' in name_upper or 'IA TORONTO' in name_upper:
            return 'IA'

        # RBC Insurance → RBC
        if 'RBC' in name_upper:
            return 'RBC'

        # Assomption / Assumption Life → Assomption
        if 'ASSOMPTION' in name_upper or 'ASSUMPTION' in name_upper or 'ASSUMPTI' in name_upper:
            return 'Assomption'

        # Beneva
        if 'BENEVA' in name_upper:
            return 'Beneva'

        # UV Assurance → UV
        if 'UV' in name_upper and ('ASSURANCE' in name_upper or len(name_upper) <= 5):
            return 'UV'

        # Manuvie / Manulife → ManuVie
        if 'MANUVIE' in name_upper or 'MANULIFE' in name_upper or 'MANU' in name_upper:
            return 'ManuVie'

        # Humania
        if 'HUMANIA' in name_upper:
            return 'Humania'

        # Autres compagnies connues - garder un nom court
        if 'SUN LIFE' in name_upper or 'SUNLIFE' in name_upper:
            return 'Sun Life'
        if 'CANADA LIFE' in name_upper or 'CANADA-LIFE' in name_upper:
            return 'Canada Life'
        if 'DESJARDINS' in name_upper:
            return 'Desjardins'
        if 'EMPIRE' in name_upper:
            return 'Empire'
        if 'EQUITABLE' in name_upper:
            return 'Equitable'

        # Par défaut, retourner le nom original nettoyé
        return name.strip()

    def _ffill_policy_by_client(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fill conditionnel du numéro de police basé sur le nom du client.

        Si une ligne n'a pas de numéro de police (vide, None, "None") mais que
        le nom du client est identique à une ligne précédente qui a un numéro
        de police, on remplit avec ce numéro.

        Args:
            df: DataFrame avec colonnes '# de Police' et 'Nom Client'

        Returns:
            DataFrame avec les numéros de police remplis
        """
        if df.empty or '# de Police' not in df.columns or 'Nom Client' not in df.columns:
            return df

        df = df.copy()

        # Dictionnaire pour stocker le dernier numéro de police connu par client
        client_policy_map: dict[str, str] = {}
        filled_count = 0

        for idx in df.index:
            client_name = df.at[idx, 'Nom Client']
            policy = df.at[idx, '# de Police']

            # Normaliser le nom du client pour la comparaison
            client_key = str(client_name).strip().upper() if pd.notna(client_name) else None

            if not client_key:
                continue

            # Vérifier si le numéro de police est vide/invalide
            policy_is_empty = (
                pd.isna(policy) or
                policy is None or
                str(policy).strip() in ('', 'None', 'nan', 'NaN')
            )

            if policy_is_empty:
                # Chercher si on a un numéro de police pour ce client
                if client_key in client_policy_map:
                    df.at[idx, '# de Police'] = client_policy_map[client_key]
                    filled_count += 1
            else:
                # Enregistrer ce numéro de police pour ce client
                policy_str = str(policy).strip()
                if policy_str and policy_str not in ('None', 'nan', 'NaN'):
                    client_policy_map[client_key] = policy_str

        if filled_count > 0:
            print(f"  ℹ️  UV: {filled_count} numéro(s) de police rempli(s) par ffill (même client)")

        return df

    # =========================================================================
    # GOOGLE SHEETS ADVISOR NORMALIZATION
    # =========================================================================

    @staticmethod
    def _get_gsheets_credentials():
        """
        Get Google Cloud credentials from multiple sources (priority order):
        1. Streamlit secrets (gcp_service_account table)
        2. Service account JSON file
        """
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]

        # Try Streamlit secrets first
        if STREAMLIT_AVAILABLE:
            try:
                if 'gcp_service_account' in st.secrets:
                    creds_dict = dict(st.secrets['gcp_service_account'])
                    return Credentials.from_service_account_info(creds_dict, scopes=scopes)
            except Exception:
                pass

        # Fallback to file-based credentials
        credentials_file = os.environ.get('GOOGLE_SHEETS_CREDENTIALS_FILE')
        if credentials_file:
            # Resolve relative path from project root
            if not os.path.isabs(credentials_file):
                credentials_file = Path(__file__).parent.parent.parent / credentials_file

            if Path(credentials_file).exists():
                return Credentials.from_service_account_file(str(credentials_file), scopes=scopes)

        return None

    def normalize_gsheet_advisors(
        self,
        worksheet_name: Optional[str] = None,
        column_name: str = 'Conseiller',
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Normalize advisor names in a Google Sheet using the AdvisorMatcher.

        Connects to the Google Sheet configured in environment variables,
        finds the specified column, and normalizes advisor names to the
        compact format "Prénom, Initiale" (e.g., "Guillaume, S").

        Args:
            worksheet_name: Name of the worksheet to process. If None, processes
                           all worksheets that have the column.
            column_name: Name of the column containing advisor names (default: 'Conseiller')
            dry_run: If True, only reports changes without applying them

        Returns:
            Dict with statistics:
            {
                'processed_sheets': list of sheet names processed,
                'total_cells': int,
                'normalized_count': int,
                'unchanged_count': int,
                'not_found_count': int,
                'changes': list of (sheet, row, original, normalized) tuples,
                'not_found': list of (sheet, row, original) tuples,
            }

        Raises:
            RuntimeError: If gspread is not available or Google Sheets not configured
        """
        from .advisor_matcher import get_advisor_matcher

        if not GSHEETS_AVAILABLE:
            raise RuntimeError("gspread library not installed. Run: pip install gspread google-auth")

        spreadsheet_id = os.environ.get('GOOGLE_SHEETS_SPREADSHEET_ID')
        if not spreadsheet_id:
            raise RuntimeError("GOOGLE_SHEETS_SPREADSHEET_ID not set in environment")

        credentials = self._get_gsheets_credentials()
        if not credentials:
            raise RuntimeError("Could not get Google Sheets credentials. Check GOOGLE_SHEETS_CREDENTIALS_FILE")

        # Connect to Google Sheets
        client = gspread.authorize(credentials)
        spreadsheet = client.open_by_key(spreadsheet_id)

        # Get matcher
        matcher = self.advisor_matcher or get_advisor_matcher()

        # Results tracking
        results = {
            'processed_sheets': [],
            'total_cells': 0,
            'normalized_count': 0,
            'unchanged_count': 0,
            'not_found_count': 0,
            'changes': [],
            'not_found': [],
        }

        # Get worksheets to process
        if worksheet_name:
            try:
                worksheets = [spreadsheet.worksheet(worksheet_name)]
            except gspread.WorksheetNotFound:
                raise ValueError(f"Worksheet '{worksheet_name}' not found in spreadsheet")
        else:
            worksheets = spreadsheet.worksheets()

        print(f"\n{'='*60}")
        print("Google Sheets Advisor Normalization")
        print(f"{'='*60}")
        print(f"Spreadsheet ID: {spreadsheet_id}")
        print(f"Column: {column_name}")
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")

        for ws in worksheets:
            try:
                # Get all values including headers
                all_values = ws.get_all_values()
                if not all_values:
                    continue

                headers = all_values[0]

                # Check if this worksheet has the target column
                if column_name not in headers:
                    continue

                col_idx = headers.index(column_name)
                col_letter = chr(ord('A') + col_idx)

                print(f"\n--- Processing: {ws.title} ---")
                print(f"  Column '{column_name}' found at index {col_idx + 1} ({col_letter})")
                results['processed_sheets'].append(ws.title)

                # Process each row (skip header)
                updates_batch = []
                for row_idx, row in enumerate(all_values[1:], start=2):  # row_idx is 1-based for gspread
                    if col_idx >= len(row):
                        continue

                    original_name = row[col_idx].strip()
                    if not original_name or original_name.lower() in ['none', 'nan', 'null', '']:
                        continue

                    results['total_cells'] += 1

                    # Try to normalize
                    normalized = matcher.match_compact(original_name)

                    if normalized:
                        if normalized != original_name:
                            results['normalized_count'] += 1
                            results['changes'].append((ws.title, row_idx, original_name, normalized))

                            if not dry_run:
                                # Batch update for efficiency
                                cell_ref = f"{col_letter}{row_idx}"
                                updates_batch.append({
                                    'range': cell_ref,
                                    'values': [[normalized]]
                                })
                        else:
                            results['unchanged_count'] += 1
                    else:
                        results['not_found_count'] += 1
                        results['not_found'].append((ws.title, row_idx, original_name))

                # Apply batch updates
                if updates_batch and not dry_run:
                    ws.batch_update(updates_batch)
                    print(f"  ✅ Updated {len(updates_batch)} cells")

            except Exception as e:
                print(f"  ⚠️  Error processing {ws.title}: {e}")

        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Sheets processed: {len(results['processed_sheets'])}")
        print(f"Total cells checked: {results['total_cells']}")
        print(f"Normalized: {results['normalized_count']}")
        print(f"Unchanged (already correct): {results['unchanged_count']}")
        print(f"Not found in database: {results['not_found_count']}")

        if results['changes']:
            print(f"\n--- Changes {'(not applied - DRY RUN)' if dry_run else '(applied)'} ---")
            for sheet, row, original, normalized in results['changes'][:20]:  # Show first 20
                print(f"  [{sheet}] Row {row}: '{original}' → '{normalized}'")
            if len(results['changes']) > 20:
                print(f"  ... and {len(results['changes']) - 20} more")

        if results['not_found']:
            print(f"\n--- Not found in advisor database ---")
            unique_not_found = set(nf[2] for nf in results['not_found'])
            for name in sorted(unique_not_found)[:20]:
                count = sum(1 for nf in results['not_found'] if nf[2] == name)
                print(f"  '{name}' ({count} occurrences)")
            if len(unique_not_found) > 20:
                print(f"  ... and {len(unique_not_found) - 20} more unique names")

        print(f"\n{'='*60}\n")

        return results
