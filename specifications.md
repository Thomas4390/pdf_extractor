# Spécifications Techniques - Vision LLM PDF Extractor v2.0

## Vue d'ensemble

Migration complète du pipeline d'extraction PDF vers une architecture unifiée dans `src/`. L'objectif est de consolider :
- L'extraction VLM (déjà implémentée)
- L'unification des données (depuis `scripts/unify_notation.py`)
- Le client Monday.com (depuis `scripts/monday_automation.py`)
- L'application Streamlit (depuis `scripts/app.py`)

---

## Architecture Cible

### Structure du répertoire `src/`

```
src/
├── __init__.py
├── pipeline.py                    # Orchestrateur principal
│
├── extractors/                    # Extraction VLM (existant)
│   ├── __init__.py
│   ├── base.py                    # BaseExtractor[T]
│   ├── uv_extractor.py
│   ├── idc_extractor.py
│   ├── idc_statement_extractor.py
│   └── assomption_extractor.py
│
├── models/                        # Schémas Pydantic (existant)
│   ├── __init__.py
│   ├── uv.py
│   ├── idc.py
│   ├── idc_statement.py
│   ├── assomption.py
│   └── common.py
│
├── clients/                       # Clients API
│   ├── __init__.py
│   ├── openrouter.py              # Client VLM (existant)
│   ├── cache.py                   # Cache local (existant)
│   ├── monday.py                  # Client Monday.com (NOUVEAU)
│   ├── json_repair.py             # (existant)
│   └── retry_handler.py           # (existant)
│
├── utils/                         # Utilitaires
│   ├── __init__.py
│   ├── config.py                  # Configuration (existant)
│   ├── pdf.py                     # PDF utils (existant)
│   ├── model_registry.py          # Registry (existant)
│   ├── advisor_matcher.py         # Matching conseillers (existant)
│   └── data_unifier.py            # Unification données (NOUVEAU)
│
├── prompts/                       # Prompts YAML (existant)
│   ├── uv.yaml
│   ├── idc.yaml
│   ├── idc_statement.yaml
│   └── assomption.yaml
│
├── app/                           # Application Streamlit (NOUVEAU)
│   ├── __init__.py
│   ├── main.py                    # Point d'entrée Streamlit
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── upload.py              # Page 1: Upload & Config
│   │   ├── preview.py             # Page 2: Preview & Edit
│   │   └── export.py              # Page 3: Export Monday.com
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_editor.py         # Éditeur de données
│   │   ├── advisor_manager.py     # Gestion conseillers
│   │   └── progress.py            # Indicateurs de progression
│   └── state.py                   # Gestion état Streamlit
│
└── tests/                         # Tests (existant)
    ├── test_uv.py
    ├── test_idc.py
    ├── test_assomption.py
    └── test_idc_statement.py
```

---

## User Stories

### US-1: Extraction VLM (existant - validé)
**En tant qu'** utilisateur
**Je veux** extraire les données des rapports PDF via un modèle de vision
**Afin d'** obtenir une extraction fiable et maintenable

**Critères d'acceptation:**
- [x] Conversion PDF → images PNG à 300 DPI via PyMuPDF
- [x] Envoi des pages sélectionnées au VLM
- [x] Validation Pydantic du résultat JSON
- [x] Retry automatique avec fallback model
- [x] Cache local par hash SHA-256

### US-2: Unification des données (NOUVEAU)
**En tant qu'** utilisateur
**Je veux** que les données extraites soient standardisées automatiquement
**Afin d'** avoir un format cohérent pour Monday.com

**Critères d'acceptation:**
- [ ] Classe `DataUnifier` convertissant les modèles Pydantic en DataFrame
- [ ] Mapping automatique vers les colonnes françaises
- [ ] Détection du type de board basée sur la source PDF
- [ ] Calcul de commission uniforme : `prime × taux_partage × taux_commission`
- [ ] Normalisation des noms de conseillers via AdvisorMatcher

### US-3: Client Monday.com (NOUVEAU)
**En tant qu'** utilisateur
**Je veux** uploader les données vers Monday.com
**Afin d'** intégrer les commissions dans mon workflow

**Critères d'acceptation:**
- [ ] Création automatique des colonnes manquantes
- [ ] Upload batch avec limite de parallélisme
- [ ] Gestion des types de colonnes (numbers, text, status, date)
- [ ] Support des deux types de boards (Historical/Sales)

### US-4: Pipeline orchestré (NOUVEAU)
**En tant qu'** utilisateur
**Je veux** un pipeline unifié extraction → unification → upload
**Afin d'** automatiser le traitement complet

**Critères d'acceptation:**
- [ ] Classe `Pipeline` orchestrant les 3 étapes
- [ ] Traitement batch parallèle (max 3 PDFs simultanés)
- [ ] Gestion des erreurs : données partielles acceptées + warning
- [ ] Logs détaillés de progression

### US-5: Application Streamlit (REFACTOR)
**En tant qu'** utilisateur
**Je veux** une interface pour gérer l'extraction et l'upload
**Afin d'** avoir un contrôle visuel sur le processus

**Critères d'acceptation:**
- [ ] Upload batch de PDFs (drag & drop multiple)
- [ ] Prévisualisation des données extraites
- [ ] Édition manuelle avant upload
- [ ] Gestion des conseillers (ajout/modification)
- [ ] Indicateurs de progression
- [ ] Workflow en 3 étapes : Upload → Preview → Export

---

## Détails Techniques d'Implémentation

### 1. DataUnifier (`src/utils/data_unifier.py`)

```python
from enum import Enum
from typing import Union
import pandas as pd

from ..models import UVReport, IDCReport, IDCStatementReport, AssomptionReport


class BoardType(Enum):
    """Type de board Monday.com."""
    HISTORICAL_PAYMENTS = "HISTORICAL_PAYMENTS"  # IDC_STATEMENT
    SALES_PRODUCTION = "SALES_PRODUCTION"        # UV, IDC, ASSOMPTION


class DataUnifier:
    """
    Convertit les modèles Pydantic extraits en DataFrames standardisés.

    Responsabilités:
    - Conversion des modèles Pydantic vers DataFrame pandas
    - Mapping vers les colonnes françaises finales
    - Calcul des commissions
    - Normalisation des noms de conseillers
    """

    # Colonnes finales pour Paiements Historiques (13 colonnes)
    FINAL_COLUMNS_HISTORICAL = [
        '# de Police', 'Nom Client', 'Compagnie', 'Statut',
        'Conseiller', 'Verifié', 'PA', 'Com', 'Boni',
        'Sur-Com', 'Reçu', 'Date', 'Texte'
    ]

    # Colonnes finales pour Ventes et Production (19 colonnes)
    FINAL_COLUMNS_SALES = [
        'Date', '# de Police', 'Nom Client', 'Compagnie', 'Statut',
        'Conseiller', 'Complet', 'PA', 'Lead/MC', 'Com', 'Reçu 1',
        'Boni', 'Reçu 2', 'Sur-Com', 'Reçu 3', 'Total',
        'Total Reçu', 'Paie', 'Texte'
    ]

    # Mapping source → type de board
    SOURCE_TO_BOARD_TYPE = {
        'UV': BoardType.SALES_PRODUCTION,
        'IDC': BoardType.SALES_PRODUCTION,
        'ASSOMPTION': BoardType.SALES_PRODUCTION,
        'IDC_STATEMENT': BoardType.HISTORICAL_PAYMENTS,
    }

    def __init__(self, advisor_matcher=None):
        self.advisor_matcher = advisor_matcher

    def unify(
        self,
        report: Union[UVReport, IDCReport, IDCStatementReport, AssomptionReport],
        source: str
    ) -> tuple[pd.DataFrame, BoardType]:
        """
        Convertit un rapport en DataFrame standardisé.

        Args:
            report: Modèle Pydantic extrait
            source: Type de source ('UV', 'IDC', 'IDC_STATEMENT', 'ASSOMPTION')

        Returns:
            Tuple (DataFrame avec colonnes françaises, BoardType)
        """
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
        else:
            raise ValueError(f"Source inconnue: {source}")

        # Appliquer le schéma de colonnes final
        df = self._apply_final_schema(df, board_type)

        # Normaliser les noms de conseillers
        if self.advisor_matcher and 'Conseiller' in df.columns:
            df['Conseiller'] = df['Conseiller'].apply(
                lambda x: self.advisor_matcher.match(x) if pd.notna(x) else x
            )

        return df, board_type

    def _calculate_commission(
        self,
        premium: float,
        sharing_rate: float,
        commission_rate: float
    ) -> float:
        """
        Calcule la commission selon la formule universelle.

        commission = prime × taux_partage × taux_commission
        """
        return premium * (sharing_rate / 100) * (commission_rate / 100)

    def _convert_uv(self, report: UVReport) -> pd.DataFrame:
        """Convertit un rapport UV en DataFrame standardisé."""
        # ... implémentation

    def _convert_idc(self, report: IDCReport) -> pd.DataFrame:
        """Convertit un rapport IDC en DataFrame standardisé."""
        # ... implémentation

    def _convert_idc_statement(self, report: IDCStatementReport) -> pd.DataFrame:
        """Convertit un relevé IDC en DataFrame standardisé."""
        # ... implémentation

    def _convert_assomption(self, report: AssomptionReport) -> pd.DataFrame:
        """Convertit un rapport Assomption en DataFrame standardisé."""
        # ... implémentation

    def _apply_final_schema(self, df: pd.DataFrame, board_type: BoardType) -> pd.DataFrame:
        """Applique le schéma de colonnes final selon le type de board."""
        if board_type == BoardType.HISTORICAL_PAYMENTS:
            columns = self.FINAL_COLUMNS_HISTORICAL
        else:
            columns = self.FINAL_COLUMNS_SALES

        # Ajouter les colonnes manquantes avec valeurs par défaut
        for col in columns:
            if col not in df.columns:
                df[col] = None

        return df[columns]
```

### 2. Client Monday.com (`src/clients/monday.py`)

```python
import httpx
from typing import Optional
import pandas as pd

from ..utils.data_unifier import BoardType


class MondayClient:
    """
    Client pour l'API Monday.com avec support GraphQL.

    Fonctionnalités:
    - CRUD sur les boards/items
    - Création automatique des colonnes manquantes
    - Upload batch avec rate limiting
    """

    BASE_URL = "https://api.monday.com/v2"

    # Mapping colonnes → types Monday.com
    COLUMN_TYPES = {
        '# de Police': 'text',
        'Nom Client': 'text',
        'Compagnie': 'text',
        'Statut': 'status',
        'Conseiller': 'text',
        'Verifié': 'checkbox',
        'PA': 'numbers',
        'Com': 'numbers',
        'Boni': 'numbers',
        'Sur-Com': 'numbers',
        'Reçu': 'numbers',
        'Reçu 1': 'numbers',
        'Reçu 2': 'numbers',
        'Reçu 3': 'numbers',
        'Total': 'numbers',
        'Total Reçu': 'numbers',
        'Date': 'date',
        'Paie': 'date',
        'Texte': 'long_text',
        'Complet': 'checkbox',
        'Lead/MC': 'text',
    }

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": api_key,
            "Content-Type": "application/json",
            "API-Version": "2024-01"
        }

    async def upload_dataframe(
        self,
        df: pd.DataFrame,
        board_id: str,
        group_id: Optional[str] = None,
        create_missing_columns: bool = True
    ) -> dict:
        """
        Upload un DataFrame vers Monday.com.

        Args:
            df: DataFrame avec colonnes françaises
            board_id: ID du board cible
            group_id: ID du groupe (optionnel)
            create_missing_columns: Créer les colonnes manquantes

        Returns:
            Résultat de l'upload avec statistiques
        """
        # 1. Récupérer les colonnes existantes
        existing_columns = await self.get_columns(board_id)

        # 2. Créer les colonnes manquantes si autorisé
        if create_missing_columns:
            for col in df.columns:
                if col not in existing_columns and col != 'Nom Client':
                    col_type = self.COLUMN_TYPES.get(col, 'text')
                    await self.create_column(board_id, col, col_type)

        # 3. Upload les items en batch
        results = await self._batch_upload(df, board_id, group_id)

        return results

    async def get_columns(self, board_id: str) -> dict:
        """Récupère les colonnes d'un board."""
        query = """
        query ($boardId: [ID!]) {
            boards(ids: $boardId) {
                columns {
                    id
                    title
                    type
                }
            }
        }
        """
        # ... implémentation

    async def create_column(
        self,
        board_id: str,
        title: str,
        column_type: str
    ) -> str:
        """Crée une colonne sur le board."""
        # ... implémentation

    async def _batch_upload(
        self,
        df: pd.DataFrame,
        board_id: str,
        group_id: Optional[str],
        batch_size: int = 50
    ) -> dict:
        """Upload les items par batch."""
        # ... implémentation
```

### 3. Pipeline (`src/pipeline.py`)

```python
import asyncio
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .extractors import (
    UVExtractor, IDCExtractor,
    IDCStatementExtractor, AssomptionExtractor
)
from .utils.data_unifier import DataUnifier, BoardType
from .utils.advisor_matcher import AdvisorMatcher
from .clients.monday import MondayClient


@dataclass
class PipelineResult:
    """Résultat du pipeline pour un PDF."""
    pdf_path: str
    source: str
    board_type: BoardType
    dataframe: 'pd.DataFrame'
    success: bool
    warnings: list[str]
    error: Optional[str] = None


class Pipeline:
    """
    Orchestrateur du pipeline complet:
    PDF → Extraction VLM → Unification → Upload Monday.com
    """

    # Mapping extension/pattern → source
    SOURCE_DETECTION = {
        'uv': 'UV',
        'idc_statement': 'IDC_STATEMENT',
        'idc': 'IDC',
        'assomption': 'ASSOMPTION',
    }

    def __init__(
        self,
        monday_api_key: Optional[str] = None,
        max_parallel: int = 3
    ):
        # Extracteurs
        self.extractors = {
            'UV': UVExtractor(),
            'IDC': IDCExtractor(),
            'IDC_STATEMENT': IDCStatementExtractor(),
            'ASSOMPTION': AssomptionExtractor(),
        }

        # Unificateur avec advisor matcher
        self.advisor_matcher = AdvisorMatcher()
        self.unifier = DataUnifier(advisor_matcher=self.advisor_matcher)

        # Client Monday (optionnel)
        self.monday_client = MondayClient(monday_api_key) if monday_api_key else None

        # Contrôle de parallélisme
        self.semaphore = asyncio.Semaphore(max_parallel)

    async def process_pdf(
        self,
        pdf_path: str | Path,
        source: Optional[str] = None
    ) -> PipelineResult:
        """
        Traite un PDF unique.

        Args:
            pdf_path: Chemin vers le PDF
            source: Type de source (auto-détecté si None)

        Returns:
            PipelineResult avec DataFrame et métadonnées
        """
        pdf_path = Path(pdf_path)
        warnings = []

        # Auto-détection de la source
        if source is None:
            source = self._detect_source(pdf_path)

        try:
            async with self.semaphore:
                # 1. Extraction VLM
                extractor = self.extractors[source]
                report = await extractor.extract(pdf_path)

                # 2. Unification
                df, board_type = self.unifier.unify(report, source)

                # Vérifier si données partielles
                if len(df) == 0:
                    warnings.append("Aucune donnée extraite")

                return PipelineResult(
                    pdf_path=str(pdf_path),
                    source=source,
                    board_type=board_type,
                    dataframe=df,
                    success=True,
                    warnings=warnings
                )

        except Exception as e:
            return PipelineResult(
                pdf_path=str(pdf_path),
                source=source,
                board_type=BoardType.SALES_PRODUCTION,
                dataframe=pd.DataFrame(),
                success=False,
                warnings=warnings,
                error=str(e)
            )

    async def process_batch(
        self,
        pdf_paths: list[str | Path],
        source: Optional[str] = None
    ) -> list[PipelineResult]:
        """
        Traite plusieurs PDFs en parallèle (max 3 simultanés).

        Args:
            pdf_paths: Liste des chemins PDF
            source: Type de source commun (auto-détecté si None)

        Returns:
            Liste des résultats
        """
        tasks = [
            self.process_pdf(path, source)
            for path in pdf_paths
        ]
        return await asyncio.gather(*tasks)

    async def upload_to_monday(
        self,
        result: PipelineResult,
        board_id: str,
        group_id: Optional[str] = None
    ) -> dict:
        """
        Upload les résultats vers Monday.com.

        Args:
            result: Résultat du pipeline
            board_id: ID du board cible
            group_id: ID du groupe (optionnel)

        Returns:
            Statistiques d'upload
        """
        if not self.monday_client:
            raise ValueError("Client Monday.com non configuré")

        return await self.monday_client.upload_dataframe(
            result.dataframe,
            board_id,
            group_id
        )

    def _detect_source(self, pdf_path: Path) -> str:
        """Détecte le type de source depuis le chemin/nom du fichier."""
        path_str = str(pdf_path).lower()

        for pattern, source in self.SOURCE_DETECTION.items():
            if pattern in path_str:
                return source

        # Fallback: demander à l'utilisateur ou lever une erreur
        raise ValueError(f"Impossible de détecter la source pour: {pdf_path}")
```

### 4. Application Streamlit (`src/app/main.py`)

```python
import streamlit as st
from pathlib import Path

from .state import init_session_state
from .pages import upload, preview, export


def main():
    st.set_page_config(
        page_title="Insurance Commission Extractor",
        page_icon="📊",
        layout="wide"
    )

    # Initialiser l'état de session
    init_session_state()

    # Navigation par étapes
    steps = ["1. Upload", "2. Preview", "3. Export"]
    current_step = st.session_state.get('current_step', 0)

    # Afficher les onglets de navigation
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i < current_step:
                st.success(step + " ✓")
            elif i == current_step:
                st.info(step + " ←")
            else:
                st.text(step)

    st.divider()

    # Afficher la page correspondante
    if current_step == 0:
        upload.render()
    elif current_step == 1:
        preview.render()
    elif current_step == 2:
        export.render()


if __name__ == "__main__":
    main()
```

---

## Schémas de données

### Colonnes Historical Payments (13 colonnes)

| Colonne | Type Monday | Description |
|---------|-------------|-------------|
| # de Police | text | Numéro de contrat |
| Nom Client | text | Nom de l'assuré (item_name) |
| Compagnie | text | Nom de l'assureur |
| Statut | status | Statut du paiement |
| Conseiller | text | Nom normalisé du conseiller |
| Verifié | checkbox | Validation manuelle |
| PA | numbers | Prime annualisée ($) |
| Com | numbers | Commission ($) |
| Boni | numbers | Bonus ($) |
| Sur-Com | numbers | Sur-commission ($) |
| Reçu | numbers | Montant reçu ($) |
| Date | date | Date du paiement |
| Texte | long_text | Commentaires |

### Colonnes Sales Production (19 colonnes)

| Colonne | Type Monday | Description |
|---------|-------------|-------------|
| Date | date | Date d'effet |
| # de Police | text | Numéro de contrat |
| Nom Client | text | Nom de l'assuré (item_name) |
| Compagnie | text | Nom de l'assureur |
| Statut | status | Statut de la vente |
| Conseiller | text | Nom normalisé du conseiller |
| Complet | checkbox | Dossier complet |
| PA | numbers | Prime annualisée ($) |
| Lead/MC | text | Type de partage |
| Com | numbers | Commission ($) |
| Reçu 1 | numbers | Commission reçue ($) |
| Boni | numbers | Bonus ($) |
| Reçu 2 | numbers | Bonus reçu ($) |
| Sur-Com | numbers | Sur-commission ($) |
| Reçu 3 | numbers | Sur-commission reçue ($) |
| Total | numbers | Total commissions ($) |
| Total Reçu | numbers | Total reçu ($) |
| Paie | date | Date de paiement |
| Texte | long_text | Commentaires |

### Mapping Source → Board Type

| Source PDF | Board Type | Raison |
|------------|------------|--------|
| UV | SALES_PRODUCTION | Rapport de ventes |
| IDC | SALES_PRODUCTION | Propositions soumises |
| ASSOMPTION | SALES_PRODUCTION | Rapport de rémunération |
| IDC_STATEMENT | HISTORICAL_PAYMENTS | Relevés de paiements historiques |

---

## Gestion des erreurs

### Stratégie de fallback VLM

```
1. Tentative avec modèle principal (qwen/qwen2.5-vl-72b-instruct)
   ↓ échec
2. Retry 1x avec le même modèle
   ↓ échec
3. Fallback vers modèle secondaire (qwen/qwen3-vl-235b-a22b-instruct)
   ↓ échec
4. Retry 1x avec modèle secondaire
   ↓ échec
5. Retourner données partielles + warning
```

### Comportement en cas de données partielles

- Les données extraites (même incomplètes) sont retournées dans le DataFrame
- Un warning est ajouté au `PipelineResult.warnings`
- L'utilisateur voit un indicateur visuel dans Streamlit
- L'upload vers Monday.com reste possible (l'utilisateur peut éditer avant)

---

## Variables d'environnement

```env
# OpenRouter API (Vision LLM)
OPENROUTER_API_KEY=sk-or-v1-xxxxx

# Monday.com API
MONDAY_API_KEY=your_jwt_token_here

# Configuration optionnelle
VLM_MAX_RETRIES=2
VLM_TIMEOUT_SECONDS=120
BATCH_MAX_PARALLEL=3
```

---

## Dépendances

```toml
[project]
dependencies = [
    # Extraction PDF
    "pymupdf>=1.24.0",

    # API clients
    "httpx>=0.27.0",

    # Data processing
    "pandas>=2.0.0",
    "pydantic>=2.6.0",

    # Fuzzy matching
    "rapidfuzz>=3.0.0",

    # Configuration
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0.0",

    # Streamlit
    "streamlit>=1.30.0",

    # JSON repair
    "json-repair>=0.25.0",
]
```

---

## Limitations connues

1. **PDF only**: Import depuis Monday.com non supporté (simplifié par design)
2. **Coût API**: ~0.01-0.05$ par page selon le modèle VLM
3. **Latence**: 5-15 secondes par PDF (extraction VLM)
4. **Parallélisme**: Maximum 3 PDFs simultanés pour éviter rate limiting
5. **Données partielles**: Acceptées avec warning (responsabilité utilisateur)

---

## Workflow complet

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Streamlit                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ 1.Upload │ →  │ 2.Preview│ →  │ 3.Export │              │
│  │          │    │ & Edit   │    │ Monday   │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│       │               │               │                     │
└───────│───────────────│───────────────│─────────────────────┘
        │               │               │
        ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   Pipeline    │ │  DataUnifier  │ │ MondayClient  │
│               │ │               │ │               │
│ ┌───────────┐ │ │ Pydantic →   │ │ DataFrame →  │
│ │ Extractor │ │ │ DataFrame    │ │ GraphQL      │
│ │ (VLM)     │ │ │ + French cols│ │ API          │
│ └───────────┘ │ │               │ │               │
└───────────────┘ └───────────────┘ └───────────────┘
        │               │               │
        ▼               ▼               ▼
   Cache local    Colonnes FR     Board Monday
   (SHA-256)      normalisées     avec items
```

---

## Prochaines étapes d'implémentation

1. **Phase 1**: Créer `src/utils/data_unifier.py`
   - Implémenter `DataUnifier` avec les 4 convertisseurs
   - Tests unitaires avec modèles Pydantic existants

2. **Phase 2**: Créer `src/clients/monday.py`
   - Migrer depuis `scripts/monday_automation.py`
   - Simplifier (retirer import Monday)
   - Tests d'intégration

3. **Phase 3**: Créer `src/pipeline.py`
   - Orchestration des composants
   - Gestion du batch parallèle
   - Tests end-to-end

4. **Phase 4**: Créer `src/app/`
   - Structure multi-pages Streamlit
   - Migration UX depuis `scripts/app.py`
   - Tests manuels

---

*Document généré le 8 janvier 2026*
