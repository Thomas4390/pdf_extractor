# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Insurance commission extraction platform with two main modes:
1. **Extraction Pipeline**: VLM-based PDF extraction using OpenRouter → Data unification → Monday.com upload
2. **Aggregation Mode**: Read data from multiple Monday.com boards → Filter by period → Aggregate by advisor → Upsert to target board

## Commands

```bash
# Run Streamlit application
streamlit run src/app/main.py

# Run individual extractor tests
python -m src.tests.test_uv
python -m src.tests.test_uv /path/to/specific.pdf

# Available test modules
python -m src.tests.test_assomption
python -m src.tests.test_idc
python -m src.tests.test_idc_statement
```

## Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Application                         │
│  src/app/main.py → sidebar.py → extraction/ or aggregation/     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Mode: Extraction                    Mode: Aggregation           │
│  ┌──────────────────┐               ┌──────────────────┐        │
│  │ Stage 1: Config  │               │ Step 1: Sources  │        │
│  │ Stage 2: Preview │               │ Step 2: Period   │        │
│  │ Stage 3: Upload  │               │ Step 3: Preview  │        │
│  └────────┬─────────┘               │ Step 4: Execute  │        │
│           │                         └────────┬─────────┘        │
│           ▼                                  ▼                  │
│  ┌────────────────┐                 ┌────────────────┐          │
│  │   Pipeline     │                 │  aggregator.py │          │
│  │ (Orchestrator) │                 │ filter_by_date │          │
│  └───┬─────┬──────┘                 │ aggregate_by_  │          │
│      │     │                        │ advisor        │          │
│      ▼     ▼                        └────────┬───────┘          │
│ Extractors  DataUnifier                      │                  │
│     │           │                            ▼                  │
│     ▼           ▼                    ┌───────────────┐          │
│ OpenRouter   Monday.com              │ MondayClient  │          │
│  (VLM API)                           │ upsert_by_    │          │
│                                      │ advisor       │          │
└──────────────────────────────────────┴───────────────┴──────────┘
```

### Extraction Pipeline Flow

```
PDF File
    ↓
Pipeline.detect_source() - Auto-detect source type from path/filename
    ↓
model_registry.get_pages_for_extraction() - Determines which pages to extract
    ↓
pdf_to_images() - PyMuPDF renders selected pages at 300 DPI
    ↓
OpenRouterClient - Sends images to VLM (model from registry)
    ↓
json_repair - Fixes malformed JSON responses
    ↓
ExtractionCache - Stores result keyed by SHA-256 hash
    ↓
Pydantic Model (validated output)
    ↓
DataUnifier - Converts to standardized DataFrame with French column names
    ↓
MondayClient.upload_dataframe() - Uploads to Monday.com
```

### Module Structure

| Module | Purpose |
|--------|---------|
| `pipeline.py` | Main orchestrator: PDF → VLM → Unify → Upload |
| `extractors/` | Source-specific extractors inheriting from `BaseExtractor[T]` |
| `models/` | Pydantic models for structured output validation |
| `clients/monday.py` | Monday.com GraphQL client with upsert support |
| `clients/openrouter.py` | OpenRouter VLM API client with fallbacks |
| `clients/cache.py` | Local JSON caching by SHA-256 hash |
| `utils/data_unifier.py` | Pydantic → DataFrame with French columns |
| `utils/aggregator.py` | Date filtering, advisor aggregation for aggregation mode |
| `utils/advisor_matcher.py` | Fuzzy matching for advisor name normalization |
| `prompts/` | YAML files containing extraction prompts per document type |
| `app/` | Streamlit UI components |

### Streamlit Application Structure

```
src/app/
├── main.py              # Entry point, routes to extraction or aggregation mode
├── state.py             # Session state initialization and management
├── sidebar.py           # Mode toggle, Monday.com connection, advisor tab
├── styles.py            # CSS styling
├── extraction/          # Extraction mode (3-stage wizard)
│   ├── stage_1.py       # File upload, source detection, model selection
│   ├── stage_2.py       # Data preview, editing, verification
│   └── stage_3.py       # Board/group selection, upload execution
├── aggregation/         # Aggregation mode (4-step wizard)
│   ├── mode.py          # Step routing and rendering
│   └── execution.py     # Data loading and upsert execution
├── aggregation_ui.py    # UI components for aggregation (stepper, previews)
└── advisor/             # Advisor management components
```

### Extractor Pattern

All extractors inherit from `BaseExtractor[T]` where `T` is the Pydantic model type:

```python
class MyExtractor(BaseExtractor[MyReport]):
    @property
    def source_name(self) -> str:
        return "my_source"

    @property
    def document_type(self) -> str:
        return "MY_SOURCE"  # Must match model_registry key

    @property
    def model_class(self) -> type[MyReport]:
        return MyReport
```

The base class provides:
- `extract(pdf_path)` → Returns validated Pydantic model
- `extract_raw(pdf_path)` → Returns raw dict
- `is_cached(pdf_path)` / `invalidate_cache(pdf_path)`
- Automatic page selection from `model_registry`

### Model Registry

`utils/model_registry.py` centralizes VLM configuration per document type:

```python
MODEL_REGISTRY = {
    "UV": ModelConfig(model_id="...", mode=VISION, page_config=None),
    "ASSOMPTION": ModelConfig(..., page_config=PageConfig(pages=[0, 2, 4])),
    "IDC_STATEMENT": ModelConfig(..., page_config=PageConfig(skip_first=2)),
}
```

### Adding a New Extractor

1. Add Pydantic model in `models/new_source.py`
2. Add config in `utils/model_registry.py` with `document_type` key
3. Create prompts in `prompts/new_source.yaml`
4. Create extractor in `extractors/new_source_extractor.py`
5. Create test in `tests/test_new_source.py`
6. Update `__init__.py` in `models/` and `extractors/`
7. Add source pattern in `pipeline.py` SOURCE_PATTERNS

### Aggregation Mode Configuration

`utils/aggregator.py` defines source board configurations:

```python
SOURCE_BOARDS = {
    "paiement_historique": SourceBoardConfig(
        display_name="Paiement historique",
        aggregate_column="Reçu",
        date_column="Date",
        output_column_name="Collected",
    ),
    "ae_tracker": SourceBoardConfig(
        ...,
        advisor_column="group_title",  # Advisor from group name
        use_group_as_advisor=True,
    ),
}
```

## Environment Variables

```env
OPENROUTER_API_KEY=your_key_here  # Required for VLM calls
MONDAY_API_KEY=your_jwt_token     # Required for Monday.com operations
```

## Supported PDF Sources

| Source | Document Type | Board Type | Pages |
|--------|---------------|------------|-------|
| UV Assurance | UV | SALES_PRODUCTION | All |
| Assomption Vie | ASSOMPTION | SALES_PRODUCTION | 1, 3, 5 |
| IDC Propositions | IDC | SALES_PRODUCTION | All |
| IDC Statements | IDC_STATEMENT | HISTORICAL_PAYMENTS | Skip 2 first |

## Monday.com Board Types

- `HISTORICAL_PAYMENTS`: 13 columns (# de Police, Nom Client, ..., Reçu, Date, Texte)
- `SALES_PRODUCTION`: 19 columns (Date, # de Police, ..., Total Reçu, Paie, Texte)

## Caching

- Results cached as JSON in `cache/` directory (project root)
- Cache key: SHA-256 hash of PDF file content
- Force re-extraction: `extractor.invalidate_cache(pdf_path)` or `force_refresh=True` in Pipeline

## VLM Configuration Defaults

- Primary model: `qwen/qwen2.5-vl-72b-instruct`
- Fallback model: `qwen/qwen3-vl-235b-a22b-instruct`
- Temperature: 0.1
- Max retries: 1
- Timeout: 120s
- PDF DPI: 300
