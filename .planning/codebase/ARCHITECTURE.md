# Architecture

**Analysis Date:** 2026-01-11

## Pattern Overview

**Overall:** Layered + Pipeline Orchestrator Architecture

**Key Characteristics:**
- Single executable with modular extractors per document source
- Configuration-driven model selection via registry
- Async-first execution model
- File-based caching (no database)
- Optional Monday.com integration for data upload

## Layers

**Presentation Layer:**
- Purpose: User interface for PDF upload and results display
- Contains: Streamlit application, progress visualization, batch processing UI
- Location: `src/app/main.py`
- Depends on: Pipeline layer
- Used by: End users via web browser

**Pipeline/Orchestration Layer:**
- Purpose: Coordinate PDF extraction workflow
- Contains: Source detection, extractor routing, data unification, upload orchestration
- Location: `src/pipeline.py`
- Depends on: Extractors, DataUnifier, Clients
- Used by: Presentation layer, programmatic API

**Extraction Service Layer:**
- Purpose: Source-specific PDF data extraction
- Contains: Base extractor class, concrete extractors (UV, IDC, Assomption, IDC_Statement)
- Location: `src/extractors/`
- Depends on: OpenRouter client, Cache, Model registry
- Used by: Pipeline

**Client Layer:**
- Purpose: External service communication
- Contains: OpenRouter VLM client, Monday.com GraphQL client, Cache service
- Location: `src/clients/`
- Depends on: httpx, configuration
- Used by: Extractors, Pipeline

**Data Model Layer:**
- Purpose: Validated data structures
- Contains: Pydantic models for extraction output
- Location: `src/models/`
- Depends on: Pydantic
- Used by: Extractors, DataUnifier

**Utility Layer:**
- Purpose: Shared helpers and configuration
- Contains: PDF processing, configuration, model registry, advisor matching
- Location: `src/utils/`
- Depends on: PyMuPDF, python-dotenv, Pydantic
- Used by: All layers

## Data Flow

**PDF Extraction Workflow:**

1. User uploads PDF via Streamlit UI (`src/app/main.py`)
2. Pipeline detects source type from filename/path (`src/pipeline.py`)
3. Pipeline routes to appropriate extractor (`src/extractors/*_extractor.py`)
4. Extractor checks cache for existing result (`src/clients/cache.py`)
5. If not cached:
   - PDF rendered to images at 300 DPI (`src/utils/pdf.py`)
   - Images sent to VLM via OpenRouter (`src/clients/openrouter.py`)
   - JSON response repaired if malformed (`src/clients/json_repair.py`)
   - Result validated against Pydantic model (`src/models/`)
   - Result cached with metadata
6. DataUnifier converts model to DataFrame (`src/utils/data_unifier.py`)
7. Optional: Upload to Monday.com (`src/clients/monday.py`)
8. Results displayed in UI

**State Management:**
- File-based: Extraction cache in `cache/` directory
- No persistent in-memory state
- Each extraction is independent

## Key Abstractions

**BaseExtractor[T]:**
- Purpose: Abstract base class for source-specific extractors
- Examples: `UVExtractor`, `IDCExtractor`, `AssomptionExtractor`, `IDCStatementExtractor`
- Pattern: Generic class with Pydantic model type parameter
- Location: `src/extractors/base.py`

**Pipeline:**
- Purpose: Main orchestrator for extraction workflow
- Examples: Single PDF processing, batch processing
- Pattern: Factory methods with async support
- Location: `src/pipeline.py`

**DataUnifier:**
- Purpose: Convert Pydantic models to standardized DataFrames
- Examples: `unify_uv()`, `unify_idc()`, `unify_assomption()`
- Pattern: Source-specific conversion methods
- Location: `src/utils/data_unifier.py`

**ModelRegistry:**
- Purpose: Map document types to VLM model configuration
- Examples: `get_model_config("UV")`, `get_pages_for_extraction("IDC")`
- Pattern: Dictionary-based configuration with dataclass values
- Location: `src/utils/model_registry.py`

## Entry Points

**Streamlit UI:**
- Location: `src/app/main.py`
- Triggers: User runs `streamlit run src/app/main.py`
- Responsibilities: PDF upload, batch processing, results display

**Pipeline API:**
- Location: `src/pipeline.py`
- Triggers: Programmatic usage via `Pipeline` class
- Responsibilities: Extraction orchestration, data unification

**Convenience Functions:**
- Location: `src/__init__.py`
- Triggers: Import and call `extract_pdf()`, `get_pipeline()`
- Responsibilities: Simple API for single PDF extraction

**Test Scripts:**
- Location: `src/tests/test_*.py`
- Triggers: `python -m src.tests.test_uv`
- Responsibilities: Functional/integration testing

## Error Handling

**Strategy:** Throw exceptions, catch at pipeline/UI level, provide user feedback

**Patterns:**
- Custom exception classes: `OpenRouterError`, `OpenRouterRateLimitError`
- Validation via Pydantic: `ValidationError` for schema mismatches
- JSON repair for malformed VLM responses (`src/clients/json_repair.py`)
- Multi-level model fallbacks (primary → fallback → secondary)

## Cross-Cutting Concerns

**Logging:**
- Python `logging` module
- Console output for progress tracking
- Streamlit UI feedback

**Validation:**
- Pydantic models at API boundary (VLM response validation)
- Source detection heuristics in Pipeline

**Configuration:**
- Pydantic `BaseSettings` for environment variables (`src/utils/config.py`)
- Model registry for document type configuration (`src/utils/model_registry.py`)
- YAML files for extraction prompts (`src/prompts/`)

**Caching:**
- SHA-256 hash-based cache keys
- JSON storage with metadata
- Automatic cache lookup before extraction

---

*Architecture analysis: 2026-01-11*
*Update when major patterns change*
