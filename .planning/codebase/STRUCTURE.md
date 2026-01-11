# Codebase Structure

**Analysis Date:** 2026-01-11

## Directory Layout

```
pdf_extractor/
├── src/                    # Main source code
│   ├── app/               # Streamlit web application
│   ├── clients/           # External API clients
│   ├── extractors/        # Source-specific PDF extractors
│   ├── models/            # Pydantic data models
│   ├── prompts/           # YAML extraction prompts
│   ├── tests/             # Test scripts
│   ├── utils/             # Configuration and utilities
│   ├── __init__.py        # Package exports
│   └── pipeline.py        # Main orchestrator
├── pdf/                   # Test PDF files by source
│   ├── uv/               # UV Assurance PDFs
│   ├── assomption/       # Assomption Vie PDFs
│   ├── idc/              # IDC Propositions PDFs
│   └── idc_statement/    # IDC Statements PDFs
├── cache/                 # Extraction cache (JSON)
├── .streamlit/            # Streamlit configuration
├── pyproject.toml         # Project metadata
├── requirements.txt       # Dependencies
├── CLAUDE.md             # AI assistant instructions
└── .env                   # Environment variables (gitignored)
```

## Directory Purposes

**src/app/**
- Purpose: Streamlit web application
- Contains: `main.py` (32KB) - Single-file web UI
- Key files: `main.py` - PDF upload, batch processing, results display
- Subdirectories: None

**src/clients/**
- Purpose: External service API clients
- Contains: Async HTTP clients, caching, JSON repair utilities
- Key files:
  - `openrouter.py` - VLM API client with fallbacks
  - `monday.py` - Monday.com GraphQL client (779 lines)
  - `cache.py` - Local JSON cache
  - `json_repair.py` - Malformed JSON recovery
  - `retry_handler.py` - Retry decorators
- Subdirectories: None

**src/extractors/**
- Purpose: Source-specific PDF data extractors
- Contains: Base class and 4 concrete extractors
- Key files:
  - `base.py` - `BaseExtractor[T]` abstract base class (9KB)
  - `uv_extractor.py` - UV Assurance extractor
  - `idc_extractor.py` - IDC Propositions extractor
  - `assomption_extractor.py` - Assomption Vie extractor
  - `idc_statement_extractor.py` - IDC Statements extractor (8KB)
- Subdirectories: None

**src/models/**
- Purpose: Pydantic data models for validated output
- Contains: One model file per source type
- Key files:
  - `uv.py` - `UVReport`, `UVActivity` (6KB)
  - `idc.py` - `IDCReport`, `IDCProposition` (4KB)
  - `assomption.py` - `AssomptionReport`, `AssomptionCommission` (4KB)
  - `idc_statement.py` - `IDCStatementReport` variants (8KB)
- Subdirectories: None

**src/prompts/**
- Purpose: YAML extraction prompts per document type
- Contains: System and user prompts for VLM extraction
- Key files:
  - `uv.yaml` - UV extraction prompts (4KB)
  - `idc.yaml` - IDC extraction prompts (4KB)
  - `assomption.yaml` - Assomption prompts (3KB)
  - `idc_statement.yaml` - IDC Statement prompts (15KB)
- Subdirectories: None

**src/tests/**
- Purpose: Functional/integration test scripts
- Contains: Standalone executable test files
- Key files:
  - `test_pipeline.py` - Pipeline orchestrator tests (448 lines)
  - `test_uv.py`, `test_idc.py`, `test_assomption.py`, `test_idc_statement.py` - Extractor tests
  - `test_batch_extraction.py` - Parallel extraction tests
  - `test_monday.py` - Monday.com integration tests (406 lines)
  - `test_data_unifier.py` - Data transformation tests
- Subdirectories: None

**src/utils/**
- Purpose: Configuration, PDF processing, shared utilities
- Contains: Helper modules for cross-cutting concerns
- Key files:
  - `config.py` - Settings via Pydantic BaseSettings
  - `model_registry.py` - Document type to VLM model mapping
  - `pdf.py` - PDF to images/text conversion
  - `data_unifier.py` - Pydantic to DataFrame conversion (large file)
  - `advisor_matcher.py` - Advisor name normalization
  - `batch.py` - Parallel extraction utilities
  - `prompt_loader.py` - YAML prompt loading
- Subdirectories: None

## Key File Locations

**Entry Points:**
- `src/app/main.py` - Streamlit UI entry (run with `streamlit run`)
- `src/pipeline.py` - Programmatic API entry
- `src/__init__.py` - Package-level exports

**Configuration:**
- `pyproject.toml` - Python version, project metadata
- `requirements.txt` - Production dependencies
- `.env` - Environment variables (secrets, API keys)
- `.streamlit/secrets.toml` - Streamlit-specific secrets
- `src/utils/config.py` - Pydantic settings class

**Core Logic:**
- `src/pipeline.py` - Main orchestrator (19KB)
- `src/extractors/base.py` - Base extractor class
- `src/utils/data_unifier.py` - Data transformation
- `src/clients/openrouter.py` - VLM API client

**Testing:**
- `src/tests/test_*.py` - All test files
- `pdf/` subdirectories - Test PDF files

**Documentation:**
- `CLAUDE.md` - AI assistant instructions

## Naming Conventions

**Files:**
- `snake_case.py` for all Python modules
- `*_extractor.py` for concrete extractor implementations
- `test_*.py` for test files
- `*.yaml` for prompt configuration

**Directories:**
- `snake_case` for all directories
- Singular names for module directories (`models/`, `utils/`)
- Plural for collections (`clients/`, `extractors/`, `tests/`)

**Special Patterns:**
- `__init__.py` for package exports
- `__pycache__/` for compiled bytecode (gitignored)

## Where to Add New Code

**New Extractor:**
- Implementation: `src/extractors/{source}_extractor.py`
- Model: `src/models/{source}.py`
- Prompts: `src/prompts/{source}.yaml`
- Tests: `src/tests/test_{source}.py`
- Registry: Update `src/utils/model_registry.py`

**New API Client:**
- Implementation: `src/clients/{service}.py`
- Tests: `src/tests/test_{service}.py`

**New Utility:**
- Implementation: `src/utils/{name}.py`
- Export in `src/utils/__init__.py` if public

**New Feature in Pipeline:**
- Modify `src/pipeline.py`
- Update `src/__init__.py` exports if new public API

## Special Directories

**cache/**
- Purpose: Local extraction cache (JSON files)
- Source: Generated by `src/clients/cache.py`
- Committed: No (gitignored)

**pdf/**
- Purpose: Test PDF files organized by source
- Source: Manually added test documents
- Committed: No (gitignored, contains sensitive data)

**.streamlit/**
- Purpose: Streamlit configuration and secrets
- Source: Manual configuration
- Committed: Partially (secrets.toml gitignored)

---

*Structure analysis: 2026-01-11*
*Update when directory structure changes*
