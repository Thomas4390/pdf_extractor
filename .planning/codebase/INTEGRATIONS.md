# External Integrations

**Analysis Date:** 2026-01-11

## APIs & External Services

**Vision Language Models (OpenRouter):**
- OpenRouter.ai - VLM API provider for PDF data extraction
  - SDK/Client: Custom async client (`src/clients/openrouter.py`)
  - Auth: API key via `OPENROUTER_API_KEY` env var
  - Primary model: `qwen/qwen2.5-vl-72b-instruct`
  - Fallback model: `qwen/qwen3-vl-235b-a22b-instruct`
  - Text fallback: `deepseek/deepseek-chat`
  - Configuration: `src/utils/config.py`, `src/utils/model_registry.py`

**Project Management (Monday.com):**
- Monday.com GraphQL API - Data upload and board management
  - SDK/Client: Custom async GraphQL client (`src/clients/monday.py`)
  - Auth: JWT token via `MONDAY_API_KEY` env var
  - API Endpoint: `https://api.monday.com/v2`
  - Rate limiting: 0.3s delay between requests
  - Batch size: 50 items per upload

**Email/SMS:**
- Not applicable

**External APIs:**
- Not applicable (beyond OpenRouter and Monday.com)

## Data Storage

**Databases:**
- None (no database dependency)
- All data processed in-memory or cached locally

**File Storage:**
- Local file system only
- PDF files stored in `pdf/` directory by source type
- Extraction cache stored in `cache/` directory

**Caching:**
- Custom JSON cache (`src/clients/cache.py`)
  - Location: `cache/` directory (project root)
  - Key: SHA-256 hash of PDF file content
  - Format: JSON with metadata (source, filename, timestamp, mode)

## Authentication & Identity

**Auth Provider:**
- None (API key-based authentication only)

**OAuth Integrations:**
- Google OAuth (for Sheets API) - Service account credentials
  - Credentials: `GOOGLE_SHEETS_CREDENTIALS_FILE` env var or Streamlit secrets
  - Client: `gspread` library (`src/utils/advisor_matcher.py`)
  - Purpose: Advisor name database sync

## Monitoring & Observability

**Error Tracking:**
- None configured (stdout/stderr logging only)

**Analytics:**
- None

**Logs:**
- Python `logging` module to stdout
- Streamlit built-in logging

## CI/CD & Deployment

**Hosting:**
- Local execution via `streamlit run src/app/main.py`
- Streamlit Cloud compatible (optional)
- No specific deployment configuration

**CI Pipeline:**
- Not configured

## Environment Configuration

**Development:**
- Required env vars: `OPENROUTER_API_KEY`
- Optional env vars: `MONDAY_API_KEY`, `GOOGLE_SHEETS_CREDENTIALS_FILE`
- Secrets location: `.env` file (gitignored), `.streamlit/secrets.toml`
- Test PDFs: `pdf/uv/`, `pdf/assomption/`, `pdf/idc/`, `pdf/idc_statement/`

**Staging:**
- Not applicable (no staging environment)

**Production:**
- Same as development (local or Streamlit Cloud)
- Secrets managed via environment variables

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

---

*Integration audit: 2026-01-11*
*Update when adding/removing external services*
