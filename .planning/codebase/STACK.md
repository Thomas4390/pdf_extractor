# Technology Stack

**Analysis Date:** 2026-01-11

## Languages

**Primary:**
- Python 3.12+ - All application code (`src/`)

**Secondary:**
- YAML - Prompt configuration files (`src/prompts/*.yaml`)

## Runtime

**Environment:**
- Python 3.12+ (CPython)
- No browser runtime (CLI/Streamlit server only)
- Virtual environment via `.venv/`

**Package Manager:**
- pip/uv
- Lockfile: Not present (pinned versions in `requirements.txt`)

## Frameworks

**Core:**
- Streamlit 1.30+ - Web UI framework (`src/app/main.py`)

**Testing:**
- Custom test scripts - Standalone executable tests (no pytest)
- asyncio - Async test execution

**Build/Dev:**
- No build step required (interpreted Python)
- python-dotenv - Environment configuration

## Key Dependencies

**Critical:**
- `pydantic` 2.6+ - Data validation, settings management (`src/utils/config.py`, `src/models/`)
- `httpx` 0.27+ - Async HTTP client for API calls (`src/clients/openrouter.py`, `src/clients/monday.py`)
- `PyMuPDF` (fitz) 1.24+ - PDF rendering at 300 DPI (`src/utils/pdf.py`)
- `pandas` 2.0+ - Data manipulation and unification (`src/utils/data_unifier.py`)

**Infrastructure:**
- `streamlit` 1.30+ - Web application framework (`src/app/main.py`)
- `python-dotenv` 1.2+ - Environment variable loading (`src/utils/config.py`)
- `PyPDF2` 3.0+ - PDF manipulation (`requirements.txt`)
- `pdfplumber` 0.10+ - PDF text extraction (`requirements.txt`)
- `gspread` 5.12+ - Google Sheets API client (`src/utils/advisor_matcher.py`)
- `google-auth` 2.23+ - Google authentication (`requirements.txt`)
- `openpyxl` 3.1+ - Excel file support (`requirements.txt`)

## Configuration

**Environment:**
- `.env` files for secrets (gitignored)
- `.streamlit/secrets.toml` for Streamlit-specific configuration
- Key env vars: `OPENROUTER_API_KEY` (required), `MONDAY_API_KEY` (optional)

**Build:**
- `pyproject.toml` - Project metadata, Python version requirement
- `requirements.txt` - Pinned production dependencies

## Platform Requirements

**Development:**
- macOS/Linux/Windows (any platform with Python 3.12+)
- No external dependencies (all Python packages)

**Production:**
- Streamlit Cloud (optional deployment target)
- Local execution via `streamlit run src/app/main.py`
- Docker compatible (no specific container setup)

---

*Stack analysis: 2026-01-11*
*Update after major dependency changes*
