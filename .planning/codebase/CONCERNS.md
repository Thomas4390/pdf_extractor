# Technical Debt & Concerns

**Analysis Date:** 2026-01-11

## Critical Issues

### Security: API Keys in Git History
- **Location:** `.env` file (project root)
- **Issue:** `.env` file containing valid API keys was committed to git
- **Affected Keys:** `MONDAY_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_SHEETS_SPREADSHEET_ID`
- **Status:** `.env` is now in `.gitignore` but history may contain secrets
- **Action Required:** Rotate all exposed API keys immediately

### Missing `.env.example` Template
- **Location:** Project root (missing file)
- **Issue:** No template file showing required environment variables
- **Impact:** Developers may commit real secrets or miss required variables
- **Action:** Create `.env.example` with placeholder values

## High Priority

### Unguarded `asyncio.gather()` Calls
- **Locations:**
  - `src/pipeline.py:377`
  - `src/clients/monday.py:693`
  - `src/utils/batch.py:167`
- **Issue:** `asyncio.gather(*tasks)` without `return_exceptions=True`
- **Impact:** Single task failure crashes entire batch processing
- **Fix:** Add `return_exceptions=True` or wrap with proper error handling

### Temporary Files Not Cleaned Up
- **Location:** `src/app/main.py:369`
- **Issue:** `tempfile.mkdtemp()` directories never cleaned up
- **Impact:** Disk space leak on long-running Streamlit applications
- **Fix:** Use `tempfile.TemporaryDirectory()` context manager

## Medium Priority

### Broad Exception Catching
- **Locations:**
  - `src/app/main.py` (lines 127, 413, 781, 869, 983)
  - `src/utils/advisor_matcher.py` (lines 62, 90, 260, 417)
  - `src/clients/openrouter.py` (lines 176, 186, 529, 539, 639)
- **Issue:** `except Exception: pass` silently swallows all errors
- **Impact:** Bugs hidden, difficult to diagnose production issues
- **Fix:** Catch specific exceptions, log warnings, add context

### Large Monolithic Files
- **Files exceeding recommended size:**
  - `src/app/main.py` (1,047 lines) - Main UI controller
  - `src/utils/data_unifier.py` (843 lines) - Data transformation
  - `src/clients/openrouter.py` (789 lines) - API client
  - `src/clients/monday.py` (778 lines) - Monday.com client
  - `src/utils/advisor_matcher.py` (578 lines) - Fuzzy matching
  - `src/pipeline.py` (575 lines) - Orchestrator
- **Impact:** Difficult to test, high maintenance burden
- **Fix:** Extract into smaller, focused modules

### Code Duplication in OpenRouter Client
- **Location:** `src/clients/openrouter.py`
- **Issue:** `extract_with_vision()` and `extract_with_text()` share ~60% code
- **Impact:** Maintenance burden, inconsistent bug fixes
- **Fix:** Extract common retry/fallback logic into base method

### Missing Input Validation
- **Location:** `src/app/main.py:373-374`
- **Issue:** PDF filenames written to filesystem without sanitization
- **Risk:** Potential path traversal if filename contains `../`
- **Fix:** Validate filenames, use `Path.name` property

## Low Priority

### Debug Print Statements
- **Locations:**
  - `src/clients/openrouter.py` (multiple lines)
  - `src/utils/advisor_matcher.py`
  - `src/utils/batch.py`
  - `src/pipeline.py`
- **Issue:** `print()` statements for progress mixed with production code
- **Impact:** Console pollution, harder debugging
- **Fix:** Use `st.info()`/`st.success()` or proper logging

### Unpinned Dependency Upper Bounds
- **Location:** `requirements.txt`
- **Issue:** Version specs use `>=` without upper bounds
- **Example:** `pandas>=2.0.0` could break with `pandas 3.x`
- **Fix:** Use bounded versions like `pandas>=2.0.0,<3.0`

### Debug Files Not Cleaned Up
- **Location:** `src/clients/json_repair.py`
- **Issue:** JSON parse failures save debug files that accumulate
- **Fix:** Implement debug file rotation or age limit

## Documentation Gaps

### Missing README.md
- **Location:** Project root
- **Issue:** No main documentation file for onboarding
- **Fix:** Create README with setup guide, architecture overview

### Complex Logic Lacking Comments
- **Locations:**
  - `src/utils/data_unifier.py` - Commission calculations
  - `src/clients/json_repair.py` - Repair strategies
  - `src/utils/advisor_matcher.py` - Fuzzy matching algorithm
- **Fix:** Add inline documentation for business logic

## Testing Gaps

### Missing Test Coverage
- **Untested Modules:**
  - `src/clients/cache.py` - Cache invalidation logic
  - `src/clients/retry_handler.py` - Retry behavior under failure
  - `src/utils/config.py` - Configuration edge cases
- **Fix:** Add unit tests for cache, retry, and config modules

## Positive Findings

**Well-implemented areas:**
- PDF file handling with proper `try/finally` in `src/utils/pdf.py`
- Pydantic validation with field validators in all models
- JSON repair strategies for malformed LLM responses
- Exponential backoff with configurable retry strategies
- Environment variable management (except for git history issue)

---

*Concerns analysis: 2026-01-11*
*Review and update periodically*
