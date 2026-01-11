# Coding Conventions

**Analysis Date:** 2026-01-11

## Naming Patterns

**Files:**
- `snake_case.py` for all modules (e.g., `uv_extractor.py`, `json_repair.py`)
- `*_extractor.py` for concrete extractor implementations
- `test_*.py` for test files (e.g., `test_uv.py`)
- `*.yaml` for prompt configuration files

**Functions:**
- `snake_case` for all functions (e.g., `pdf_to_images()`, `get_pdf_hash()`)
- `@property` methods use lowercase (e.g., `source_name`, `document_type`)
- Private methods prefixed with underscore (e.g., `_encode_image()`, `_build_messages()`)

**Variables:**
- `snake_case` for variables (e.g., `pdf_path`, `model_config`)
- `UPPER_SNAKE_CASE` for module-level constants (e.g., `API_URL`, `DEFAULT_BATCH_SIZE`)

**Types:**
- `PascalCase` for classes (e.g., `UVExtractor`, `BaseExtractor`, `OpenRouterClient`)
- `PascalCase` for Pydantic models (e.g., `UVReport`, `IDCProposition`)
- `PascalCase` for Enums (e.g., `SourceType`, `ExtractionMode`, `BoardType`)
- No prefix for interfaces/protocols

## Code Style

**Formatting:**
- 4 spaces per indentation level
- Double quotes for strings (`"hello"` not `'hello'`)
- No explicit formatter configuration (follows PEP 8)
- Line length: Not strictly enforced but generally <120 characters

**Linting:**
- No explicit linter configuration (`.flake8`, `setup.cfg` not present)
- Type hints expected throughout
- Run: Manual (no automated linting)

## Import Organization

**Order:**
1. Standard library imports (`import asyncio`, `from pathlib import Path`)
2. Third-party imports (`import pandas as pd`, `from pydantic import BaseModel`)
3. Local imports (`from ..clients.cache import ExtractionCache`)

**Grouping:**
- Blank line between groups
- No alphabetical sorting required
- Type imports mixed with regular imports

**Path Aliases:**
- Relative imports within package (`from ..utils.config import Settings`)
- Absolute imports for top-level (`from src.pipeline import Pipeline`)

## Error Handling

**Patterns:**
- Custom exception classes inherit from `Exception` (e.g., `OpenRouterError`, `MondayError`)
- Exceptions use `@dataclass` decorator for structured attributes
- Async functions use try/except blocks

**Error Types:**
- `OpenRouterError` - VLM API failures
- `OpenRouterRateLimitError` - Rate limiting
- `RetryExhaustedError` - All retries failed
- `ValidationError` (Pydantic) - Schema validation failures
- `FileNotFoundError` - Missing PDF files

**Error Strategy:**
- Throw at source, catch at boundaries (pipeline, UI)
- Log error context before raising
- Multi-level fallbacks for VLM models

## Logging

**Framework:**
- Python `logging` module
- Console output (`print()` in tests for progress)

**Patterns:**
- `logger = logging.getLogger(__name__)` in modules
- Debug/info/error levels used
- Progress feedback via print statements in test scripts

**When:**
- API call failures
- Cache operations
- Extraction progress

## Comments

**When to Comment:**
- Module-level docstrings explaining purpose
- Class docstrings with description
- Function docstrings for public APIs
- Inline comments for non-obvious logic

**JSDoc/TSDoc:**
- Not applicable (Python codebase)

**Docstring Format:**
```python
"""
Short description of the function.

Args:
    param_name: Description of parameter

Returns:
    Description of return value

Raises:
    ExceptionType: When this happens
"""
```

**TODO Comments:**
- Format: `# TODO: description`
- No username prefix required

## Function Design

**Size:**
- No strict limit, but prefer focused functions
- Extract helpers for complex logic
- Some large files exist (e.g., `data_unifier.py`)

**Parameters:**
- Type hints on all parameters: `def extract(pdf_path: Union[str, Path]) -> T:`
- Default values for optional params: `max_concurrent: int = 3`
- `Optional[T]` for nullable params

**Return Values:**
- Explicit return type hints
- Generic types: `BaseExtractor[T]`
- Union types: `Union[str, Path]` or `str | Path` (modern syntax)

## Module Design

**Exports:**
- Public API exported in `__init__.py`
- Example: `src/__init__.py` exports `Pipeline`, `DataUnifier`, `extract_pdf()`

**Barrel Files:**
- `__init__.py` in each package directory
- Re-exports key classes/functions
- Example: `src/models/__init__.py` exports all model classes

## Type Hints

**Coverage:**
- Extensive type hints throughout codebase
- All function signatures typed
- Class attributes typed via Pydantic or dataclass

**Patterns:**
- `Union[str, Path]` for path parameters (also `str | Path` modern syntax)
- `Optional[T]` for nullable types
- `Generic[T]` for parameterized classes
- `TypeVar("T")` for generic type parameters

## Pydantic Patterns

**Model Definition:**
```python
class UVActivity(BaseModel):
    contrat: str = Field(
        ...,
        description="Contract number",
        examples=["110970886"],
    )
```

**Validation:**
- `@field_validator` for custom validation
- `Field(...)` for required fields
- `Field(default=None)` for optional fields

---

*Convention analysis: 2026-01-11*
*Update when patterns change*
