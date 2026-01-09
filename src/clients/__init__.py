"""API clients for external services and caching."""

from .cache import ExtractionCache
from .json_repair import (
    repair_json,
    strip_markdown,
    extract_json_from_response,
    safe_json_parse,
    save_debug_json,
)
from .openrouter import OpenRouterClient, OpenRouterError, OpenRouterRateLimitError
from .retry_handler import (
    RetryExhaustedError,
    RetryStrategy,
    async_retry,
    default_strategy,
    with_retry,
)
from .monday import (
    MondayClient,
    MondayError,
    ColumnType,
    CreateResult,
    UploadResult,
    get_monday_client,
    get_board_id_for_type,
)

__all__ = [
    # Cache
    "ExtractionCache",
    # OpenRouter client
    "OpenRouterClient",
    "OpenRouterError",
    "OpenRouterRateLimitError",
    # Monday.com client
    "MondayClient",
    "MondayError",
    "ColumnType",
    "CreateResult",
    "UploadResult",
    "get_monday_client",
    "get_board_id_for_type",
    # JSON repair utilities
    "repair_json",
    "strip_markdown",
    "extract_json_from_response",
    "safe_json_parse",
    "save_debug_json",
    # Retry utilities
    "async_retry",
    "with_retry",
    "RetryStrategy",
    "RetryExhaustedError",
    "default_strategy",
]
