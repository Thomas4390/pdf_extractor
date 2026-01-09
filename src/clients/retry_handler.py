"""
Retry handling utilities for API calls.

Provides decorators and utilities for implementing exponential backoff
retry logic with configurable strategies for different error types.
"""

import asyncio
import functools
import logging
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, message: str, last_error: Optional[Exception] = None):
        super().__init__(message)
        self.last_error = last_error


async def async_retry(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    **kwargs: Any,
) -> T:
    """
    Execute an async function with exponential backoff retry.

    Args:
        func: Async function to execute
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        retryable_exceptions: Tuple of exception types to retry on
        on_retry: Optional callback called on each retry (attempt, exception)
        **kwargs: Keyword arguments for the function

    Returns:
        Result from the function

    Raises:
        RetryExhaustedError: When all retries are exhausted
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as e:
            last_error = e

            if attempt < max_retries - 1:
                delay = min(base_delay * (exponential_base**attempt), max_delay)

                if on_retry:
                    on_retry(attempt + 1, e)
                else:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                await asyncio.sleep(delay)

    raise RetryExhaustedError(
        f"Failed after {max_retries} attempts", last_error=last_error
    )


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable:
    """
    Decorator for async functions with exponential backoff retry.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        retryable_exceptions: Tuple of exception types to retry on
        on_retry: Optional callback called on each retry

    Returns:
        Decorated function

    Example:
        @with_retry(max_retries=3, retryable_exceptions=(httpx.HTTPError,))
        async def fetch_data():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await async_retry(
                func,
                *args,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                retryable_exceptions=retryable_exceptions,
                on_retry=on_retry,
                **kwargs,
            )

        return wrapper

    return decorator


class RetryStrategy:
    """
    Configurable retry strategy for API calls.

    Supports different behaviors for different error types:
    - Rate limits: longer delays
    - Network errors: standard exponential backoff
    - Validation errors: may or may not retry based on config
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        rate_limit_delay: float = 30.0,
        rate_limit_multiplier: float = 1.5,
    ):
        """
        Initialize retry strategy.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            rate_limit_delay: Base delay for rate limit errors
            rate_limit_multiplier: Multiplier for rate limit delays
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.rate_limit_delay = rate_limit_delay
        self.rate_limit_multiplier = rate_limit_multiplier

    def get_delay(
        self,
        attempt: int,
        is_rate_limit: bool = False,
    ) -> float:
        """
        Calculate delay for a given attempt.

        Args:
            attempt: Current attempt number (0-indexed)
            is_rate_limit: Whether this is a rate limit error

        Returns:
            Delay in seconds
        """
        if is_rate_limit:
            delay = self.rate_limit_delay * (self.rate_limit_multiplier**attempt)
        else:
            delay = self.base_delay * (2**attempt)

        return min(delay, self.max_delay)

    def should_retry(self, attempt: int) -> bool:
        """Check if another retry should be attempted."""
        return attempt < self.max_retries - 1


# Default strategy for API calls
default_strategy = RetryStrategy(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    rate_limit_delay=30.0,
)
