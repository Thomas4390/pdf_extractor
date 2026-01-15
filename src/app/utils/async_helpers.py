"""
Async helper utilities for Streamlit application.

Provides safe async execution in environments that may already
have a running event loop (like Streamlit or Jupyter).
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine, TypeVar

T = TypeVar('T')


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine safely, handling existing event loops.

    This function handles the common issue where asyncio.run() fails
    when called from within an existing event loop (e.g., in Streamlit
    or Jupyter environments).

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(coro)

    # Event loop already running - use nest_asyncio if available,
    # otherwise run in a separate thread
    try:
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except ImportError:
        # nest_asyncio not available - run in thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
