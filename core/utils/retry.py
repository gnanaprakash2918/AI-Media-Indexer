"""Retry utilities for asynchronous operations."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


async def retry(
    fn: Callable[[], Awaitable[None]],
    retries: int = 2,
    delay: float = 3.0,
) -> None:
    """Retry an async function with exponential backoff on failure.

    Logs each retry attempt with the error message for debugging.
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            await fn()
            return
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                logger.warning(
                    f"[RETRY] Attempt {attempt + 1}/{retries + 1} failed: {type(exc).__name__}: {exc}. "
                    f"Retrying in {delay}s..."
                )
            else:
                logger.error(
                    f"[RETRY] All {retries + 1} attempts failed. Last error: {type(exc).__name__}: {exc}"
                )
            await asyncio.sleep(delay)

    if last_exc:
        raise last_exc
    raise RuntimeError("Retries exhausted without exception capture")
