from __future__ import annotations

import asyncio
from typing import Awaitable, Callable


async def retry(
    fn: Callable[[], Awaitable[None]],
    retries: int = 2,
    delay: float = 3.0,
) -> None:
    """Retry an async function with exponential backoff on failure."""
    last_exc: Exception | None = None
    for _ in range(retries + 1):
        try:
            await fn()
            return
        except Exception as exc:
            last_exc = exc
            await asyncio.sleep(delay)

    if last_exc:
        raise last_exc
    raise RuntimeError("Retries exhausted without exception capture")
