"""Decorator for observability span generation."""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Callable, TypeVar

from core.utils.observability import end_span, start_span

F = TypeVar("F", bound=Callable[..., Any])


def observe(name: str) -> Callable[[F], F]:
    """Decorator to observe a function execution as a span."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            # Capture args safely (concise)
            metadata = {
                "args": [str(a)[:200] for a in args],
                "kwargs": {k: str(v)[:200] for k, v in kwargs.items()},
            }
            start_span(name, metadata=metadata)
            try:
                result = await func(*args, **kwargs)
                end_span("success")
                return result
            except Exception as exc:
                end_span("error", str(exc))
                raise
            finally:
                _ = time.perf_counter() - start

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            # Capture args safely (concise)
            metadata = {
                "args": [str(a)[:200] for a in args],
                "kwargs": {k: str(v)[:200] for k, v in kwargs.items()},
            }
            start_span(name, metadata=metadata)
            try:
                result = func(*args, **kwargs)
                end_span("success")
                return result
            except Exception as exc:
                end_span("error", str(exc))
                raise
            finally:
                _ = time.perf_counter() - start

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator
