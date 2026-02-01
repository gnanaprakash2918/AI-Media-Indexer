"""Decorator for observability span generation."""

from __future__ import annotations

import asyncio
import functools
import time
import traceback
from collections.abc import Callable
from typing import Any, TypeVar

from core.errors import MediaIndexerError
from core.utils.logger import log
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
                if isinstance(exc, MediaIndexerError):
                    log(f"[OBSERVE] {name} failed with known error: {exc}")
                else:
                    log(f"[OBSERVE] {name} CRASHED: {exc}")
                    log(traceback.format_exc())
                end_span("error", str(exc))
                raise
            finally:
                duration = time.perf_counter() - start
                log(f"[PERF] {name} took {duration:.4f}s")

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
                if isinstance(exc, MediaIndexerError):
                    log(f"[OBSERVE] {name} failed with known error: {exc}")
                else:
                    log(f"[OBSERVE] {name} CRASHED: {exc}")
                    log(traceback.format_exc())
                end_span("error", str(exc))
                raise
            finally:
                duration = time.perf_counter() - start
                log(f"[PERF] {name} took {duration:.4f}s")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator
