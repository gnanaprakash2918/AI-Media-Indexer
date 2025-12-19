"""Logging configuration and utilities."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from loguru import logger

from config import settings

trace_id_ctx: ContextVar[str | None] = ContextVar("trace_id", default=None)
span_ctx: ContextVar[str | None] = ContextVar("span", default=None)
component_ctx: ContextVar[str | None] = ContextVar("component", default=None)

class InterceptHandler(logging.Handler):
    """Redirects standard logging into Loguru while preserving correct caller info."""

    def emit(self, record: logging.LogRecord) -> None:
        """Log the specified logging record."""
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        _base_logger().opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


def setup_logger() -> None:
    """Configure Loguru for console and file logging.

    Features:
    - Human-friendly console logs
    - Structured JSON file logs (Grafana Loki / ELK)
    - Full interception of stdlib logging
    """
    logger.remove()
    
    def _patcher(record):
        record["extra"].setdefault("trace_id", None)
        record["extra"].setdefault("span", None)
        record["extra"].setdefault("component", None)

    # Ensure trace_id always exists to prevent KeyErrors in format string
    logger.configure(patcher=_patcher)

    # Ensure log directory exists
    log_dir: Path = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    logger.add(
        sys.stderr,
        level=getattr(settings, "log_level", "INFO"),
        colorize=True,
        backtrace=True,
        diagnose=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level:<8}</level> | "
            "trace=<cyan>{extra[trace_id]}</cyan> "
            "span=<cyan>{extra[span]}</cyan> "
            "comp=<cyan>{extra[component]}</cyan> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    logger.add(
        str(log_file),
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        serialize=True,
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    logging.basicConfig(
        handlers=[InterceptHandler()],
        level=0,
        force=True,
    )

    for noisy in (
        "uvicorn.access",
        "uvicorn.error",
        "httpx",
        "httpcore",
        "asyncio",
    ):
        _logger = logging.getLogger(noisy)
        _logger.handlers = [InterceptHandler()]
        _logger.propagate = False


def bind_context(
    *,
    trace_id: str | None = None,
    span: str | None = None,
    component: str | None = None,
) -> None:
    """Bind context vars globally (used by middleware / pipeline)."""
    if trace_id is not None:
        trace_id_ctx.set(trace_id)
    if span is not None:
        span_ctx.set(span)
    if component is not None:
        component_ctx.set(component)


def clear_context() -> None:
    """Clear all context variables."""
    trace_id_ctx.set(None)
    span_ctx.set(None)
    component_ctx.set(None)


def _base_logger(extra: dict[str, Any] | None = None):
    return logger.bind(
        trace_id=trace_id_ctx.get(),
        span=span_ctx.get(),
        component=component_ctx.get(),
        **(extra or {}),
    )


def log(message: str, level: str = "INFO", **kwargs: Any) -> None:
    """Backward-compatible logging helper.

    Usage:
        log("Starting transcription", video=path)
        log("Whisper failed", level="ERROR", error=str(exc))
    """
    lvl = level.upper()
    _base_logger(kwargs).log(lvl, message)


def _handle_uncaught(exc_type, exc, tb):
    logger.bind(
        trace_id=trace_id_ctx.get(),
        span=span_ctx.get(),
        component=component_ctx.get(),
    ).opt(exception=(exc_type, exc, tb)).critical("Unhandled exception")


sys.excepthook = _handle_uncaught

setup_logger()


def get_logger(name: str | None = None):
    """Get a logger instance (optionally bound to a name)."""
    return logger.bind(name=name)
