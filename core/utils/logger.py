"""Application-wide structured logging configuration.

Provides:

- Colorized, human-readable console logs (stderr).
- Structured JSONL file logs (agent.jsonl) for querying and ingestion.
- Full stdlib logging integration (third-party libraries included).
- Callsite, timestamp, level, and bound context support.

Initialize once at startup; import ``logger`` or use ``get_logger()``.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import structlog
from structlog.stdlib import BoundLogger
from structlog.types import EventDict, Processor, WrappedLogger

from ...config import settings

LOG_FILE_NAME = "agent.jsonl"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5


def _ensure_utf8_stderr() -> None:
    """Best-effort patch to make stderr UTF-8 capable.

    This prevents ``UnicodeEncodeError: 'charmap' codec can't encode...``
    when printing log lines that contain characters unsupported by the
    legacy Windows console encoding (e.g. full-width punctuation, emojis,
    Tamil, etc.).

    It uses ``TextIOBase.reconfigure`` when available (Python 3.7+),
    while guarding with ``hasattr`` so static analyzers like Pylance
    don't complain.
    """
    stderr = sys.stderr

    # Some environments wrap stderr with objects that don't expose
    # reconfigure; don't assume it always exists.
    if hasattr(stderr, "reconfigure"):
        try:
            stderr.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[call-arg]
        except Exception:
            pass


def _add_logger_name(
    logger: logging.Logger | WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add the logger name to the event dict.

    Mimics structlog.processors.add_logger_name in a type-checker-friendly way.

    Args:
        logger: The stdlib or wrapped logger instance.
        method_name: The log method name (for example, "info").
        event_dict: The current event dictionary.

    Returns:
        EventDict: The updated event dictionary including the "logger" field.
    """
    event_dict.setdefault("logger", logger.name)
    return event_dict


def _get_log_level() -> int:
    """Resolve the global log level.

    Precedence:

    1. settings.log_level
    2. LOG_LEVEL environment variable
    3. logging.INFO default

    Returns:
        int: Resolved stdlib logging level (for example, logging.INFO).
    """
    level_name = str(
        getattr(settings, "log_level", os.getenv("LOG_LEVEL", "INFO"))
    ).upper()
    return getattr(logging, level_name, logging.INFO)


def _ensure_log_dir(log_dir: Path) -> None:
    """Ensure that the log directory exists."""
    log_dir.mkdir(parents=True, exist_ok=True)


def configure_logger() -> None:
    """Configure global structured logging.

    Sets up:

    - Console logging (pretty, colorized).
    - Rotating JSONL file logging.
    - structlog + stdlib logging integration.

    Safe to call multiple times, but typically done once at process startup.
    """
    # Make console less fragile with Unicode first.
    _ensure_utf8_stderr()

    log_dir: Path = settings.log_dir
    _ensure_log_dir(log_dir)

    log_file = log_dir / LOG_FILE_NAME
    log_level = _get_log_level()

    timestamper: Processor = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        _add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.PATHNAME,
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
        timestamper,
    ]

    # Console handler: human-readable, colorized output.
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True, pad_event=2),
            foreign_pre_chain=shared_processors,
        )
    )

    # File handler: JSON Lines, rotated.
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=DEFAULT_MAX_BYTES,
        backupCount=DEFAULT_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=shared_processors,
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            _add_logger_name,
            timestamper,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None) -> BoundLogger:
    """Return a structured logger.

    Args:
        name: Optional logger name (usually __name__) for source tagging.

    Returns:
        BoundLogger: A structlog bound logger instance.
    """
    return structlog.get_logger(name) if name else structlog.get_logger()


def log(msg: str, **kwargs: Any) -> None:
    """Log an INFO-level message using the default logger.

    Compatibility wrapper for legacy calls like: ``log("event", key=value)``.

    Args:
        msg: Log event name or message.
        **kwargs: Additional structured fields to attach to the event.
    """
    logger.info(msg, **kwargs)


# Configure on import so all logs use this pipeline.
configure_logger()

# Default application logger.
logger = get_logger(__name__)
