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

VERBOSE_PATTERNS = (
    "Frame ",
    "Processing frame",
    "Processed frame",
    "chunk_",
    "frames_chunk",
    "Embedding",
    "Vector",
    "Upserting",
    "Flushing buffer",
    "batch",
    "CLAP",
    "Whisper",
    "Transcribing",
    "segment ",
)

PROGRESS_PATTERNS = (
    "[PROGRESS]",
    "% complete",
    "Stage:",
    "ETA:",
)


def _is_verbose_message(msg: str) -> bool:
    return any(p in msg for p in VERBOSE_PATTERNS)


def _is_progress_message(msg: str) -> bool:
    return any(p in msg for p in PROGRESS_PATTERNS)


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


def _console_filter(record) -> bool:
    msg = record["message"]
    if record["extra"].get("file_only"):
        return False
    if record["level"].name == "DEBUG" and _is_verbose_message(msg):
        return False
    return True


def setup_logger() -> None:
    """Configure Loguru for console and file logging.

    Features:
    - Clean console with important logs only (INFO+, progress updates)
    - Full DEBUG logs to file for debugging
    - Progress-specific formatting
    - Verbose internal logs filtered from console
    """
    logger.remove()

    def _patcher(record):
        if record["extra"].get("trace_id") is None:
            record["extra"]["trace_id"] = trace_id_ctx.get()
        if record["extra"].get("span") is None:
            record["extra"]["span"] = span_ctx.get()
        if record["extra"].get("component") is None:
            record["extra"]["component"] = component_ctx.get()
        if "file_only" not in record["extra"]:
            record["extra"]["file_only"] = False

    logger.configure(patcher=_patcher)

    log_dir: Path = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        backtrace=False,
        diagnose=False,
        filter=_console_filter,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<7}</level> | "
            "<level>{message}</level>"
        ),
    )

    logger.add(
        str(log_file),
        level="DEBUG",
        rotation="00:00",
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

    class ShardingFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            if "The following layers were not sharded" in msg:
                return False
            if "unexpected keyword argument 'dtype'" in msg:
                return False
            return True

    sharding_filter = ShardingFilter()

    noisy_ml_frameworks = (
        "speechbrain",
        "transformers",
        "transformers.modeling_utils",
        "accelerate",
        "torch",
        "torch.distributed",
        "torch.cuda",
        "whisper",
        "faster_whisper",
        "ctranslate2",
        "onnxruntime",
        "PIL",
        "matplotlib",
        "numba",
        "filelock",
        "huggingface_hub",
        "datasets",
    )
    for framework in noisy_ml_frameworks:
        framework_logger = logging.getLogger(framework)
        framework_logger.setLevel(logging.WARNING)
        framework_logger.addFilter(sharding_filter)


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


def log_progress(
    stage: str,
    percent: float,
    message: str = "",
    eta_seconds: float | None = None,
    speed: str = "",
) -> None:
    """Log a clean progress update to terminal.

    Shows a single-line progress update that's easy to read.
    Format: [PROGRESS] Stage: 45% | message | ETA: 2m 30s | 2.5 fps

    Args:
        stage: Current pipeline stage name
        percent: Progress percentage (0-100)
        message: Optional status message
        eta_seconds: Optional ETA in seconds
        speed: Optional speed string (e.g., "2.5 fps")
    """
    parts = [f"[PROGRESS] {stage}: {percent:.0f}%"]
    if message:
        parts.append(message)
    if eta_seconds is not None and eta_seconds > 0:
        if eta_seconds < 60:
            parts.append(f"ETA: {eta_seconds:.0f}s")
        elif eta_seconds < 3600:
            parts.append(f"ETA: {eta_seconds / 60:.1f}m")
        else:
            parts.append(f"ETA: {eta_seconds / 3600:.1f}h")
    if speed:
        parts.append(speed)

    _base_logger().info(" | ".join(parts))


def log_verbose(message: str, **kwargs: Any) -> None:
    """Log a verbose debug message to file only.

    Use this for high-frequency internal operations that would spam the terminal.
    These logs still go to the log file for debugging.

    Args:
        message: The log message
        **kwargs: Additional context to include
    """
    _base_logger({"file_only": True, **kwargs}).debug(message)


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


# Alias for backward compatibility
configure_logger = setup_logger


class LokiSink:
    """Loguru sink for Grafana Loki."""

    def __init__(self, url: str) -> None:
        """Initializes the Loki sink.

        Args:
            url: The HTTP URL of the Loki push endpoint.
        """
        self.url = url
        self.session = None

    def write(self, message: str) -> None:
        """Send log record to Loki."""
        import json
        import time

        import httpx

        # message is a serialized JSON string from loguru
        try:
            record = json.loads(message)
            text = record["text"]
            record_level = record["record"]["level"]["name"]
            timestamp_ns = str(int(time.time() * 1e9))

            payload = {
                "streams": [
                    {
                        "stream": {
                            "app": "ai-media-indexer",
                            "level": record_level,
                            "component": record["extra"].get(
                                "component", "unknown"
                            ),
                        },
                        "values": [[timestamp_ns, text]],
                    }
                ]
            }

            # Use sync client (since this runs in a thread via enqueue=True)
            # We create a new client per request or use a session?
            # Ideally session, but thread safety...
            # requests/httpx clients are thread safe?
            # Just use one-off for simplicity or a module-level client.
            # For 2-3 logs/sec it's fine.

            httpx.post(
                self.url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=2.0,
            )
        except Exception:
            # Failsafe: don't crash logging
            pass


if settings.enable_loki:
    logger.add(
        LokiSink(settings.loki_url).write,
        level="INFO",
        serialize=True,  # Pass JSON string to sink
        enqueue=True,  # Run in background thread
    )
