"""Observability module for distributed tracing with Langfuse."""

from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

from langfuse import Langfuse
from loguru import logger

from config import settings
from core.utils.logger import bind_context, clear_context

langfuse_client: Langfuse | None = None

trace_id_ctx: ContextVar[str | None] = ContextVar("obs_trace_id", default=None)

span_stack_ctx: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "obs_span_stack", default=None
)

trace_obj_ctx: ContextVar[Any | None] = ContextVar("obs_trace_obj", default=None)


def init_langfuse() -> None:
    """Initialize Langfuse client based on configuration."""
    global langfuse_client
    
    logger.info(f"🔍 Langfuse init: backend={settings.langfuse_backend}")
    
    if settings.langfuse_backend == "disabled":
        langfuse_client = None
        logger.info("  Langfuse disabled")
        return
        
    if settings.langfuse_backend == "cloud" and settings.langfuse_public_key:
        langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info(f"  ✅ Langfuse cloud initialized: {settings.langfuse_host}")
        
    elif settings.langfuse_backend == "docker":
        public_key = settings.langfuse_public_key or "local"
        secret_key = settings.langfuse_secret_key or "local"
        # Use LANGFUSE_HOST if set, otherwise default docker host
        host = settings.langfuse_host if settings.langfuse_host != "https://cloud.langfuse.com" else settings.langfuse_docker_host
        langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info(f"  ✅ Langfuse docker initialized: {host}")
    else:
        logger.warning(f"  ⚠️ Langfuse not initialized: backend={settings.langfuse_backend}, key_present={bool(settings.langfuse_public_key)}")


def start_trace(name: str, metadata: dict[str, Any] | None = None) -> str:
    """Start a new trace."""
    
    # Reset contexts
    clear_context()
    
    if langfuse_client:
        trace_obj = langfuse_client.start_span(
            name=name,
            metadata=metadata or {},
        )
        trace_obj_ctx.set(trace_obj)
        trace_id = trace_obj.trace_id
    else:
        trace_id = str(uuid4())
        
    trace_id_ctx.set(trace_id)
    bind_context(trace_id=trace_id)
        
    logger.info("trace_start", trace=name)
    return trace_id


def end_trace(status: str = "success", error: str | None = None) -> None:
    """End the current trace."""
    trace_obj = trace_obj_ctx.get()
    
    if trace_obj:
        update_kwargs: dict[str, Any] = {"metadata": {"status": status, "error": error}}
        if status == "error":
            update_kwargs["level"] = "ERROR"
            update_kwargs["status_message"] = error
            
        trace_obj.update(**update_kwargs)
        trace_obj.end()
        
        # Flush to ensure trace is sent to server
        if langfuse_client:
            try:
                langfuse_client.flush()
            except Exception:
                pass
        
    logger.info("trace_end", status=status, error=error)
    trace_id_ctx.set(None)
    span_stack_ctx.set(None)
    trace_obj_ctx.set(None)
    clear_context()


def start_span(name: str, metadata: dict[str, Any] | None = None) -> None:
    """Start a new span."""
    stack = span_stack_ctx.get() or []
    parent_item = stack[-1] if stack else None
    bind_context(span=name)
    span_client = None
    if langfuse_client:
        # Get parent: either the last span or the root trace object
        parent = None
        if parent_item and parent_item.get("span"):
             parent = parent_item["span"]
        else:
             parent = trace_obj_ctx.get()

        if parent:
            span_client = parent.start_span(
                name=name,
                metadata=metadata or {},
            )
            
    # Push to stack
    new_item = {"name": name, "span": span_client}
    span_stack_ctx.set(stack + [new_item])
    logger.info("span_start", span=name)


def end_span(status: str = "success", error: str | None = None) -> None:
    """End the current span."""
    stack = span_stack_ctx.get()
    if not stack:
        return

    item = stack[-1]
    name = item["name"]
    span_client = item["span"]

    if span_client:
        update_kwargs = {}
        if status == "error":
            update_kwargs["level"] = "ERROR"
            update_kwargs["status_message"] = error
            
        span_client.update(**update_kwargs)
        span_client.end()

    logger.info("span_end", span=name, status=status, error=error)
    
    # Pop
    new_stack = stack[:-1]
    span_stack_ctx.set(new_stack)
    
    # Restore context
    if new_stack:
        bind_context(span=new_stack[-1]["name"])
    else:
        bind_context(span=None)


class Span:
    """Context manager for observing a code block as a span."""

    def __init__(self, name: str, metadata: dict[str, Any] | None = None):
        """Initialize the span context manager."""
        self.name = name
        self.metadata = metadata
        self.start = 0.0

    def __enter__(self):
        """Start the span."""
        self.start = time.perf_counter()
        start_span(self.name, self.metadata)

    def __exit__(self, exc_type, exc, tb):
        """End the span, recording any exceptions."""
        duration = time.perf_counter() - self.start
        if exc:
            end_span("error", str(exc))
            logger.error(
                "span_failed",
                span=self.name,
                duration=duration,
                error=str(exc),
            )
        else:
            end_span("success")
            logger.info(
                "span_success",
                span=self.name,
                duration=duration,
            )
        return False
