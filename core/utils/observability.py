"""Observability module for distributed tracing with Langfuse SDK v3."""

from __future__ import annotations

import os
import time
import base64
import json
import logging
from contextvars import ContextVar
from datetime import datetime
from typing import Any
from uuid import uuid4

import requests
from loguru import logger

from config import settings
from core.utils.logger import bind_context, clear_context

# Module-level flag for initialization
_langfuse_initialized = False

# ContextVars
trace_id_ctx: ContextVar[str | None] = ContextVar("obs_trace_id", default=None)
# Stack items: {"name": str, "id": str, "start_time": float}
span_stack_ctx: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "obs_span_stack", default=None
)
# We no longer store SDK objects
trace_obj_ctx: ContextVar[Any | None] = ContextVar("obs_trace_obj", default=None)


def init_langfuse() -> None:
    """Initialize Langfuse via environment variables (checking only)."""
    global _langfuse_initialized
    
    backend = settings.langfuse_backend
    
    # Auto-enable cloud if keys are present and backend is disabled (default)
    if backend == "disabled" and settings.langfuse_public_key and settings.langfuse_secret_key:
        backend = "cloud"
        logger.info("🔍 Langfuse: Auto-enabling cloud mode (keys detected)")
    
    logger.info(f"🔍 Langfuse init: backend={backend}")
    
    if backend == "disabled":
        return
    
    # Set environment variables (kept for compatibility or if we use SDK later)
    if settings.langfuse_public_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    if settings.langfuse_secret_key:
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    
    if backend == "cloud":
        os.environ["LANGFUSE_HOST"] = settings.langfuse_host
        logger.info(f"  ✅ Langfuse cloud configured: {settings.langfuse_host}")
    elif backend == "docker":
        host = settings.langfuse_docker_host or "http://localhost:3300"
        os.environ["LANGFUSE_HOST"] = host
        logger.info(f"  ✅ Langfuse docker configured: {host}")
        
    _langfuse_initialized = True
    logger.info("  ✅ Langfuse raw client ready")


def _send_ingestion_event(event_type: str, body: dict[str, Any]) -> None:
    """Send a raw ingestion event to Langfuse API."""
    try:
        # Resolving config with fallback to settings
        host = os.environ.get("LANGFUSE_HOST") or settings.langfuse_docker_host or "http://localhost:3300"
        pk = os.environ.get("LANGFUSE_PUBLIC_KEY") or settings.langfuse_public_key
        sk = os.environ.get("LANGFUSE_SECRET_KEY") or settings.langfuse_secret_key
        
        if not pk or not sk:
            # logger.warning(f"Langfuse skipped: Missing keys (pk={bool(pk)}, sk={bool(sk)})")
            return

        auth_str = f"{pk}:{sk}"
        b64_auth = base64.b64encode(auth_str.encode()).decode()
        
        url = f"{host}/api/public/ingestion"
        headers = {
            "Authorization": f"Basic {b64_auth}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "batch": [
                {
                    "id": str(uuid4()),
                    "type": event_type,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "body": body
                }
            ]
        }
        
        # logger.info(f"Langfuse Sending {event_type} to {url}...")
        
        try:
             resp = requests.post(url, headers=headers, json=payload, timeout=2.0)
             if resp.status_code not in (200, 201, 207):
                 logger.warning(f"Langfuse API Error {resp.status_code}: {resp.text}")
             # else:
             #    logger.info(f"Langfuse Sent {event_type}: {resp.status_code}")
                 
        except requests.exceptions.ReadTimeout:
             logger.warning(f"Langfuse timeout sending {event_type}")
        except Exception as e:
             logger.warning(f"Langfuse ingestion error: {e}")
             
    except Exception as e:
        logger.warning(f"Langfuse raw send failed: {e}")


def start_trace(name: str, metadata: dict[str, Any] | None = None) -> str:
    """Start a new trace."""
    clear_context()
    
    trace_id = str(uuid4())
    trace_id_ctx.set(trace_id)
    bind_context(trace_id=trace_id)
    
    # Send trace-create
    _send_ingestion_event("trace-create", {
        "id": trace_id,
        "name": name,
        "userId": "system",
        "metadata": metadata or {},
        # "timestamp": datetime.utcnow().isoformat() + "Z"  # Let server set timestamp to avoid 'delayed' queue
    })
        
    logger.info("trace_start", trace=name)
    return trace_id


def end_trace(status: str = "success", error: str | None = None) -> None:
    """End the current trace (not strictly required for Langfuse trace-create but good for logging)."""
    logger.info("trace_end", status=status, error=error)
    trace_id_ctx.set(None)
    span_stack_ctx.set(None)
    trace_obj_ctx.set(None)
    clear_context()


def start_span(name: str, metadata: dict[str, Any] | None = None) -> None:
    """Start a new span using raw API."""
    trace_id = trace_id_ctx.get()
    if not trace_id:
        return # Cannot start span without trace
        
    stack = span_stack_ctx.get() or []
    
    # Determine parent
    parent_id = None
    if stack:
        parent_id = stack[-1]["id"]
    
    span_id = str(uuid4())
    start_time = time.time()
    
    bind_context(span=name)
    
    # Send span-create
    body = {
        "id": span_id,
        "traceId": trace_id,
        "name": name,
        # "startTime": datetime.utcfromtimestamp(start_time).isoformat() + "Z", # Let server set start
        "metadata": metadata or {},
        "type": "span"
    }
    if parent_id:
        body["parentObservationId"] = parent_id
        
    _send_ingestion_event("span-create", body)
    
    # Push to stack
    new_item = {"name": name, "id": span_id, "start_time": start_time}
    span_stack_ctx.set(stack + [new_item])
    
    logger.info("span_start", span=name)


def end_span(status: str = "success", error: str | None = None) -> None:
    """End the current span using raw API."""
    stack = span_stack_ctx.get()
    if not stack:
        return

    item = stack[-1]
    name = item["name"]
    span_id = item["id"]
    start_time = item["start_time"]
    
    end_time = time.time()
    
    trace_id = trace_id_ctx.get()
    
    # Send span-create (upsert) with end time
    if trace_id:
        body = {
            "id": span_id,
            "traceId": trace_id,
            "name": name,
            "startTime": datetime.utcfromtimestamp(start_time).isoformat() + "Z",
            "endTime": datetime.utcfromtimestamp(end_time).isoformat() + "Z",
            "type": "span"
        }
        
        if status == "error":
            body["level"] = "ERROR"
            body["statusMessage"] = error or "Unknown Error"
        
        # We assume parentId hasn't changed. Sending minimal fields + endTime might behave as upsert
        # but Langfuse ingestion usually expects full object or at least required fields.
        # We re-send required fields.
        
        _send_ingestion_event("span-create", body)

    logger.info("span_end", span=name, status=status, error=error)
    
    new_stack = stack[:-1]
    span_stack_ctx.set(new_stack)
    
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
