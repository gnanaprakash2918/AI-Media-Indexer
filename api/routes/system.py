"""System configuration and health check routes."""
import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from config import settings
from core.utils.hardware import (
    get_available_vram,
    get_cached_profile,
    get_used_vram,
    refresh_profile,
)
from core.utils.logger import logger
from core.utils.progress import progress_tracker

router = APIRouter()


@router.get("/config/system")
async def get_system_config():
    """Get detected hardware profile and current settings."""
    profile = get_cached_profile()
    return {
        "profile": profile.to_dict(),
        "vram_used_gb": round(get_used_vram(), 2),
        "vram_free_gb": round(get_available_vram() - get_used_vram(), 2),
        "settings": {
            "embedding_model": settings.embedding_model_override,
            "vision_model": settings.ollama_vision_model,
            "batch_size": settings.batch_size,
            "max_concurrent_jobs": settings.max_concurrent_jobs,
            "lazy_unload": settings.lazy_unload,
            "high_performance_mode": settings.high_performance_mode,
        },
    }


@router.post("/config/system")
async def update_system_config(updates: dict):
    """Runtime override of system settings."""
    # Note: We use dict instead of ConfigUpdate schema to allow partial updates more flexibly,
    # but validation is good. The original code used dict, schemas uses ConfigUpdate.
    # We should support both or stick to one. The original code iterated over allowed_keys.

    allowed_keys = {
        "batch_size",
        "max_concurrent_jobs",
        "lazy_unload",
        "high_performance_mode",
    }
    applied = {}
    for key, value in updates.items():
        if key in allowed_keys:
            setattr(settings, key, value)
            applied[key] = value

    new_profile = refresh_profile()

    return {"applied": applied, "new_profile": new_profile.to_dict()}


@router.get("/health")
async def health_check(request: Request):
    """Health check endpoint with Performance and Observability checks."""
    try:
        pipeline = getattr(request.app.state, "pipeline", None)

        # 1. DB Stats
        stats = None
        if pipeline and pipeline.db:
            try:
                stats = pipeline.db.get_collection_stats()
            except Exception:
                pass

        # 2. Observability Check
        observability_status = "disabled"
        if settings.langfuse_public_key:
            observability_status = "configured"

        return {
            "status": "ok",
            "device": settings.device,
            "pipeline": "ready" if pipeline else "unavailable",
            "qdrant": "connected" if pipeline and pipeline.db else "disconnected",
            "stats": stats,
            "observability": observability_status,
            "asr_mode": "Native" if settings.use_native_nemo else "Docker/Whisper",
        }
    except Exception as e:
        logger.exception("Health check failed")
        return {"status": "error", "message": str(e), "type": type(e).__name__}


@router.get("/events")
async def sse_events():
    """Server-Sent Events endpoint for real-time updates with heartbeat."""

    async def event_generator():
        queue = progress_tracker.subscribe()
        heartbeat_interval = 15  # seconds
        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=heartbeat_interval,
                    )
                    data = json.dumps(event)
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield ": heartbeat\n\n"
        finally:
            progress_tracker.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
