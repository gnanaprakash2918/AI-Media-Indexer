"""System configuration and health check routes."""

import asyncio
import json
from typing import Any

import numpy as np
from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from api.file_dialog import open_file_dialog
from config import settings
from core.utils.hardware import (
    get_available_vram,
    get_cached_profile,
    get_used_vram,
    refresh_profile,
)
from core.utils.logger import logger
from core.utils.progress import progress_tracker


def _numpy_serializer(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Unable to serialize {type(obj)}")


router = APIRouter()


@router.get("/system/browse")
async def browse_file_system(
    directory: bool = Query(
        default=False, description="Select folder instead of file"
    ),
) -> dict:
    """Opens a native file dialog for the user to select a file or folder.

    Spawns a separate process so tkinter gets its own main thread,
    avoiding conflicts with FastAPI/uvicorn's event loop.

    Args:
        directory: If True, opens folder picker. Otherwise opens file picker.

    Returns:
        Dictionary with selected 'path' or null if cancelled.
    """
    path = await open_file_dialog(directory)
    return {"path": path}


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
            # VLM / Video Understanding options
            "enable_frame_vlm": getattr(settings, "enable_frame_vlm", True),
            "enable_video_embeddings": getattr(
                settings, "enable_video_embeddings", True
            ),
            "enable_hybrid_vlm": getattr(settings, "enable_hybrid_vlm", True),
            "enable_visual_embeddings": getattr(
                settings, "enable_visual_embeddings", True
            ),
            "enable_voice_analysis": getattr(
                settings, "enable_voice_analysis", True
            ),
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
        # VLM / Video Understanding options
        "enable_frame_vlm",
        "enable_video_embeddings",
        "enable_hybrid_vlm",
        # Visual features
        "enable_visual_embeddings",
        # Voice analysis
        "enable_voice_analysis",
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
            "qdrant": "connected"
            if pipeline and pipeline.db
            else "disconnected",
            "stats": stats,
            "observability": observability_status,
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
                    data = json.dumps(event, default=_numpy_serializer)
                    yield f"data: {data}\n\n"
                except TimeoutError:
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
