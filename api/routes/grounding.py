"""API routes for visual grounding and segment management."""

from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from core.processing.grounding_pipeline import GroundingPipeline
from core.storage.db import VectorDB
from core.utils.logger import get_logger

router = APIRouter()
log = get_logger(__name__)

# Initialize pipeline lazily or globally?
# For now, instantiate per request or globally. Global is better.
grounding_pipeline = GroundingPipeline()
db = VectorDB()


class TriggerRequest(BaseModel):
    """Request schema for triggering visual grounding."""

    video_path: str
    concepts: list[str] | None = None


class MaskletUpdate(BaseModel):
    """Schema for updating a masklet or segment."""

    label: str | None = None
    confidence: float | None = None
    payload: dict[str, Any] | None = None


@router.post("/grounding/trigger")
async def trigger_grounding(
    request: TriggerRequest, background_tasks: BackgroundTasks
):
    """Trigger visual grounding for a video in the background.

    Args:
        request: The grounding request (video path, concepts).
        background_tasks: FastAPI background task manager.

    Returns:
        Status message confirming the job is queued.
    """
    try:
        # Run in background to avoid blocking
        # We wrap the coroutine because BackgroundTasks expects a callable
        background_tasks.add_task(
            grounding_pipeline.process_video,
            request.video_path,
            request.concepts,
        )
        return {"status": "queued", "video_path": request.video_path}
    except Exception as e:
        log.error(f"Failed to trigger grounding: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.patch("/masklets/{masklet_id}")
async def update_masklet(masklet_id: str, update: MaskletUpdate):
    """Update a masklet's metadata (e.g. for UI corrections).

    Args:
        masklet_id: Unique identifier for the masklet.
        update: Metadata updates to apply.

    Returns:
        Update status confirming changes.
    """
    updates = update.model_dump(exclude_unset=True)
    if not updates:
        return {"status": "no_changes"}

    success = db.update_masklet(masklet_id, updates)
    if not success:
        raise HTTPException(
            status_code=404, detail="Masklet not found or update failed"
        )

    return {"status": "updated", "masklet_id": masklet_id}


@router.get("/masklets")
async def get_masklets(
    video_path: str,
    start_time: float | None = None,
    end_time: float | None = None,
):
    """Retrieve masklets for a video and time range.

    Args:
        video_path: Path of the video to query.
        start_time: Optional start time filter.
        end_time: Optional end time filter.

    Returns:
        List of found masklets/segments.
    """
    try:
        return db.get_masklets(video_path, start_time, end_time)
    except Exception as e:
        log.error(f"Failed to get masklets: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
