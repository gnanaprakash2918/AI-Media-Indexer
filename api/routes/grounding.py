"""API routes for visual grounding and segment management."""

from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from api.deps import get_pipeline
from core.ingestion.pipeline import IngestionPipeline
from core.processing.grounding_pipeline import GroundingPipeline
from core.utils.logger import get_logger

router = APIRouter()
log = get_logger(__name__)

# Lazily created per-request; avoids module-level side effects.
_grounding_pipeline: GroundingPipeline | None = None


def _get_grounding_pipeline() -> GroundingPipeline:
    global _grounding_pipeline
    if _grounding_pipeline is None:
        _grounding_pipeline = GroundingPipeline()
    return _grounding_pipeline


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
    request: TriggerRequest,
    background_tasks: BackgroundTasks,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Trigger visual grounding for a video in the background.

    Args:
        request: The grounding request (video path, concepts).
        background_tasks: FastAPI background task manager.
        pipeline: Injected ingestion pipeline.

    Returns:
        Status message confirming the job is queued.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        gp = _get_grounding_pipeline()
        background_tasks.add_task(
            gp.process_video,
            request.video_path,
            request.concepts,
        )
        return {"status": "queued", "video_path": request.video_path}
    except Exception as e:
        log.error(f"Failed to trigger grounding: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.patch("/masklets/{masklet_id}")
async def update_masklet(
    masklet_id: str,
    update: MaskletUpdate,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Update a masklet's metadata (e.g. for UI corrections).

    Args:
        masklet_id: Unique identifier for the masklet.
        update: Metadata updates to apply.
        pipeline: Injected ingestion pipeline.

    Returns:
        Update status confirming changes.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    updates = update.model_dump(exclude_unset=True)
    if not updates:
        return {"status": "no_changes"}

    success = pipeline.db.update_masklet(masklet_id, updates)
    if not success:
        raise HTTPException(
            status_code=404, detail="Masklet not found or update failed"
        )

    return {"status": "updated", "masklet_id": masklet_id}


@router.get("/masklets")
async def get_masklets(
    video_path: str,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
    start_time: float | None = None,
    end_time: float | None = None,
):
    """Retrieve masklets for a video and time range.

    Args:
        video_path: Path of the video to query.
        pipeline: Injected ingestion pipeline.
        start_time: Optional start time filter.
        end_time: Optional end time filter.

    Returns:
        List of found masklets/segments.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        return pipeline.db.get_masklets(video_path, start_time, end_time)
    except Exception as e:
        log.error(f"Failed to get masklets: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
