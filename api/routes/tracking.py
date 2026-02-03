from typing import List, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from core.storage.db import get_vector_db
from core.tracking.sam3_tracker import SAM3Tracker
from core.utils.logger import get_logger

router = APIRouter(prefix="/tracking", tags=["Object Tracking"])
log = get_logger(__name__)

class TagObjectRequest(BaseModel):
    video_path: str
    concept_label: Optional[str] = None
    points: Optional[List[Tuple[int, int]]] = None
    point_labels: Optional[List[int]] = None # 1 for foreground, 0 for background
    start_frame: int = 0
    start_time: float = 0.0

class TrackingResponse(BaseModel):
    task_id: str
    status: str
    message: str

tracker = SAM3Tracker()

@router.post("/tag-object", response_model=TrackingResponse)
async def tag_object(
    request: TagObjectRequest,
    background_tasks: BackgroundTasks
):
    """Tag an object in a video using SAM 3 (Zero-Shot or Interactive).
    
    Modes:
    1. Concept Tracking: Provide 'concept_label' (e.g., "red bag").
    2. Point Tracking: Provide 'points' (clicks) and 'point_labels'.
    """
    if not request.concept_label and not request.points:
        raise HTTPException(
            status_code=400,
            detail="Must provide either 'concept_label' or 'points'."
        )

    task_id = f"track_{request.video_path}_{request.start_time}"

    # Enqueue background task
    background_tasks.add_task(
        _run_tracking_task,
        request
    )

    return TrackingResponse(
        task_id=task_id,
        status="queued",
        message="Tracking task started in background."
    )

async def _run_tracking_task(req: TagObjectRequest):
    """Background worker to run SAM 3 tracking and upsert masklets."""
    log.info(f"[Tracking] Starting task for {req.video_path}...")

    db = get_vector_db()

    try:
        segments = []
        if req.concept_label:
            log.info(f"[Tracking] Mode: Concept ('{req.concept_label}')")
            segments = await tracker.track_concept(
                req.video_path,
                req.concept_label,
                start_time=req.start_time
            )
        elif req.points:
            log.info(f"[Tracking] Mode: Points ({len(req.points)} clicks)")
            segments = await tracker.track_points(
                req.video_path,
                req.points,
                req.point_labels or [1]*len(req.points),
                start_frame_idx=req.start_frame
            )

        if not segments:
            log.warning("[Tracking] No segments found.")
            return

        # Persist to DB
        log.info(f"[Tracking] Persisting {len(segments)} masklets...")
        for seg in segments:
            # Expecting seg to contain: mask, frame_idx, bbox (optional)
            mask = seg.get("mask")
            frame_idx = seg.get("frame_idx", 0)

            if mask is None:
                continue

            # Generate visual embedding (crop + SigLIP)
            try:
                visual_vector = await tracker.extract_visual_embedding(
                    req.video_path,
                    mask,
                    frame_idx
                )
            except Exception as e:
                log.warning(f"[Tracking] Embedding extraction failed for frame {frame_idx}: {e}")
                visual_vector = None

            # Get actual FPS from video metadata
            fps = 30.0
            try:
                import cv2
                cap = cv2.VideoCapture(req.video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()
            except ImportError:
                log.warning("[Tracking] OpenCV not found, defaulting to 30.0 FPS")
            except Exception as e:
                log.warning(f"[Tracking] Failed to get FPS: {e}, defaulting to 30.0")

            start_ts = frame_idx / fps
            end_ts = start_ts + (1.0 / fps) # Single frame duration

            db.insert_masklet(
                video_path=req.video_path,
                concept=req.concept_label or "tracked_object",
                start_time=start_ts,
                end_time=end_ts,
                confidence=seg.get("score", 1.0),
                payload={
                    "frame_idx": frame_idx,
                    "bbox": seg.get("bbox"),
                    "source": "sam3_hitl"
                },
                embedding=visual_vector
            )

        log.info("[Tracking] Task complete.")

    except Exception as e:
        log.error(f"[Tracking] Task failed: {e}")
