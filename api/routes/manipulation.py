from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional
from pathlib import Path

from core.manipulation.pipeline import get_manipulation_pipeline, JobStatus

router = APIRouter(prefix="/manipulation", tags=["manipulation"])

class MaskFrame(BaseModel):
    frame_index: int
    rle: str  # Run-Length Encoding or just a placeholder for now. 
              # For MVP, we might accept bbox and generate mask backend-side.

class InpaintRequest(BaseModel):
    video_path: str
    regions: List[Dict]  # List of objects with {frame, bbox} or similar
    # For a real MVP, let's stick to simplest: BBox over a time range
    
class RegionRequest(BaseModel):
    video_path: str
    start_time: float
    end_time: float
    bbox: List[int]  # [x, y, w, h]

class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    result_path: Optional[str] = None
    error: Optional[str] = None

@router.post("/inpaint", response_model=JobResponse)
async def trigger_inpaint(req: RegionRequest):
    """Trigger an inpainting job for a specific region over time."""
    pipeline = get_manipulation_pipeline()
    path = Path(req.video_path)
    
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
        
    # GENERATE MASKS GENERATOR
    # For this MVP, we will convert the static BBox over time into per-frame masks
    # In a real system, we would use SAM tracker here.
    # For now, we assume a static bbox for the duration.
    
    import cv2
    import numpy as np
    
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(req.start_time * fps)
    end_frame = int(req.end_time * fps)
    
    masks = {}
    x, y, w, h = req.bbox
    
    # Validation
    if x < 0 or y < 0 or x + w > width or y + h > height:
         raise HTTPException(status_code=400, detail="Bounding box out of bounds")
         
    # Create mask for each frame in range
    # Note: efficiently we'd lazily generate this, but for < 10s clips this is fine memory-wise (~300 masks)
    # Mask format: uint8 0=bg, 255=fg
    # Implementation detail: The pipeline expects full frame masks
    
    for f_idx in range(start_frame, end_frame + 1):
        if f_idx >= 0:
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y:y+h, x:x+w] = 255
            masks[f_idx] = mask
            
    cap.release()
    
    if not masks:
        raise HTTPException(status_code=400, detail="No frames in selected time range")

    job_id = pipeline.submit_inpaint(path, masks)
    return _get_job_response(job_id)

@router.post("/redact", response_model=JobResponse)
async def trigger_redact(req: RegionRequest):
    """Trigger a privacy redaction (blur) job."""
    pipeline = get_manipulation_pipeline()
    path = Path(req.video_path)
    
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    import cv2
    import numpy as np
    
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(req.start_time * fps)
    end_frame = int(req.end_time * fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    masks = {}
    x, y, w, h = req.bbox
    
    for f_idx in range(start_frame, end_frame + 1):
        if f_idx >= 0:
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y:y+h, x:x+w] = 255
            masks[f_idx] = mask
            
    cap.release()
    
    job_id = pipeline.submit_redact(path, masks)
    return _get_job_response(job_id)

@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    return _get_job_response(job_id)

def _get_job_response(job_id: str) -> JobResponse:
    pipeline = get_manipulation_pipeline()
    job = pipeline.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return JobResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress=job.progress,
        result_path=str(job.result_path) if job.result_path else None,
        error=job.error
    )
