"""Pipeline for Async Video Manipulation (Inpainting/Redaction).

Manages a job queue for potentially long-running video processing tasks.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional
import numpy as np

from core.manipulation.painters import PrivacyBlur

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ManipulationJob:
    job_id: str
    job_type: str  # "inpaint" or "redact"
    video_path: Path
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    result_path: Optional[Path] = None
    error: Optional[str] = None
    created_at: float = 0.0


class ManipulationPipeline:
    """Manages video manipulation jobs."""

    def __init__(self):
        self._jobs: Dict[str, ManipulationJob] = {}
        self._queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._privacy_blur = PrivacyBlur(blur_type="gaussian", kernel_size=51)

    def start(self):
        """Start the background worker."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())
            logger.info("ManipulationPipeline worker started")

    async def stop(self):
        """Stop the background worker."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    def get_job(self, job_id: str) -> Optional[ManipulationJob]:
        return self._jobs.get(job_id)

    def submit_inpaint(
        self, video_path: Path, mask_frames: Dict[int, "np.ndarray"]
    ) -> str:
        """Submit an inpainting job."""
        import time

        job_id = str(uuid.uuid4())
        job = ManipulationJob(
            job_id=job_id,
            job_type="inpaint",
            video_path=video_path,
            created_at=time.time(),
        )
        self._jobs[job_id] = job
        
        # We need to pass the mask data separately as it's not pickle-safe or just large
        # Ideally, we should persist it, but for now we keep it in memory queue
        self._queue.put_nowait((job_id, mask_frames))
        return job_id

    def submit_redact(
        self, video_path: Path, mask_frames: Dict[int, "np.ndarray"]
    ) -> str:
        """Submit a redaction (blur) job."""
        import time

        job_id = str(uuid.uuid4())
        job = ManipulationJob(
            job_id=job_id,
            job_type="redact",
            video_path=video_path,
            created_at=time.time(),
        )
        self._jobs[job_id] = job
        self._queue.put_nowait((job_id, mask_frames))
        return job_id

    async def _worker(self):
        """Process jobs from the queue."""
        while True:
            try:
                job_id, mask_data = await self._queue.get()
                job = self._jobs.get(job_id)
                if not job:
                    self._queue.task_done()
                    continue

                job.status = JobStatus.RUNNING
                logger.info(f"Starting manipulation job {job_id} ({job.job_type})")

                try:
                    if job.job_type == "inpaint":
                        await self._process_inpaint(job, mask_data)
                    elif job.job_type == "redact":
                        await self._process_redact(job, mask_data)
                    
                    job.status = JobStatus.COMPLETED
                    job.progress = 100.0
                    logger.info(f"Job {job_id} completed successfully")

                except Exception as e:
                    logger.error(f"Job {job_id} failed: {e}", exc_info=True)
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                finally:
                    self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker crashed: {e}")
                await asyncio.sleep(1)

    async def _process_inpaint(self, job: ManipulationJob, mask_frames: Dict[int, "np.ndarray"]):
        """Run inpainting logic (CPU blocking, run in executor)."""
        import asyncio
        from core.manipulation.inpainting import get_inpainter, InpaintRequest

        loop = asyncio.get_running_loop()
        
        # Offload to thread to not block FastAPI
        def _run_sync():
            inpainter = get_inpainter()
            req = InpaintRequest(
                video_path=job.video_path,
                mask_frames=mask_frames
            )
            return inpainter.inpaint_video(req)

        result = await loop.run_in_executor(None, _run_sync)
        
        if result.success and result.output_path:
            job.result_path = result.output_path
        else:
            raise RuntimeError(result.error or "Inpainting returned failed status")

    async def _process_redact(self, job: ManipulationJob, mask_frames: Dict[int, "np.ndarray"]):
        """Run redaction logic."""
        import cv2
        
        video_path = str(job.video_path)
        output_path = job.video_path.parent / f"{job.video_path.stem}_redacted{job.video_path.suffix}"
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                mask = mask_frames.get(frame_idx)
                if mask is not None and mask.any():
                    frame = self._privacy_blur.apply(frame, mask)
                
                writer.write(frame)
                frame_idx += 1
                
                # Update progress occasionally
                if frame_idx % 30 == 0 and total_frames > 0:
                    job.progress = (frame_idx / total_frames) * 100
                    
        finally:
            cap.release()
            writer.release()
            
        job.result_path = output_path


# Global singleton
_pipeline: Optional[ManipulationPipeline] = None

def get_manipulation_pipeline() -> ManipulationPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ManipulationPipeline()
        _pipeline.start()
    return _pipeline
