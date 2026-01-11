"""FastAPI server configuration and routes with full backend functionality exposed."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import time
from pathlib import Path
from uuid import uuid4
import sys

# Windows-specific asyncio fix for "WinError 10054" noise
if sys.platform == "win32":
    import asyncio
    import logging
    
    # Custom filter to suppress ConnectionResetError from logging
    class ConnectionResetFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = str(record.getMessage())
            if "ConnectionResetError" in msg or "WinError 10054" in msg:
                return False
            if record.exc_info:
                exc_type = record.exc_info[0]
                if exc_type and issubclass(exc_type, ConnectionResetError):
                    return False
            return True
    
    # Apply filter to root logger and common loggers
    for logger_name in ['', 'uvicorn', 'uvicorn.error', 'asyncio']:
        logging.getLogger(logger_name).addFilter(ConnectionResetFilter())
    
    class SilenceEventLoopPolicy(asyncio.WindowsProactorEventLoopPolicy):
        def _loop_factory(self) -> asyncio.AbstractEventLoop:
            loop = super()._loop_factory()
            
            def exception_handler(loop, context):
                # Suppress connection reset errors typical in Windows
                exc = context.get("exception")
                if exc:
                    if isinstance(exc, ConnectionResetError):
                        return
                    exc_str = str(exc)
                    if "ConnectionResetError" in exc_str or "WinError 10054" in exc_str:
                        return
                # Default handler for everything else
                loop.default_exception_handler(context)

            loop.set_exception_handler(exception_handler)
            return loop

    asyncio.set_event_loop_policy(SilenceEventLoopPolicy())

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import numpy as np
from qdrant_client.http import models

from config import settings
from core.ingestion.pipeline import IngestionPipeline
from core.ingestion.jobs import job_manager, JobStatus
from core.ingestion.scanner import LibraryScanner
from core.utils.logger import bind_context, clear_context, logger
from core.utils.observability import end_trace, init_langfuse, start_trace
from core.utils.progress import progress_tracker

pipeline: IngestionPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    init_langfuse()
    logger.info("startup")

    thumb_dir = settings.cache_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    global pipeline
    try:
        pipeline = IngestionPipeline()
        logger.info("Pipeline initialized")
        
        # Crash Recovery: Auto-detect stuck jobs from previous crashes
        # Uses heartbeat timeout (60s) to detect jobs that died mid-processing
        recovery_stats = job_manager.recover_on_startup(timeout_seconds=60.0)
        if recovery_stats["paused"] > 0:
            logger.warning(f"Crash recovery: Marked {recovery_stats['paused']} interrupted jobs as PAUSED (resumable)")
                
    except Exception as exc:
        pipeline = None
        logger.error(f"Pipeline init failed: {exc}")

    yield

    if pipeline and pipeline.db:
        pipeline.db.close()
    logger.info("shutdown")

# Media file validation
ALLOWED_MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv",  # Video
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg",  # Audio
}


# Models
class IngestRequest(BaseModel):
    """Request body for ingestion."""
    path: str
    media_type_hint: str = "unknown"
    start_time: float | None = None  # Seconds (e.g., 600 = 10:00)
    end_time: float | None = None    # Seconds (e.g., 1200 = 20:00)

class ScanRequest(BaseModel):
    """Request body for folder scanning."""
    directory: str
    recursive: bool = True
    extensions: list[str] = Field(
        default=[".mp4", ".mkv", ".avi", ".mov", ".webm"]
    )

class ConfigUpdate(BaseModel):
    """Configuration update model."""
    device: str | None = None
    compute_type: str | None = None
    frame_interval: int | None = None
    frame_sample_ratio: int | None = None
    face_detection_threshold: float | None = None
    face_detection_resolution: int | None = None
    language: str | None = None
    llm_provider: str | None = None
    ollama_base_url: str | None = None
    ollama_model: str | None = None
    google_api_key: str | None = None
    hf_token: str | None = None
    enable_voice_analysis: bool | None = None
    enable_resource_monitoring: bool | None = None

class NameFaceRequest(BaseModel):
    """Request body for naming a face cluster."""
    name: str

class AdvancedSearchRequest(BaseModel):
    query: str
    use_rerank: bool = False
    limit: int = 20
    min_confidence: float = 0.0

class IdentityMergeRequest(BaseModel):
    target_identity_id: str

class IdentityRenameRequest(BaseModel):
    name: str

class RedactRequest(BaseModel):
    video_path: str
    identity_id: str
    output_path: str | None = None

class FrameDescriptionRequest(BaseModel):
    description: str

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AI Media Indexer",
        version="2.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # System Configuration Endpoints
    @app.get("/config/system")
    async def get_system_config():
        """Get detected hardware profile and current settings."""
        from core.utils.hardware import get_cached_profile, get_available_vram, get_used_vram
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
            }
        }

    @app.post("/config/system")
    async def update_system_config(updates: dict):
        """Runtime override of system settings."""
        allowed_keys = {"batch_size", "max_concurrent_jobs", "lazy_unload", "high_performance_mode"}
        applied = {}
        for key, value in updates.items():
            if key in allowed_keys:
                setattr(settings, key, value)
                applied[key] = value
        
        # Refresh hardware profile
        from core.utils.hardware import refresh_profile
        new_profile = refresh_profile()
        
        return {"applied": applied, "new_profile": new_profile.to_dict()}

    # Mount static files for pre-generated thumbnails
    thumb_dir = settings.cache_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (thumb_dir / "faces").mkdir(exist_ok=True)
    (thumb_dir / "voices").mkdir(exist_ok=True)


    # Dynamic face thumbnail endpoint - generates on-demand if file doesn't exist
    @app.get("/thumbnails/faces/{filename}")
    async def get_face_thumbnail(filename: str):
        """Serve face thumbnail, generating on-demand if missing using FFmpeg fast-seek."""
        from fastapi.responses import FileResponse, Response
        import subprocess
        from subprocess import DEVNULL
        
        file_path = thumb_dir / "faces" / filename
        
        # If file exists, serve it immediately
        if file_path.exists():
            return FileResponse(file_path, media_type="image/jpeg")
        
        # Generate on-demand using FFmpeg (10-50x faster than OpenCV)
        try:
            if not pipeline or not pipeline.db:
                raise HTTPException(status_code=503, detail="Database not ready")
            
            # Look up face by thumbnail_path
            face_data = pipeline.db.get_face_by_thumbnail(f"/thumbnails/faces/{filename}")
            if not face_data:
                raise HTTPException(status_code=404, detail="Face not found")
            
            media_path = face_data.get("media_path")
            timestamp = face_data.get("timestamp", 0)
            
            if not media_path or not Path(media_path).exists():
                raise HTTPException(status_code=404, detail="Source video not found")
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # FFmpeg fast-seek: -ss BEFORE -i for fast seeking (vs slow decode-seek)
            # Use scale filter for 192px thumbnail, quality 5 for small file
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(timestamp),      # Fast seek before input
                "-i", str(media_path),
                "-frames:v", "1",
                "-vf", "scale=192:-2",       # 192px width, maintain aspect
                "-q:v", "5",                 # Quality 5 (~75% JPEG)
                str(file_path)
            ]
            
            result = subprocess.run(
                cmd, 
                stdout=DEVNULL, 
                stderr=DEVNULL, 
                timeout=10  # 10 second timeout
            )
            
            if result.returncode == 0 and file_path.exists():
                return FileResponse(file_path, media_type="image/jpeg")
            else:
                raise HTTPException(status_code=500, detail="FFmpeg extraction failed")
                
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Thumbnail generation timeout")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Face thumbnail generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Dynamic voice audio endpoint - generates on-demand if file doesn't exist
    @app.get("/thumbnails/voices/{filename}")
    async def get_voice_audio(filename: str):
        """Serve voice audio clip, generating on-demand if missing."""
        from fastapi.responses import FileResponse
        import subprocess
        
        file_path = thumb_dir / "voices" / filename
        
        # If file exists, serve it
        if file_path.exists():
            return FileResponse(file_path, media_type="audio/mpeg")
        
        # Parse filename: {hash}_{start}_{end}.mp3
        try:
            if not pipeline or not pipeline.db:
                raise HTTPException(status_code=503, detail="Database not ready")
            
            # Look up voice segment by audio_path
            segment = pipeline.db.get_voice_by_audio_path(f"/thumbnails/voices/{filename}")
            if not segment:
                raise HTTPException(status_code=404, detail="Voice segment not found")
            
            media_path = segment.get("media_path")
            start = segment.get("start", 0)
            end = segment.get("end", start + 5)
            
            if not media_path or not Path(media_path).exists():
                raise HTTPException(status_code=404, detail="Source video not found")
            
            # Extract audio segment using FFmpeg
            file_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", media_path,
                "-t", str(end - start),
                "-q:a", "2",
                "-map", "a",
                str(file_path)
            ]
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if file_path.exists():
                return FileResponse(file_path, media_type="audio/mpeg")
            else:
                raise HTTPException(status_code=500, detail="Audio extraction failed")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Voice audio generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Static fallback for other thumbnail files
    app.mount("/thumbnails", StaticFiles(directory=str(thumb_dir)), name="thumbnails")

    @app.middleware("http")
    async def observability_middleware(request: Request, call_next):
        _ = request.headers.get("x-trace-id", str(uuid4()))
        bind_context(component="api")
        start_trace(
            name=request.url.path,
            metadata={
                "method": request.method,
                "client": request.client.host if request.client else None,
            },
        )
        try:
            response = await call_next(request)
            end_trace("success")
            return response
        except Exception as exc:
            end_trace("error", str(exc))
            raise
        finally:
            clear_context()

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        stats = None
        if pipeline and pipeline.db:
            try:
                stats = pipeline.db.get_collection_stats()
            except Exception:
                pass
        return {
            "status": "ok",
            "device": settings.device,
            "pipeline": "ready" if pipeline else "unavailable",
            "qdrant": "connected" if pipeline and pipeline.db else "disconnected",
            "stats": stats,
        }

    @app.get("/events")
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

    @app.get("/media")
    async def stream_media(
        request: Request,
        path: str = Query(...),
        start: float = Query(0.0, description="Start time in seconds for segment playback"),
        end: float = Query(None, description="End time in seconds for segment playback"),
    ):
        """Stream a media file with HTTP Range support for proper seeking and audio.
        
        Supports:
        - HTTP Range requests (required for video seeking)
        - Optional start/end params for segment playback
        """
        from fastapi.responses import StreamingResponse, Response
        import os
        import stat
        
        file_path = Path(path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get file stats
        file_size = file_path.stat().st_size
        
        # Get mime type
        suffix = file_path.suffix.lower()
        mime_types = {
            '.mp4': 'video/mp4', '.mkv': 'video/x-matroska', '.webm': 'video/webm',
            '.avi': 'video/x-msvideo', '.mov': 'video/quicktime', '.m4v': 'video/x-m4v',
            '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.flac': 'audio/flac',
            '.m4a': 'audio/mp4', '.aac': 'audio/aac', '.ogg': 'audio/ogg',
        }
        media_type = mime_types.get(suffix, 'application/octet-stream')
        
        # Parse Range header
        range_header = request.headers.get("range")
        
        if range_header:
            # Parse "bytes=start-end"
            range_match = range_header.replace("bytes=", "").split("-")
            range_start = int(range_match[0]) if range_match[0] else 0
            range_end = int(range_match[1]) if range_match[1] else file_size - 1
            
            # Ensure valid range
            range_start = max(0, min(range_start, file_size - 1))
            range_end = max(range_start, min(range_end, file_size - 1))
            
            content_length = range_end - range_start + 1
            
            def iterfile():
                with open(file_path, "rb") as f:
                    f.seek(range_start)
                    remaining = content_length
                    chunk_size = 64 * 1024  # 64KB chunks
                    while remaining > 0:
                        read_size = min(chunk_size, remaining)
                        data = f.read(read_size)
                        if not data:
                            break
                        remaining -= len(data)
                        yield data
            
            headers = {
                "Content-Range": f"bytes {range_start}-{range_end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Content-Type": media_type,
            }
            
            return StreamingResponse(
                iterfile(),
                status_code=206,  # Partial Content
                headers=headers,
                media_type=media_type,
            )
        else:
            # No range request - stream entire file
            def iterfile():
                with open(file_path, "rb") as f:
                    chunk_size = 64 * 1024
                    while chunk := f.read(chunk_size):
                        yield chunk
            
            headers = {
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
            }
            
            return StreamingResponse(
                iterfile(),
                headers=headers,
                media_type=media_type,
            )

    @app.get("/media/segment")
    async def stream_segment(
        path: str = Query(...),
        start: float = Query(..., description="Start time in seconds"),
        end: float = Query(None, description="End time in seconds (default: start + 10)"),
    ):
        """Stream a specific video segment with caching.
        
        Strategy: 
        - First request: Fast H264 encode and cache to disk (~5-10s)
        - Subsequent requests: Serve from cache (instant)
        """
        from fastapi.responses import FileResponse
        import subprocess
        import hashlib
        
        file_path = Path(path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Default segment duration
        duration = (end - start) if end else 10.0
        
        # Create cache directory for segments
        cache_dir = settings.cache_dir / "segments"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate cache filename based on path + start + end
        cache_key = f"{file_path.stem}_{start:.2f}_{duration:.2f}"
        cache_hash = hashlib.md5(f"{path}_{start}_{duration}".encode()).hexdigest()[:8]
        cache_file = cache_dir / f"{cache_key}_{cache_hash}.mp4"
        
        # Serve from cache if exists
        if cache_file.exists():
            return FileResponse(
                cache_file, 
                media_type="video/mp4",
                headers={"Cache-Control": "public, max-age=86400"}
            )
        
        # Encode segment using fast H264
        # Using ultrafast preset + CRF 28 for speed over quality
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),           # Fast seek before input
            "-i", str(file_path),
            "-t", str(duration),
            "-c:v", "libx264",           # H264 encoding
            "-preset", "ultrafast",       # Fastest encoding
            "-crf", "28",                 # Good quality/speed balance
            "-c:a", "aac",               # AAC audio
            "-b:a", "128k",
            "-movflags", "+faststart",    # Enable fast start for streaming
            "-f", "mp4",
            str(cache_file)
        ]
        
        # Run synchronously (blocking but cached)
        result = subprocess.run(
            cmd, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE,
            timeout=60  # Max 60 seconds for encoding
        )
        
        if cache_file.exists():
            return FileResponse(
                cache_file,
                media_type="video/mp4", 
                headers={"Cache-Control": "public, max-age=86400"}
            )
        else:
            logger.error(f"Segment encoding failed: {result.stderr.decode()[:500]}")
            raise HTTPException(status_code=500, detail="Segment encoding failed")

    @app.get("/media/thumbnail")
    async def get_media_thumbnail(path: str = Query(...), time: float = 0.0):
        """Generate a thumbnail for a video at a specific timestamp."""
        from fastapi.responses import Response
        import cv2
        import numpy as np
        
        file_path = Path(path)
        if not file_path.exists():
             # Return a placeholder or 404
            raise HTTPException(status_code=404, detail="File not found")
            
        # Use OpenCV for fast seeking
        try:
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                raise ValueError("Could not open video")
                
            # Seek
            # Convert seconds to msec
            cap.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                raise ValueError("Could not read frame")
                
            # Resize for thumbnail (e.g. 320px width)
            h, w = frame.shape[:2]
            target_w = 320
            scale = target_w / w
            target_h = int(h * scale)
            frame = cv2.resize(frame, (target_w, target_h))
            
            # Encode as JPEG
            # [1] is quality (0-100)
            success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not success:
                raise ValueError("Encoding failed")
                
            return Response(content=buffer.tobytes(), media_type="image/jpeg")
            
        except Exception as e:
            logger.error(f"Thumbnail generation failed: {e}")
            raise HTTPException(status_code=500, detail="Could not generate thumbnail")

    @app.post("/ingest")
    async def ingest_media(
        ingest_request: IngestRequest,
        background_tasks: BackgroundTasks
    ):
        """Trigger processing of a local file in the background."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        file_path = Path(ingest_request.path)
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found on server: {ingest_request.path}"
            )
        
        # Validate media file extension
        ext = file_path.suffix.lower()
        if ext not in ALLOWED_MEDIA_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_MEDIA_EXTENSIONS))}"
            )

        # Generate Job ID upfront so we can return it
        job_id = str(uuid4())

        async def run_pipeline():
            # Start a new trace for the background task
            trace_name = f"ingest_{file_path.name}"
            start_trace(name="background_ingest", metadata={"file": str(file_path), "job_id": job_id})
            try:
                assert pipeline is not None
                await pipeline.process_video(
                    file_path,
                    ingest_request.media_type_hint,
                    start_time=ingest_request.start_time,
                    end_time=ingest_request.end_time,
                    job_id=job_id,
                )
                end_trace("success")
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                end_trace("error", str(e))

        background_tasks.add_task(run_pipeline)

        return {
            "status": "queued",
            "job_id": job_id,
            "file": str(file_path),
            "start_time": ingest_request.start_time,
            "end_time": ingest_request.end_time,
            "message": "Processing started. Use /events for live updates.",
        }

    @app.get("/jobs")
    async def list_jobs():
        """List all processing jobs with granular progress details."""
        jobs = progress_tracker.get_all()
        return {
            "jobs": [
                {
                    "job_id": j.job_id,
                    "status": j.status.value,
                    "progress": j.progress,
                    "file_path": j.file_path,
                    "media_type": j.media_type,
                    "current_stage": j.current_stage,
                    "pipeline_stage": getattr(j, 'pipeline_stage', 'init'),
                    "message": j.message,
                    "started_at": j.started_at,
                    "completed_at": j.completed_at,
                    "error": j.error,
                    # Granular stats
                    "total_frames": j.total_frames,
                    "processed_frames": j.processed_frames,
                    "current_item_index": getattr(j, 'current_item_index', 0),
                    "total_items": getattr(j, 'total_items', 0),
                    "timestamp": j.current_frame_timestamp,
                    "duration": j.total_duration,
                    "last_heartbeat": getattr(j, 'last_heartbeat', 0.0),
                }
                for j in jobs
            ]
        }

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        """Get details of a specific job."""
        job = progress_tracker.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": job.progress,
            "file_path": job.file_path,
            "media_type": job.media_type,
            "current_stage": job.current_stage,
            "pipeline_stage": getattr(job, 'pipeline_stage', 'init'),
            "message": job.message,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "error": job.error,
            # Granular stats
            "total_frames": job.total_frames,
            "processed_frames": job.processed_frames,
            "current_item_index": getattr(job, 'current_item_index', 0),
            "total_items": getattr(job, 'total_items', 0),
            "timestamp": job.current_frame_timestamp,
            "duration": job.total_duration,
            "last_heartbeat": getattr(job, 'last_heartbeat', 0.0),
            "checkpoint_data": job.checkpoint_data,
        }

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        """Cancel a running job."""
        success = progress_tracker.cancel(job_id)
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Job not found or not running",
            )
        return {"status": "cancelled", "job_id": job_id}


        
        # 3. Determine resume point
        start_time = 0.0
        if job.checkpoint_data and "timestamp" in job.checkpoint_data:
            start_time = float(job.checkpoint_data["timestamp"])
            
        logger.info(f"Resuming job {job_id} ({job.file_path}) from {start_time:.1f}s")

        # 4. Respawn pipeline
        async def run_pipeline_resume():
            # Start trace
            start_trace(name="resume_ingest", metadata={"file": job.file_path, "job_id": job_id})
            try:
                assert pipeline is not None
                await pipeline.process_video(
                    job.file_path,
                    job.media_type,
                    start_time=start_time,
                    job_id=job_id
                )
                end_trace("success")
            except Exception as e:
                logger.error(f"Pipeline resume error: {e}")
                progress_tracker.fail(job_id, error=str(e))
                end_trace("error", str(e))

        background_tasks.add_task(run_pipeline_resume)
        
        return {"status": "resumed", "job_id": job_id, "start_time": start_time}
    
    @app.delete("/library")
    async def delete_library_item(path: str = Query(...)):
        """Remove a media file from the library (DB + Thumbnails + Jobs)."""
        if not pipeline or not pipeline.db:
             raise HTTPException(status_code=503, detail="Database not ready")
        
        logger.info(f"Deleting media: {path}")
        
        # 1. Delete from Qdrant
        pipeline.db.delete_media_by_path(path)
        
        # 2. Delete job history
        all_jobs = job_manager.get_all_jobs(limit=1000)
        for job in all_jobs:
            if job.file_path == path:
                job_manager.delete_job(job.job_id)
        
        # 3. Clean up thumbnails
        import hashlib
        try:
             safe_stem = hashlib.md5(Path(path).stem.encode()).hexdigest()
             thumb_dir = settings.cache_dir / "thumbnails"
             
             # Delete generic thumbnails
             for f in (thumb_dir / "faces").glob(f"{safe_stem}_*"):
                 try: f.unlink()
                 except: pass
             for f in (thumb_dir / "voices").glob(f"{safe_stem}_*"):
                 try: f.unlink()
                 except: pass
        except Exception as e:
            logger.warning(f"Thumbnail cleanup failed: {e}")
            
        return {"status": "deleted", "path": path}

    @app.post("/search/advanced")
    async def advanced_search(req: AdvancedSearchRequest):
        from core.retrieval.engine import get_search_engine
        if not pipeline or not pipeline.db:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        engine = get_search_engine(db=pipeline.db)
        results = await engine.search(
            query=req.query,
            use_rerank=req.use_rerank,
            limit=req.limit,
        )
        
        filtered = [r for r in results if r.score >= req.min_confidence]
        
        return {
            "query": req.query,
            "total": len(filtered),
            "results": [r.model_dump() for r in filtered],
        }

    @app.get("/api/media/thumbnail")
    async def get_video_thumbnail(path: str = Query(...), time: float = Query(0.0)):
        from fastapi.responses import Response
        import subprocess
        video_path = Path(path)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")
        
        cmd = [
            "ffmpeg", "-ss", str(time),
            "-i", str(video_path),
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", "5",
            "pipe:1",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.returncode == 0 and result.stdout:
                return Response(content=result.stdout, media_type="image/jpeg")
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Failed to extract thumbnail")

    # === IDENTITY MANAGEMENT ===
    @app.get("/identities")
    async def list_identities():
        from core.storage.identity_graph import identity_graph
        identities = identity_graph.get_all_identities()
        result = []
        for ident in identities:
            tracks = identity_graph.get_face_tracks_for_identity(ident.id)
            result.append({
                "id": ident.id,
                "name": ident.name,
                "is_verified": ident.is_verified,
                "face_track_count": len(tracks),
                "created_at": ident.created_at,
            })
        return {"identities": result, "total": len(result)}

    @app.post("/identities/{identity_id}/merge")
    async def merge_identities(identity_id: str, req: IdentityMergeRequest):
        from core.storage.identity_graph import identity_graph
        try:
            identity_graph.merge_identities(identity_id, req.target_identity_id)
            return {"status": "merged", "source": identity_id, "target": req.target_identity_id}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.patch("/identities/{identity_id}")
    async def rename_identity(identity_id: str, req: IdentityRenameRequest):
        from core.storage.identity_graph import identity_graph
        identity = identity_graph.get_identity(identity_id)
        if not identity:
            raise HTTPException(status_code=404, detail="Identity not found")
        identity_graph.update_identity_name(identity_id, req.name)
        return {"status": "renamed", "id": identity_id, "name": req.name}

    @app.delete("/identities/{identity_id}")
    async def delete_identity(identity_id: str):
        from core.storage.identity_graph import identity_graph
        identity_graph.delete_identity(identity_id)
        return {"status": "deleted", "id": identity_id}

    # === JOB CONTROL (SQLite-backed) ===
    @app.post("/jobs/{job_id}/pause")
    async def pause_job(job_id: str):
        from core.ingestion.jobs import JobStatus
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job_manager.update_job(job_id, status=JobStatus.PAUSED)
        return {"status": "paused", "job_id": job_id}

    @app.post("/jobs/{job_id}/resume")
    async def resume_job(job_id: str):
        from core.ingestion.jobs import JobStatus
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job_manager.update_job(job_id, status=JobStatus.PENDING)
        return {"status": "resumed", "job_id": job_id}

    @app.delete("/jobs/{job_id}")
    async def delete_job(job_id: str, background_tasks: BackgroundTasks):
        """Delete a job and notify UI via SSE."""
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        # 1. Cancel running tasks
        progress_tracker.cancel(job_id)
        # 2. Remove from DB
        job_manager.delete_job(job_id)
        # 3. Emit SSE Event so UI updates immediately
        progress_tracker.emit_event(
            job_id,
            status=JobStatus.CANCELLED,
            progress=0,
            message="Job deleted",
            payload={"action": "delete", "job_id": job_id}
        )
        return {"status": "deleted", "job_id": job_id}

    # === PRIVACY TOOLS ===
    @app.post("/tools/redact")
    async def redact_identity(req: RedactRequest, background_tasks: BackgroundTasks):
        from core.tools.privacy import VideoRedactor
        video_path = Path(req.video_path)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")
        
        output_path = Path(req.output_path) if req.output_path else None
        
        async def run_redaction():
            redactor = VideoRedactor()
            result = await redactor.redact_identity(
                video_path=video_path,
                target_identity_id=req.identity_id,
                output_path=output_path,
            )
            logger.info(f"Redaction complete: {result}")
        
        background_tasks.add_task(asyncio.create_task, run_redaction())
        return {
            "status": "started",
            "video": req.video_path,
            "identity_id": req.identity_id,
        }

    @app.put("/frames/{frame_id}/description")
    async def update_frame_description(frame_id: str, req: FrameDescriptionRequest):
        """Manually correct a frame's description and re-index it."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        description = req.description.strip()
        if not description:
            raise HTTPException(status_code=400, detail="Description cannot be empty")
        success = pipeline.db.update_frame_description(frame_id, description)
        if not success:
            raise HTTPException(status_code=404, detail="Frame not found or update failed")
        return {"status": "updated", "id": frame_id, "new_description": description, "re_embedded": True}

    @app.post("/media/regenerate-thumbnails")
    async def regenerate_thumbnails(path: str = Query(...)):
        """Regenerate all face thumbnails and voice clips for a media file.
        
        Uses parallel batch extraction for speed (4 concurrent FFmpeg processes).
        """
        from core.utils.batch_extract import extract_frames_batch, extract_audio_clips_batch
        import hashlib
        
        file_path = Path(path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if not pipeline or not pipeline.db:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # Get prefix for output files
        prefix = hashlib.md5(file_path.stem.encode()).hexdigest()
        
        face_output_dir = settings.cache_dir / "thumbnails" / "faces"
        voice_output_dir = settings.cache_dir / "thumbnails" / "voices"
        face_output_dir.mkdir(parents=True, exist_ok=True)
        voice_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all faces and voice segments for this media from database
        faces = pipeline.db.get_faces_by_media(str(file_path))
        voices = pipeline.db.get_voice_segments(str(file_path))
        
        # Extract timestamps and segments
        face_timestamps = list(set(f.get("timestamp", 0) for f in faces if f.get("timestamp") is not None))
        voice_segments = [(v.get("start", 0), v.get("end", v.get("start", 0) + 3)) for v in voices]
        
        # Run batch extraction in thread pool
        import asyncio
        
        face_results = {}
        voice_results = {}
        
        if face_timestamps:
            face_results = await asyncio.get_event_loop().run_in_executor(
                None,
                extract_frames_batch,
                file_path, face_timestamps, face_output_dir, prefix
            )
        
        if voice_segments:
            voice_results = await asyncio.get_event_loop().run_in_executor(
                None,
                extract_audio_clips_batch,
                file_path, voice_segments, voice_output_dir, prefix
            )
        
        return {
            "status": "completed",
            "faces_generated": len(face_results),
            "voices_generated": len(voice_results),
            "total_faces": len(faces),
            "total_voices": len(voices),
        }

    @app.get("/search")
    async def search(
        q: str,
        limit: int = 20,
        search_type: str = Query(
            default="all",
            description="Type of search: all, dialogue, visual, voice",
        ),
        video_path: str | None = Query(
            default=None,
            description="Filter results to specific video path",
        ),
    ):
        """Semantic search across audio, visual, and voice with detailed stats.
        
        Returns results with:
        - Occurrence stats (frame count, time ranges)
        - Thumbnail URLs
        - Segment timestamps for playback
        - Debug info to diagnose search quality issues
        """
        if not pipeline:
            return {"error": "Pipeline not initialized", "results": [], "stats": {}}
        
        from urllib.parse import quote
        from collections import defaultdict
        
        start_time_search = time.perf_counter()
        logger.info(f"Search query: '{q}' | type: {search_type} | limit: {limit} | video_filter: {video_path}")

        results = []
        stats = {
            "query": q,
            "search_type": search_type,
            "video_filter": video_path,
            "dialogue_count": 0,
            "visual_count": 0,
            "total_frames_scanned": 0,
        }

        # Dialogue/transcript search
        if search_type in ("all", "dialogue"):
            dialogue_results = pipeline.db.search_media(q, limit=limit * 2)
            
            # Post-filter by video_path if specified
            if video_path:
                dialogue_results = [r for r in dialogue_results if r.get("video_path") == video_path]
            
            stats["dialogue_count"] = len(dialogue_results)
            
            for hit in dialogue_results[:limit]:
                hit["result_type"] = "dialogue"
                hit["thumbnail_url"] = None
                video = hit.get("video_path")
                start_time = hit.get("start", 0)
                if video:
                    safe_path = quote(str(video))
                    hit["thumbnail_url"] = f"/media/thumbnail?path={safe_path}&time={start_time}"
                    hit["playback_url"] = f"/media?path={safe_path}#t={start_time}"
            
            results.extend(dialogue_results[:limit])
            logger.info(f"  Dialogue results: {len(dialogue_results)}")

        # Visual/frame search
        if search_type in ("all", "visual"):
            # Fetch more results to allow for deduplication and filtering
            frame_results = pipeline.db.search_frames(q, limit=limit * 4)
            
            # Post-filter by video_path if specified
            if video_path:
                frame_results = [r for r in frame_results if r.get("video_path") == video_path]
            
            stats["total_frames_scanned"] = len(frame_results)
            
            # Log raw results to debug identical results issue
            if frame_results:
                logger.info(f"  Raw frame results: {len(frame_results)}")
                # Log top 3 scores and descriptions to see if they're discriminative
                for i, hit in enumerate(frame_results[:3]):
                    desc = (hit.get("action") or "")[:80]
                    logger.info(f"     Match #{i+1}: score={hit.get('score', 0):.4f} | '{desc}...'")
            
            # Group by video and deduplicate
            unique_frames = []
            seen_timestamps: dict[str, list[float]] = {}
            occurrence_tracker: dict[str, dict] = defaultdict(lambda: {"count": 0, "timestamps": []})
            
            for hit in frame_results:
                video = hit.get("video_path")
                ts = hit.get("timestamp", 0)
                score = hit.get("score", 0)
                
                # Track all occurrences for stats
                if video:
                    occurrence_tracker[video]["count"] += 1
                    occurrence_tracker[video]["timestamps"].append(ts)
                
                # Deduplication: Skip if we already have a frame from this video within 5 seconds
                # (reduced from 10s to get more diverse results)
                if video in seen_timestamps:
                    if any(abs(ts - existing) < 5.0 for existing in seen_timestamps[video]):
                        continue
                elif video is not None:
                    seen_timestamps[video] = []
                
                if video is not None:
                    seen_timestamps[video].append(ts)
                
                # Expand to 7-second context (timestamp center)
                start_context = max(0.0, ts - 3.5)
                end_context = ts + 3.5
                
                # Add rich metadata
                hit["result_type"] = "visual"
                hit["start"] = start_context
                hit["end"] = end_context
                hit["original_timestamp"] = ts
                
                # Add URLs for thumbnail and playback
                if video:
                    safe_path = quote(str(video))
                    hit["thumbnail_url"] = f"/media/thumbnail?path={safe_path}&time={ts}"
                    hit["playback_url"] = f"/media?path={safe_path}#t={start_context}"
                    # NEW: Segment URL with proper audio extraction (more reliable than #t= fragment)
                    hit["segment_url"] = f"/media/segment?path={safe_path}&start={start_context:.2f}&end={end_context:.2f}"
                
                unique_frames.append(hit)
                if len(unique_frames) >= limit:
                    break
            
            # Add occurrence stats to results
            for hit in unique_frames:
                video = hit.get("video_path")
                if video and video in occurrence_tracker:
                    hit["occurrence_count"] = occurrence_tracker[video]["count"]
                    hit["occurrence_timestamps"] = sorted(occurrence_tracker[video]["timestamps"])[:10]  # Top 10
            
            stats["visual_count"] = len(unique_frames)
            results.extend(unique_frames)
            logger.info(f"  Unique visual results: {len(unique_frames)} (from {len(frame_results)} raw)")

        # Sort by score (highest first)
        results.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
        final_results = results[:limit]
        
        stats["returned_count"] = len(final_results)
        
        # Log final result summary
        if final_results:
            scores = [float(r.get("score", 0)) for r in final_results]
            stats["score_range"] = {"min": min(scores), "max": max(scores), "avg": sum(scores)/len(scores)}
            duration = time.perf_counter() - start_time_search
            logger.info(f"  Search complete: {len(final_results)} results | Score={min(scores):.4f}-{max(scores):.4f} | Duration={duration:.3f}s")
        
        if not final_results:
             logger.warning(f"No results found for query: '{q}'. Using fallback.")
             # Fallback: return most recent indexed frames
             fallback_results = pipeline.db.get_recent_frames(limit=10)
             # Enrich fallback results like normal results
             for hit in fallback_results:
                 video = hit.get("video_path")
                 ts = hit.get("timestamp", 0)
                 if video:
                     safe_path = quote(str(video))
                     hit["thumbnail_url"] = f"/media/thumbnail?path={safe_path}&time={ts}"
                     hit["playback_url"] = f"/media?path={safe_path}#t={ max(0, ts-3) }"
             
             final_results = fallback_results
             stats["fallback"] = True
             stats["message"] = "No exact matches found. Showing recent indexed content."

        return {
            "results": final_results,
            "stats": stats,
        }

    @app.get("/search/agentic")
    async def agentic_search(
        q: str,
        limit: int = 20,
        use_expansion: bool = True,
    ):
        """FAANG-level search with LLM query expansion and identity resolution.

        This endpoint uses:
        1. LLM to parse query and extract entities (persons, actions, objects, brands)
        2. Query expansion (e.g., 'South Indian food' → 'idli, dosa, sambar')
        3. Identity resolution (person names → face cluster IDs)
        4. Filtered vector search (frames with matching face IDs)

        Args:
            q: Natural language query (e.g., 'Prakash bowling at Brunswick')
            limit: Maximum results to return
            use_expansion: Whether to use LLM for query expansion

        Returns:
            Dict with results, parsed query, resolved identity, and metadata
        """
        if not pipeline:
            return {"error": "Pipeline not initialized", "results": []}

        try:
            from core.retrieval.agentic_search import SearchAgent
            agent = SearchAgent(db=pipeline.db)
            result = await agent.search(q, limit=limit, use_expansion=use_expansion)
            return result
        except Exception as e:
            logger.error(f"Agentic search failed: {e}")
            # Fallback to regular search
            regular_results = pipeline.db.search_frames(query=q, limit=limit)
            return {
                "query": q,
                "parsed": None,
                "error": str(e),
                "fallback": True,
                "results": regular_results,
            }

    @app.get("/search/scenes")
    async def scene_search(
        q: str,
        limit: int = 20,
        use_expansion: bool = True,
        video_path: str | None = Query(None, description="Filter to specific video"),
    ):
        """Production-grade scene-level search for complex queries.

        This is the advanced search endpoint that:
        1. Uses LLM to parse complex queries (clothing, accessories, location, actions)
        2. Searches scene-level embeddings (visual, motion, dialogue vectors)
        3. Filters by identity, clothing color/type, accessories, location
        4. Returns scenes with start/end timestamps for precise playback

        Example queries:
        - "Prakash wearing blue t-shirt with spectacles bowling at Brunswick hitting a strike"
        - "Someone eating idli at a South Indian restaurant in the morning"

        Args:
            q: Natural language query (can be paragraph-length)
            limit: Maximum results
            use_expansion: Whether to use LLM for query expansion
            video_path: Optional filter to specific video

        Returns:
            Dict with scene results, timestamps, and parsed query
        """
        if not pipeline:
            return {"error": "Pipeline not initialized", "results": []}

        from urllib.parse import quote

        start_time_search = time.perf_counter()
        logger.info(f"[SceneSearch] Query: '{q[:100]}...' | video_filter: {video_path}")

        try:
            from core.retrieval.agentic_search import SearchAgent
            agent = SearchAgent(db=pipeline.db)
            result = await agent.search_scenes(
                query=q,
                limit=limit,
                use_expansion=use_expansion,
                video_path=video_path,
            )

            # Enrich results with thumbnail and playback URLs
            for hit in result.get("results", []):
                video = hit.get("media_path")
                start = hit.get("start_time", 0)
                if video:
                    safe_path = quote(str(video))
                    hit["thumbnail_url"] = f"/media/thumbnail?path={safe_path}&time={start}"
                    hit["playback_url"] = f"/media?path={safe_path}#t={start}"

            duration = time.perf_counter() - start_time_search
            result["stats"] = {
                "duration_seconds": duration,
                "search_mode": "scene",
            }
            logger.info(f"[SceneSearch] Returned {len(result.get('results', []))} scenes in {duration:.3f}s")
            return result

        except Exception as e:
            logger.error(f"[SceneSearch] Error: {e}")
            # Fallback to frame search
            return {
                "query": q,
                "error": str(e),
                "fallback": True,
                "results": pipeline.db.search_frames(query=q, limit=limit),
            }

    @app.get("/search/hybrid")
    async def hybrid_search(
        q: str,
        limit: int = 20,
        video_path: str | None = Query(None, description="Filter to specific video"),
    ):
        """100% accuracy hybrid search with HITL identity integration.
        
        This endpoint provides the highest retrieval accuracy by:
        1. Detecting HITL names in query → automatic identity filtering
        2. Vector search with BGE-M3 embeddings
        3. Keyword boost on structured fields (entities, visible_text, scene)
        4. Face names and speaker names in results
        
        Example queries:
        - "Prakash eating idly" → filters to frames with "Prakash" face
        - "bowling at Brunswick" → matches scene location and action
        - "tilak on forehead" → matches entity descriptions
        
        Args:
            q: Natural language query
            limit: Maximum results
            video_path: Optional filter to specific video
            
        Returns:
            High-precision search results with identity metadata
        """
        if not pipeline:
            return {"error": "Pipeline not initialized", "results": []}
        
        from urllib.parse import quote
        
        start_time_search = time.perf_counter()
        logger.info(f"[HybridSearch] Query: '{q}' | video_filter: {video_path}")
        
        try:
            results = pipeline.db.search_frames_hybrid(
                query=q,
                video_paths=video_path,
                limit=limit,
            )
            
            # Enrich results with URLs
            for hit in results:
                video = hit.get("video_path")
                ts = hit.get("timestamp", 0)
                if video:
                    safe_path = quote(str(video))
                    hit["thumbnail_url"] = f"/media/thumbnail?path={safe_path}&time={ts}"
                    hit["playback_url"] = f"/media?path={safe_path}#t={max(0, ts-3)}"
            
            duration = time.perf_counter() - start_time_search
            logger.info(f"[HybridSearch] Returned {len(results)} results in {duration:.3f}s")
            
            return {
                "query": q,
                "video_filter": video_path,
                "results": results,
                "stats": {
                    "total": len(results),
                    "duration_seconds": duration,
                },
            }
            
        except Exception as e:
            logger.error(f"[HybridSearch] Error: {e}")
            return {"error": str(e), "results": []}

    @app.get("/library")
    async def get_library():
        """Get list of all indexed media files."""
        if not pipeline:
            return {"error": "Pipeline not initialized", "media": []}
        media = pipeline.db.get_indexed_media()
        return {"media": media}

    @app.delete("/library/{path:path}")
    async def delete_from_library(path: str):
        """Delete a media file from the index."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        deleted = pipeline.db.delete_media(path)
        return {"deleted": deleted, "path": path}

    @app.post("/scan")
    async def scan_directory(
        scan_request: ScanRequest,
        background_tasks: BackgroundTasks
    ):
        """Scan a directory for media files and queue them for processing."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        dir_path = Path(scan_request.directory)
        if not dir_path.exists() or not dir_path.is_dir():
            raise HTTPException(
                status_code=404,
                detail=f"Directory not found: {scan_request.directory}",
            )

        scanner = LibraryScanner()
        files = list(scanner.scan(dir_path))

        async def process_all():
            # Start a new trace for the background scan
            start_trace(name="background_scan", metadata={"directory": str(dir_path), "files": len(files)})
            try:
                for media_asset in files:
                    try:
                        assert pipeline is not None
                        await pipeline.process_video(
                            media_asset.file_path,
                            media_asset.media_type.value,
                        )
                    except Exception as e:
                        logger.error(f"Error processing {media_asset.file_path}: {e}")
                end_trace("success")
            except Exception as e:
                 end_trace("error", str(e))

        background_tasks.add_task(process_all)

        return {
            "status": "scanning",
            "directory": str(dir_path),
            "files_found": len(files),
            "files": [str(f.file_path) for f in files[:10]],
            "message": "Processing started. Use /events for live updates.",
        }

    @app.get("/config")
    async def get_config():
        """Get current configuration."""
        return {
            "device": settings.device,
            "compute_type": settings.compute_type,
            "qdrant_backend": settings.qdrant_backend,
            "qdrant_host": settings.qdrant_host,
            "qdrant_port": settings.qdrant_port,
            "frame_interval": settings.frame_interval,
            "frame_sample_ratio": settings.frame_sample_ratio,
            "face_detection_threshold": settings.face_detection_threshold,
            "face_detection_resolution": settings.face_detection_resolution,
            "language": settings.language,
            "llm_provider": settings.llm_provider.value,
            "enable_voice_analysis": settings.enable_voice_analysis,
            "enable_resource_monitoring": settings.enable_resource_monitoring,
            "max_cpu_percent": settings.max_cpu_percent,
            "max_ram_percent": settings.max_ram_percent,
        }

    @app.post("/config")
    async def update_config(config_update: ConfigUpdate):
        """Update configuration."""
        if config_update.device is not None:
            from typing import Literal, cast
            settings.device_override = cast(
                Literal["cuda", "cpu", "mps"] | None,
                config_update.device
            )
        if config_update.frame_interval is not None:
            settings.frame_interval = config_update.frame_interval
        if config_update.language is not None:
            settings.language = config_update.language
        if config_update.llm_provider is not None:
            from config import LLMProvider
            try:
                settings.llm_provider = LLMProvider(config_update.llm_provider)
            except ValueError:
                pass
        if config_update.ollama_base_url is not None:
            settings.ollama_base_url = config_update.ollama_base_url
        if config_update.ollama_model is not None:
            settings.ollama_model = config_update.ollama_model
        if config_update.google_api_key is not None:
            from pydantic import SecretStr
            settings.gemini_api_key = SecretStr(config_update.google_api_key)
        if config_update.hf_token is not None:
            settings.hf_token = config_update.hf_token
        if config_update.enable_voice_analysis is not None:
            settings.enable_voice_analysis = config_update.enable_voice_analysis
        if config_update.enable_resource_monitoring is not None:
            settings.enable_resource_monitoring = \
                config_update.enable_resource_monitoring
        if config_update.frame_sample_ratio is not None:
            settings.frame_sample_ratio = config_update.frame_sample_ratio
        if config_update.face_detection_threshold is not None:
            settings.face_detection_threshold = config_update.face_detection_threshold
        if config_update.face_detection_resolution is not None:
            settings.face_detection_resolution = config_update.face_detection_resolution

        return {"status": "updated", "requires_restart": True}

    @app.get("/system/browse")
    async def browse_file(initial_dir: str = "") -> dict:
        """Open a native file dialog on the server to select a file."""
        def open_dialog():
            import tkinter as tk
            from tkinter import filedialog
            try:
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                file_path = filedialog.askopenfilename(
                    initialdir=initial_dir or None,
                    title="Select Media File to Ingest"
                )
                root.destroy()
                return file_path
            except Exception as e:
                logger.error(f"Failed to open native dialog: {e}")
                return None

        # Run in a separate thread to not block the event loop
        path = await asyncio.to_thread(open_dialog)
        return {"path": path if path else None}


    @app.get("/faces/unresolved")
    async def get_unresolved_faces(limit: int = 50):
        """Get face clusters that need naming."""
        if not pipeline:
            return {"error": "Pipeline not initialized", "faces": []}
        faces = pipeline.db.get_unresolved_faces(limit=limit)
        return {"faces": faces}

    @app.get("/faces/named")
    async def get_named_faces():
        """Get all named face clusters."""
        if not pipeline:
            return {"error": "Pipeline not initialized", "faces": []}
        faces = pipeline.db.get_named_faces()
        return {"faces": faces}

    @app.post("/faces/{cluster_id}/name")
    async def name_face_cluster(cluster_id: int, name_request: NameFaceRequest):
        """Assign a name to a face cluster and update all related embeddings.
        
        HITL naming triggers:
        1. Check if another cluster already has this name → AUTO-MERGE
        2. Update all faces in cluster with the name
        3. Update all frames containing this face with identity text
        4. Propagate name to linked speaker clusters (if mapped)
        """
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        name = name_request.name.strip()
        merged_from = None
        
        # 0. Check if another face cluster already has this name → AUTO-MERGE
        existing_cluster = pipeline.db.get_face_cluster_by_name(name)
        if existing_cluster and existing_cluster != cluster_id:
            # Merge target cluster into the existing named cluster
            logger.info(f"[AUTO-MERGE] Cluster {cluster_id} → Cluster {existing_cluster} (same name: '{name}')")
            pipeline.db.merge_face_clusters(cluster_id, existing_cluster)
            merged_from = cluster_id
            cluster_id = existing_cluster  # Continue with merged cluster
        
        # 1. Update face name in DB
        updated = pipeline.db.update_face_name(cluster_id, name)
        
        # 2. Update frames containing this face with identity text
        frames = pipeline.db.get_frames_by_face_cluster(cluster_id)
        frames_updated = 0
        for frame in frames:
            # Get all face names for this frame
            face_cluster_ids = (frame.get("payload") or {}).get("face_cluster_ids", [])
            face_names = []
            for cid in face_cluster_ids:
                fname = pipeline.db.get_face_name_by_cluster(cid)
                if fname:
                    face_names.append(fname)
            
            if face_names:
                pipeline.db.update_frame_identity_text(
                    frame["id"],
                    face_names=face_names,
                    speaker_names=[],  # Updated below if found
                )
                frames_updated += 1
            
            # 3. Propagate Name to Overlapping Speakers
            try:
                # Access payload correctly - data is nested under "payload" key
                payload = frame.get("payload") or {}
                video_path = payload.get("video_path")
                timestamp = payload.get("timestamp", 0)
                
                if video_path:
                    speaker_cluster_ids = pipeline._get_speaker_clusters_at_time(
                        video_path, 
                        timestamp
                    )
                    if len(speaker_cluster_ids) == 1:
                        spk_cid = speaker_cluster_ids[0]
                        existing_spk_name = pipeline.db.get_speaker_name_by_cluster(spk_cid)
                        if not existing_spk_name:
                            logger.info(f"Auto-mapping Face Cluster {cluster_id} ('{name}') -> Speaker Cluster {spk_cid}")
                            pipeline.db.set_speaker_name(spk_cid, name)
            except Exception as e:
                logger.warning(f"Failed to auto-map speaker: {e}")
        
        logger.info(f"[HITL] Named cluster {cluster_id} as '{name}', updated {frames_updated} frames")
        
        result = {
            "updated": updated, 
            "cluster_id": cluster_id, 
            "name": name,
            "frames_updated": frames_updated,
        }
        
        if merged_from:
            result["merged_from"] = merged_from
            result["message"] = f"Auto-merged with existing cluster named '{name}'"
        
        return result

    @app.put("/faces/{face_id}/name")
    async def name_single_face(face_id: str, name_request: NameFaceRequest):
        """Assign a name to a single face."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        success = pipeline.db.update_single_face_name(face_id, name_request.name)
        if not success:
            raise HTTPException(status_code=404, detail="Face not found")
        return {"success": True, "face_id": face_id, "name": name_request.name}

    @app.delete("/faces/{face_id}")
    async def delete_face(face_id: str):
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        success = pipeline.db.delete_face(face_id)
        if not success:
            raise HTTPException(status_code=404, detail="Face not found")
        return {"success": True, "face_id": face_id}

    @app.post("/faces/{cluster_id}/main")
    async def set_main_character(cluster_id: int, is_main: bool = True):
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        success = pipeline.db.set_face_main(cluster_id, is_main)
        return {"success": success, "cluster_id": cluster_id, "is_main": is_main}

    @app.get("/voices")
    async def get_voice_segments(
        media_path: str | None = None,
        limit: int = 100,
    ):
        """Get voice segments."""
        if not pipeline:
            return {"error": "Pipeline not initialized", "segments": []}
        segments = pipeline.db.get_voice_segments(media_path=media_path, limit=limit)
        return {"segments": segments}

    @app.post("/voices/{cluster_id}/name")
    async def name_voice_cluster(cluster_id: int, name_request: NameFaceRequest):
        """Assign a name to a voice cluster and auto-label overlapping faces."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # 0. Get old name (for removal from frame embeddings)
        old_name = pipeline.db.get_speaker_name_by_cluster(cluster_id)

        # 1. Update speaker name in DB
        updated_count = pipeline.db.set_speaker_name(cluster_id, name_request.name)
        
        # 2. Propagate Name to Overlapping Faces
        # Find frames where this speaker is active
        # Since we don't have a direct "get_frames_by_speaker" method, we can query segments
        # and then find frames in those time ranges.
        # Implementation Detail: This is expensive if we do it for ALL segments.
        # Optimization: Just do it for a sample or if we have an index.
        # Provide a Best-Effort propagation:
        try:
            segments = pipeline.db.client.scroll(
                collection_name=pipeline.db.VOICE_COLLECTION,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="cluster_id", match=models.MatchValue(value=cluster_id))]
                ),
                limit=50, # Limit to 50 segments to avoid timeout
                with_payload=True
            )[0]
            
            propagated_count = 0
            for point in segments:
                payload = point.payload or {}
                video_path = payload.get("video_path")
                start = payload.get("start_time")
                end = payload.get("end_time")
                
                if not video_path or start is None or end is None:
                    continue
                
                # Find faces in this time range
                # We can sample the midpoint
                midpoint = (start + end) / 2
                
                # Get frames/faces around this midpoint?
                # Using existing db method if possible or standard logic
                # For now, let's use the pipeline helper we used before
                # But pipeline.get_faces_at_time isn't exposed nicely.
                # Let's search frames in this video near this time.
                
                # BETTER APPROACH:
                # Use the Face Collection directly. Find faces in this video between start/end.
                # But Qdrant filtering by range is what we need.
                
                face_points = pipeline.db.client.scroll(
                    collection_name=pipeline.db.FACES_COLLECTION,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(key="media_path", match=models.MatchValue(value=video_path)),
                            models.FieldCondition(key="timestamp", range=models.Range(gte=start, lte=end))
                        ]
                    ),
                    limit=10,
                    with_payload=True
                )[0]
                
                for face_pt in face_points:
                    face_payload = face_pt.payload or {}
                    face_cluster_id = face_payload.get("cluster_id")
                    face_name = face_payload.get("name")
                    
                    if face_cluster_id and not face_name:
                         # Found an unnamed face during this named speech segment
                         # Auto-label it!
                         pipeline.db.set_face_name(face_cluster_id, name_request.name)
                         propagated_count += 1
                         logger.info(f"Auto-mapping Speaker Cluster {cluster_id} ('{name_request.name}') -> Face Cluster {face_cluster_id}")
            
            logger.info(f"Propagated speaker name to {propagated_count} overlapping face clusters")

        except Exception as e:
            logger.warning(f"Failed to auto-map face: {e}")

        # 3. TRIGGER RE-EMBEDDING for accuracy!
        # Voice naming changes "Speaking: Name" parts of Identity Text
        # We must find frames where this speaker is active and update them.
        try:
             # Fully implemented re-embedding logic
             re_embedded_count = pipeline.db.re_embed_voice_cluster_frames(
                 cluster_id=cluster_id,
                 new_name=name_request.name,
                 old_name=old_name
             )
             logger.info(f"Re-embedded {re_embedded_count} frames for voice cluster {cluster_id}")
        except Exception as e:
             logger.error(f"Failed to re-embed frames for voice cluster {cluster_id}: {e}")
             re_embedded_count = 0

        return {"success": True, "cluster_id": cluster_id, "name": name_request.name, "updated": updated_count}

    @app.delete("/voices/{segment_id}")
    async def delete_voice_segment(segment_id: str):
        """Delete a voice segment and its audio file."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        success = pipeline.db.delete_voice_segment(segment_id)
        if not success:
            raise HTTPException(status_code=404, detail="Segment not found")
        return {"success": True, "segment_id": segment_id}

    @app.get("/stats")
    async def get_stats():
        """Get database statistics."""
        if not pipeline:
            return {"error": "Pipeline not initialized"}
        stats = pipeline.db.get_collection_stats()
        jobs = progress_tracker.get_all()
        active_jobs = len([j for j in jobs if j.status == JobStatus.RUNNING])
        completed_jobs = len([j for j in jobs if j.status == JobStatus.COMPLETED])
        failed_jobs = len([j for j in jobs if j.status == JobStatus.FAILED])
        
        # Add identity graph stats
        from core.storage.identity_graph import identity_graph
        try:
             identity_stats = identity_graph.get_stats()
        except Exception:
             identity_stats = {"identities": 0, "tracks": 0}

        return {
            "collections": stats,
            "jobs": {
                "active": active_jobs,
                "completed": completed_jobs,
                "failed": failed_jobs,
                "total": len(jobs),
            },
            "identity_graph": identity_stats,
        }

    # Face Clustering Endpoints
    @app.post("/faces/cluster")
    async def trigger_face_clustering(
        threshold: float = Query(None, description="Cross-video cosine distance threshold (default 0.35 = 65% similarity)"),
        intra_video_threshold: float = Query(0.50, description="Same-video threshold (0.50 = 50% similarity, balanced for pose)"),
        algorithm: str = Query("chinese_whispers", description="Algorithm: chinese_whispers, connected_components, or agglomerative"),
        min_bbox_size: int = Query(None, description="Min face size in pixels (default from config)"),
        min_det_score: float = Query(None, description="Min detection confidence (default from config)"),
        temporal_boost: bool = Query(True, description="Apply temporal boost for same-video faces"),
    ):
        """Run production-grade face clustering with BALANCED intra-video grouping.
        
        Key features:
        - **Balanced same-video clustering**: Uses 0.50 threshold (50% similarity) for faces
          in the same video - handles moderate pose variations without over-merging
        - **Moderate temporal boosting**: +0.12 for faces <5s, +0.08 for <30s, +0.05 for <2min
          (Max effective threshold: 0.62 for close faces)
        - **Quality filtering**: Skip small/low-confidence faces
        - **Transitive closure**: If A~B and B~C, then A,B,C all grouped together
        """
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        import numpy as np
        from sklearn.metrics.pairwise import cosine_distances
        
        # Use config defaults if not specified
        if threshold is None:
            threshold = settings.face_clustering_threshold
        if min_bbox_size is None:
            min_bbox_size = settings.face_min_bbox_size
        if min_det_score is None:
            min_det_score = settings.face_min_det_score
        
        all_faces = pipeline.db.get_all_face_embeddings()
        if not all_faces:
            return {"status": "no_faces", "clusters": 0}
        
        # Quality filtering - skip low-quality faces
        quality_faces = []
        filtered_count = 0
        for f in all_faces:
            bbox_ok = f.get("bbox_size") is None or f.get("bbox_size", 100) >= min_bbox_size
            det_ok = f.get("det_score") is None or f.get("det_score", 1.0) >= min_det_score
            if bbox_ok and det_ok:
                quality_faces.append(f)
            else:
                filtered_count += 1
        
        if len(quality_faces) < 2:
            return {
                "status": "insufficient_quality_faces", 
                "clusters": 0, 
                "message": f"Only {len(quality_faces)} quality faces (filtered {filtered_count} low-quality)",
                "total_faces": len(all_faces),
            }
        
        embeddings = np.array([f["embedding"] for f in quality_faces])
        
        # Check for zero-padding (SFace 128-dim padded to 512-dim)
        is_sface = False
        if embeddings.shape[1] == 512:
            tail_energy = np.sum(np.abs(embeddings[:, 128:]))
            if tail_energy < 1e-3:
                is_sface = True
                embeddings = embeddings[:, :128]
                logger.warning("Using SFace (128d) fallback. Install InsightFace for better quality!")
                # SFace needs STRICTER threshold (less discriminative embeddings)
                threshold = min(threshold, 0.28)
        
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_norm = embeddings / norms
        
        # Compute pairwise COSINE DISTANCE matrix
        dist_matrix = cosine_distances(embeddings_norm)
        
        start_time_cluster = time.perf_counter()
        logger.info(f"Face Clustering: {len(quality_faces)} faces (filtered {filtered_count}) | "
                   f"Model={'SFace' if is_sface else 'InsightFace'} | "
                   f"Algo={algorithm} | Threshold={threshold:.2f}")
        
        if algorithm == "chinese_whispers":
            labels = _chinese_whispers_cluster(
                dist_matrix, 
                threshold, 
                quality_faces if temporal_boost else None,
                intra_video_threshold,
            )
        elif algorithm == "connected_components":
            # Guaranteed transitive closure via graph connected components
            try:
                import networkx as nx
            except ImportError:
                logger.warning("NetworkX not installed, falling back to chinese_whispers")
                labels = _chinese_whispers_cluster(dist_matrix, threshold, quality_faces if temporal_boost else None, intra_video_threshold)
            else:
                G = nx.Graph()
                G.add_nodes_from(range(len(quality_faces)))
                for i in range(len(quality_faces)):
                    for j in range(i + 1, len(quality_faces)):
                        same_video = quality_faces[i].get("media_path") == quality_faces[j].get("media_path")
                        
                        if same_video and temporal_boost:
                            # AGGRESSIVE same-video clustering
                            # Use higher threshold since same-video = high prior of same person
                            effective_threshold = intra_video_threshold
                            
                            # Additional temporal boost based on time proximity
                            ts_i = quality_faces[i].get("timestamp", 0) or 0
                            ts_j = quality_faces[j].get("timestamp", 0) or 0
                            time_diff = abs(ts_i - ts_j)
                            
                            if time_diff < 5.0:       # Within 5 seconds
                                effective_threshold += 0.12
                            elif time_diff < 30.0:    # Within 30 seconds
                                effective_threshold += 0.08
                            elif time_diff < 120.0:   # Within 2 minutes
                                effective_threshold += 0.05
                        else:
                            # Cross-video: use stricter threshold
                            effective_threshold = threshold
                        
                        if dist_matrix[i, j] < effective_threshold:
                            G.add_edge(i, j)
                
                components = list(nx.connected_components(G))
                labels = [0] * len(quality_faces)
                for cluster_id, component in enumerate(components):
                    for node in component:
                        labels[node] = cluster_id
        else:
            # Fallback to Agglomerative
            from sklearn.cluster import AgglomerativeClustering
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage="single",  # Single linkage = "friends of friends" - most permissive
                distance_threshold=threshold,
            )
            labels = clusterer.fit_predict(dist_matrix)
        
        # Update each face with its cluster ID
        updated = 0
        for i, face in enumerate(quality_faces):
            cluster_id = int(labels[i])
            pipeline.db.update_face_cluster_id(face["id"], cluster_id)
            updated += 1
        
        n_clusters = len(set(labels))
        
        # Log cluster size distribution
        from collections import Counter
        cluster_sizes = Counter(labels)
        size_dist = dict(sorted(Counter(cluster_sizes.values()).items()))
        
        # Find largest clusters
        largest = cluster_sizes.most_common(5)
        
        duration = time.perf_counter() - start_time_cluster
        logger.info(f"Face Clustering complete: {n_clusters} clusters | "
                   f"Largest: {largest[0][1] if largest else 0} | "
                   f"Duration={duration:.3f}s")
        
        return {
            "status": "clustered",
            "total_faces": len(all_faces),
            "quality_faces": len(quality_faces),
            "filtered_faces": filtered_count,
            "clusters": n_clusters,
            "updated": updated,
            "algorithm": algorithm,
            "model_type": "SFace (128d)" if is_sface else "InsightFace (512d)",
            "cross_video_threshold": threshold,
            "cross_video_similarity": f"{(1-threshold)*100:.0f}%",
            "intra_video_threshold": intra_video_threshold,
            "intra_video_similarity": f"{(1-intra_video_threshold)*100:.0f}%",
            "temporal_boost": temporal_boost,
            "cluster_size_distribution": size_dist,
            "largest_clusters": largest,
        }

    def _chinese_whispers_cluster(
        dist_matrix: "np.ndarray", 
        threshold: float,
        faces: list[dict] | None = None,
        intra_video_threshold: float = 0.55,
    ) -> list[int]:
        """Chinese Whispers clustering algorithm with aggressive intra-video clustering.
        
        This is the same algorithm used by dlib/face_recognition library.
        It's a graph-based clustering that naturally handles:
        - Varying cluster sizes
        - Transitive relationships (A~B, B~C → A,B,C same cluster)
        - No need to specify number of clusters
        
        If faces metadata is provided, applies aggressive same-video clustering:
        - Uses intra_video_threshold (0.55) for any same-video faces
        - Additional boost: +0.25 for <5s, +0.15 for <30s, +0.10 for <2min
        
        Algorithm:
        1. Build similarity graph where edge exists if distance < threshold
        2. Each node starts with unique label
        3. Iterate: each node adopts the most common label among its neighbors
        4. Repeat until convergence
        """
        import numpy as np
        import random
        
        n = len(dist_matrix)
        
        # Build adjacency list from distance matrix
        # Edge exists if cosine distance < threshold (i.e., similar enough)
        adjacency = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                same_video = False
                if faces is not None:
                    same_video = faces[i].get("media_path") == faces[j].get("media_path")
                
                if same_video and faces is not None:
                    # AGGRESSIVE same-video clustering
                    effective_threshold = intra_video_threshold
                    
                    ts_i = faces[i].get("timestamp", 0) or 0
                    ts_j = faces[j].get("timestamp", 0) or 0
                    time_diff = abs(ts_i - ts_j)
                    
                    if time_diff < 5.0:       # Within 5 seconds
                        effective_threshold += 0.12
                    elif time_diff < 30.0:    # Within 30 seconds
                        effective_threshold += 0.08
                    elif time_diff < 120.0:   # Within 2 minutes
                        effective_threshold += 0.05
                else:
                    effective_threshold = threshold
                
                if dist_matrix[i, j] < effective_threshold:
                    adjacency[i].append(j)
                    adjacency[j].append(i)
        
        # Log connectivity
        edges = sum(len(adj) for adj in adjacency) // 2
        logger.info(f"  Chinese Whispers: {n} nodes, {edges} edges (threshold={threshold:.2f}, intra_video={intra_video_threshold:.2f})")
        
        # Initialize each node with unique label
        labels = list(range(n))
        
        # Iterate until convergence (or max iterations)
        max_iterations = 100
        for iteration in range(max_iterations):
            changed = False
            # Random order to avoid bias
            order = list(range(n))
            random.shuffle(order)
            
            for node in order:
                if not adjacency[node]:
                    continue
                
                # Count neighbor labels
                neighbor_labels = [labels[neighbor] for neighbor in adjacency[node]]
                from collections import Counter
                label_counts = Counter(neighbor_labels)
                
                # Adopt most common neighbor label
                most_common_label = label_counts.most_common(1)[0][0]
                if labels[node] != most_common_label:
                    labels[node] = most_common_label
                    changed = True
            
            if not changed:
                logger.info(f"  Converged after {iteration + 1} iterations")
                break
        
        # Renumber labels to be consecutive 0, 1, 2, ...
        unique_labels = sorted(set(labels))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = [label_map[l] for l in labels]
        
        return labels

    @app.get("/faces/clusters")
    async def get_face_clusters():
        """Get all faces grouped by cluster."""
        if not pipeline:
            return {"error": "Pipeline not initialized", "clusters": {}}
        clusters = pipeline.db.get_faces_grouped_by_cluster()
        
        # Transform to API-friendly format
        result = []
        for cluster_id, faces in clusters.items():
            # Pick the first face as representative
            representative = faces[0] if faces else None
            
            # Check if any face in the cluster is marked as main
            is_main = any(f.get("is_main", False) for f in faces)
            
            result.append({
                "cluster_id": cluster_id,
                "name": faces[0].get("name") if faces else None,
                "face_count": len(faces),
                "representative": representative,
                "faces": faces,
                "is_main": is_main,
            })
        
        # Sort by is_main (desc) -> face_count (desc)
        result.sort(key=lambda x: (x["is_main"], x["face_count"]), reverse=True)
        return {"clusters": result}

    @app.get("/identity/suggestions")
    async def get_identity_suggestions():
        """Get AI-powered identity linking suggestions (Face-Voice, TMDB, Merge)."""
        if not pipeline:
            return {"suggestions": [], "error": "Pipeline not initialized"}
        
        from core.processing.identity_linker import get_identity_linker
        
        try:
            face_clusters_raw = pipeline.db.get_faces_grouped_by_cluster()
            face_clusters = []
            for cluster_id, faces in face_clusters_raw.items():
                face_clusters.append({
                    "cluster_id": cluster_id,
                    "name": faces[0].get("name") if faces else None,
                    "timestamps": [f.get("timestamp", 0) for f in faces],
                })
            
            voice_clusters = []
            try:
                voices = pipeline.db.get_all_voice_segments()
                cluster_map: dict[int, list[dict]] = {}
                for v in voices:
                    cid = v.get("speaker_cluster_id", -1)
                    if cid not in cluster_map:
                        cluster_map[cid] = []
                    cluster_map[cid].append({"start": v.get("start", 0), "end": v.get("end", 0)})
                for cid, segments in cluster_map.items():
                    voice_clusters.append({
                        "cluster_id": cid,
                        "name": segments[0].get("speaker_name") if segments else None,
                        "timestamps": segments,
                    })
            except Exception:
                pass
            
            linker = get_identity_linker()
            suggestions = linker.get_all_suggestions(face_clusters, voice_clusters)
            
            return {"suggestions": suggestions}
        except Exception as e:
            return {"suggestions": [], "error": str(e)}

    @app.post("/faces/merge")
    async def merge_face_clusters(from_cluster: int, to_cluster: int):
        """Merge two face clusters into one."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        updated = pipeline.db.merge_face_clusters(from_cluster, to_cluster)
        return {"merged": updated, "from": from_cluster, "to": to_cluster}

    # Voice Clustering and HITL Endpoints
    @app.post("/voices/cluster")
    async def trigger_voice_clustering(
        threshold: float = Query(0.5, description="Cosine distance threshold (0.5 = 50% similarity)"),
        algorithm: str = Query("chinese_whispers", description="Algorithm: chinese_whispers (best), hdbscan, agglomerative"),
    ):
        """Run production-grade voice clustering.
        
        Algorithms:
        - chinese_whispers: Graph-based, handles transitive relationships (RECOMMENDED)
        - hdbscan: Density-based, good for varying cluster densities
        - agglomerative: Hierarchical, good fallback for small datasets
        
        Chinese Whispers is the industry standard (used by dlib, face_recognition).
        Default threshold 0.5 = 50% cosine similarity needed to consider same speaker.
        """
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        import numpy as np
        from sklearn.metrics.pairwise import cosine_distances
        
        voices = pipeline.db.get_all_voice_embeddings()
        if not voices:
            return {"status": "no_voices", "clusters": 0}
        
        if len(voices) < 2:
            return {"status": "insufficient_voices", "clusters": 1, "message": "Need at least 2 voice segments"}
        
        embeddings = np.array([v["embedding"] for v in voices])
        
        # Step 1: L2 normalize embeddings (should already be normalized, but ensure)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_norm = embeddings / norms
        
        # Step 2: Compute pairwise COSINE DISTANCE matrix
        dist_matrix = cosine_distances(embeddings_norm)
        
        start_time_voice = time.perf_counter()
        logger.info(f"Voice Clustering: {len(voices)} segments | Algo={algorithm} | Threshold={threshold:.2f}")
        
        if algorithm == "chinese_whispers":
            # Use same proven algorithm as face clustering
            labels = _chinese_whispers_cluster(dist_matrix, threshold)
        elif algorithm == "hdbscan":
            import hdbscan
            # Use precomputed distance matrix with HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=2,
                min_samples=1,
                metric="precomputed",
                cluster_selection_epsilon=threshold * 0.5,  # Tighter for speaker ID
                cluster_selection_method="leaf",  # More fine-grained clusters
            )
            labels = clusterer.fit_predict(dist_matrix)
        else:
            # Fallback: Agglomerative with single linkage (friends-of-friends)
            from sklearn.cluster import AgglomerativeClustering
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage="average",  # Average linkage for voice (more stable)
                distance_threshold=threshold,
            )
            labels = clusterer.fit_predict(dist_matrix)
        
        # Update each voice segment with its cluster ID
        updated = 0
        for i, voice in enumerate(voices):
            cluster_id = int(labels[i])
            # Handle HDBSCAN noise points (-1) by giving them unique IDs
            if cluster_id == -1:
                cluster_id = -abs(hash(voice["id"])) % (10**9)
            pipeline.db.update_voice_cluster_id(voice["id"], cluster_id)
            updated += 1
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1) if hasattr(labels, '__iter__') else 0
        
        # Log cluster size distribution
        from collections import Counter
        cluster_sizes = Counter(labels)
        size_dist = dict(sorted(Counter(cluster_sizes.values()).items()))
        largest = cluster_sizes.most_common(5)
        
        duration = time.perf_counter() - start_time_voice
        logger.info(f"Voice Clustering complete: {n_clusters} clusters | Largest cluster size: {largest[0][1] if largest else 0} | Duration={duration:.3f}s")
        
        return {
            "status": "clustered",
            "total_segments": len(voices),
            "clusters": n_clusters,
            "noise_points": n_noise,
            "updated": updated,
            "algorithm": algorithm,
            "distance_threshold": threshold,
            "similarity_threshold": f"{(1-threshold)*100:.0f}%",
            "cluster_size_distribution": size_dist,
            "largest_clusters": largest,
        }

    @app.get("/voices/clusters")
    async def get_voice_clusters():
        """Get all voice segments grouped by cluster."""
        if not pipeline:
            return {"error": "Pipeline not initialized", "clusters": {}}
        clusters = pipeline.db.get_voices_grouped_by_cluster()
        
        result = []
        for cluster_id, segments in clusters.items():
            representative = segments[0] if segments else None
            result.append({
                "cluster_id": cluster_id,
                "speaker_name": segments[0].get("speaker_name") if segments else None,
                "segment_count": len(segments),
                "representative": representative,
                "segments": segments,
            })
        
        result.sort(key=lambda x: x["segment_count"], reverse=True)
        return {"clusters": result}

    @app.put("/voices/{segment_id}/name")
    async def rename_voice_speaker(segment_id: str, name_request: NameFaceRequest):
        """Rename a speaker for a voice segment."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        success = pipeline.db.update_voice_speaker_name(segment_id, name_request.name)
        if not success:
            raise HTTPException(status_code=404, detail="Segment not found")
        return {"success": True, "segment_id": segment_id, "name": name_request.name}

    @app.post("/voices/merge")
    async def merge_voice_clusters(from_cluster: int, to_cluster: int):
        """Merge two voice clusters into one."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        updated = pipeline.db.merge_voice_clusters(from_cluster, to_cluster)
        return {"merged": updated, "from": from_cluster, "to": to_cluster}

    # Manual Cluster Management Endpoints
    @app.put("/faces/{face_id}/cluster")
    async def move_face_to_cluster(face_id: str, cluster_id: int):
        """Move a single face to a different cluster."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        success = pipeline.db.update_face_cluster_id(face_id, cluster_id)
        if not success:
            raise HTTPException(status_code=404, detail="Face not found")
        return {"success": True, "face_id": face_id, "cluster_id": cluster_id}

    @app.post("/faces/new-cluster")
    async def create_new_face_cluster(face_ids: list[str]):
        """Move faces to a new cluster (generates new cluster ID)."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        import random
        new_cluster_id = random.randint(100000, 999999)
        updated = 0
        for face_id in face_ids:
            if pipeline.db.update_face_cluster_id(face_id, new_cluster_id):
                updated += 1
        return {"success": True, "new_cluster_id": new_cluster_id, "faces_moved": updated}

    @app.put("/voices/{segment_id}/cluster")
    async def move_voice_to_cluster(segment_id: str, cluster_id: int):
        """Move a voice segment to a different cluster."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        success = pipeline.db.update_voice_cluster_id(segment_id, cluster_id)
        if not success:
            raise HTTPException(status_code=404, detail="Segment not found")
        return {"success": True, "segment_id": segment_id, "cluster_id": cluster_id}

    @app.post("/voices/new-cluster")
    async def create_new_voice_cluster(segment_ids: list[str]):
        """Move voice segments to a new cluster."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        import random
        new_cluster_id = random.randint(100000, 999999)
        updated = 0
        for seg_id in segment_ids:
            if pipeline.db.update_voice_cluster_id(seg_id, new_cluster_id):
                updated += 1
        return {"success": True, "new_cluster_id": new_cluster_id, "segments_moved": updated}

    # Name-Based Search
    @app.get("/search/by-name")
    async def search_by_name(name: str, limit: int = 20):
        """Search for media by face name or speaker name."""
        if not pipeline:
            return {"error": "Pipeline not initialized", "results": []}
        
        results = []
        name_lower = name.lower()
        
        # Search faces by name
        try:
            face_clusters = pipeline.db.get_faces_grouped_by_cluster()
            for cluster_id, faces in face_clusters.items():
                for face in faces:
                    face_name = face.get("name") or ""
                    if name_lower in face_name.lower():
                        results.append({
                            "type": "face",
                            "name": face_name,
                            "media_path": face.get("media_path"),
                            "timestamp": face.get("timestamp"),
                            "thumbnail_path": face.get("thumbnail_path"),
                            "cluster_id": cluster_id,
                        })
        except Exception:
            pass
        
        # Search voices by speaker name
        try:
            voice_clusters = pipeline.db.get_voices_grouped_by_cluster()
            for cluster_id, segments in voice_clusters.items():
                for seg in segments:
                    speaker_name = seg.get("speaker_name") or ""
                    if name_lower in speaker_name.lower():
                        results.append({
                            "type": "voice",
                            "name": speaker_name,
                            "media_path": seg.get("media_path"),
                            "start": seg.get("start"),
                            "end": seg.get("end"),
                            "audio_path": seg.get("audio_path"),
                            "cluster_id": cluster_id,
                        })
        except Exception:
            pass
        
        return {"results": results[:limit], "total": len(results)}

    @app.get("/debug/frames")
    async def debug_frame_descriptions(limit: int = 50):
        """Debug endpoint to inspect stored frame descriptions.
        
        Helps diagnose why different queries might return identical results
        by showing what descriptions are actually stored in the database.
        """
        if not pipeline:
            return {"error": "Pipeline not initialized"}
        
        try:
            # Scroll through all frames in the media_frames collection
            resp = pipeline.db.client.scroll(
                collection_name=pipeline.db.MEDIA_COLLECTION,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            
            frames = []
            for point in resp[0]:
                payload = point.payload or {}
                frames.append({
                    "id": str(point.id),
                    "video_path": payload.get("video_path"),
                    "timestamp": payload.get("timestamp"),
                    "description": payload.get("action"),
                    "type": payload.get("type"),
                })
            
            # Analyze description diversity
            descriptions = [f["description"] or "" for f in frames]
            unique_descriptions = set(descriptions)
            
            # Check for identical descriptions
            from collections import Counter
            desc_counts = Counter(descriptions)
            duplicates = {k: v for k, v in desc_counts.items() if v > 1}
            
            return {
                "total_frames": len(frames),
                "unique_descriptions": len(unique_descriptions),
                "duplicate_descriptions": duplicates,
                "diversity_ratio": len(unique_descriptions) / max(1, len(frames)),
                "sample_frames": frames[:20],  # First 20 samples
            }
        except Exception as e:
            return {"error": str(e)}

    thumb_path = settings.cache_dir / "thumbnails"
    thumb_path.mkdir(parents=True, exist_ok=True)
    app.mount("/thumbnails", StaticFiles(directory=str(thumb_path)), name="thumbnails")

    return app


app = create_app()
