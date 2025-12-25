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

from config import settings
from core.ingestion.pipeline import IngestionPipeline
from core.ingestion.scanner import LibraryScanner
from core.utils.logger import bind_context, clear_context, logger
from core.utils.observability import end_trace, init_langfuse, start_trace
from core.utils.progress import JobStatus, progress_tracker

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


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AI Media Indexer",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

        async def run_pipeline():
            # Start a new trace for the background task
            trace_name = f"ingest_{file_path.name}"
            start_trace(name="background_ingest", metadata={"file": str(file_path)})
            try:
                assert pipeline is not None
                await pipeline.process_video(
                    file_path,
                    ingest_request.media_type_hint,
                    start_time=ingest_request.start_time,
                    end_time=ingest_request.end_time,
                )
                end_trace("success")
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                end_trace("error", str(e))

        background_tasks.add_task(run_pipeline)

        return {
            "status": "queued",
            "file": str(file_path),
            "start_time": ingest_request.start_time,
            "end_time": ingest_request.end_time,
            "message": "Processing started. Use /events for live updates.",
        }

    @app.get("/jobs")
    async def list_jobs():
        """List all processing jobs."""
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
                    "message": j.message,
                    "started_at": j.started_at,
                    "completed_at": j.completed_at,
                    "error": j.error,
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
            "message": job.message,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "error": job.error,
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
        logger.info(f"Search query: '{q}' | type: {search_type} | limit: {limit}")

        results = []
        stats = {
            "query": q,
            "search_type": search_type,
            "dialogue_count": 0,
            "visual_count": 0,
            "total_frames_scanned": 0,
        }

        # Dialogue/transcript search
        if search_type in ("all", "dialogue"):
            dialogue_results = pipeline.db.search_media(q, limit=limit)
            stats["dialogue_count"] = len(dialogue_results)
            
            for hit in dialogue_results:
                hit["result_type"] = "dialogue"
                hit["thumbnail_url"] = None  # Could generate from video_path + start time
                video = hit.get("video_path")
                start_time = hit.get("start", 0)
                if video:
                    safe_path = quote(str(video))
                    hit["thumbnail_url"] = f"/media/thumbnail?path={safe_path}&time={start_time}"
                    hit["playback_url"] = f"/media?path={safe_path}#t={start_time}"
            
            results.extend(dialogue_results)
            logger.info(f"  Dialogue results: {len(dialogue_results)}")

        # Visual/frame search
        if search_type in ("all", "visual"):
            # Fetch more results to allow for deduplication
            frame_results = pipeline.db.search_frames(q, limit=limit * 3)
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
                else:
                    seen_timestamps[video] = []
                
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
        
        return {
            "results": final_results,
            "stats": stats,
        }

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
        """Assign a name to a face cluster."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        updated = pipeline.db.update_face_name(cluster_id, name_request.name)
        return {"updated": updated, "cluster_id": cluster_id, "name": name_request.name}

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
        return {
            "collections": stats,
            "jobs": {
                "active": active_jobs,
                "completed": completed_jobs,
                "failed": failed_jobs,
                "total": len(jobs),
            },
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
