"""FastAPI server configuration and routes with full backend functionality exposed."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

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

    # Mount static files for thumbnails
    thumb_dir = settings.cache_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
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
    async def stream_media(path: str = Query(...)):
        """Stream a media file for playback in browser."""
        from fastapi.responses import FileResponse
        file_path = Path(path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get mime type
        suffix = file_path.suffix.lower()
        mime_types = {
            '.mp4': 'video/mp4', '.mkv': 'video/x-matroska', '.webm': 'video/webm',
            '.avi': 'video/x-msvideo', '.mov': 'video/quicktime',
            '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.flac': 'audio/flac',
        }
        media_type = mime_types.get(suffix, 'application/octet-stream')
        
        return FileResponse(
            file_path,
            media_type=media_type,
            filename=file_path.name,
        )

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
            try:
                assert pipeline is not None
                await pipeline.process_video(
                    file_path,
                    ingest_request.media_type_hint,
                    start_time=ingest_request.start_time,
                    end_time=ingest_request.end_time,
                )
            except Exception as e:
                logger.error(f"Pipeline error: {e}")

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

    @app.get("/search")
    async def search(
        q: str,
        limit: int = 20,
        search_type: str = Query(
            default="all",
            description="Type of search: all, dialogue, visual, voice",
        ),
    ):
        """Semantic search across audio, visual, and voice."""
        if not pipeline:
            return {"error": "Pipeline not initialized", "results": []}

        results = []

        if search_type in ("all", "dialogue"):
            dialogue_results = pipeline.db.search_media(q, limit=limit)
            results.extend(dialogue_results)

        if search_type in ("all", "visual"):
            frame_results = pipeline.db.search_frames(q, limit=limit)
            results.extend(frame_results)

        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:limit]

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
            for media_asset in files:
                try:
                    assert pipeline is not None
                    await pipeline.process_video(
                        media_asset.file_path,
                        media_asset.media_type.value,
                    )
                except Exception as e:
                    logger.error(f"Error processing {media_asset.file_path}: {e}")

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
    async def trigger_face_clustering():
        """Run production-grade HDBSCAN clustering on face embeddings.
        
        Best practices implemented:
        - L2 normalization for cosine similarity
        - PCA dimensionality reduction (improves clustering in high-dim space)
        - HDBSCAN (handles varying cluster densities better than DBSCAN)
        """
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        import numpy as np
        from sklearn.decomposition import PCA
        import hdbscan
        
        faces = pipeline.db.get_all_face_embeddings()
        if not faces:
            return {"status": "no_faces", "clusters": 0}
        
        if len(faces) < 2:
            return {"status": "insufficient_faces", "clusters": 0, "message": "Need at least 2 faces"}
        
        embeddings = np.array([f["embedding"] for f in faces])
        
        # Check for zero-padding (SFace 128-dim padded to 512-dim)
        # If the last 300 dimensions are all close to zero for all vectors, it's likely SFace data.
        # We need to slice it to 128-dim for accurate clustering.
        is_padded = False
        if embeddings.shape[1] == 512:
            # Check if dimensions 128:512 are effectively zero
            tail_energy = np.sum(np.abs(embeddings[:, 128:]))
            if tail_energy < 1e-3:
                is_padded = True
                embeddings = embeddings[:, :128]
                logger.info("Detected zero-padded embeddings (SFace). Sliced to 128-dim.")
        
        # Step 1: L2 normalize embeddings (Critical for Cosine/Euclidean)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        # Step 2: PCA dimensionality reduction
        # Production Best Practice:
        # - For small datasets (N < 50), PCA removes too much information (reducing 512d -> N-1 dims).
        #   Better to use raw Euclidean distance on normalized vectors.
        # - For large datasets, PCA helps denoise and speed up.
        target_dims = min(128, embeddings.shape[0] - 1)
        use_pca = (not is_padded) and (embeddings.shape[0] >= 50) and (target_dims > 5)
        
        if use_pca:
            pca = PCA(n_components=target_dims)
            embeddings_reduced = pca.fit_transform(embeddings)
        else:
            embeddings_reduced = embeddings
        
        # Step 3: HDBSCAN clustering (Tuned for Google Photos-like grouping)
        # - min_cluster_size=2: Smallest valid group.
        # - cluster_selection_epsilon=0.65: VERY AGGRESSIVE MERGING.
        #   Standard ArcFace verification threshold is ~0.6-0.7 for hard cases.
        #   0.65 helps merge profile/frontal views for small datasets.
        epsilon = 0.65
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric="euclidean",
            cluster_selection_epsilon=epsilon,
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(embeddings_reduced)
        
        # Update each face with its cluster ID
        updated = 0
        for i, face in enumerate(faces):
            cluster_id = int(labels[i])
            # Noise points (label=-1) get unique negative cluster IDs to keep them separate
            if cluster_id == -1:
                cluster_id = -abs(hash(face["id"])) % (10**9)
            
            # Crucial: Update logical cluster ID in Qdrant payload so it persists
            pipeline.db.update_face_cluster_id(face["id"], cluster_id)
            updated += 1
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        return {
            "status": "clustered",
            "total_faces": len(faces),
            "clusters": n_clusters,
            "noise_points": n_noise,
            "updated": updated,
            "algorithm": "HDBSCAN",
            "model_type": "SFace (128d)" if is_padded else "InsightFace (512d)",
            "pca_components": target_dims,
        }

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
            result.append({
                "cluster_id": cluster_id,
                "name": faces[0].get("name") if faces else None,
                "face_count": len(faces),
                "representative": representative,
                "faces": faces,
            })
        
        # Sort by face count descending
        result.sort(key=lambda x: x["face_count"], reverse=True)
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
    async def trigger_voice_clustering():
        """Run production-grade HDBSCAN clustering on voice embeddings.
        
        Best practices:
        - L2 normalization
        - PCA dimensionality reduction (256 → 50)
        - HDBSCAN for varying cluster densities
        """
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        import numpy as np
        from sklearn.decomposition import PCA
        import hdbscan
        
        voices = pipeline.db.get_all_voice_embeddings()
        if not voices:
            return {"status": "no_voices", "clusters": 0}
        
        if len(voices) < 2:
            return {"status": "insufficient_voices", "clusters": 0}
        
        embeddings = np.array([v["embedding"] for v in voices])
        
        # Step 1: L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        # Step 2: PCA dimensionality reduction
        # Voice vectors are 256-dim. Reduce to ~50 to focus on speaker identity features.
        target_dims = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
        if target_dims > 5:
            pca = PCA(n_components=target_dims)
            embeddings_reduced = pca.fit_transform(embeddings)
        else:
            embeddings_reduced = embeddings
        
        # Step 3: HDBSCAN clustering
        # epsilon=0.4 allows for some variation in speaker tone/mic quality
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric="euclidean",
            cluster_selection_epsilon=0.4,
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(embeddings_reduced)
        
        # Update each voice segment with its cluster ID
        updated = 0
        for i, voice in enumerate(voices):
            cluster_id = int(labels[i])
            if cluster_id == -1:
                cluster_id = -abs(hash(voice["id"])) % (10**9)
            pipeline.db.update_voice_cluster_id(voice["id"], cluster_id)
            updated += 1
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        return {
            "status": "clustered",
            "total_segments": len(voices),
            "clusters": n_clusters,
            "noise_points": n_noise,
            "updated": updated,
            "algorithm": "HDBSCAN",
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

    thumb_path = settings.cache_dir / "thumbnails"
    thumb_path.mkdir(parents=True, exist_ok=True)
    app.mount("/thumbnails", StaticFiles(directory=str(thumb_path)), name="thumbnails")

    return app


app = create_app()
