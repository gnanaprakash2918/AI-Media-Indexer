"""Face identity processing utilities with graceful fallback chain.

Model priority:
1. InsightFace ArcFace (512-dim, best accuracy, requires onnxruntime)
2. SFace (128-dim, OpenCV built-in, good fallback)
3. Basic YuNet detection only (no embeddings - last resort)
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypeVar

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image
from sklearn.cluster import HDBSCAN  # type: ignore

from config import settings
from core.schemas import DetectedFace
from core.utils.resource_arbiter import GPU_SEMAPHORE
from core.utils.observe import observe

# =========================================================================
# SYSTEM CAPABILITY DETECTION
# =========================================================================


def _detect_system_capabilities() -> dict:
    """Detect system RAM and VRAM to determine loading strategy.

    Returns:
        Dict with 'ram_gb', 'vram_gb', 'is_high_end' keys.
    """
    import gc

    capabilities = {
        "ram_gb": 8.0,  # Default conservative
        "vram_gb": 0.0,
        "is_high_end": False,
    }

    # Detect RAM
    try:
        import psutil

        capabilities["ram_gb"] = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        # Fallback: try os-specific methods
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulong

            class MEMORYSTATUS(ctypes.Structure):
                _fields_ = [
                    ("dwLength", c_ulong),
                    ("dwMemoryLoad", c_ulong),
                    ("dwTotalPhys", c_ulong),
                    ("dwAvailPhys", c_ulong),
                    ("dwTotalPageFile", c_ulong),
                    ("dwAvailPageFile", c_ulong),
                    ("dwTotalVirtual", c_ulong),
                    ("dwAvailVirtual", c_ulong),
                ]

            mem_status = MEMORYSTATUS()
            mem_status.dwLength = ctypes.sizeof(MEMORYSTATUS)
            kernel32.GlobalMemoryStatus(ctypes.byref(mem_status))
            capabilities["ram_gb"] = mem_status.dwTotalPhys / (1024**3)
        except Exception:
            pass

    # Detect VRAM
    try:
        import torch

        if torch.cuda.is_available():
            # Get total VRAM of first GPU
            capabilities["vram_gb"] = torch.cuda.get_device_properties(
                0
            ).total_memory / (1024**3)
    except ImportError:
        pass

    # High-end: 32GB+ RAM or 8GB+ VRAM
    capabilities["is_high_end"] = (
        capabilities["ram_gb"] >= 32.0 or capabilities["vram_gb"] >= 8.0
    )

    gc.collect()
    return capabilities


# Cache system capabilities
_SYSTEM_CAPS: dict | None = None


def get_system_capabilities() -> dict:
    """Get cached system capabilities."""
    global _SYSTEM_CAPS
    if _SYSTEM_CAPS is None:
        _SYSTEM_CAPS = _detect_system_capabilities()
        print(
            f"[System] Detected: RAM={_SYSTEM_CAPS['ram_gb']:.1f}GB, VRAM={_SYSTEM_CAPS['vram_gb']:.1f}GB, High-end={_SYSTEM_CAPS['is_high_end']}"
        )
    return _SYSTEM_CAPS


# =========================================================================
# FACE TRACK BUILDER - Temporal Face Grouping for Accurate Clustering
# =========================================================================


@dataclass
class ActiveFaceTrack:
    """A face track being built during video processing.

    Tracks group face detections across consecutive frames based on:
    1. Spatial continuity (IoU overlap of bounding boxes)
    2. Appearance consistency (cosine similarity of embeddings)
    """

    track_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    embeddings: list[list[float]]  # All embeddings in this track
    bboxes: list[tuple[int, int, int, int]]  # All bounding boxes
    confidences: list[float]  # Detection confidences
    best_thumbnail_frame: int = 0  # Frame with highest confidence
    best_confidence: float = 0.0


class FaceTrackBuilder:
    """Build face tracks from frame-by-frame detections.

    The key insight: Instead of clustering all faces globally (which fails
    with lighting/angle variation), we first group faces into temporal TRACKS
    within each video. A track is a sequence of the same face across frames.

    Algorithm:
    1. For each new frame, match detected faces to existing active tracks
    2. Matching uses IoU (spatial overlap) + cosine similarity (appearance)
    3. If no match, start a new track
    4. If a track has no matches for N frames, finalize it
    5. Output: List of finalized tracks with averaged embeddings

    This solves the "Prakash in dark room" vs "Prakash outside" problem
    because the track contains multiple angles/lighting conditions, and
    the averaged embedding is more robust.
    """

    # Thresholds
    IOU_THRESHOLD: float = settings.face_track_iou_threshold
    COSINE_THRESHOLD: float = settings.face_track_cosine_threshold
    MAX_FRAMES_MISSING: int = settings.face_track_max_missing_frames
    MIN_TRACK_LENGTH: int = 2  # Minimum frames to consider a valid track

    def __init__(self, frame_interval: float = 1.0) -> None:
        """Initialize the track builder.

        Args:
            frame_interval: Seconds between frames (for timestamp calculation).
        """
        self.frame_interval = frame_interval
        self.active_tracks: dict[int, ActiveFaceTrack] = {}
        self.finalized_tracks: list[ActiveFaceTrack] = []
        self._next_track_id = 1
        self._current_frame = 0
        self._frames_since_last_seen: dict[int, int] = {}  # track_id -> frames

    def process_frame(
        self,
        faces: list[DetectedFace],
        frame_index: int,
        timestamp: float,
    ) -> None:
        """Process detected faces from a single frame.

        Args:
            faces: List of detected faces with embeddings and bboxes.
            frame_index: Current frame number.
            timestamp: Current timestamp in seconds.
        """
        self._current_frame = frame_index

        # Track which active tracks got matched this frame
        matched_track_ids: set[int] = set()

        for face in faces:
            if face.embedding is None:
                continue

            # Try to match to an existing active track
            best_track_id = self._find_best_matching_track(face)

            if best_track_id is not None:
                # Extend existing track
                track = self.active_tracks[best_track_id]
                track.end_frame = frame_index
                track.end_time = timestamp
                track.embeddings.append(face.embedding)
                track.bboxes.append(face.bbox)
                track.confidences.append(face.confidence)

                # Update best thumbnail if this is higher confidence
                if face.confidence > track.best_confidence:
                    track.best_confidence = face.confidence
                    track.best_thumbnail_frame = frame_index

                matched_track_ids.add(best_track_id)
                self._frames_since_last_seen[best_track_id] = 0
            else:
                # Start new track
                new_track = ActiveFaceTrack(
                    track_id=self._next_track_id,
                    start_frame=frame_index,
                    end_frame=frame_index,
                    start_time=timestamp,
                    end_time=timestamp,
                    embeddings=[face.embedding],
                    bboxes=[face.bbox],
                    confidences=[face.confidence],
                    best_thumbnail_frame=frame_index,
                    best_confidence=face.confidence,
                )
                self.active_tracks[self._next_track_id] = new_track
                self._frames_since_last_seen[self._next_track_id] = 0
                self._next_track_id += 1

        # Update frames since last seen for unmatched tracks
        tracks_to_finalize = []
        for track_id in list(self.active_tracks.keys()):
            if track_id not in matched_track_ids:
                self._frames_since_last_seen[track_id] = (
                    self._frames_since_last_seen.get(track_id, 0) + 1
                )
                if (
                    self._frames_since_last_seen[track_id]
                    >= self.MAX_FRAMES_MISSING
                ):
                    tracks_to_finalize.append(track_id)

        # Finalize tracks that have been missing too long
        for track_id in tracks_to_finalize:
            self._finalize_track(track_id)

    def _find_best_matching_track(self, face: DetectedFace) -> int | None:
        """Find the best matching active track for a face detection.

        Uses both IoU (spatial) and cosine similarity (appearance) matching.

        Returns:
            Track ID if a good match is found, None otherwise.
        """
        if not self.active_tracks:
            return None

        best_track_id = None
        best_score = 0.0

        face_emb = np.array(face.embedding, dtype=np.float64)
        face_emb_norm = face_emb / (np.linalg.norm(face_emb) + 1e-9)

        for track_id, track in self.active_tracks.items():
            # Skip if track hasn't been seen recently (likely a different person)
            if self._frames_since_last_seen.get(track_id, 0) > 2:
                continue

            # Compute IoU with last known bbox
            last_bbox = track.bboxes[-1]
            iou = self._compute_iou(face.bbox, last_bbox)

            if iou < self.IOU_THRESHOLD:
                continue

            # Compute cosine similarity with track centroid
            track_centroid = self._compute_track_centroid(track)
            similarity = float(np.dot(face_emb_norm, track_centroid))

            if similarity < self.COSINE_THRESHOLD:
                continue

            # Combined score: weighted average of IoU and similarity
            combined_score = 0.4 * iou + 0.6 * similarity

            if combined_score > best_score:
                best_score = combined_score
                best_track_id = track_id

        return best_track_id

    def _compute_iou(
        self,
        box1: tuple[int, int, int, int],
        box2: tuple[int, int, int, int],
    ) -> float:
        """Compute Intersection over Union between two bboxes.

        Boxes are in (top, right, bottom, left) format.
        """
        top1, right1, bottom1, left1 = box1
        top2, right2, bottom2, left2 = box2

        # Intersection
        inter_left = max(left1, left2)
        inter_top = max(top1, top2)
        inter_right = min(right1, right2)
        inter_bottom = min(bottom1, bottom2)

        if inter_right <= inter_left or inter_bottom <= inter_top:
            return 0.0

        inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)

        # Union
        area1 = (right1 - left1) * (bottom1 - top1)
        area2 = (right2 - left2) * (bottom2 - top2)
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def _compute_track_centroid(self, track: ActiveFaceTrack) -> np.ndarray:
        """Compute normalized centroid embedding for a track."""
        if not track.embeddings:
            return np.zeros(512, dtype=np.float64)

        embeddings = np.array(track.embeddings, dtype=np.float64)
        centroid = np.mean(embeddings, axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        return centroid

    def _finalize_track(self, track_id: int) -> None:
        """Move a track from active to finalized."""
        if track_id not in self.active_tracks:
            return

        track = self.active_tracks.pop(track_id)
        del self._frames_since_last_seen[track_id]

        # Only keep tracks with minimum length
        if len(track.embeddings) >= self.MIN_TRACK_LENGTH:
            self.finalized_tracks.append(track)

    def finalize_all(self) -> list[ActiveFaceTrack]:
        """Finalize all remaining active tracks and return all tracks.

        Call this at the end of video processing.
        """
        for track_id in list(self.active_tracks.keys()):
            self._finalize_track(track_id)

        return self.finalized_tracks

    def get_track_embeddings(self) -> list[tuple[int, list[float], dict]]:
        """Get averaged embeddings for all finalized tracks.

        Returns:
            List of (track_id, averaged_embedding, metadata) tuples.
        """
        results = []
        for track in self.finalized_tracks:
            centroid = self._compute_track_centroid(track)
            metadata = {
                "start_frame": track.start_frame,
                "end_frame": track.end_frame,
                "start_time": track.start_time,
                "end_time": track.end_time,
                "frame_count": len(track.embeddings),
                "best_thumbnail_frame": track.best_thumbnail_frame,
                "avg_confidence": np.mean(track.confidences)
                if track.confidences
                else 0.0,
            }
            results.append((track.track_id, centroid.tolist(), metadata))

        return results


# Directories
MODELS_DIR = settings.project_root() / "models"
CACHE_DIR = settings.project_root() / ".face_cache"

# OpenCV models (fallback)
YUNET_MODEL: Final = "face_detection_yunet_2023mar.onnx"
YUNET_URL: Final = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_MODEL: Final = "face_recognition_sface_2021dec.onnx"
SFACE_URL: Final = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

# GPU_SEMAPHORE moved to core.utils.concurrency

# Model type for tracking which engine is in use
ModelType = Literal["insightface", "sface", "yunet_only"]


def _ensure_opencv_models() -> None:
    """Download OpenCV models if missing."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in ((YUNET_MODEL, YUNET_URL), (SFACE_MODEL, SFACE_URL)):
        path = MODELS_DIR / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)


def _try_import_insightface():
    """Try to import InsightFace, return None if not available."""
    try:
        from insightface.app import FaceAnalysis

        return FaceAnalysis
    except ImportError:
        return None


class FaceManager:
    """Face detection and recognition with graceful model fallback.

    Tries models in order:
    1. InsightFace ArcFace (512-dim) - best accuracy
    2. SFace (128-dim) - good fallback
    3. YuNet only (detection, no embeddings) - last resort
    """

    def __init__(
        self,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 3,
        dbscan_metric: str = "euclidean",
        use_gpu: bool = False,
        max_fps: float = 8.0,
        batch_size: int = 16,
        global_clusters: dict[int, list[float]] | None = None,
        db_client: Any = None,
    ) -> None:
        """Initialize FaceManager.

        Args:
            dbscan_eps: Eps for DBSCAN clustering.
            dbscan_min_samples: Min samples for DBSCAN.
            dbscan_metric: Metric for DBSCAN.
            use_gpu: Whether to use GPU.
            max_fps: Max processing FPS.
            batch_size: Batch size for inference.
            global_clusters: Existing global face clusters.
            db_client: Database client.
        """
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.dbscan_metric = dbscan_metric
        self.max_fps = max_fps
        self.batch_size = batch_size
        self._use_gpu = use_gpu
        self._initialized = False
        self._init_lock = asyncio.Lock()

        self.global_clusters: dict[int, list[float]] = global_clusters or {}
        self._next_cluster_id = max(self.global_clusters.keys(), default=0) + 1

        self._model_type: ModelType = "yunet_only"
        self._insightface_app = None
        self._opencv_detector = None
        self._opencv_recognizer = None
        self._embedding_dim = 128

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if self.global_clusters:
            print(
                f"[FaceManager] Loaded {len(self.global_clusters)} global clusters for cross-video matching"
            )

    def unload_gpu(self) -> None:
        """Unload models from GPU to free VRAM for other processes (like Ollama).

        Call this after face detection but before vision/LLM analysis.
        The models will be reloaded on next detect_faces() call.
        """
        import gc

        import torch

        # Clear InsightFace
        if self._insightface_app is not None:
            del self._insightface_app
            self._insightface_app = None

        # Clear OpenCV models (less important but be thorough)
        if self._opencv_detector is not None:
            del self._opencv_detector
            self._opencv_detector = None
        if self._opencv_recognizer is not None:
            del self._opencv_recognizer
            self._opencv_recognizer = None

        # Reset init flag so models reload on next use
        self._initialized = False

        # Force garbage collection and GPU memory release
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("[FaceManager] GPU models unloaded - VRAM freed for Ollama")

    @property
    def embedding_version(self) -> str:
        """Return embedding version string for cache invalidation."""
        return f"{self._model_type}_v1_{self._embedding_dim}d"

    @observe("face_model_init")
    async def _lazy_init(self) -> None:
        """Initialize face models with fallback chain."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return

            # Try InsightFace first
            if await self._try_init_insightface():
                self._model_type = "insightface"
                self._embedding_dim = 512
                self._initialized = True
                return

            # Fallback to SFace
            # SFace natively produces 128-dim embeddings, which we pad to 512-dim
            # The clustering logic in server.py will detect this padding and handle it
            if await self._try_init_sface():
                self._model_type = "sface"
                self._embedding_dim = 128
                self._initialized = True
                return

            # Last resort: YuNet detection only
            await self._init_yunet_only()
            self._model_type = "yunet_only"
            self._embedding_dim = 0
            self._initialized = True

    async def _try_init_insightface(self) -> bool:
        """Try to initialize InsightFace. Returns True on success.

        On low-resource systems, loads models sequentially with allowed_modules
        to prevent memory allocation failures.
        """
        import gc

        face_analysis_cls = _try_import_insightface()
        if face_analysis_cls is None:
            print("[FaceManager] InsightFace library not found. Falling back.")
            return False

        # Check system capabilities
        caps = get_system_capabilities()
        is_high_end = caps["is_high_end"]

        # CRITICAL: Clean VRAM before loading InsightFace to prevent OOM
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("[FaceManager] Cleaned VRAM before InsightFace init")
        except ImportError:
            pass

        try:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self._use_gpu
                else ["CPUExecutionProvider"]
            )

            if is_high_end:
                # High-end system: load all models at once (faster)
                print(
                    "[FaceManager] High-end system detected, loading all InsightFace models..."
                )
                app = face_analysis_cls(
                    name="buffalo_l",
                    root=str(MODELS_DIR),
                    providers=providers,
                )
                app.prepare(
                    ctx_id=0 if self._use_gpu else -1, det_size=(640, 640)
                )
            else:
                # Low-resource system: load only essential models (detection + recognition)
                # Skip genderage and 3d landmark models to reduce memory usage
                print(
                    "[FaceManager] Low-resource system detected, loading InsightFace models sequentially..."
                )
                gc.collect()

                # Force garbage collection before loading
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except ImportError:
                    pass

                # Only load detection and recognition models
                # This skips: genderage.onnx, 1k3d68.onnx, 2d106det.onnx
                app = face_analysis_cls(
                    name="buffalo_l",
                    root=str(MODELS_DIR),
                    providers=providers,
                    allowed_modules=[
                        "detection",
                        "recognition",
                    ],  # Skip age/gender/3d landmarks
                )

                # Use smaller detection size to reduce memory
                det_size = (480, 480) if caps["ram_gb"] < 16 else (640, 640)
                app.prepare(
                    ctx_id=0 if self._use_gpu else -1, det_size=det_size
                )

                # Clean up after model load
                gc.collect()

            self._insightface_app = app
            print(
                f"[FaceManager] SUCCESS: Using InsightFace ArcFace (512-dim). Providers: {providers}"
            )
            return True
        except Exception as e:
            error_msg = str(e).lower()
            if (
                "bad allocation" in error_msg
                or "out of memory" in error_msg
                or "oom" in error_msg
            ):
                print(
                    "[FaceManager] InsightFace OOM error, trying minimal config..."
                )
                gc.collect()

                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

                # Last resort: CPU only with minimal modules and smallest detection size
                try:
                    app = face_analysis_cls(
                        name="buffalo_l",
                        root=str(MODELS_DIR),
                        providers=["CPUExecutionProvider"],  # Force CPU
                        allowed_modules=["detection", "recognition"],
                    )
                    app.prepare(ctx_id=-1, det_size=(320, 320))  # Smallest size
                    self._insightface_app = app
                    print(
                        "[FaceManager] SUCCESS: InsightFace loaded with minimal config (CPU, 320x320)"
                    )
                    return True
                except Exception as e2:
                    print(
                        f"[FaceManager] InsightFace minimal config also failed: {e2}"
                    )
                    return False

            print(f"[FaceManager] CRITICAL: InsightFace init failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def _try_init_sface(self) -> bool:
        """Try to initialize SFace. Returns True on success."""
        try:
            _ensure_opencv_models()
            backend = cv2.dnn.DNN_BACKEND_DEFAULT
            target = cv2.dnn.DNN_TARGET_CPU

            if self._use_gpu:
                try:
                    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        backend = cv2.dnn.DNN_BACKEND_CUDA
                        target = cv2.dnn.DNN_TARGET_CUDA
                except Exception:
                    pass

            self._opencv_detector = cv2.FaceDetectorYN.create(
                model=str(MODELS_DIR / YUNET_MODEL),
                config="",
                input_size=(
                    settings.face_detection_resolution,
                    settings.face_detection_resolution,
                ),
                score_threshold=settings.face_detection_threshold,
                nms_threshold=0.3,
                top_k=5000,
                backend_id=backend,
                target_id=target,
            )
            self._opencv_recognizer = cv2.FaceRecognizerSF.create(
                model=str(MODELS_DIR / SFACE_MODEL),
                config="",
                backend_id=backend,
                target_id=target,
            )
            print("[FaceManager] Using SFace (128-dim) - FALLBACK")
            return True
        except Exception as e:
            print(f"[FaceManager] SFace init failed: {e}")
            return False

    async def _init_yunet_only(self) -> None:
        """Initialize YuNet for detection only (no embeddings)."""
        try:
            _ensure_opencv_models()
            self._opencv_detector = cv2.FaceDetectorYN.create(
                model=str(MODELS_DIR / YUNET_MODEL),
                config="",
                input_size=(
                    settings.face_detection_resolution,
                    settings.face_detection_resolution,
                ),
                score_threshold=settings.face_detection_threshold,
                nms_threshold=0.3,
                top_k=5000,
            )
            print("[FaceManager] Using YuNet detection only - NO EMBEDDINGS")
        except Exception as e:
            print(f"[FaceManager] YuNet init failed: {e}")

    @observe("face_detect")
    async def detect_faces(self, image_path: Path | str) -> list[DetectedFace]:
        """Detect faces in an image with automatic model selection."""
        await self._lazy_init()
        path = Path(image_path)
        image = await asyncio.to_thread(self._load_image, path)

        if image.size == 0:
            return []

        if self._model_type == "insightface":
            return await self._detect_insightface(image)
        elif self._model_type == "sface":
            return await self._detect_sface(image)
        else:
            return await self._detect_yunet_only(image)

    @observe("face_detect_batch")
    async def detect_faces_batch(
        self, image_paths: list[Path | str | NDArray[np.uint8]]
    ) -> list[list[DetectedFace]]:
        """Detect faces in a batch of images (optimized)."""
        await self._lazy_init()
        
        # Parallel Async Image Loading (Fixes Blocking I/O)
        tasks = []
        for p in image_paths:
            if isinstance(p, (str, Path)):
                # Correctly load image synchronously in thread
                tasks.append(asyncio.to_thread(self._load_image, Path(p)))
            else:
                async def _noop(img): return img
                tasks.append(_noop(p))
        
        images = await asyncio.gather(*tasks)
                
        results = []
        
        # Resource Arbiter manages VRAM + Locking
        from core.utils.resource_arbiter import RESOURCE_ARBITER

        # Process in chunks
        for i in range(0, len(images), self.batch_size):
            chunk = images[i : i + self.batch_size]
            chunk_results = []
            
            # Acquire GPU lock & Register VRAM usage
            # InsightFace = ~1.5GB
            async with RESOURCE_ARBITER.acquire("insightface", vram_gb=1.5):
                for img in chunk:
                     if self._model_type == "insightface":
                        # Direct call to avoid double-locking deadlock
                        # _detect_insightface acquires lock, so we copy logic here
                        if img.size == 0:
                            chunk_results.append([])
                            continue
                            
                        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        # Inference in thread
                        try:
                            faces = await asyncio.to_thread(self._insightface_app.get, bgr)
                            # Process results
                            res = []
                            for face in faces:
                                bbox = face.bbox.astype(int)
                                box = (int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[0]))
                                det_score = float(face.det_score) if hasattr(face, "det_score") else 1.0
                                bbox_size = min(int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1]))
                                emb = face.embedding.astype(np.float64)
                                emb /= np.linalg.norm(emb) + 1e-9
                                
                                dface = DetectedFace(
                                    bbox=box,
                                    embedding=emb.tolist(),
                                    confidence=det_score
                                )
                                dface._bbox_size = bbox_size
                                res.append(dface)
                            chunk_results.append(res)
                        except Exception as e:
                            print(f"[FaceManager] Batch inference failed: {e}")
                            chunk_results.append([])

                     elif self._model_type == "sface":
                        chunk_results.append(await self._detect_sface(img))
                     else:
                        chunk_results.append(await self._detect_yunet_only(img))
            
            results.extend(chunk_results)
            
        return results

    async def _detect_insightface(self, image: NDArray[np.uint8]) -> list[DetectedFace]:
        if image.size == 0: return []
        assert self._insightface_app is not None, "InsightFace not initialized"
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        from core.utils.resource_arbiter import RESOURCE_ARBITER
        # Use Arbiter to track VRAM
        async with RESOURCE_ARBITER.acquire("insightface", vram_gb=1.5):
            # Run InsightFace in a thread to avoid blocking the event loop
            faces = await asyncio.to_thread(self._insightface_app.get, bgr)
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            box = (int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[0]))
            det_score = float(face.det_score) if hasattr(face, "det_score") else 1.0
            bbox_size = min(int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))
            emb = face.embedding.astype(np.float64)
            emb /= np.linalg.norm(emb) + 1e-9
            results.append(DetectedFace(bbox=box, embedding=emb.tolist(), confidence=det_score))
            results[-1]._bbox_size = bbox_size
        return results

    async def _detect_sface(self, image: NDArray[np.uint8]) -> list[DetectedFace]:
        if image.size == 0: return []
        assert self._opencv_detector is not None and self._opencv_recognizer is not None
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        async with GPU_SEMAPHORE:
            return await asyncio.to_thread(self._detect_sface_sync, bgr)

    def _detect_sface_sync(self, bgr: NDArray[np.uint8]) -> list[DetectedFace]:
        h, w = bgr.shape[:2]
        self._opencv_detector.setInputSize((w, h))
        _, dets = self._opencv_detector.detect(bgr)
        if dets is None: return []
        results = []
        for d in dets:
            d_arr = np.array(d)
            x, y, bw, bh = d_arr[:4]
            box = (int(y), int(x + bw), int(y + bh), int(x))
            face_box = np.array([[x, y, bw, bh]], dtype=np.int32)
            aligned = self._opencv_recognizer.alignCrop(bgr, face_box)
            lab = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
            aligned = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            emb = self._opencv_recognizer.feature(aligned).reshape(-1).astype(np.float64)
            emb /= np.linalg.norm(emb) + 1e-9
            emb_padded = np.zeros(512, dtype=np.float64)
            emb_padded[: len(emb)] = emb
            results.append(DetectedFace(bbox=box, embedding=emb_padded.tolist(), confidence=float(d[14]) if len(d)>14 else 0.9))
            results[-1]._bbox_size = int(min(bw, bh))
        return results

    async def _detect_yunet_only(self, image: NDArray[np.uint8]) -> list[DetectedFace]:
        if image.size == 0: return []
        assert self._opencv_detector is not None
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        async with GPU_SEMAPHORE:
            def _inf():
                h, w = bgr.shape[:2]
                self._opencv_detector.setInputSize((w, h))
                return self._opencv_detector.detect(bgr)
            _, dets = await asyncio.to_thread(_inf)
        if dets is None: return []
        results = []
        for d in dets:
            x, y, bw, bh = d[:4]
            results.append(DetectedFace(bbox=(int(y), int(x+bw), int(y+bh), int(x)), embedding=None))
            results[-1]._bbox_size = int(min(bw, bh))
        return results

    @observe("face_cluster")
    def cluster_faces(self, encodings: Sequence[ArrayLike]) -> NDArray[np.int64]:
        if not encodings: return np.array([], dtype=np.int64)
        return HDBSCAN(min_cluster_size=max(3, self.dbscan_min_samples), min_samples=self.dbscan_min_samples,
                       cluster_selection_epsilon=self.dbscan_eps, metric=self.dbscan_metric,
                       allow_single_cluster=True).fit_predict(self._to_2d_array(encodings))

    def match_or_create_cluster(self, embedding: list[float], existing_clusters: dict[int, list[float]], threshold: float = 0.4) -> tuple[int, dict[int, list[float]]]:
        emb = np.array(embedding, dtype=np.float64)
        emb_norm = emb / (np.linalg.norm(emb) + 1e-9)
        best_id, best_sim = None, -1.0
        for cid, centroid in existing_clusters.items():
            cen_norm = np.array(centroid) / (np.linalg.norm(centroid) + 1e-9)
            sim = float(np.dot(emb_norm, cen_norm))
            if sim > (1.0 - threshold) and sim > best_sim:
                best_id, best_sim = cid, sim
        if best_id is not None and best_sim < 0.6:
            try:
                from core.processing.biometric_arbitrator import BIOMETRIC_ARBITRATOR
                if not BIOMETRIC_ARBITRATOR.should_merge_sync(emb_norm, np.array(existing_clusters[best_id]), primary_sim=best_sim):
                    best_id = None
            except ImportError: pass
        if best_id is not None:
            new_cen = (np.array(existing_clusters[best_id]) + emb) / 2.0
            existing_clusters[best_id] = (new_cen / (np.linalg.norm(new_cen) + 1e-9)).tolist()
            return best_id, existing_clusters
        nid = max(existing_clusters.keys(), default=0) + 1
        existing_clusters[nid] = emb_norm.tolist()
        return nid, existing_clusters

    def resolve_identity_conflict(self, track_id: int, crop: NDArray[np.uint8], known: dict[int, NDArray[np.float32]]) -> int:
        try:
            from core.processing.biometrics import get_biometric_arbitrator
            arb = get_biometric_arbitrator()
            emb = arb.get_embedding(crop)
            if emb is None: return track_id
            if track_id in known and arb.verify_identity(emb, known[track_id]): return track_id
            mid = arb.find_matching_identity(emb, known)
            return mid if mid is not None else track_id
        except ImportError: return track_id

    @staticmethod
    async def _load_image(path: Path) -> NDArray[np.uint8]:
        if not path.exists(): return np.array([], dtype=np.uint8)
        def _read():
            try:
                with Image.open(path) as img: return np.asarray(img.convert("RGB"), dtype=np.uint8)
            except Exception: return np.array([], dtype=np.uint8)
        return await asyncio.to_thread(_read)

    def process_video_frames(self, frames: Iterable[NDArray[np.uint8]]) -> list[list[DetectedFace]]:
        out = []
        for batch in self._batch(frames, self.batch_size):
            out.extend(asyncio.run(self.detect_faces_batch(batch)))
        return out

    T = TypeVar("T")

    @staticmethod
    def _batch(it: Iterable[T], sz: int) -> Iterable[list[T]]:
        batch = []
        for i in it:
            batch.append(i)
            if len(batch) == sz: yield batch; batch = []
        if batch: yield batch

    def _cache_key(self, image: NDArray[Any], box: tuple[int, int, int, int]) -> str:
        return f"{hashlib.sha1(image.data[:2048]).hexdigest()}_{box}_{self.embedding_version}"

    def _disk_cache_get(self, key: str) -> NDArray[np.float64] | None:
        p = CACHE_DIR / f"{key}.npy"
        return np.load(p) if p.exists() else None

    def _disk_cache_put(self, key: str, val: NDArray[np.float64]) -> None:
        p = CACHE_DIR / f"{key}.npy"
        if not p.exists(): np.save(p, val)

    @staticmethod
    def _to_2d_array(encodings: Sequence[ArrayLike]) -> NDArray[np.float64]:
        arrs = [np.asarray(e, dtype=np.float64) for e in encodings]
        if len({a.shape for a in arrs}) > 1: raise ValueError("Inconsistent shapes")
        return np.vstack(arrs)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("image_path")
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    fm = FaceManager(use_gpu=args.gpu)
    faces = asyncio.run(fm.detect_faces(args.image_path))
    print(f"Model: {fm._model_type}, Found {len(faces)} faces")
    if faces:
        embs = [f.embedding for f in faces if f.embedding]
        if embs: print(fm.cluster_faces(embs))
