"""Face identity processing utilities with graceful fallback chain.

Model priority:
1. InsightFace ArcFace (512-dim, best accuracy, requires onnxruntime)
2. SFace (128-dim, OpenCV built-in, good fallback)
3. Basic YuNet detection only (no embeddings - last resort)
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final, Literal

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image
from sklearn.cluster import DBSCAN

from config import settings
from core.schemas import DetectedFace
from core.utils.observe import observe

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

GPU_SEMAPHORE = asyncio.Semaphore(1)

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
    ) -> None:
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.dbscan_metric = dbscan_metric
        self.max_fps = max_fps
        self.batch_size = batch_size
        self._use_gpu = use_gpu
        self._initialized = False
        self._init_lock = asyncio.Lock()
        
        # Model references
        self._model_type: ModelType = "yunet_only"
        self._insightface_app = None
        self._opencv_detector = None
        self._opencv_recognizer = None
        self._embedding_dim = 128  # Updated during init
        
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
        """Try to initialize InsightFace. Returns True on success."""
        FaceAnalysis = _try_import_insightface()
        if FaceAnalysis is None:
            print("[FaceManager] InsightFace library not found. Falling back.")
            return False
        
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self._use_gpu else ["CPUExecutionProvider"]
            # Buffalo_l is the 512-dim model with highest accuracy
            app = FaceAnalysis(
                name="buffalo_l",
                root=str(MODELS_DIR),
                providers=providers,
            )
            app.prepare(ctx_id=0 if self._use_gpu else -1, det_size=(640, 640))
            self._insightface_app = app
            print(f"[FaceManager] SUCCESS: Using InsightFace ArcFace (512-dim). Providers: {providers}")
            return True
        except Exception as e:
            print(f"[FaceManager] CRITICAL: InsightFace init failed despite library presence: {e}")
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
                input_size=(settings.face_detection_resolution, settings.face_detection_resolution),
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
                input_size=(settings.face_detection_resolution, settings.face_detection_resolution),
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
        image = self._load_image(path)
        
        if self._model_type == "insightface":
            return await self._detect_insightface(image)
        elif self._model_type == "sface":
            return await self._detect_sface(image)
        else:
            return await self._detect_yunet_only(image)

    async def _detect_insightface(self, image: NDArray[np.uint8]) -> list[DetectedFace]:
        """Detect faces using InsightFace with quality metrics."""
        assert self._insightface_app is not None, "InsightFace not initialized"
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        async with GPU_SEMAPHORE:
            faces = self._insightface_app.get(bgr)
        
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            # Convert to (top, right, bottom, left) format
            box = (int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[0]))
            
            # Quality metrics for clustering
            det_score = float(face.det_score) if hasattr(face, 'det_score') else 1.0
            bbox_width = int(bbox[2] - bbox[0])
            bbox_height = int(bbox[3] - bbox[1])
            bbox_size = min(bbox_width, bbox_height)
            
            # Get embedding and normalize
            emb = face.embedding.astype(np.float64)
            emb /= np.linalg.norm(emb) + 1e-9
            
            results.append(DetectedFace(
                bbox=box, 
                embedding=emb.tolist(),
                confidence=det_score,  # Store detection confidence
                # Note: bbox_size will be passed separately to insert_face
            ))
            # Store bbox_size as an attribute for pipeline to access
            results[-1]._bbox_size = bbox_size
        
        return results

    async def _detect_sface(self, image: NDArray[np.uint8]) -> list[DetectedFace]:
        """Detect faces using SFace with CLAHE normalization and quality metrics."""
        assert self._opencv_detector is not None, "OpenCV detector not initialized"
        assert self._opencv_recognizer is not None, "OpenCV recognizer not initialized"
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        async with GPU_SEMAPHORE:
            h, w = bgr.shape[:2]
            self._opencv_detector.setInputSize((w, h))
            _, dets = self._opencv_detector.detect(bgr)
        
        if dets is None:
            return []
        
        results = []
        for d in dets:
            x, y, bw, bh = d[:4]
            box = (int(y), int(x + bw), int(y + bh), int(x))
            
            # Quality metrics for clustering
            det_score = float(d[14]) if len(d) > 14 else 0.9  # YuNet confidence at index 14
            bbox_size = int(min(bw, bh))
            
            # Get face crop and apply CLAHE for lighting normalization
            face_box = np.array([[x, y, bw, bh]], dtype=np.int32)
            aligned = self._opencv_recognizer.alignCrop(bgr, face_box)
            
            # CLAHE on L channel
            lab = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            aligned = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Get embedding and normalize
            emb = self._opencv_recognizer.feature(aligned).reshape(-1).astype(np.float64)
            emb /= np.linalg.norm(emb) + 1e-9
            
            # Pad 128-dim to 512-dim for Qdrant compatibility
            emb_padded = np.zeros(512, dtype=np.float64)
            emb_padded[:len(emb)] = emb
            
            results.append(DetectedFace(
                bbox=box, 
                embedding=emb_padded.tolist(),
                confidence=det_score,
            ))
            # Store bbox_size as an attribute for pipeline to access
            results[-1]._bbox_size = bbox_size
        
        return results

    async def _detect_yunet_only(self, image: NDArray[np.uint8]) -> list[DetectedFace]:
        """Detect faces with YuNet only (no embeddings)."""
        assert self._opencv_detector is not None, "OpenCV detector not initialized"
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        async with GPU_SEMAPHORE:
            h, w = bgr.shape[:2]
            self._opencv_detector.setInputSize((w, h))
            _, dets = self._opencv_detector.detect(bgr)
        
        if dets is None:
            return []
        
        results = []
        for d in dets:
            x, y, bw, bh = d[:4]
            box = (int(y), int(x + bw), int(y + bh), int(x))
            results.append(DetectedFace(bbox=box, embedding=None))
        
        return results

    @observe("face_cluster")
    def cluster_faces(self, all_encodings: Sequence[ArrayLike]) -> NDArray[np.int64]:
        """Cluster face encodings using DBSCAN."""
        if not all_encodings:
            return np.array([], dtype=np.int64)
        data = self._to_2d_array(all_encodings)
        return DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric=self.dbscan_metric,
        ).fit_predict(data)

    def match_or_create_cluster(
        self,
        embedding: list[float],
        existing_clusters: dict[int, list[float]],
        threshold: float = 0.4,
    ) -> tuple[int, dict[int, list[float]]]:
        """Match face to existing cluster or create new one (incremental clustering).

        This solves the hash-based ID problem by using actual embedding similarity.
        Each cluster is represented by its centroid embedding.

        Args:
            embedding: The 512-dim face embedding to match.
            existing_clusters: Dict mapping cluster_id -> centroid embedding.
            threshold: Cosine distance threshold (0.4 = 60% similarity required).

        Returns:
            Tuple of (cluster_id, updated_clusters_dict).
        """
        import numpy as np

        emb = np.array(embedding, dtype=np.float64)
        emb_norm = emb / (np.linalg.norm(emb) + 1e-9)

        best_cluster_id = None
        best_similarity = -1.0

        for cluster_id, centroid in existing_clusters.items():
            centroid_arr = np.array(centroid, dtype=np.float64)
            centroid_norm = centroid_arr / (np.linalg.norm(centroid_arr) + 1e-9)

            # Cosine similarity
            similarity = float(np.dot(emb_norm, centroid_norm))

            # Convert to distance: distance = 1 - similarity
            # threshold of 0.4 means we need similarity >= 0.6
            if similarity > (1.0 - threshold) and similarity > best_similarity:
                best_similarity = similarity
                best_cluster_id = cluster_id

        if best_cluster_id is not None:
            # Update centroid with running average
            old_centroid = np.array(existing_clusters[best_cluster_id], dtype=np.float64)
            # Simple average (could weight by count for better accuracy)
            new_centroid = (old_centroid + emb) / 2.0
            new_centroid = new_centroid / (np.linalg.norm(new_centroid) + 1e-9)
            existing_clusters[best_cluster_id] = new_centroid.tolist()
            return best_cluster_id, existing_clusters

        # Create new cluster
        new_cluster_id = max(existing_clusters.keys(), default=0) + 1
        existing_clusters[new_cluster_id] = emb_norm.tolist()
        return new_cluster_id, existing_clusters

    def resolve_identity_conflict(
        self,
        track_id: int,
        current_crop: NDArray[np.uint8],
        known_identities: dict[int, NDArray[np.float32]],
    ) -> int:
        """Verify track ID against biometric identity, correcting if needed.
        
        Args:
            track_id: ID suggested by visual tracker.
            current_crop: BGR face crop from current frame.
            known_identities: Dict mapping cluster_id -> ArcFace embedding.
            
        Returns:
            Verified cluster_id (may differ from track_id if biometrics disagree).
        """
        from core.processing.biometrics import get_biometric_arbitrator
        
        arbitrator = get_biometric_arbitrator()
        current_emb = arbitrator.get_embedding(current_crop)
        
        if current_emb is None:
            return track_id
            
        # Check if claimed identity matches
        if track_id in known_identities:
            if arbitrator.verify_identity(current_emb, known_identities[track_id]):
                return track_id
            print(f"[FaceManager] Identity conflict: track {track_id} rejected by biometrics")
        
        # Search for correct identity
        match_id = arbitrator.find_matching_identity(current_emb, known_identities)
        if match_id is not None:
            print(f"[FaceManager] Identity corrected: {track_id} -> {match_id}")
            return match_id
            
        return track_id


    @staticmethod
    def _load_image(path: Path) -> NDArray[np.uint8]:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    def process_video_frames(
        self, frames: Iterable[NDArray[np.uint8]]
    ) -> list[list[DetectedFace]]:
        """Process video frames to detect and encode faces."""
        out = []
        last = 0.0
        interval = 1.0 / self.max_fps
        for batch in self._batch(frames, self.batch_size):
            now = time.perf_counter()
            if now - last < interval:
                time.sleep(interval - (now - last))
            last = time.perf_counter()
            for frame in batch:
                faces = asyncio.run(self._detect_frame(frame))
                out.append(faces)
        return out

    @observe("face_detect_frame")
    async def _detect_frame(self, frame: NDArray[np.uint8]) -> list[DetectedFace]:
        """Detect faces in a single frame."""
        await self._lazy_init()
        
        if self._model_type == "insightface":
            return await self._detect_insightface(frame)
        elif self._model_type == "sface":
            return await self._detect_sface(frame)
        else:
            return await self._detect_yunet_only(frame)

    @staticmethod
    def _batch(
        iterable: Iterable[NDArray[np.uint8]],
        size: int,
    ) -> Iterable[list[NDArray[np.uint8]]]:
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _cache_key(
        self, image: NDArray[Any], box: tuple[int, int, int, int]
    ) -> str:
        h = hashlib.sha1(image.data[:2048]).hexdigest()
        return f"{h}_{box}_{self.embedding_version}"

    def _disk_cache_get(self, key: str) -> NDArray[np.float64] | None:
        path = CACHE_DIR / f"{key}.npy"
        if path.exists():
            return np.load(path)
        return None

    def _disk_cache_put(self, key: str, value: NDArray[np.float64]) -> None:
        path = CACHE_DIR / f"{key}.npy"
        if not path.exists():
            np.save(path, value)

    @staticmethod
    def _to_2d_array(encodings: Sequence[ArrayLike]) -> NDArray[np.float64]:
        arrs = [np.asarray(e, dtype=np.float64) for e in encodings]
        shapes = {a.shape for a in arrs}
        if len(shapes) != 1:
            raise ValueError(shapes)
        return np.vstack(arrs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    fm = FaceManager(use_gpu=args.gpu)
    faces = asyncio.run(fm.detect_faces(args.image_path))
    print(f"Model: {fm._model_type}, Embedding dim: {fm._embedding_dim}")
    print(f"Detected {len(faces)} faces")
    if faces:
        valid_embeddings = [f.embedding for f in faces if f.embedding is not None]
        if valid_embeddings:
            print(f"Clustering {len(valid_embeddings)} embeddings...")
            print(fm.cluster_faces(valid_embeddings))
