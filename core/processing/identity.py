"""Face identity processing utilities (Async)."""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image
from sklearn.cluster import DBSCAN

from config import settings
from core.schemas import DetectedFace
from core.utils.observe import observe

MODELS_DIR = settings.project_root() / "models"
CACHE_DIR = settings.project_root() / ".face_cache"

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

EMBEDDING_VERSION: Final = "sface_v2_128d_l2"

GPU_SEMAPHORE = asyncio.Semaphore(1)


def _ensure_models() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in (
        (YUNET_MODEL, YUNET_URL),
        (SFACE_MODEL, SFACE_URL),
    ):
        path = MODELS_DIR / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)


def _migrate_cache(old_version: str, new_version: str) -> None:
    if not CACHE_DIR.exists():
        return
    for p in CACHE_DIR.glob(f"*_{old_version}.npy"):
        new = p.with_name(p.name.replace(old_version, new_version))
        if not new.exists():
            os.rename(p, new)


class FaceManager:
    """Manages face detection, recognition, and clustering operations (Async).

    This class handles loading of face detection (YuNet) and recognition (SFace)
    models, ensuring they are downloaded if missing. It supports lazy initialization
    to save resources until needed and can utilize GPU acceleration if available.
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
        """Initialize the FaceManager with configuration parameters.

        Args:
            dbscan_eps: The maximum distance between two samples for one to be
                considered as in the neighborhood of the other.
            dbscan_min_samples: The number of samples (or total weight) in a
                neighborhood for a point to be considered as a core point.
            dbscan_metric: The metric to use when calculating distance between
                instances in a feature array.
            use_gpu: Whether to attempt using a CUDA-enabled GPU for inference.
            max_fps: Maximum frames per second to process when batching generic
                inputs (simulated processing rate).
            batch_size: Number of frames to process in a single batch.
        """
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.dbscan_metric = dbscan_metric
        self.max_fps = max_fps
        self.batch_size = batch_size
        self._use_gpu_requested = use_gpu
        self._initialized = False
        self._init_lock = asyncio.Lock()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _ensure_models()
        _migrate_cache("sface_v1_128d_l2", EMBEDDING_VERSION)

    async def _lazy_init(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            backend = cv2.dnn.DNN_BACKEND_DEFAULT
            target = cv2.dnn.DNN_TARGET_CPU
            try:
                if self._use_gpu_requested and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    backend = cv2.dnn.DNN_BACKEND_CUDA
                    target = cv2.dnn.DNN_TARGET_CUDA
            except Exception:
                pass
            self.detector = cv2.FaceDetectorYN.create(
                model=str(MODELS_DIR / YUNET_MODEL),
                config="",
                input_size=(320, 320),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000,
                backend_id=backend,
                target_id=target,
            )
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=str(MODELS_DIR / SFACE_MODEL),
                config="",
                backend_id=backend,
                target_id=target,
            )
            self._initialized = True

    @observe("face_detect")
    async def detect_faces(self, image_path: Path | str) -> list[DetectedFace]:
        """Detect faces in a single image file asynchronously.

        Args:
            image_path: Path to the image file.

        Returns:
            A list of DetectedFace objects containing bounding boxes and
            encodings for all detected faces.
        """
        await self._lazy_init()
        path = Path(image_path)
        image = self._load_image(path)
        boxes = await self._detect_boxes(image)
        if not boxes:
            return []
        encodings = await self._compute_encodings(image, boxes)
        return [
            DetectedFace(bbox=b, embedding=e.tolist())
            for b, e in zip(boxes, encodings, strict=True)
        ]

    @observe("face_cluster")
    def cluster_faces(self, all_encodings: Sequence[ArrayLike]) -> NDArray[np.int64]:
        """Cluster face encodings using DBSCAN.

        Args:
            all_encodings: A sequence of face encodings (vectors).

        Returns:
            A numpy array of cluster labels for each encoding. -1 indicates
            noise (no cluster found).
        """
        if not all_encodings:
            return np.array([], dtype=np.int64)
        data = self._to_2d_array(all_encodings)
        return DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric=self.dbscan_metric,
        ).fit_predict(data)

    @staticmethod
    def _load_image(path: Path) -> NDArray[np.uint8]:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    async def _detect_boxes(
        self, image: NDArray[np.uint8]
    ) -> list[tuple[int, int, int, int]]:
        async with GPU_SEMAPHORE:
            h, w = image.shape[:2]
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.detector.setInputSize((w, h))
            _, dets = self.detector.detect(bgr)
        if dets is None:
            return []
        boxes = []
        for d in dets:
            x, y, bw, bh = d[:4]
            boxes.append(
                (
                    int(y),
                    int(x + bw),
                    int(y + bh),
                    int(x),
                )
            )
        return boxes

    async def _compute_encodings(
        self,
        image: NDArray[np.uint8],
        boxes: list[tuple[int, int, int, int]],
    ) -> list[NDArray[np.float64]]:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = []
        for box in boxes:
            key = self._cache_key(bgr, box)
            cached = self._disk_cache_get(key)
            if cached is not None:
                results.append(cached)
                continue
            async with GPU_SEMAPHORE:
                top, right, bottom, left = box
                face_box = np.array(
                    [[left, top, right - left, bottom - top]], dtype=np.int32
                )
                aligned = self.recognizer.alignCrop(bgr, face_box)
                emb = self.recognizer.feature(aligned).reshape(-1).astype(np.float64)
                emb /= np.linalg.norm(emb) + 1e-9
            self._disk_cache_put(key, emb)
            results.append(emb)
        return results

    def process_video_frames(
        self, frames: Iterable[NDArray[np.uint8]]
    ) -> list[list[DetectedFace]]:
        """Process a stream of video frames to detect and encode faces.

        Args:
            frames: An iterable of video frames as numpy arrays (RGB).

        Returns:
            A list where each element corresponds to a frame and contains a list
            of DetectedFace objects found in that frame.
        """
        out = []
        last = 0.0
        interval = 1.0 / self.max_fps
        for batch in self._batch(frames, self.batch_size):
            now = time.perf_counter()
            if now - last < interval:
                time.sleep(interval - (now - last))
            last = time.perf_counter()
            for frame in batch:
                boxes = asyncio.run(self._detect_boxes(frame))
                if not boxes:
                    out.append([])
                    continue
                encs = asyncio.run(self._compute_encodings(frame, boxes))
                out.append(
                    [
                        DetectedFace(bbox=b, embedding=e.tolist())
                        for b, e in zip(boxes, encs, strict=True)
                    ]
                )
        return out

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
        return f"{h}_{box}_{EMBEDDING_VERSION}"

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
    print(len(faces))
    if faces:
        valid_embeddings = [f.embedding for f in faces if f.embedding is not None]
        if valid_embeddings:
            print(fm.cluster_faces(valid_embeddings))
