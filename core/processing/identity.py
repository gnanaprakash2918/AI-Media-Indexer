"""Face identity processing utilities.

This module provides the `FaceManager` class, which can:

1. Detect faces in an image and compute 128-d face embeddings using
   `face_recognition`.
2. Cluster those embeddings into identities using DBSCAN from scikit-learn.
"""

from __future__ import annotations

import bz2
import sys
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import dlib
import numpy as np
from dlib import (
    cnn_face_detection_model_v1,  # type: ignore
    face_recognition_model_v1,  # type: ignore
    get_frontal_face_detector,  # type: ignore
    rectangle,  # type: ignore
    shape_predictor,  # type: ignore
)
from mediapipe.python.solutions import face_detection as mp_face_detection
from numpy.typing import ArrayLike, NDArray
from PIL import Image
from sklearn.cluster import DBSCAN

from config import Settings
from core.schemas import DetectedFace
from core.utils.logger import log

# Where to store dlib model files relative to the project root.
MODELS_DIR = Settings.project_root() / "models"

DLIB_MODEL_URLS: dict[str, tuple[str, int]] = {
    "shape_predictor_68_face_landmarks.dat": (
        "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        10_000_000,
    ),
    "dlib_face_recognition_resnet_model_v1.dat": (
        "https://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
        10_000_000,
    ),
    "mmod_human_face_detector.dat": (
        "https://dlib.net/files/mmod_human_face_detector.dat.bz2",
        500_000,
    ),
}


def _ensure_dlib_models(models_dir: Path, use_cnn: bool = True) -> None:
    """Ensure required dlib model files exist; download missing ones."""
    models_dir.mkdir(parents=True, exist_ok=True)

    for name, (url, min_size) in DLIB_MODEL_URLS.items():
        if not use_cnn and name == "mmod_human_face_detector.dat":
            continue

        target = models_dir / name
        if target.exists() and target.stat().st_size >= min_size:
            continue

        log(f"[dlib] Downloading model: {name}", file=sys.stderr)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            urllib.request.urlretrieve(url, tmp_path)
            with bz2.open(tmp_path, "rb") as src, open(target, "wb") as dst:
                dst.write(src.read())

            if target.stat().st_size < min_size:
                raise RuntimeError(f"Downloaded {name} looks corrupted (too small)")

            log(f"[dlib] Installed {name}", file=sys.stderr)
        finally:
            tmp_path.unlink(missing_ok=True)


class FaceManager:
    """High-level interface for face detection and identity clustering."""

    def __init__(
        self,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 3,
        dbscan_metric: str = "euclidean",
        use_gpu: bool = False,
    ) -> None:
        """Initialize the FaceManager.

        Args:
            dbscan_eps: Maximum distance between two samples for them
                to be considered neighbors in DBSCAN.
            dbscan_min_samples: Minimum number of samples required to
                form a cluster in DBSCAN.
            dbscan_metric: Distance metric used by DBSCAN
                (e.g., "euclidean", "cosine").
            use_gpu: Whether to attempt using GPU for face detection.
                NOTE: On 8GB VRAM cards (like RTX 4060), running this
                alongside Ollama/Whisper may cause OOM.
        """
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.dbscan_metric = dbscan_metric

        # --- CUDA Check ---
        # Safely check for CUDA support to fix Pylance/Attribute errors
        self.dlib_has_cuda = False
        try:
            if dlib.cuda.get_num_devices() > 0:  # type: ignore[attr-defined]
                self.dlib_has_cuda = True
        except AttributeError:
            self.dlib_has_cuda = False

        # Only use GPU if requested AND dlib supports it
        self.use_gpu = use_gpu and self.dlib_has_cuda

        # Ensure models are available (download if needed).
        _ensure_dlib_models(MODELS_DIR, use_cnn=self.use_gpu)

        # 1. MediaPipe Detector (Primary CPU - VRAM Free & High Accuracy)
        # Use direct import so Pylance understands the attribute
        self.mp_detector = mp_face_detection.FaceDetection(
            model_selection=1,  # 0=Short-range, 1=Full-range (better for movies)
            min_detection_confidence=0.5,
        )

        # 2. HOG detector (Legacy CPU fallback)
        self.hog_detector = get_frontal_face_detector()

        # 3. CNN detector (GPU - High Accuracy but VRAM heavy)
        self.cnn_detector: cnn_face_detection_model_v1 | None = None
        if self.use_gpu:
            try:
                self.cnn_detector = cnn_face_detection_model_v1(
                    str(MODELS_DIR / "mmod_human_face_detector.dat")
                )
            except Exception as e:
                log(f"[WARN] Failed to load Dlib CNN detector: {e}. Fallback to CPU.")
                self.cnn_detector = None
                self.use_gpu = False

        # Landmark predictor and face embedding model (128-d vectors).
        self.shape_predictor = shape_predictor(
            str(MODELS_DIR / "shape_predictor_68_face_landmarks.dat")
        )
        self.face_rec_model = face_recognition_model_v1(
            str(MODELS_DIR / "dlib_face_recognition_resnet_model_v1.dat")
        )

    def detect_faces(self, image_path: Path | str) -> list[DetectedFace]:
        """Detect faces in an image and compute their 128-d encodings."""
        path = Path(image_path)

        if not path.exists():
            msg = f"Image file not found: {path}"
            raise FileNotFoundError(msg)

        try:
            image = self._load_image(path)
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to load image: {path}"
            raise ValueError(msg) from exc

        boxes = self._detect_face_boxes(image)

        if not boxes:
            return []

        encodings = self._compute_encodings(image, boxes)

        results: list[DetectedFace] = []
        for box, enc in zip(boxes, encodings, strict=True):
            top, right, bottom, left = box
            results.append(
                DetectedFace(
                    box=(int(top), int(right), int(bottom), int(left)),
                    encoding=enc.tolist(),
                ),
            )

        return results

    @staticmethod
    def _load_image(path: Path) -> NDArray[np.uint8]:
        """Load image as RGB uint8 numpy array."""
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    def _detect_face_boxes(
        self,
        image: NDArray[np.uint8],
    ) -> list[tuple[int, int, int, int]]:
        """Detect face bounding boxes using GPU -> MediaPipe -> HOG fallback."""
        boxes: list[tuple[int, int, int, int]] = []

        # Priority 1: Try CNN (GPU) if enabled.
        # This is the most accurate but risky for VRAM.
        if self.use_gpu and self.cnn_detector is not None:
            try:
                detections = self.cnn_detector(image, 1)
                boxes = [
                    (
                        int(d.rect.top()),
                        int(d.rect.right()),
                        int(d.rect.bottom()),
                        int(d.rect.left()),
                    )
                    for d in detections
                ]
            except Exception:
                # Silently fail on OOM/RuntimeError and fall through to MediaPipe
                boxes = []

        # Priority 2: MediaPipe (CPU).
        # Used if GPU is disabled, failed, or found nothing.
        # Accurate for side profiles and uses 0 VRAM.
        if not boxes:
            results = self.mp_detector.process(image)
            detections = getattr(results, "detections", None)

            if detections:
                h, w, _ = image.shape
                for detection in detections:
                    bboxc = detection.location_data.relative_bounding_box

                    # Convert MP relative coords to CSS absolute
                    # (top, right, bottom, left)
                    left = int(bboxc.xmin * w)
                    top = int(bboxc.ymin * h)
                    width = int(bboxc.width * w)
                    height = int(bboxc.height * h)

                    # Ensure coordinates are within image bounds
                    left = max(0, left)
                    top = max(0, top)
                    right = min(w, left + width)
                    bottom = min(h, top + height)

                    boxes.append((top, right, bottom, left))

        # Priority 3: HOG (Legacy CPU).
        # Only used if MediaPipe fails entirely (rare).
        if not boxes:
            detections = self.hog_detector(image, 1)
            boxes = [
                (
                    int(d.top()),
                    int(d.right()),
                    int(d.bottom()),
                    int(d.left()),
                )
                for d in detections
            ]

        return boxes

    def _compute_encodings(
        self,
        image: NDArray[np.uint8],
        boxes: list[tuple[int, int, int, int]],
    ) -> list[NDArray[np.float64]]:
        """Compute 128-d face encodings for given boxes (dlib ResNet)."""
        encodings: list[NDArray[np.float64]] = []

        for top, right, bottom, left in boxes:
            rect = rectangle(left, top, right, bottom)
            shape = self.shape_predictor(image, rect)
            descriptor = self.face_rec_model.compute_face_descriptor(image, shape)
            encodings.append(np.asarray(descriptor, dtype=np.float64))

        return encodings

    def cluster_faces(
        self,
        all_encodings: Sequence[ArrayLike],
    ) -> NDArray[np.int64]:
        """Cluster face encodings into identities using DBSCAN.

        Args:
            all_encodings: A sequence of 128-d face encodings.

        Returns:
            A 1D numpy array of shape (n_samples,) with the cluster label for
            each encoding. ``-1`` indicates noise or outliers.
        """
        if not all_encodings:
            return np.array([], dtype=np.int64)

        data = self._to_2d_array(all_encodings)

        dbscan = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric=self.dbscan_metric,
        )

        dbscan.fit(data)

        return dbscan.labels_

    @staticmethod
    def _to_2d_array(
        encodings: Sequence[ArrayLike],
    ) -> NDArray[np.float64]:
        """Validate and stack face encodings into a 2D numpy array.

        Args:
            encodings: A sequence of 128-d face encodings.

        Returns:
            A 2D numpy array of shape (n_samples, 128).
        """
        processed: list[NDArray[np.float64]] = []

        for idx, enc in enumerate(encodings):
            arr = np.asarray(enc, dtype=np.float64)
            if arr.ndim != 1:
                msg = f"Encoding at index {idx} is not 1D. Got shape {arr.shape!r}."
                raise ValueError(msg)
            processed.append(arr)

        lengths = {arr.shape[0] for arr in processed}
        if len(lengths) != 1:
            msg = (
                f"Encodings have inconsistent lengths: {sorted(lengths)}. "
                "All encodings must be the same dimensionality."
            )
            raise ValueError(msg)

        return np.vstack(processed)
