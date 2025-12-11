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

import cv2
import dlib
import numpy as np
from dlib import (
    cnn_face_detection_model_v1,  # type: ignore
    face_recognition_model_v1,  # type: ignore
    get_frontal_face_detector,  # type: ignore
    rectangle,  # type: ignore
    shape_predictor,  # type: ignore
)
from numpy.typing import ArrayLike, NDArray
from PIL import Image
from sklearn.cluster import DBSCAN

from config import Settings
from core.schemas import DetectedFace
from core.utils.logger import log

MODELS_DIR = Settings.project_root() / "models"
YUNET_MODEL_NAME = "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

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


def _ensure_models(models_dir: Path, use_cnn: bool = True) -> None:
    """Ensures all required Dlib and YuNet models are downloaded."""
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download Dlib models (BZ2 compressed)
    for name, (url, min_size) in DLIB_MODEL_URLS.items():
        if not use_cnn and name == "mmod_human_face_detector.dat":
            continue

        target = models_dir / name
        if target.exists() and target.stat().st_size >= min_size:
            continue

        log(f"[models] Downloading: {name}", file=sys.stderr)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            urllib.request.urlretrieve(url, tmp_path)
            with bz2.open(tmp_path, "rb") as src, open(target, "wb") as dst:
                dst.write(src.read())

            if target.stat().st_size < min_size:
                raise RuntimeError(f"Downloaded {name} looks corrupted (too small)")
            log(f"[models] Installed {name}", file=sys.stderr)
        finally:
            tmp_path.unlink(missing_ok=True)

    # 2. Download YuNet model (Rew ONNX)
    yunet_target = models_dir / YUNET_MODEL_NAME
    if not yunet_target.exists():
        log(f"[models] Downloading: {YUNET_MODEL_NAME}", file=sys.stderr)
        try:
            urllib.request.urlretrieve(YUNET_URL, yunet_target)
            log(f"[models] Installed {YUNET_MODEL_NAME}", file=sys.stderr)
        except Exception as e:
            log(f"[WARN] Failed to download YuNet model: {e}", file=sys.stderr)


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

        # Check for Dlib CUDA support
        try:
            self.dlib_has_cuda = dlib.cuda.get_num_devices() > 0  # type: ignore[attr-defined]
        except AttributeError:
            self.dlib_has_cuda = False

        # Determine strict GPU usage
        self.use_gpu = use_gpu and self.dlib_has_cuda

        _ensure_models(MODELS_DIR, use_cnn=self.use_gpu)

        self.yunet_path = str(MODELS_DIR / YUNET_MODEL_NAME)
        self.yunet_detector = None
        try:
            cv_has_cuda = False
            try:
                cv_has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
            except AttributeError:
                pass

            backend_id = (
                cv2.dnn.DNN_BACKEND_CUDA
                if (self.use_gpu and cv_has_cuda)
                else cv2.dnn.DNN_BACKEND_DEFAULT
            )
            target_id = (
                cv2.dnn.DNN_TARGET_CUDA
                if (self.use_gpu and cv_has_cuda)
                else cv2.dnn.DNN_TARGET_CPU
            )

            self.yunet_detector = cv2.FaceDetectorYN.create(
                model=self.yunet_path,
                config="",
                input_size=(320, 320),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000,
                backend_id=backend_id,
                target_id=target_id,
            )
        except Exception as e:
            log(f"[WARN] Failed to init YuNet: {e}. Fallback to Dlib HOG only.")

        # Initialize Dlib Detectors
        self.hog_detector = get_frontal_face_detector()
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
            raise FileNotFoundError(f"Image file not found: {path}")

        try:
            image = self._load_image(path)
        except Exception as exc:
            raise ValueError(f"Failed to load image: {path}") from exc

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

        # 1. Try Dlib CNN (GPU)
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
                boxes = []

        if not boxes and self.yunet_detector is not None:
            h, w = image.shape[:2]
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            self.yunet_detector.setInputSize((w, h))
            _, detections = self.yunet_detector.detect(image_bgr)

            if detections is not None:
                for det in detections:
                    x, y, w_box, h_box = det[:4]

                    left = int(x)
                    top = int(y)
                    right = int(x + w_box)
                    bottom = int(y + h_box)

                    left = max(0, left)
                    top = max(0, top)
                    right = min(w, right)
                    bottom = min(h, bottom)

                    boxes.append((top, right, bottom, left))

        # 3. Fallback to Dlib HOG (CPU)
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
