"""Face identity processing utilities.

This module provides the `FaceManager` class, which can:

1. Detect faces in an image and compute 128-d face embeddings using
   `face_recognition`.
2. Cluster those embeddings into identities using DBSCAN from scikit-learn.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import face_recognition
import numpy as np
from numpy.typing import ArrayLike, NDArray
from schemas import DetectedFace
from sklearn.cluster import DBSCAN


class FaceManager:
    """High-level interface for face detection and identity clustering."""

    def __init__(
        self,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 3,
        dbscan_metric: str = "euclidean",
    ) -> None:
        """Initialize the FaceManager with default DBSCAN hyperparameters.

        Args:
            dbscan_eps: Maximum distance between two samples for them to be
                considered as in the same neighborhood.
            dbscan_min_samples: Minimum number of samples required to form a
                dense region (cluster).
            dbscan_metric: Distance metric to use for DBSCAN.
        """
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.dbscan_metric = dbscan_metric

    def detect_faces(self, image_path: Path | str) -> list[DetectedFace]:
        """Detect faces in an image and compute their 128-d encodings.

        Args:
            image_path: Path to the input image file.

        Returns:
            A list of dictionaries, one per detected face. Each dict has:
            - ``box``: (top, right, bottom, left) bounding box in pixels.
            - ``encoding``: list[float] of length 128.
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        try:
            image = face_recognition.load_image_file(str(path))
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to load image: {path}"
            raise ValueError(msg) from exc

        boxes = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, boxes)

        results: list[DetectedFace] = []
        for loc, enc in zip(boxes, encodings, strict=False):
            top, right, bottom, left = loc
            results.append(
                DetectedFace(
                    box=(int(top), int(right), int(bottom), int(left)),
                    encoding=enc.tolist(),
                )
            )

        return results

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
            msg = "No encodings provided for clustering."
            raise ValueError(msg)

        encodings_array = self._to_2d_array(all_encodings)

        dbscan = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric=self.dbscan_metric,
        )
        dbscan.fit(encodings_array)

        return dbscan.labels_

    @staticmethod
    def _to_2d_array(encodings: Sequence[ArrayLike]) -> NDArray[np.float64]:
        """Convert a sequence of encodings into a 2D numpy array.

        Args:
            encodings: Sequence of encodings (each 1D, length 128).

        Returns:
            2D numpy array of shape (n_samples, 128).
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
