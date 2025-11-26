"""Face identity processing utilities.

This module provides the `FaceManager` class, which can:

1. Detect faces in an image and compute 128-d face embeddings using
   `face_recognition`.
2. Cluster those embeddings into identities using DBSCAN from scikit-learn.
"""

from __future__ import annotations

from pathlib import Path

import face_recognition
from schemas import DetectedFace


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
