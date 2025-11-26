"""Face identity processing utilities.

This module provides the `FaceManager` class, which can:

1. Detect faces in an image and compute 128-d face embeddings using
   `face_recognition`.
2. Cluster those embeddings into identities using DBSCAN from scikit-learn.
"""

from __future__ import annotations


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
