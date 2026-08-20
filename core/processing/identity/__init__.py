"""Identity processing package."""

from typing import Any

from .face_clusterer import FaceClusterer
from .face_detector import FaceDetector
from .face_embedder import FaceEmbedder
from .face_tracker import ActiveFaceTrack, FaceTrackBuilder


class FaceManager:
    """Facade composing face detection, embedding, tracking, and clustering."""

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
        self.detector = FaceDetector(
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            dbscan_metric=dbscan_metric,
            use_gpu=use_gpu,
            max_fps=max_fps,
            batch_size=batch_size,
            global_clusters=global_clusters,
            db_client=db_client,
        )
        self.clusterer = FaceClusterer(
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            metric=dbscan_metric,
        )
        self.embedder = FaceEmbedder()

    # Delegate detector methods
    def unload_gpu(self):
        return self.detector.unload_gpu()

    @property
    def embedding_version(self):
        return self.detector.embedding_version

    async def _lazy_init(self):
        return await self.detector._lazy_init()

    async def detect_faces(self, image_path):
        return await self.detector.detect_faces(image_path)

    async def detect_faces_batch(self, bgr_images):
        return await self.detector.detect_faces_batch(bgr_images)

    def process_video_frames(self, frames):
        return self.detector.process_video_frames(frames)

    # Delegate clusterer methods
    def cluster_faces(self, encodings):
        return self.clusterer.cluster_faces(encodings)

    def match_or_create_cluster(self, track_embedding):
        return self.clusterer.match_or_create_cluster(track_embedding)

    async def resolve_identity_conflict(self, face_id, voice_id):
        return await self.clusterer.resolve_identity_conflict(face_id, voice_id)

    # Delegate embedder methods
    def _cache_key(self, image, box):
        return self.embedder._cache_key(image, box, self.embedding_version)

    def _disk_cache_get(self, key):
        return self.embedder._disk_cache_get(key)

    def _disk_cache_put(self, key, val):
        return self.embedder._disk_cache_put(key, val)

    def _to_2d_array(self, encodings):
        return self.embedder._to_2d_array(encodings)


__all__ = [
    "FaceManager",
    "FaceTrackBuilder",
    "ActiveFaceTrack",
]
