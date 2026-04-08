"""Mock implementations of storage protocols for isolated unit testing."""

from typing import Any

from core.ports.storage import StorageBackend


class MockStorageBackend(StorageBackend):
    """In-memory mock database that conforms to StorageBackend protocol."""

    def __init__(self):
        self.faces = []
        self.voices = []
        self.scenes = []
        self.frames = []
        self.media = []
        self.masklets = []

    async def get_face_clusters(self) -> list[dict[str, Any]]:
        return self.faces

    async def store_face_cluster(
        self,
        cluster_id: int,
        embeddings: list[list[float]],
        metadata: dict[str, Any],
    ) -> None:
        self.faces.append({"cluster_id": cluster_id, "metadata": metadata})

    async def get_voice_clusters(self) -> list[dict[str, Any]]:
        return self.voices

    async def store_voice_cluster(
        self, cluster_id: int, embedding: list[float], metadata: dict[str, Any]
    ) -> None:
        self.voices.append({"cluster_id": cluster_id, "metadata": metadata})

    def store_scene(
        self,
        media_path: str,
        start_time: float,
        end_time: float,
        visual_text: str = "",
        motion_text: str = "",
        dialogue_text: str = "",
        visual_features: list[float] | None = None,
        internvideo_features: list[float] | None = None,
        languagebind_features: list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.scenes.append(
            {
                "media_path": media_path,
                "start_time": start_time,
                "end_time": end_time,
                "payload": payload or {},
            }
        )

    def insert_masklet(
        self,
        video_path: str,
        concept: str,
        start_time: float,
        end_time: float,
        confidence: float,
        payload: dict,
        embedding: list[float],
    ) -> None:
        self.masklets.append(
            {
                "video_path": video_path,
                "concept": concept,
                "start_time": start_time,
                "end_time": end_time,
            }
        )

    def get_masklets_for_media(self, video_path: str) -> list[dict]:
        return [m for m in self.masklets if m["video_path"] == video_path]
